"""REST endpoints for job lifecycle, uploads, media range streaming, speakers, and downloads."""

from __future__ import annotations

import copy
import json
import mimetypes
import os
import shutil
import tempfile
import uuid
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Body, Depends, File, Form, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.background import BackgroundTask

from aurisdeck.api.deps import (
    ApiError,
    get_coordinator,
    get_current_user,
    get_job_repo,
    get_paths,
    get_segment_repo,
)
from aurisdeck.db.connection import open_connection
from aurisdeck.db.repositories import (
    JobRecord,
    JobRepo,
    SegmentRecord,
    SegmentRepo,
    SpeakerRecord,
    TrackRecord,
)
from aurisdeck.domain.models import (
    ProfileAsrConfig,
    ProfileContextConfig,
    ProfileDiarizationConfig,
    ProfileHardwareConfig,
    ProfileInputConfig,
    ProfileOutputConfig,
    ProfilePreprocessingConfig,
    ProfileSchema,
    validate_slug,
)
from aurisdeck.export.exporters import build_batch_zip
from aurisdeck.media.ffmpeg import MediaInvalidError
from aurisdeck.media.ingest import ZipSlipError, unpack_multitrack_zip
from aurisdeck.storage.paths import DataPaths, sanitize_filename
from aurisdeck.storage.stores import (
    GlossaryStore,
    ProfileStore,
    ReplacementStore,
    build_effective_context,
)
from aurisdeck.workers.coordinator import Coordinator

router = APIRouter(prefix="/api/v1/jobs", dependencies=[Depends(get_current_user)])

_VALID_MODES = frozenset({"single", "multitrack", "batch"})
_VALID_FORMATS = ("vtt", "srt", "md", "json")
_ACTIVE_JOB_STATUSES = frozenset(
    {
        "queued",
        "preprocessing",
        "downloading_model",
        "loading_model",
        "transcribing",
        "diarizing",
        "merging",
        "exporting",
        "cancelling",
    }
)
_COMPLETED_JOB_STATUSES = frozenset({"completed", "completed_partial"})
_FORMAT_MEDIA_TYPES = {
    "vtt": "text/vtt; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "md": "text/markdown; charset=utf-8",
    "json": "application/json; charset=utf-8",
}
_UPLOAD_CHUNK_BYTES = 1024 * 1024


class SpeakersPatchRequest(BaseModel):
    """Payload for updating per-job speaker display aliases."""

    model_config = ConfigDict(extra="forbid")

    aliases: dict[str, str] = Field(default_factory=dict)


class DiarizeRequest(BaseModel):
    """Optional speaker-count constraints for post-hoc diarization."""

    model_config = ConfigDict(extra="forbid")

    min_speakers: int | None = Field(default=None, ge=1)
    max_speakers: int | None = Field(default=None, ge=1)


class ExportRequest(BaseModel):
    """Optional output format and subtitle styling overrides for re-export."""

    model_config = ConfigDict(extra="forbid")

    voice_style: Literal["v_tag", "prefix", "none"] | None = None
    formats: list[Literal["vtt", "srt", "md", "json"]] | None = Field(default=None, min_length=1)
    subtitles: dict[str, Any] | None = None


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _serialize_track(track: TrackRecord) -> dict[str, Any]:
    return {
        "id": track.id,
        "job_id": track.job_id,
        "track_index": track.track_index,
        "original_filename": track.original_filename,
        "media_relpath": track.media_relpath,
        "duration_ms": track.duration_ms,
        "offset_ms": track.offset_ms,
        "speaker_key": track.speaker_key,
        "asr_status": track.asr_status,
        "asr_checkpoint_relpath": track.asr_checkpoint_relpath,
    }


def _serialize_speaker(speaker: SpeakerRecord) -> dict[str, Any]:
    return {
        "job_id": speaker.job_id,
        "speaker_key": speaker.speaker_key,
        "display_name": speaker.display_name,
        "track_id": speaker.track_id,
    }


def _serialize_segment(segment: SegmentRecord) -> dict[str, Any]:
    return {
        "id": segment.id,
        "job_id": segment.job_id,
        "track_id": segment.track_id,
        "seq": segment.seq,
        "start_ms": segment.start_ms,
        "end_ms": segment.end_ms,
        "speaker_key": segment.speaker_key,
        "raw_text": segment.raw_text,
        "text": segment.text,
        "base_text": segment.base_text,
        "edited_text": segment.edited_text,
        "is_edited": segment.is_edited,
        "words": list(segment.words),
    }


def _serialize_job(job: JobRecord) -> dict[str, Any]:
    return {
        "id": job.id,
        "owner_user_id": job.owner_user_id,
        "visibility": job.visibility,
        "batch_id": job.batch_id,
        "collection_id": job.collection_id,
        "title": job.title,
        "mode": job.mode,
        "status": job.status,
        "diarization_status": job.diarization_status,
        "language_requested": job.language_requested,
        "language_detected": job.language_detected,
        "source_type": job.source_type,
        "source_fingerprint": job.source_fingerprint,
        "media_retained": job.media_retained,
        "effective_config": copy.deepcopy(job.effective_config),
        "error_code": job.error_code,
        "error_params": copy.deepcopy(job.error_params) if job.error_params is not None else None,
        "error_detail": job.error_detail,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "completed_at": job.completed_at,
        "tracks": [_serialize_track(track) for track in job.tracks],
        "speakers": [_serialize_speaker(speaker) for speaker in job.speakers],
    }


def _get_job_or_404(job_repo: JobRepo, job_id: str) -> JobRecord:
    job = job_repo.get_job(job_id)
    if job is None:
        raise ApiError(
            status_code=404,
            error_code="JOB_NOT_FOUND",
            error_params={"job_id": job_id},
            detail=f"Job {job_id!r} not found",
        )
    return job


def _scan_available_formats(paths: DataPaths, job_id: str) -> list[str]:
    try:
        out_dir = paths.job_outputs_dir(job_id)
    except ValueError:
        return []
    if not out_dir.is_dir():
        return []

    present: set[str] = set()
    outputs_root = paths.outputs.resolve()
    for entry in out_dir.iterdir():
        resolved = entry.resolve()
        if not resolved.is_relative_to(outputs_root) or not resolved.is_file():
            continue
        ext = resolved.suffix.lower().lstrip(".")
        if ext in _VALID_FORMATS:
            present.add(ext)
    return [fmt for fmt in _VALID_FORMATS if fmt in present]


def _find_output_file(paths: DataPaths, job: JobRecord, fmt: str) -> Path | None:
    try:
        out_dir = paths.job_outputs_dir(job.id)
    except ValueError:
        return None
    if not out_dir.is_dir():
        return None

    outputs_root = paths.outputs.resolve()
    candidates = sorted(
        path
        for path in out_dir.glob(f"*.{fmt}")
        if path.resolve().is_relative_to(outputs_root) and path.resolve().is_file()
    )
    if not candidates:
        return None

    title_stem = sanitize_filename(job.title).lower() if job.title else ""
    for candidate in candidates:
        if title_stem and candidate.stem.lower() == title_stem:
            return candidate.resolve()
    return candidates[0].resolve()


def _resolve_base_profile(
    paths: DataPaths,
    user: Mapping[str, Any],
    profile_id: str | None,
) -> ProfileSchema:
    store = ProfileStore(paths)
    if profile_id is not None and profile_id.strip():
        candidate = profile_id.strip()
        try:
            slug = validate_slug(candidate)
        except ValueError as exc:
            raise ApiError(
                status_code=422,
                error_code="VALIDATION_ERROR",
                error_params={"profile_id": candidate},
                detail=str(exc),
            ) from exc
        profile = store.get(slug)
        if profile is None:
            raise ApiError(
                status_code=404,
                error_code="PROFILE_NOT_FOUND",
                error_params={"profile_id": slug},
                detail=f"Profile {slug!r} not found",
            )
        return profile

    prefs = user.get("preferences")
    preferred_id = prefs.get("default_profile_id") if isinstance(prefs, Mapping) else None
    if isinstance(preferred_id, str) and preferred_id.strip():
        try:
            preferred = store.get(validate_slug(preferred_id.strip()))
            if preferred is not None:
                return preferred
        except ValueError:
            pass

    fallback = store.get("es-tecnico")
    if fallback is not None:
        return fallback

    existing = store.list()
    if existing:
        return existing[0]

    return ProfileSchema(id="default", display_name="Default")


def _normalize_hotwords(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    raise ValueError("hotwords must be a list of strings or comma-separated string")


def _normalize_gpus_override(raw: Any) -> str | list[int]:
    if isinstance(raw, bool):
        raise TypeError("gpus cannot be a boolean")
    if raw in ("auto", "cpu"):
        return str(raw)
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if not parts:
            raise ValueError("gpus string cannot be empty")
        return [int(part) for part in parts]
    if isinstance(raw, list):
        return [int(item) for item in raw]
    raise ValueError(f"Unsupported gpus value: {raw!r}")


def _normalize_language(raw: Any) -> str | None:
    if raw is None:
        return None
    cleaned = str(raw).strip()
    if not cleaned or cleaned.lower() == "auto":
        return None
    return cleaned


def _apply_config_overrides(
    base_cfg: Mapping[str, Any],
    overrides: Mapping[str, Any],
    *,
    paths: DataPaths,
    mode: str | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(base_cfg))
    cfg.setdefault("schema_version", 1)
    cfg.setdefault("input", {})
    cfg.setdefault("preprocessing", {})
    cfg.setdefault("asr", {})
    cfg.setdefault("diarization", {})
    cfg.setdefault("context", {})
    cfg.setdefault("hardware", {})
    cfg.setdefault("output", {})
    cfg["output"].setdefault("subtitles", {})

    if mode is not None:
        cfg["input"]["mode"] = mode

    if isinstance(overrides.get("preprocessing"), Mapping):
        cfg["preprocessing"].update(dict(overrides["preprocessing"]))
    if "normalize_audio" in overrides:
        cfg["preprocessing"]["normalize_audio"] = bool(overrides["normalize_audio"])

    if isinstance(overrides.get("asr"), Mapping):
        cfg["asr"].update(dict(overrides["asr"]))
    if "model" in overrides:
        cfg["asr"]["model"] = str(overrides["model"])
    if "beam_size" in overrides:
        cfg["asr"]["beam_size"] = overrides["beam_size"]
    if "vad_filter" in overrides:
        cfg["asr"]["vad_filter"] = bool(overrides["vad_filter"])
    if "word_timestamps" in overrides:
        cfg["asr"]["word_timestamps"] = bool(overrides["word_timestamps"])
    if "compute_type" in overrides:
        cfg["asr"]["compute_type"] = str(overrides["compute_type"])
    if "language" in overrides:
        cfg["asr"]["language"] = _normalize_language(overrides["language"])

    if isinstance(overrides.get("diarization"), Mapping):
        cfg["diarization"].update(dict(overrides["diarization"]))
    if "diarize" in overrides:
        cfg["diarization"]["enabled"] = bool(overrides["diarize"])
    if "min_speakers" in overrides:
        cfg["diarization"]["min_speakers"] = overrides["min_speakers"]
    if "max_speakers" in overrides:
        cfg["diarization"]["max_speakers"] = overrides["max_speakers"]

    if isinstance(overrides.get("context"), Mapping):
        cfg["context"].update(dict(overrides["context"]))
    if "initial_prompt" in overrides:
        cfg["context"]["initial_prompt"] = str(overrides["initial_prompt"] or "")
    if "hotwords" in overrides:
        cfg["context"]["hotwords"] = _normalize_hotwords(overrides["hotwords"])
    if "glossary_ids" in overrides:
        cfg["context"]["glossary_ids"] = list(overrides["glossary_ids"])
    if "replacement_ids" in overrides:
        cfg["context"]["replacement_ids"] = list(overrides["replacement_ids"])
    if "inline_replacements" in overrides:
        cfg["context"]["inline_replacements"] = dict(overrides["inline_replacements"])
    if "apply_replacements" in overrides:
        cfg["context"]["apply_replacements"] = bool(overrides["apply_replacements"])

    if isinstance(overrides.get("hardware"), Mapping):
        cfg["hardware"].update(dict(overrides["hardware"]))
    if "gpus" in overrides:
        cfg["hardware"]["gpus"] = _normalize_gpus_override(overrides["gpus"])
    if "cpu_fallback" in overrides:
        cfg["hardware"]["cpu_fallback"] = bool(overrides["cpu_fallback"])
    if "cpu_threads" in overrides:
        cfg["hardware"]["cpu_threads"] = overrides["cpu_threads"]

    if isinstance(overrides.get("output"), Mapping):
        output_override = dict(overrides["output"])
        subtitles_override = output_override.pop("subtitles", None)
        cfg["output"].update(output_override)
        if isinstance(subtitles_override, Mapping):
            cfg["output"]["subtitles"].update(dict(subtitles_override))
    if "formats" in overrides:
        cfg["output"]["formats"] = list(overrides["formats"])
    if isinstance(overrides.get("subtitles"), Mapping):
        cfg["output"]["subtitles"].update(dict(overrides["subtitles"]))
    if "voice_style" in overrides:
        cfg["output"]["subtitles"]["voice_style"] = overrides["voice_style"]

    if isinstance(overrides.get("speaker_aliases"), Mapping):
        cfg["speaker_aliases"] = {
            str(k): str(v) for k, v in overrides["speaker_aliases"].items()
        }
    if isinstance(overrides.get("track_offsets"), Mapping):
        cfg["track_offsets"] = {
            str(k): int(v) for k, v in overrides["track_offsets"].items()
        }

    # Validate each section via domain schemas before snapshotting.
    input_cfg = ProfileInputConfig.model_validate(cfg["input"])
    prep_cfg = ProfilePreprocessingConfig.model_validate(cfg["preprocessing"])
    asr_cfg = ProfileAsrConfig.model_validate(cfg["asr"])
    diar_cfg = ProfileDiarizationConfig.model_validate(cfg["diarization"])
    ctx_cfg = ProfileContextConfig.model_validate(cfg["context"])
    hw_cfg = ProfileHardwareConfig.model_validate(cfg["hardware"])
    out_cfg = ProfileOutputConfig.model_validate(cfg["output"])

    cfg["input"] = input_cfg.model_dump(mode="json")
    cfg["preprocessing"] = prep_cfg.model_dump(mode="json")
    cfg["asr"] = asr_cfg.model_dump(mode="json")
    cfg["diarization"] = diar_cfg.model_dump(mode="json")
    cfg["context"] = ctx_cfg.model_dump(mode="json")
    cfg["hardware"] = hw_cfg.model_dump(mode="json")
    cfg["output"] = out_cfg.model_dump(mode="json")

    snapshot_profile = ProfileSchema(
        id="effective",
        display_name="Effective",
        input=input_cfg,
        preprocessing=prep_cfg,
        asr=asr_cfg,
        diarization=diar_cfg,
        context=ctx_cfg,
        hardware=hw_cfg,
        output=out_cfg,
    )
    extra_hotwords = (
        _normalize_hotwords(overrides.get("extra_hotwords"))
        if "extra_hotwords" in overrides
        else None
    )
    extra_prompt = (
        str(overrides["extra_prompt"]) if overrides.get("extra_prompt") is not None else None
    )
    effective_ctx = build_effective_context(
        snapshot_profile,
        glossary_store=GlossaryStore(paths),
        replacement_store=ReplacementStore(paths),
        extra_hotwords=extra_hotwords,
        extra_prompt=extra_prompt,
    )
    cfg["effective_context"] = effective_ctx.model_dump(mode="json")
    return cfg


def _update_job_effective_config(
    paths: DataPaths,
    job_id: str,
    effective_config: Mapping[str, Any],
    *,
    language_requested: str | None = None,
    update_language: bool = False,
) -> None:
    now = _utc_now()
    payload = json.dumps(dict(effective_config), ensure_ascii=False)
    conn = open_connection(paths.db_path)
    try:
        with conn:
            if update_language:
                conn.execute(
                    """
                    UPDATE jobs
                    SET effective_config_json = ?, language_requested = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (payload, language_requested, now, job_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE jobs
                    SET effective_config_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (payload, now, job_id),
                )
    finally:
        conn.close()


async def _save_upload_file(upload: UploadFile, dest: Path, tmp_dir: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=tmp_dir,
            prefix=f".{dest.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            while chunk := await upload.read(_UPLOAD_CHUNK_BYTES):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, dest)
        tmp_path = None
    finally:
        await upload.close()
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return dest


def _resolve_track_override(
    mapping: Mapping[str, Any],
    *keys: str,
) -> Any | None:
    for key in keys:
        if key and key in mapping:
            return mapping[key]
    return None


def _resolve_track_speaker_name(
    speaker_aliases: Mapping[str, Any],
    *,
    default_name: str,
    keys: tuple[str, ...],
) -> str:
    raw = _resolve_track_override(speaker_aliases, *keys)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return default_name


def _resolve_track_offset_ms(
    track_offsets: Mapping[str, Any],
    *,
    default_offset_ms: int,
    keys: tuple[str, ...],
) -> int:
    raw = _resolve_track_override(track_offsets, *keys)
    if raw is None:
        return default_offset_ms
    if isinstance(raw, bool):
        raise TypeError("Track offset cannot be a boolean")
    offset = int(raw)
    if offset < 0:
        raise ValueError("Track offset cannot be negative")
    return offset


def _derive_job_title(raw_title: str | None, fallback_filename: str, default_label: str) -> str:
    if raw_title is not None and raw_title.strip():
        return raw_title.strip()
    stem = Path(sanitize_filename(fallback_filename)).stem
    return stem or default_label


def _cleanup_job_dirs(paths: DataPaths, job_id: str) -> None:
    for target in (
        paths.job_media_dir(job_id),
        paths.job_checkpoints_dir(job_id),
        paths.job_outputs_dir(job_id),
        paths.job_tmp_dir(job_id),
    ):
        shutil.rmtree(target, ignore_errors=True)


# ---------------------------------------------------------------------------
# Static / specific prefix routes (declared before /{job_id})
# ---------------------------------------------------------------------------


@router.post("/inspect-zip")
async def inspect_zip(
    file: Annotated[UploadFile, File(...)],
    paths: Annotated[DataPaths, Depends(get_paths)],
) -> list[dict[str, Any]]:
    """Inspect a multitrack ZIP archive in tmp/, returning track hints without creating a job."""
    inspect_dir = paths.tmp / f"inspect_{uuid.uuid4().hex}"
    zip_name = sanitize_filename(file.filename or "archive.zip")
    zip_path = inspect_dir / zip_name
    extract_dir = inspect_dir / "extracted"

    try:
        await _save_upload_file(file, zip_path, paths.tmp)
        extracted = await run_in_threadpool(unpack_multitrack_zip, zip_path, extract_dir)
    except (ZipSlipError, MediaInvalidError, ValueError, zipfile.BadZipFile) as exc:
        raise ApiError(
            status_code=400,
            error_code="INVALID_ARCHIVE",
            detail=str(exc),
        ) from exc
    finally:
        shutil.rmtree(inspect_dir, ignore_errors=True)

    return [
        {
            "filename": hint.filename,
            "speaker_name": hint.speaker_name,
            "offset_ms": hint.offset_ms,
        }
        for _, hint in extracted
    ]


@router.get("/batches/{batch_id}/download.zip")
async def download_batch_zip(
    batch_id: str,
    paths: Annotated[DataPaths, Depends(get_paths)],
    job_repo: Annotated[JobRepo, Depends(get_job_repo)],
    fmt: Annotated[str, Query()] = "vtt",
) -> FileResponse:
    """Build and download a ZIP archive containing completed outputs for a batch."""
    normalized_fmt = fmt.strip().lower().lstrip(".")
    if normalized_fmt not in _VALID_FORMATS:
        raise ApiError(
            status_code=400,
            error_code="VALIDATION_ERROR",
            error_params={"fmt": fmt},
            detail=f"Unsupported output format: {fmt!r}",
        )

    jobs = job_repo.list_jobs(batch_id=batch_id, limit=1000, offset=0)
    if not jobs:
        raise ApiError(
            status_code=404,
            error_code="JOB_NOT_FOUND",
            error_params={"batch_id": batch_id},
            detail=f"Batch {batch_id!r} not found",
        )

    output_files: list[Path] = []
    for job in jobs:
        if job.status not in _COMPLETED_JOB_STATUSES:
            continue
        found = _find_output_file(paths, job, normalized_fmt)
        if found is not None:
            output_files.append(found)

    if not output_files:
        raise ApiError(
            status_code=404,
            error_code="VALIDATION_ERROR",
            error_params={"batch_id": batch_id, "fmt": normalized_fmt},
            detail=f"No completed {normalized_fmt} outputs available for batch {batch_id!r}",
        )

    safe_batch = sanitize_filename(batch_id)
    bundle_dir = paths.tmp / f"batch_{safe_batch}_{uuid.uuid4().hex}"
    dest_zip = bundle_dir / f"{safe_batch}_{normalized_fmt}.zip"
    await run_in_threadpool(build_batch_zip, output_files, dest_zip, paths.tmp)

    return FileResponse(
        path=dest_zip,
        media_type="application/zip",
        filename=dest_zip.name,
        content_disposition_type="attachment",
        background=BackgroundTask(shutil.rmtree, bundle_dir, ignore_errors=True),
    )


# ---------------------------------------------------------------------------
# Collection routes (/api/v1/jobs)
# ---------------------------------------------------------------------------


@router.post("", status_code=201)
async def create_job(
    user: Annotated[dict[str, Any], Depends(get_current_user)],
    paths: Annotated[DataPaths, Depends(get_paths)],
    job_repo: Annotated[JobRepo, Depends(get_job_repo)],
    segment_repo: Annotated[SegmentRepo, Depends(get_segment_repo)],
    coordinator: Annotated[Coordinator, Depends(get_coordinator)],
    files: Annotated[list[UploadFile], File(...)],
    mode: Annotated[str, Form()] = "single",
    profile_id: Annotated[str | None, Form()] = None,
    overrides_json: Annotated[str | None, Form()] = None,
    title: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    """Create one or more transcription jobs in single, multitrack, or batch mode."""
    normalized_mode = mode.strip().lower()
    if normalized_mode not in _VALID_MODES:
        raise ApiError(
            status_code=422,
            error_code="VALIDATION_ERROR",
            error_params={"mode": mode},
            detail=f"Unsupported mode: {mode!r}",
        )
    if not files:
        raise ApiError(
            status_code=400,
            error_code="VALIDATION_ERROR",
            detail="At least one file is required",
        )
    if normalized_mode == "single" and len(files) != 1:
        raise ApiError(
            status_code=400,
            error_code="VALIDATION_ERROR",
            detail="Single mode requires exactly one file",
        )

    base_profile = _resolve_base_profile(paths, user, profile_id)

    overrides: dict[str, Any] = {}
    if overrides_json is not None and overrides_json.strip():
        try:
            parsed_overrides = json.loads(overrides_json)
        except json.JSONDecodeError as exc:
            raise ApiError(
                status_code=400,
                error_code="VALIDATION_ERROR",
                detail=f"Invalid overrides_json: {exc}",
            ) from exc
        if not isinstance(parsed_overrides, dict):
            raise ApiError(
                status_code=400,
                error_code="VALIDATION_ERROR",
                detail="overrides_json must be a JSON object",
            )
        overrides = parsed_overrides

    base_cfg = base_profile.model_dump(mode="json")
    base_cfg["profile_id"] = base_profile.id
    try:
        effective_config = _apply_config_overrides(
            base_cfg,
            overrides,
            paths=paths,
            mode=normalized_mode,
        )
    except (ValidationError, ValueError, TypeError) as exc:
        raise ApiError(
            status_code=422,
            error_code="VALIDATION_ERROR",
            detail=str(exc),
        ) from exc

    owner_user_id = str(user.get("id") or "default")
    language_requested = effective_config["asr"].get("language")
    speaker_aliases = effective_config.get("speaker_aliases", {})
    track_offsets = effective_config.get("track_offsets", {})

    if normalized_mode == "batch":
        batch_id = uuid.uuid4().hex
        created_jobs: list[dict[str, Any]] = []
        created_ids: list[str] = []
        try:
            for upload in files:
                job_id = uuid.uuid4().hex
                created_ids.append(job_id)
                track_id = uuid.uuid4().hex
                safe_name = sanitize_filename(upload.filename or "audio")
                media_dir = paths.job_media_dir(job_id)
                dest_path = media_dir / f"{track_id}_{safe_name}"
                await _save_upload_file(upload, dest_path, paths.tmp)
                media_relpath = dest_path.relative_to(paths.root).as_posix()

                job_title = (
                    _derive_job_title(title, safe_name, safe_name)
                    if len(files) == 1
                    else _derive_job_title(None, safe_name, safe_name)
                )
                job_repo.create_job(
                    title=job_title,
                    mode="batch",
                    effective_config=effective_config,
                    job_id=job_id,
                    owner_user_id=owner_user_id,
                    batch_id=batch_id,
                    language_requested=language_requested,
                )
                job_repo.add_track(
                    job_id=job_id,
                    track_index=0,
                    original_filename=safe_name,
                    media_relpath=media_relpath,
                    speaker_key="SPEAKER_00",
                    track_id=track_id,
                )
                if isinstance(speaker_aliases, Mapping) and "SPEAKER_00" in speaker_aliases:
                    alias = str(speaker_aliases["SPEAKER_00"]).strip()
                    if alias:
                        segment_repo.upsert_speakers(
                            job_id,
                            [
                                {
                                    "speaker_key": "SPEAKER_00",
                                    "display_name": alias,
                                    "track_id": track_id,
                                }
                            ],
                        )
                await run_in_threadpool(coordinator.enqueue_job, job_id)
                refreshed = job_repo.get_job(job_id)
                assert refreshed is not None
                created_jobs.append(_serialize_job(refreshed))
        except Exception:
            for jid in created_ids:
                job_repo.delete_job(jid)
                _cleanup_job_dirs(paths, jid)
            raise

        return {
            "batch_id": batch_id,
            "jobs": created_jobs,
        }

    job_id = uuid.uuid4().hex
    media_dir = paths.job_media_dir(job_id)

    if normalized_mode == "single":
        upload = files[0]
        track_id = uuid.uuid4().hex
        safe_name = sanitize_filename(upload.filename or "audio")
        dest_path = media_dir / f"{track_id}_{safe_name}"
        try:
            await _save_upload_file(upload, dest_path, paths.tmp)
            media_relpath = dest_path.relative_to(paths.root).as_posix()
            job_title = _derive_job_title(title, safe_name, "Transcription")

            job_repo.create_job(
                title=job_title,
                mode="single",
                effective_config=effective_config,
                job_id=job_id,
                owner_user_id=owner_user_id,
                language_requested=language_requested,
            )
            job_repo.add_track(
                job_id=job_id,
                track_index=0,
                original_filename=safe_name,
                media_relpath=media_relpath,
                speaker_key="SPEAKER_00",
                track_id=track_id,
            )
            if isinstance(speaker_aliases, Mapping) and speaker_aliases:
                speaker_rows = [
                    {
                        "speaker_key": str(key),
                        "display_name": str(val).strip(),
                        "track_id": track_id if str(key) == "SPEAKER_00" else None,
                    }
                    for key, val in speaker_aliases.items()
                    if str(val).strip()
                ]
                if speaker_rows:
                    segment_repo.upsert_speakers(job_id, speaker_rows)

            await run_in_threadpool(coordinator.enqueue_job, job_id)
            refreshed = job_repo.get_job(job_id)
            assert refreshed is not None
        except Exception:
            job_repo.delete_job(job_id)
            _cleanup_job_dirs(paths, job_id)
            raise

        serialized = _serialize_job(refreshed)
        return {**serialized, "job": serialized, "jobs": [serialized]}

    # mode == "multitrack"
    is_zip_upload = len(files) == 1 and Path(files[0].filename or "").suffix.lower() == ".zip"
    pending_tracks: list[tuple[int, str, str, str, int, str, str]] = []
    # Tuple fields: (idx, track_id, original_filename, media_relpath, offset_ms, speaker_key, display_name)

    try:
        if is_zip_upload:
            upload = files[0]
            tmp_upload_dir = paths.tmp / f"upload_{uuid.uuid4().hex}"
            zip_name = sanitize_filename(upload.filename or "multitrack.zip")
            zip_path = tmp_upload_dir / zip_name
            try:
                await _save_upload_file(upload, zip_path, paths.tmp)
                extracted = await run_in_threadpool(unpack_multitrack_zip, zip_path, media_dir)
            except (ZipSlipError, MediaInvalidError, ValueError, zipfile.BadZipFile) as exc:
                raise ApiError(
                    status_code=400,
                    error_code="INVALID_ARCHIVE",
                    detail=str(exc),
                ) from exc
            finally:
                shutil.rmtree(tmp_upload_dir, ignore_errors=True)

            if not extracted:
                raise ApiError(
                    status_code=400,
                    error_code="INVALID_ARCHIVE",
                    detail="ZIP archive contains no supported audio/video tracks",
                )

            job_title = _derive_job_title(title, zip_name, "Multitrack")
            for idx, (extracted_path, hint) in enumerate(extracted):
                track_id = uuid.uuid4().hex
                speaker_key = f"SPEAKER_{idx:02d}"
                safe_name = extracted_path.name
                final_path = media_dir / f"{track_id}_{safe_name}"
                os.replace(extracted_path, final_path)
                media_relpath = final_path.relative_to(paths.root).as_posix()

                lookup_keys = (
                    safe_name,
                    Path(safe_name).stem,
                    str(idx),
                    speaker_key,
                    hint.speaker_name,
                )
                try:
                    resolved_name = _resolve_track_speaker_name(
                        speaker_aliases,
                        default_name=hint.speaker_name or Path(safe_name).stem or speaker_key,
                        keys=lookup_keys,
                    )
                    resolved_offset = _resolve_track_offset_ms(
                        track_offsets,
                        default_offset_ms=hint.offset_ms,
                        keys=lookup_keys,
                    )
                except (ValueError, TypeError) as exc:
                    raise ApiError(
                        status_code=422,
                        error_code="VALIDATION_ERROR",
                        detail=str(exc),
                    ) from exc

                pending_tracks.append(
                    (
                        idx,
                        track_id,
                        safe_name,
                        media_relpath,
                        resolved_offset,
                        speaker_key,
                        resolved_name,
                    )
                )
        else:
            first_name = sanitize_filename(files[0].filename or "multitrack")
            job_title = _derive_job_title(title, first_name, "Multitrack")
            for idx, upload in enumerate(files):
                track_id = uuid.uuid4().hex
                speaker_key = f"SPEAKER_{idx:02d}"
                safe_name = sanitize_filename(upload.filename or f"track_{idx}")
                dest_path = media_dir / f"{track_id}_{safe_name}"
                await _save_upload_file(upload, dest_path, paths.tmp)
                media_relpath = dest_path.relative_to(paths.root).as_posix()

                stem_name = Path(safe_name).stem or speaker_key
                lookup_keys = (safe_name, stem_name, str(idx), speaker_key)
                try:
                    resolved_name = _resolve_track_speaker_name(
                        speaker_aliases,
                        default_name=stem_name,
                        keys=lookup_keys,
                    )
                    resolved_offset = _resolve_track_offset_ms(
                        track_offsets,
                        default_offset_ms=0,
                        keys=lookup_keys,
                    )
                except (ValueError, TypeError) as exc:
                    raise ApiError(
                        status_code=422,
                        error_code="VALIDATION_ERROR",
                        detail=str(exc),
                    ) from exc

                pending_tracks.append(
                    (
                        idx,
                        track_id,
                        safe_name,
                        media_relpath,
                        resolved_offset,
                        speaker_key,
                        resolved_name,
                    )
                )

        job_repo.create_job(
            title=job_title,
            mode="multitrack",
            effective_config=effective_config,
            job_id=job_id,
            owner_user_id=owner_user_id,
            language_requested=language_requested,
        )
        speaker_rows: list[dict[str, Any]] = []
        for idx, track_id, orig_name, relpath, offset_ms, speaker_key, display_name in pending_tracks:
            job_repo.add_track(
                job_id=job_id,
                track_index=idx,
                original_filename=orig_name,
                media_relpath=relpath,
                speaker_key=speaker_key,
                track_id=track_id,
                offset_ms=offset_ms,
            )
            speaker_rows.append(
                {
                    "speaker_key": speaker_key,
                    "display_name": display_name,
                    "track_id": track_id,
                }
            )
        segment_repo.upsert_speakers(job_id, speaker_rows)

        await run_in_threadpool(coordinator.enqueue_job, job_id)
        refreshed = job_repo.get_job(job_id)
        assert refreshed is not None
    except Exception:
        job_repo.delete_job(job_id)
        _cleanup_job_dirs(paths, job_id)
        raise

    serialized = _serialize_job(refreshed)
    return {**serialized, "job": serialized, "jobs": [serialized]}


@router.get("")
async def list_jobs(
    job_repo: Annotated[JobRepo, Depends(get_job_repo)],
    status: Annotated[str | None, Query()] = None,
    batch_id: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    """List jobs with optional status and batch_id filters."""
    jobs = job_repo.list_jobs(
        batch_id=batch_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    return {
        "jobs": [_serialize_job(job) for job in jobs],
        "limit": limit,
        "offset": offset,
    }


# ---------------------------------------------------------------------------
# Item routes (/api/v1/jobs/{job_id}...)
# ---------------------------------------------------------------------------


@router.get("/{job_id}")
async def get_job_detail(
    job_id: str,
    paths: Annotated[DataPaths, Depends(get_paths)],
    job_repo: Annotated[JobRepo, Depends(get_job_repo)],
    segment_repo: Annotated[SegmentRepo, Depends(get_segment_repo)],
) -> dict[str, Any]:
    """Return full job metadata, tracks, speakers, transcript segments, and output formats."""
    job = _get_job_or_404(job_repo, job_id)
    segments = segment_repo.list_segments(job_id)
    available_formats = _scan_available_formats(paths, job_id)
    payload = _serialize_job(job)
    payload["segments"] = [_serialize_segment(seg) for seg in segments]
    payload["available_formats"] = available_formats
    return payload


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    job_repo: Annotated[JobRepo, Depends(get_job_repo)],
    coordinator: Annotated[Coordinator, Depends(get_coordinator)],
) -> dict[str, Any]:
    """Cancel a queued or running job."""
    _get_job_or_404(job_repo, job_id)
    await run_in_threadpool(coordinator.cancel_job, job_id)
    updated = _get_job_or_404(job_repo, job_id)
    serialized = _serialize_job(updated)
    return {**serialized, "job": serialized}


@router.post("/{job_id}/retry")
async def retry_job(
    job_id: str,
    user: Annotated[dict[str, Any], Depends(get_current_user)],
    paths: Annotated[DataPaths, Depends(get_paths)],
    job_repo: Annotated[JobRepo, Depends(get_job_repo)],
    coordinator: Annotated[Coordinator, Depends(get_coordinator)],
    body: Annotated[dict[str, Any] | None, Body()] = None,
) -> dict[str, Any]:
    """Retry a job, optionally applying new profile or context overrides first."""
    job = _get_job_or_404(job_repo, job_id)
    if body:
        overrides = dict(body)
        requested_profile_id = overrides.pop("profile_id", None)
        if requested_profile_id is not None:
            base_profile = _resolve_base_profile(paths, user, str(requested_profile_id))
            base_cfg = base_profile.model_dump(mode="json")
            base_cfg["profile_id"] = base_profile.id
        else:
            base_cfg = job.effective_config

        try:
            updated_cfg = _apply_config_overrides(
                base_cfg,
                overrides,
                paths=paths,
                mode=job.mode,
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise ApiError(
                status_code=422,
                error_code="VALIDATION_ERROR",
                detail=str(exc),
            ) from exc

        _update_job_effective_config(
            paths,
            job_id,
            updated_cfg,
            language_requested=updated_cfg["asr"].get("language"),
            update_language=True,
        )

    await run_in_threadpool(coordinator.retry_job, job_id)
    updated = _get_job_or_404(job_repo, job_id)
    serialized = _serialize_job(updated)
    return {**serialized, "job": serialized}


@router.post("/{job_id}/diarize")
async def diarize_job(
    job_id: str,
    paths: Annotated[DataPaths, Depends(get_paths)],
    job_repo: Annotated[JobRepo, Depends(get_job_repo)],
    coordinator: Annotated[Coordinator, Depends(get_coordinator)],
    body: Annotated[DiarizeRequest | None, Body()] = None,
) -> dict[str, Any]:
    """Trigger post-hoc speaker diarization on a completed or partial job."""
    job = _get_job_or_404(job_repo, job_id)

    if body is not None:
        overrides: dict[str, Any] = {"diarize": True}
        if body.min_speakers is not None:
            overrides["min_speakers"] = body.min_speakers
        if body.max_speakers is not None:
            overrides["max_speakers"] = body.max_speakers
        try:
            updated_cfg = _apply_config_overrides(job.effective_config, overrides, paths=paths)
        except (ValidationError, ValueError, TypeError) as exc:
            raise ApiError(
                status_code=422,
                error_code="VALIDATION_ERROR",
                detail=str(exc),
            ) from exc
        _update_job_effective_config(paths, job_id, updated_cfg)

    try:
        await run_in_threadpool(coordinator.trigger_diarize, job_id)
    except ValueError as exc:
        raise ApiError(
            status_code=400,
            error_code="VALIDATION_ERROR",
            detail=str(exc),
        ) from exc

    updated = _get_job_or_404(job_repo, job_id)
    serialized = _serialize_job(updated)
    return {**serialized, "job": serialized}


@router.patch("/{job_id}/speakers")
async def update_job_speakers(
    job_id: str,
    body: SpeakersPatchRequest,
    paths: Annotated[DataPaths, Depends(get_paths)],
    job_repo: Annotated[JobRepo, Depends(get_job_repo)],
    segment_repo: Annotated[SegmentRepo, Depends(get_segment_repo)],
    coordinator: Annotated[Coordinator, Depends(get_coordinator)],
) -> dict[str, Any]:
    """Update speaker display aliases and re-export outputs immediately on CPU."""
    _get_job_or_404(job_repo, job_id)

    for speaker_key, raw_name in body.aliases.items():
        cleaned_key = speaker_key.strip()
        cleaned_name = raw_name.strip()
        if not cleaned_key or not cleaned_name:
            raise ApiError(
                status_code=422,
                error_code="VALIDATION_ERROR",
                detail="Speaker key and display name cannot be empty",
            )
        renamed = segment_repo.rename_speaker(job_id, cleaned_key, cleaned_name)
        if renamed is None:
            segment_repo.upsert_speakers(
                job_id,
                [
                    {
                        "speaker_key": cleaned_key,
                        "display_name": cleaned_name,
                        "track_id": None,
                    }
                ],
            )

    await run_in_threadpool(coordinator.reexport_job, job_id)
    return {
        "job_id": job_id,
        "speakers": [_serialize_speaker(sp) for sp in segment_repo.list_speakers(job_id)],
        "segments": [_serialize_segment(seg) for seg in segment_repo.list_segments(job_id)],
        "available_formats": _scan_available_formats(paths, job_id),
    }


@router.post("/{job_id}/export")
async def reexport_job(
    job_id: str,
    paths: Annotated[DataPaths, Depends(get_paths)],
    job_repo: Annotated[JobRepo, Depends(get_job_repo)],
    coordinator: Annotated[Coordinator, Depends(get_coordinator)],
    body: Annotated[ExportRequest | None, Body()] = None,
) -> dict[str, Any]:
    """Re-render output files on CPU with optional format or voice_style updates."""
    job = _get_job_or_404(job_repo, job_id)
    if body is not None:
        overrides: dict[str, Any] = {}
        if body.formats is not None:
            overrides["formats"] = list(body.formats)
        if body.subtitles is not None:
            overrides["subtitles"] = dict(body.subtitles)
        if body.voice_style is not None:
            overrides["voice_style"] = body.voice_style
        if overrides:
            try:
                updated_cfg = _apply_config_overrides(job.effective_config, overrides, paths=paths)
            except (ValidationError, ValueError, TypeError) as exc:
                raise ApiError(
                    status_code=422,
                    error_code="VALIDATION_ERROR",
                    detail=str(exc),
                ) from exc
            _update_job_effective_config(paths, job_id, updated_cfg)

    written = await run_in_threadpool(coordinator.reexport_job, job_id)
    return {
        "job_id": job_id,
        "available_formats": _scan_available_formats(paths, job_id),
        "files": [Path(path).name for path in written],
    }


@router.get("/{job_id}/media/{track_id}")
async def stream_track_media(
    job_id: str,
    track_id: str,
    paths: Annotated[DataPaths, Depends(get_paths)],
    job_repo: Annotated[JobRepo, Depends(get_job_repo)],
) -> FileResponse:
    """Serve original track media with native HTTP Range / 206 Partial Content support."""
    job = _get_job_or_404(job_repo, job_id)
    track = next((item for item in job.tracks if item.id == track_id), None)
    if track is None:
        raise ApiError(
            status_code=404,
            error_code="MEDIA_INVALID",
            error_params={"job_id": job_id, "track_id": track_id},
            detail=f"Track {track_id!r} not found in job {job_id!r}",
        )

    media_root = paths.media.resolve()
    raw_path = Path(track.media_relpath)
    candidate = raw_path if raw_path.is_absolute() else (paths.root / raw_path)
    resolved = candidate.resolve()

    if not resolved.is_relative_to(media_root):
        raise ApiError(
            status_code=403,
            error_code="FORBIDDEN",
            detail="Track media path resolves outside media directory",
        )
    if not resolved.is_file():
        raise ApiError(
            status_code=404,
            error_code="MEDIA_INVALID",
            detail=f"Media file for track {track_id!r} not found on disk",
        )

    media_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    return FileResponse(path=resolved, media_type=media_type)


@router.get("/{job_id}/download/{fmt}")
async def download_job_output(
    job_id: str,
    fmt: str,
    paths: Annotated[DataPaths, Depends(get_paths)],
    job_repo: Annotated[JobRepo, Depends(get_job_repo)],
) -> FileResponse:
    """Download a rendered transcript file (.vtt, .srt, .md, .json) preserving filename dots."""
    normalized_fmt = fmt.strip().lower().lstrip(".")
    if normalized_fmt not in _VALID_FORMATS:
        raise ApiError(
            status_code=400,
            error_code="VALIDATION_ERROR",
            error_params={"fmt": fmt},
            detail=f"Unsupported download format: {fmt!r}",
        )

    job = _get_job_or_404(job_repo, job_id)
    resolved = _find_output_file(paths, job, normalized_fmt)
    if resolved is None:
        raise ApiError(
            status_code=404,
            error_code="VALIDATION_ERROR",
            error_params={"job_id": job_id, "fmt": normalized_fmt},
            detail=f"Output file in format {normalized_fmt!r} not found for job {job_id!r}",
        )

    return FileResponse(
        path=resolved,
        media_type=_FORMAT_MEDIA_TYPES[normalized_fmt],
        filename=resolved.name,
        content_disposition_type="attachment",
    )


@router.delete("/{job_id}")
async def delete_job(
    job_id: str,
    paths: Annotated[DataPaths, Depends(get_paths)],
    job_repo: Annotated[JobRepo, Depends(get_job_repo)],
    coordinator: Annotated[Coordinator, Depends(get_coordinator)],
) -> dict[str, Any]:
    """Cancel an active job if running, delete its SQLite record, and remove all job directories."""
    job = _get_job_or_404(job_repo, job_id)
    if job.status in _ACTIVE_JOB_STATUSES:
        await run_in_threadpool(coordinator.cancel_job, job_id)

    job_repo.delete_job(job_id)
    _cleanup_job_dirs(paths, job_id)
    return {"deleted": True, "id": job_id}
