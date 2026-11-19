"""Command-line interface (`aurisdeck`) for server, transcription, profiles, auth, and models."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import shutil
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import click
from pydantic import ValidationError
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from aurisdeck import __version__
from aurisdeck.api.app import create_app
from aurisdeck.auth.service import AuthError, AuthService
from aurisdeck.config import Settings
from aurisdeck.db.migrations import run_migrations
from aurisdeck.db.repositories import JobRecord, JobRepo, SegmentRepo, UserRepo
from aurisdeck.domain.models import ProfileSchema
from aurisdeck.media.ffmpeg import MediaInvalidError
from aurisdeck.media.ingest import ZipSlipError, unpack_multitrack_zip
from aurisdeck.storage.defaults import seed_default_profiles
from aurisdeck.storage.paths import DataPaths, sanitize_filename
from aurisdeck.storage.stores import (
    GlossaryStore,
    ProfileStore,
    ReplacementStore,
    build_effective_context,
)
from aurisdeck.workers.coordinator import Coordinator
from aurisdeck.workers.events import EventBus

try:
    import httpx
except ImportError:  # pragma: no cover - optional only if installed without dev/test extras
    httpx = None  # type: ignore[assignment]

_SUPPORTED_FORMATS = ("vtt", "srt", "md", "json")
_TERMINAL_STATUSES = frozenset({"completed", "completed_partial", "failed", "cancelled"})
_FAILURE_STATUSES = frozenset({"failed", "cancelled"})
_CD_FILENAME_STAR_RE = re.compile(r"filename\*\s*=\s*(?:utf-8''|UTF-8'')([^;]+)")
_CD_FILENAME_RE = re.compile(r'filename\s*=\s*"([^"]+)"|filename\s*=\s*([^;\s]+)')


@dataclass(slots=True)
class _TrackedJobState:
    """Mutable progress and completion state for a CLI-tracked transcription job."""

    job_id: str
    title: str
    status: str = "queued"
    error_code: str | None = None
    error_detail: str = ""
    stage: str = "queued"
    device_id: str | None = None
    rtfx: float | None = None
    processed_ms: int = 0
    total_ms: int = 0
    output_dir: Path = field(default_factory=lambda: Path("."))


def _parse_formats(raw_values: Iterable[str]) -> list[str]:
    """Parse and validate comma-separated or repeated --format values."""
    items: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for part in raw.split(","):
            fmt = part.strip().lower().lstrip(".")
            if not fmt:
                continue
            if fmt not in _SUPPORTED_FORMATS:
                raise click.BadParameter(
                    f"Unsupported format {fmt!r}. Expected one of: {', '.join(_SUPPORTED_FORMATS)}",
                    param_hint="--format",
                )
            if fmt not in seen:
                seen.add(fmt)
                items.append(fmt)
    return items or ["vtt"]


def _resolve_mode(mode: str | None, inputs: tuple[Path, ...]) -> str:
    """Resolve effective job mode from --mode or input file count/extension."""
    if mode is not None:
        normalized = mode.strip().lower()
        if normalized == "single" and len(inputs) != 1:
            raise click.BadParameter(
                "Single mode requires exactly one input file",
                param_hint="--mode",
            )
        return normalized
    if len(inputs) > 1:
        return "batch"
    if len(inputs) == 1 and inputs[0].suffix.lower() == ".zip":
        return "multitrack"
    return "single"


def _parse_gpus_spec(raw: str) -> str | list[int]:
    """Normalize --gpus flag ('auto', 'cpu', or comma-separated GPU indices)."""
    cleaned = raw.strip().lower()
    if cleaned in {"auto", "cpu"}:
        return cleaned
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise click.BadParameter("GPU specification cannot be empty", param_hint="--gpus")
    indices: list[int] = []
    for part in parts:
        try:
            idx = int(part)
        except ValueError as exc:
            raise click.BadParameter(
                f"Invalid GPU specification {raw!r}; expected 'auto', 'cpu', or integer indices",
                param_hint="--gpus",
            ) from exc
        if idx < 0:
            raise click.BadParameter("GPU indices must be non-negative", param_hint="--gpus")
        indices.append(idx)
    return indices


def _extract_http_error(response: Any) -> tuple[str, str]:
    """Extract structured (error_code, detail) from an HTTP error response."""
    fallback_code = f"HTTP_{response.status_code}"
    try:
        payload = response.json()
    except ValueError:
        return fallback_code, (response.text or "").strip()
    if isinstance(payload, Mapping):
        code = str(payload.get("error_code") or fallback_code)
        detail = str(payload.get("detail") or "")
        return code, detail
    return fallback_code, (response.text or "").strip()


def _filename_from_response(response: Any, fallback: str) -> str:
    """Extract a safe local filename from Content-Disposition or return fallback."""
    header = response.headers.get("content-disposition", "")
    if header:
        star_match = _CD_FILENAME_STAR_RE.search(header)
        if star_match:
            decoded = unquote(star_match.group(1).strip().strip('"'))
            if decoded:
                return sanitize_filename(decoded)
        plain_match = _CD_FILENAME_RE.search(header)
        if plain_match:
            raw_name = plain_match.group(1) or plain_match.group(2) or ""
            raw_name = raw_name.strip()
            if raw_name:
                return sanitize_filename(raw_name)
    return sanitize_filename(fallback)


def _iter_sse_events(lines: Iterable[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Parse SSE text lines into (event_type, parsed_json_payload) tuples."""
    current_event: str | None = None
    data_lines: list[str] = []

    def _flush() -> tuple[str, dict[str, Any]] | None:
        nonlocal current_event, data_lines
        if not data_lines:
            current_event = None
            return None
        raw_data = "\n".join(data_lines)
        ev_name = current_event or "message"
        current_event = None
        data_lines = []
        try:
            parsed = json.loads(raw_data)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        resolved_type = str(parsed.get("type") or ev_name)
        return resolved_type, parsed

    for raw_line in lines:
        line = raw_line.rstrip("\r")
        if line == "":
            flushed = _flush()
            if flushed is not None:
                yield flushed
            continue
        if line.startswith(":"):
            continue
        field_name, sep, value = line.partition(":")
        if not sep:
            continue
        value = value.removeprefix(" ")
        if field_name == "event":
            current_event = value
        elif field_name == "data":
            data_lines.append(value)

    flushed = _flush()
    if flushed is not None:
        yield flushed


def _build_progress(console: Console) -> Progress:
    """Create the shared Rich progress bar configured for job stage/RTFx display."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def _format_task_description(state: _TrackedJobState) -> str:
    """Format a concise Rich progress label showing stage, device, and RTFx."""
    meta: list[str] = []
    if state.device_id:
        meta.append(state.device_id)
    if state.rtfx is not None and state.rtfx > 0:
        meta.append(f"{state.rtfx:.1f}x")
    elif state.stage:
        meta.append(state.stage)
    suffix = f" [{' · '.join(meta)}]" if meta else ""
    return f"{state.title}{suffix}"


def _apply_event_to_state(
    state: _TrackedJobState,
    event_type: str,
    envelope: Mapping[str, Any],
    progress: Progress,
    task_id: TaskID,
) -> None:
    """Update tracked job state and Rich progress bar from a v1 EventBus/SSE envelope."""
    payload = envelope.get("payload")
    data: Mapping[str, Any] = payload if isinstance(payload, Mapping) else {}

    if event_type == "job.stage":
        stage = str(data.get("stage") or data.get("status") or "")
        device_id = data.get("device_id")
        if stage:
            state.stage = stage
            state.status = stage
        if device_id:
            state.device_id = str(device_id)
        progress.update(task_id, description=_format_task_description(state))
        return

    if event_type == "job.progress":
        processed_ms = int(data.get("processed_ms") or 0)
        total_ms = int(data.get("total_ms") or 0)
        device_id = data.get("device_id")
        rtfx = data.get("rtfx")
        state.stage = "transcribing"
        state.processed_ms = processed_ms
        if total_ms > 0:
            state.total_ms = total_ms
        if device_id:
            state.device_id = str(device_id)
        if isinstance(rtfx, (int, float)) and not isinstance(rtfx, bool):
            state.rtfx = float(rtfx)
        update_kwargs: dict[str, Any] = {"description": _format_task_description(state)}
        if state.total_ms > 0:
            update_kwargs["total"] = state.total_ms
            update_kwargs["completed"] = min(state.processed_ms, state.total_ms)
        progress.update(task_id, **update_kwargs)
        return

    if event_type == "job.completed":
        state.status = "completed"
        state.stage = "completed"
        state.rtfx = None
        total = state.total_ms if state.total_ms > 0 else 100
        progress.update(
            task_id,
            total=total,
            completed=total,
            description=_format_task_description(state),
        )
        return

    if event_type == "job.completed_partial":
        state.status = "completed_partial"
        state.stage = "completed_partial"
        state.rtfx = None
        if data.get("error_code"):
            state.error_code = str(data["error_code"])
        state.error_detail = str(data.get("detail") or "")
        total = state.total_ms if state.total_ms > 0 else 100
        progress.update(
            task_id,
            total=total,
            completed=total,
            description=_format_task_description(state),
        )
        return

    if event_type in {"job.error", "job.failed"}:
        state.status = "failed"
        state.stage = "failed"
        state.rtfx = None
        state.error_code = str(data.get("error_code") or "JOB_FAILED")
        state.error_detail = str(data.get("detail") or "")
        progress.update(task_id, description=f"{state.title} [{state.error_code}]")
        return

    if event_type == "job.cancelled":
        state.status = "cancelled"
        state.stage = "cancelled"
        state.rtfx = None
        state.error_code = str(data.get("error_code") or "JOB_CANCELLED")
        state.error_detail = str(data.get("detail") or "")
        progress.update(task_id, description=f"{state.title} [{state.error_code}]")


def _build_local_effective_config(
    paths: DataPaths,
    *,
    profile_id: str | None,
    mode: str,
    lang: str | None,
    diarize: bool,
    hotwords: str | None,
    gpus: str | None,
    formats: list[str],
) -> tuple[dict[str, Any], str | None]:
    """Resolve profile + CLI overrides into a validated effective_config snapshot."""
    profile_store = ProfileStore(paths)
    user_repo = UserRepo(paths.db_path)

    base_profile: ProfileSchema | None = None
    if profile_id is not None and profile_id.strip():
        candidate = profile_id.strip()
        try:
            base_profile = profile_store.get(candidate)
        except ValueError as exc:
            raise click.ClickException(f"VALIDATION_ERROR: {exc}") from exc
        if base_profile is None:
            raise click.ClickException(f"PROFILE_NOT_FOUND: Profile {candidate!r} not found")
    else:
        default_user = user_repo.get_by_username("default")
        preferred_id = (
            default_user.preferences.get("default_profile_id")
            if default_user is not None and isinstance(default_user.preferences, Mapping)
            else None
        )
        if isinstance(preferred_id, str) and preferred_id.strip():
            try:
                base_profile = profile_store.get(preferred_id.strip())
            except ValueError:
                base_profile = None
        if base_profile is None:
            base_profile = profile_store.get("es-tecnico")
        if base_profile is None:
            existing = profile_store.list()
            base_profile = (
                existing[0] if existing else ProfileSchema(id="default", display_name="Default")
            )

    cfg = copy.deepcopy(base_profile.model_dump(mode="json"))
    cfg["input"]["mode"] = mode
    cfg["output"]["formats"] = list(formats)

    if lang is not None:
        cleaned_lang = lang.strip()
        cfg["asr"]["language"] = (
            None if not cleaned_lang or cleaned_lang.lower() == "auto" else cleaned_lang
        )
    if diarize:
        cfg["diarization"]["enabled"] = True
    if hotwords is not None:
        cfg["context"]["hotwords"] = [p.strip() for p in hotwords.split(",") if p.strip()]
    if gpus is not None:
        cfg["hardware"]["gpus"] = _parse_gpus_spec(gpus)

    try:
        validated = ProfileSchema.model_validate(cfg)
    except ValidationError as exc:
        raise click.ClickException(f"VALIDATION_ERROR: {exc}") from exc

    effective_ctx = build_effective_context(
        validated,
        glossary_store=GlossaryStore(paths),
        replacement_store=ReplacementStore(paths),
    )
    effective_config = validated.model_dump(mode="json")
    effective_config["profile_id"] = base_profile.id
    effective_config["effective_context"] = effective_ctx.model_dump(mode="json")
    return effective_config, validated.asr.language


def _create_local_jobs(
    paths: DataPaths,
    job_repo: JobRepo,
    segment_repo: SegmentRepo,
    *,
    inputs: tuple[Path, ...],
    mode: str,
    effective_config: dict[str, Any],
    language_requested: str | None,
    out_dir: Path | None,
) -> dict[str, _TrackedJobState]:
    """Create local SQLite job(s) and stage media under data/media/<job_id>/."""
    states: dict[str, _TrackedJobState] = {}

    if mode == "batch":
        batch_id = uuid.uuid4().hex
        for src_path in inputs:
            job_id = uuid.uuid4().hex
            track_id = uuid.uuid4().hex
            safe_name = sanitize_filename(src_path.name)
            media_dir = paths.job_media_dir(job_id)
            media_dir.mkdir(parents=True, exist_ok=True)
            dest_path = media_dir / f"{track_id}_{safe_name}"
            shutil.copy2(src_path, dest_path)

            title = Path(safe_name).stem or safe_name
            job_repo.create_job(
                title=title,
                mode="batch",
                effective_config=effective_config,
                job_id=job_id,
                owner_user_id="default",
                batch_id=batch_id,
                language_requested=language_requested,
            )
            job_repo.add_track(
                job_id=job_id,
                track_index=0,
                original_filename=safe_name,
                media_relpath=dest_path.relative_to(paths.root).as_posix(),
                speaker_key="SPEAKER_00",
                track_id=track_id,
            )
            states[job_id] = _TrackedJobState(
                job_id=job_id,
                title=title,
                output_dir=out_dir if out_dir is not None else src_path.parent,
            )
        return states

    if mode == "single":
        src_path = inputs[0]
        job_id = uuid.uuid4().hex
        track_id = uuid.uuid4().hex
        safe_name = sanitize_filename(src_path.name)
        media_dir = paths.job_media_dir(job_id)
        media_dir.mkdir(parents=True, exist_ok=True)
        dest_path = media_dir / f"{track_id}_{safe_name}"
        shutil.copy2(src_path, dest_path)

        title = Path(safe_name).stem or "Transcription"
        job_repo.create_job(
            title=title,
            mode="single",
            effective_config=effective_config,
            job_id=job_id,
            owner_user_id="default",
            language_requested=language_requested,
        )
        job_repo.add_track(
            job_id=job_id,
            track_index=0,
            original_filename=safe_name,
            media_relpath=dest_path.relative_to(paths.root).as_posix(),
            speaker_key="SPEAKER_00",
            track_id=track_id,
        )
        states[job_id] = _TrackedJobState(
            job_id=job_id,
            title=title,
            output_dir=out_dir if out_dir is not None else src_path.parent,
        )
        return states

    # mode == "multitrack"
    job_id = uuid.uuid4().hex
    media_dir = paths.job_media_dir(job_id)
    media_dir.mkdir(parents=True, exist_ok=True)
    is_zip = len(inputs) == 1 and inputs[0].suffix.lower() == ".zip"
    staged_tracks: list[tuple[int, str, str, str, int, str, str]] = []

    if is_zip:
        try:
            extracted = unpack_multitrack_zip(inputs[0], media_dir)
        except (ZipSlipError, MediaInvalidError, ValueError) as exc:
            shutil.rmtree(media_dir, ignore_errors=True)
            code = getattr(exc, "error_code", "INVALID_ARCHIVE")
            raise click.ClickException(f"{code}: {exc}") from exc
        if not extracted:
            shutil.rmtree(media_dir, ignore_errors=True)
            raise click.ClickException("INVALID_ARCHIVE: ZIP archive contains no supported tracks")

        title = Path(sanitize_filename(inputs[0].name)).stem or "Multitrack"
        for idx, extracted_path in enumerate(extracted):
            track_id = uuid.uuid4().hex
            speaker_key = f"SPEAKER_{idx:02d}"
            safe_name = extracted_path.name
            final_path = media_dir / f"{track_id}_{safe_name}"
            os.replace(extracted_path, final_path)
            staged_tracks.append(
                (
                    idx,
                    track_id,
                    safe_name,
                    final_path.relative_to(paths.root).as_posix(),
                    0,
                    speaker_key,
                    Path(safe_name).stem or speaker_key,
                )
            )
    else:
        first_name = sanitize_filename(inputs[0].name)
        title = Path(first_name).stem or "Multitrack"
        for idx, src_path in enumerate(inputs):
            track_id = uuid.uuid4().hex
            speaker_key = f"SPEAKER_{idx:02d}"
            safe_name = sanitize_filename(src_path.name)
            dest_path = media_dir / f"{track_id}_{safe_name}"
            shutil.copy2(src_path, dest_path)
            staged_tracks.append(
                (
                    idx,
                    track_id,
                    safe_name,
                    dest_path.relative_to(paths.root).as_posix(),
                    0,
                    speaker_key,
                    Path(safe_name).stem or speaker_key,
                )
            )

    job_repo.create_job(
        title=title,
        mode="multitrack",
        effective_config=effective_config,
        job_id=job_id,
        owner_user_id="default",
        language_requested=language_requested,
    )
    speaker_rows: list[dict[str, Any]] = []
    for idx, track_id, orig_name, relpath, offset_ms, speaker_key, display_name in staged_tracks:
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
    states[job_id] = _TrackedJobState(
        job_id=job_id,
        title=title,
        output_dir=out_dir if out_dir is not None else inputs[0].parent,
    )
    return states


def _sync_state_from_job_record(
    state: _TrackedJobState,
    job: JobRecord,
    progress: Progress,
    task_id: TaskID,
) -> None:
    """Reconcile tracked state against a persisted JobRecord if SSE missed an update."""
    state.status = job.status
    if job.error_code:
        state.error_code = job.error_code
    if job.error_detail:
        state.error_detail = job.error_detail
    if job.status in {"completed", "completed_partial"}:
        total = state.total_ms if state.total_ms > 0 else 100
        progress.update(
            task_id,
            total=total,
            completed=total,
            description=f"{state.title} [{job.status}]",
        )
    elif job.status in _FAILURE_STATUSES:
        code = state.error_code or "JOB_FAILED"
        state.error_code = code
        progress.update(task_id, description=f"{state.title} [{code}]")


def _run_remote_transcribe(
    *,
    inputs: tuple[Path, ...],
    mode: str,
    profile: str | None,
    overrides: dict[str, Any],
    formats: list[str],
    out_dir: Path | None,
    server: str,
    token: str | None,
    console: Console,
) -> None:
    """Execute transcription remotely against an AurisDeck HTTP server."""
    if httpx is None:
        raise click.ClickException("DEPENDENCY_MISSING: httpx is required for --server mode")

    base_url = server.rstrip("/")
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    dest_root = out_dir if out_dir is not None else Path(".")
    timeout = httpx.Timeout(connect=10.0, read=None, write=120.0, pool=10.0)

    try:
        with httpx.Client(base_url=base_url, headers=headers, timeout=timeout) as client:
            form_data: dict[str, str] = {"mode": mode}
            if profile:
                form_data["profile_id"] = profile
            if overrides:
                form_data["overrides_json"] = json.dumps(overrides)

            with ExitStack() as stack:
                upload_files = [
                    (
                        "files",
                        (
                            path.name,
                            stack.enter_context(path.open("rb")),
                            "application/octet-stream",
                        ),
                    )
                    for path in inputs
                ]
                create_resp = client.post("/api/v1/jobs", data=form_data, files=upload_files)

            if create_resp.status_code >= 400:
                code, detail = _extract_http_error(create_resp)
                raise click.ClickException(f"{code}: {detail}" if detail else code)

            payload = create_resp.json()
            batch_id = payload.get("batch_id")
            raw_jobs = payload.get("jobs")
            if not isinstance(raw_jobs, list) or not raw_jobs:
                single_job = payload.get("job") if isinstance(payload.get("job"), Mapping) else payload
                raw_jobs = [single_job] if isinstance(single_job, Mapping) and "id" in single_job else []

            if not raw_jobs:
                raise click.ClickException("INVALID_RESPONSE: Server returned no job IDs")

            states: dict[str, _TrackedJobState] = {}
            for item in raw_jobs:
                jid = str(item["id"])
                states[jid] = _TrackedJobState(
                    job_id=jid,
                    title=str(item.get("title") or jid),
                    status=str(item.get("status") or "queued"),
                    output_dir=dest_root,
                )

            with _build_progress(console) as progress:
                tasks = {
                    jid: progress.add_task(_format_task_description(state), total=100)
                    for jid, state in states.items()
                }

                if not all(s.status in _TERMINAL_STATUSES for s in states.values()):
                    with client.stream("GET", "/api/v1/events") as stream_resp:
                        if stream_resp.status_code >= 400:
                            stream_resp.read()
                            code, detail = _extract_http_error(stream_resp)
                            raise click.ClickException(f"{code}: {detail}" if detail else code)

                        for event_type, envelope in _iter_sse_events(stream_resp.iter_lines()):
                            ev_job_id = envelope.get("job_id")
                            if not ev_job_id or ev_job_id not in states:
                                continue
                            _apply_event_to_state(
                                states[ev_job_id],
                                event_type,
                                envelope,
                                progress,
                                tasks[ev_job_id],
                            )
                            if all(s.status in _TERMINAL_STATUSES for s in states.values()):
                                break

                # Fallback polling if the SSE stream closed before all jobs became terminal.
                while any(s.status not in _TERMINAL_STATUSES for s in states.values()):
                    for jid, state in states.items():
                        if state.status in _TERMINAL_STATUSES:
                            continue
                        poll_resp = client.get(f"/api/v1/jobs/{jid}")
                        if poll_resp.status_code >= 400:
                            code, detail = _extract_http_error(poll_resp)
                            raise click.ClickException(f"{code}: {detail}" if detail else code)
                        job_data = poll_resp.json()
                        state.status = str(job_data.get("status") or state.status)
                        if job_data.get("error_code"):
                            state.error_code = str(job_data["error_code"])
                        if job_data.get("error_detail"):
                            state.error_detail = str(job_data["error_detail"])
                        if state.status in {"completed", "completed_partial"}:
                            total = state.total_ms if state.total_ms > 0 else 100
                            progress.update(
                                tasks[jid],
                                total=total,
                                completed=total,
                                description=f"{state.title} [{state.status}]",
                            )
                        elif state.status in _FAILURE_STATUSES:
                            err_code = state.error_code or "JOB_FAILED"
                            state.error_code = err_code
                            progress.update(tasks[jid], description=f"{state.title} [{err_code}]")
                    if any(s.status not in _TERMINAL_STATUSES for s in states.values()):
                        time.sleep(0.5)

            failed = [s for s in states.values() if s.status in _FAILURE_STATUSES]
            if failed:
                first = failed[0]
                code = first.error_code or ("JOB_CANCELLED" if first.status == "cancelled" else "JOB_FAILED")
                detail = f": {first.error_detail}" if first.error_detail else ""
                raise click.ClickException(f"{code}{detail}")

            for state in states.values():
                if state.status == "completed_partial" and state.error_code:
                    console.print(
                        f"[yellow]Warning:[/yellow] job {state.job_id} completed partially ({state.error_code})"
                    )

            dest_root.mkdir(parents=True, exist_ok=True)
            written_paths: list[Path] = []
            if batch_id and len(states) > 1:
                for fmt in formats:
                    dl_resp = client.get(
                        f"/api/v1/jobs/batches/{batch_id}/download.zip",
                        params={"fmt": fmt},
                    )
                    if dl_resp.status_code >= 400:
                        code, detail = _extract_http_error(dl_resp)
                        raise click.ClickException(f"{code}: {detail}" if detail else code)
                    fname = _filename_from_response(
                        dl_resp,
                        fallback=f"{sanitize_filename(str(batch_id))}_{fmt}.zip",
                    )
                    dest_file = dest_root / fname
                    dest_file.write_bytes(dl_resp.content)
                    written_paths.append(dest_file)
            else:
                for state in states.values():
                    for fmt in formats:
                        dl_resp = client.get(f"/api/v1/jobs/{state.job_id}/download/{fmt}")
                        if dl_resp.status_code >= 400:
                            code, detail = _extract_http_error(dl_resp)
                            raise click.ClickException(f"{code}: {detail}" if detail else code)
                        fname = _filename_from_response(
                            dl_resp,
                            fallback=f"{sanitize_filename(state.title)}.{fmt}",
                        )
                        dest_file = dest_root / fname
                        dest_file.write_bytes(dl_resp.content)
                        written_paths.append(dest_file)

            for written in written_paths:
                click.echo(str(written))
    except httpx.HTTPError as exc:
        raise click.ClickException(f"CONNECTION_ERROR: {exc}") from exc


def _run_local_transcribe(
    *,
    inputs: tuple[Path, ...],
    mode: str,
    profile: str | None,
    lang: str | None,
    diarize: bool,
    hotwords: str | None,
    gpus: str | None,
    formats: list[str],
    out_dir: Path | None,
    data_dir: Path,
    console: Console,
) -> None:
    """Execute transcription locally via an in-process Coordinator."""
    settings = Settings(data_dir=data_dir)
    paths = DataPaths(settings.data_dir)
    paths.ensure_dirs()
    run_migrations(paths.db_path)
    seed_default_profiles(ProfileStore(paths))

    effective_config, language_requested = _build_local_effective_config(
        paths,
        profile_id=profile,
        mode=mode,
        lang=lang,
        diarize=diarize,
        hotwords=hotwords,
        gpus=gpus,
        formats=formats,
    )

    job_repo = JobRepo(paths.db_path)
    segment_repo = SegmentRepo(paths.db_path)
    event_bus = EventBus()

    states = _create_local_jobs(
        paths,
        job_repo,
        segment_repo,
        inputs=inputs,
        mode=mode,
        effective_config=effective_config,
        language_requested=language_requested,
        out_dir=out_dir,
    )

    coordinator = Coordinator(
        settings=settings,
        paths=paths,
        job_repo=job_repo,
        segment_repo=segment_repo,
        event_bus=event_bus,
    )

    async def _execute_and_wait() -> None:
        queue = event_bus.subscribe()
        coordinator.start()
        try:
            for job_id in states:
                coordinator.enqueue_job(job_id)

            with _build_progress(console) as progress:
                tasks = {
                    jid: progress.add_task(_format_task_description(state), total=100)
                    for jid, state in states.items()
                }

                while not all(s.status in _TERMINAL_STATUSES for s in states.values()):
                    try:
                        envelope = await asyncio.wait_for(queue.get(), timeout=0.5)
                        ev_job_id = envelope.get("job_id")
                        if ev_job_id and ev_job_id in states:
                            _apply_event_to_state(
                                states[ev_job_id],
                                str(envelope.get("type") or ""),
                                envelope,
                                progress,
                                tasks[ev_job_id],
                            )
                    except TimeoutError:
                        for jid, state in states.items():
                            if state.status in _TERMINAL_STATUSES:
                                continue
                            record = job_repo.get_job(jid)
                            if record is not None and record.status in _TERMINAL_STATUSES:
                                _sync_state_from_job_record(state, record, progress, tasks[jid])
        finally:
            event_bus.unsubscribe(queue)
            coordinator.stop()

    asyncio.run(_execute_and_wait())

    failed = [s for s in states.values() if s.status in _FAILURE_STATUSES]
    if failed:
        first = failed[0]
        code = first.error_code or ("JOB_CANCELLED" if first.status == "cancelled" else "JOB_FAILED")
        detail = f": {first.error_detail}" if first.error_detail else ""
        raise click.ClickException(f"{code}{detail}")

    written_paths: list[Path] = []
    for state in states.values():
        if state.status == "completed_partial" and state.error_code:
            console.print(
                f"[yellow]Warning:[/yellow] job {state.job_id} completed partially ({state.error_code})"
            )
        job_out_dir = paths.job_outputs_dir(state.job_id)
        if not job_out_dir.is_dir():
            continue
        state.output_dir.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            for src_out in sorted(job_out_dir.glob(f"*.{fmt}")):
                if src_out.is_file():
                    dest_file = state.output_dir / src_out.name
                    shutil.copy2(src_out, dest_file)
                    written_paths.append(dest_file)

    for written in written_paths:
        click.echo(str(written))


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="aurisdeck")
def main() -> None:
    """AurisDeck CLI for serving, transcribing, managing profiles/auth, and pulling models."""


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host address.")
@click.option("--port", type=int, default=8080, show_default=True, help="Bind TCP port.")
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=Path("./data"),
    envvar="AURISDECK_DATA_DIR",
    show_default=True,
    help="Persistent AurisDeck data directory.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=None,
    envvar="AURISDECK_LOG_LEVEL",
    help="Minimum logging level (overrides AURISDECK_LOG_LEVEL).",
)
def serve(host: str, port: int, data_dir: Path, log_level: str | None) -> None:
    """Start the AurisDeck web server and job coordinator."""
    import uvicorn

    from aurisdeck.logging_setup import configure_main_logging
    from aurisdeck.storage.paths import DataPaths

    settings_kwargs: dict[str, Any] = {"data_dir": data_dir}
    if log_level is not None:
        settings_kwargs["log_level"] = log_level.upper()
    settings = Settings(**settings_kwargs)
    paths = DataPaths(settings.data_dir)
    paths.ensure_dirs()
    configure_main_logging(
        paths,
        level=settings.log_level,
        max_bytes=settings.log_max_bytes,
        backup_count=settings.log_backup_count,
    )
    app = create_app(settings)
    uvicorn.run(
        app,
        host=host,
        port=port,
        root_path=settings.base_path,
        log_level=settings.log_level.lower(),
        timeout_graceful_shutdown=5,
    )


@main.command()
@click.argument(
    "inputs",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--profile", default=None, help="Profile slug to use.")
@click.option(
    "--mode",
    type=click.Choice(["single", "multitrack", "batch"], case_sensitive=False),
    default=None,
    help="Input mode: single, multitrack, or batch.",
)
@click.option("--lang", default=None, help="Language code (e.g. es, en, auto).")
@click.option("--diarize", is_flag=True, default=False, help="Enable speaker diarization.")
@click.option("--hotwords", default=None, help="Comma-separated vocabulary hotwords.")
@click.option("--gpus", default=None, help="GPU selection: auto, cpu, or comma-separated indices.")
@click.option(
    "--format",
    "formats_raw",
    multiple=True,
    help="Output format(s), e.g. vtt,md (defaults to vtt).",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Destination directory (defaults to '.' with --server, or next to input locally).",
)
@click.option(
    "--server",
    default=None,
    envvar="AURISDECK_SERVER_URL",
    help="Remote AurisDeck server URL (e.g. http://localhost:8080).",
)
@click.option(
    "--token",
    default=None,
    envvar="AURISDECK_TOKEN",
    help="Bearer API token for authenticating with --server.",
)
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=Path("./data"),
    envvar="AURISDECK_DATA_DIR",
    show_default=True,
    help="Local data directory when running without --server.",
)
def transcribe(
    inputs: tuple[Path, ...],
    profile: str | None,
    mode: str | None,
    lang: str | None,
    diarize: bool,
    hotwords: str | None,
    gpus: str | None,
    formats_raw: tuple[str, ...],
    out_dir: Path | None,
    server: str | None,
    token: str | None,
    data_dir: Path,
) -> None:
    """Transcribe audio/video files locally or via a running AurisDeck server."""
    formats = _parse_formats(formats_raw)
    resolved_mode = _resolve_mode(mode, inputs)
    console = Console(stderr=True)

    if server:
        overrides: dict[str, Any] = {"formats": formats}
        if lang is not None:
            overrides["language"] = lang
        if diarize:
            overrides["diarize"] = True
        if hotwords is not None:
            overrides["hotwords"] = hotwords
        if gpus is not None:
            overrides["gpus"] = gpus

        _run_remote_transcribe(
            inputs=inputs,
            mode=resolved_mode,
            profile=profile,
            overrides=overrides,
            formats=formats,
            out_dir=out_dir,
            server=server,
            token=token,
            console=console,
        )
        return

    _run_local_transcribe(
        inputs=inputs,
        mode=resolved_mode,
        profile=profile,
        lang=lang,
        diarize=diarize,
        hotwords=hotwords,
        gpus=gpus,
        formats=formats,
        out_dir=out_dir,
        data_dir=data_dir,
        console=console,
    )


@main.group()
def profiles() -> None:
    """Manage transcription profiles."""


@profiles.command("list")
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=Path("./data"),
    envvar="AURISDECK_DATA_DIR",
    show_default=True,
    help="Persistent AurisDeck data directory.",
)
def list_profiles(data_dir: Path) -> None:
    """List available JSON transcription profiles."""
    paths = DataPaths(data_dir)
    paths.ensure_dirs()
    store = ProfileStore(paths)
    seed_default_profiles(store)
    items = store.list()

    console = Console()
    table = Table(title="AurisDeck Profiles")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Mode")
    table.add_column("Model")
    table.add_column("Language")
    table.add_column("Diarize")
    table.add_column("Subtitles")

    for prof in items:
        table.add_row(
            prof.id,
            prof.display_name,
            prof.input.mode,
            prof.asr.model,
            prof.asr.language or "auto",
            "yes" if prof.diarization.enabled else "no",
            prof.output.subtitles.voice_style,
        )
    console.print(table)


@main.group()
def auth() -> None:
    """Manage authentication credentials and API tokens."""


@auth.command("reset-password")
@click.option(
    "--username",
    default=None,
    help="Target username (defaults to primary administrator).",
)
@click.option("--password", default=None, help="New password (prompted securely if omitted).")
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=Path("./data"),
    envvar="AURISDECK_DATA_DIR",
    show_default=True,
    help="Persistent AurisDeck data directory.",
)
def reset_password(username: str | None, password: str | None, data_dir: Path) -> None:
    """Reset the stored password hash for a user account (or primary admin)."""
    resolved_password = password
    if resolved_password is None:
        resolved_password = click.prompt(
            "New password",
            hide_input=True,
            confirmation_prompt=True,
        )

    settings = Settings(data_dir=data_dir)
    paths = DataPaths(settings.data_dir)
    paths.ensure_dirs()
    run_migrations(paths.db_path)

    user_repo = UserRepo(paths.db_path)
    auth_service = AuthService(user_repo=user_repo, settings=settings)
    try:
        updated_username = auth_service.reset_password(resolved_password, username=username)
    except AuthError as exc:
        raise click.ClickException(f"{exc.error_code}: {exc}") from exc

    click.echo(f"Password updated for user {updated_username!r}.")


@auth.command("list-users")
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=Path("./data"),
    envvar="AURISDECK_DATA_DIR",
    show_default=True,
    help="Persistent AurisDeck data directory.",
)
def list_users(data_dir: Path) -> None:
    """List configured AurisDeck user accounts."""
    settings = Settings(data_dir=data_dir)
    paths = DataPaths(settings.data_dir)
    paths.ensure_dirs()
    run_migrations(paths.db_path)

    users = UserRepo(paths.db_path).list_users()
    for user in users:
        click.echo(f"{user.username}\t{user.role}")


@auth.command("create-token")
@click.option("--username", default="default", show_default=True, help="Target username.")
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=Path("./data"),
    envvar="AURISDECK_DATA_DIR",
    show_default=True,
    help="Persistent AurisDeck data directory.",
)
def create_token(username: str, data_dir: Path) -> None:
    """Generate and display a new Bearer API token once."""
    settings = Settings(data_dir=data_dir)
    paths = DataPaths(settings.data_dir)
    paths.ensure_dirs()
    run_migrations(paths.db_path)

    user_repo = UserRepo(paths.db_path)
    auth_service = AuthService(user_repo=user_repo, settings=settings)
    try:
        raw_token = auth_service.rotate_api_token(username=username)
    except AuthError as exc:
        raise click.ClickException(f"{exc.error_code}: {exc}") from exc

    click.echo(f"Bearer API token for {username!r} (shown once, store it securely):", err=True)
    click.echo(raw_token)


@main.command("pull-model")
@click.argument("model_name")
@click.option(
    "--hf-endpoint",
    default=None,
    envvar="HF_ENDPOINT",
    help="Custom Hugging Face Hub endpoint URL.",
)
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=Path("./data"),
    envvar="AURISDECK_DATA_DIR",
    show_default=True,
    help="Persistent AurisDeck data directory.",
)
def pull_model(model_name: str, hf_endpoint: str | None, data_dir: Path) -> None:
    """Pre-download a faster-whisper model into data/models/."""
    settings = Settings(data_dir=data_dir)
    paths = DataPaths(settings.data_dir)
    paths.ensure_dirs()

    os.environ["HF_HOME"] = str(paths.models)
    resolved_endpoint = hf_endpoint or settings.hf_endpoint
    if resolved_endpoint:
        os.environ["HF_ENDPOINT"] = resolved_endpoint

    try:
        from faster_whisper.utils import download_model

        kwargs: dict[str, Any] = {"cache_dir": str(paths.models)}
        if paths.db_path.is_file():
            from aurisdeck.db.repositories import SystemSettingsRepo

            db_token = SystemSettingsRepo(paths.db_path).get_hf_token()
            if db_token:
                kwargs["use_auth_token"] = db_token
        downloaded_path = download_model(model_name, **kwargs)
    except Exception as exc:
        raise click.ClickException(f"MODEL_DOWNLOAD_FAILED: {exc}") from exc

    click.echo(str(downloaded_path))


if __name__ == "__main__":
    main()
