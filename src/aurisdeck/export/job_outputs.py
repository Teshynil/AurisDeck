"""On-demand rendering and disk caching for job transcript outputs."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from aurisdeck.db.repositories import JobRecord, SegmentRecord
from aurisdeck.export.cue_segmenter import SegmentItem, WordTiming, segment_to_cues
from aurisdeck.export.exporters import (
    export_json,
    export_markdown,
    export_srt,
    export_vtt,
)
from aurisdeck.media.ingest import SUPPORTED_MEDIA_EXTENSIONS
from aurisdeck.storage.paths import DataPaths, atomic_write_text, sanitize_filename

logger = logging.getLogger(__name__)

VALID_OUTPUT_FORMATS: tuple[str, ...] = ("vtt", "srt", "md", "json")


def job_output_stem(job: JobRecord) -> str:
    """Derive a safe output filename stem preserving intermediate dots."""
    raw_title = (job.title or "").strip()
    if not raw_title and job.tracks:
        raw_title = Path(sanitize_filename(job.tracks[0].original_filename)).stem
    if not raw_title:
        return job.id
    safe = sanitize_filename(raw_title)
    if Path(safe).suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS:
        return Path(safe).stem or job.id
    return safe


def invalidate_job_outputs(paths: DataPaths, job_id: str) -> None:
    """Remove cached rendered output files for a job so they regenerate on next request."""
    out_dir = paths.job_outputs_dir(job_id)
    if not out_dir.is_dir():
        return
    for entry in out_dir.iterdir():
        if entry.is_file() and entry.suffix.lower().lstrip(".") in VALID_OUTPUT_FORMATS:
            entry.unlink(missing_ok=True)


def _to_segment_items(segments: Sequence[SegmentRecord]) -> list[SegmentItem]:
    return [
        SegmentItem(
            seq=seg.seq,
            start_ms=seg.start_ms,
            end_ms=seg.end_ms,
            speaker_key=seg.speaker_key or None,
            raw_text=seg.raw_text,
            text=seg.text,
            words=[
                WordTiming(
                    start_ms=int(w.get("start_ms", 0)),
                    end_ms=int(w.get("end_ms", 0)),
                    text=str(w.get("text", "")),
                )
                for w in seg.words
            ],
        )
        for seg in segments
    ]


def ensure_job_output(
    paths: DataPaths,
    job: JobRecord,
    segments: Sequence[SegmentRecord],
    speakers: Sequence[Any],
    fmt: str,
) -> Path:
    """Return cached output file path for `fmt`, rendering and caching it if missing."""
    normalized = str(fmt).strip().lower().lstrip(".")
    if normalized not in VALID_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {fmt!r}")

    out_dir = paths.job_outputs_dir(job.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = job_output_stem(job)
    dest = out_dir / f"{stem}.{normalized}"
    if dest.is_file():
        return dest

    speaker_aliases = {sp.speaker_key: sp.display_name for sp in speakers}
    segment_items = _to_segment_items(segments)

    if normalized in {"vtt", "srt"}:
        output_cfg = job.effective_config.get("output", {})
        sub_cfg = output_cfg.get("subtitles", {}) if isinstance(output_cfg, dict) else {}
        voice_style = str(sub_cfg.get("voice_style", "v_tag"))
        max_cue_seconds = float(sub_cfg.get("max_cue_seconds", 6.0))
        max_chars_per_line = int(sub_cfg.get("max_chars_per_line", 68))
        max_lines_per_cue = int(sub_cfg.get("max_lines_per_cue", 2))
        language = job.language_detected or job.language_requested

        cues = segment_to_cues(
            segment_items,
            speaker_aliases=speaker_aliases,
            max_chars_per_line=max_chars_per_line,
            max_lines_per_cue=max_lines_per_cue,
            max_cue_seconds=max_cue_seconds,
        )
        if normalized == "vtt":
            content = export_vtt(cues, language=language, voice_style=voice_style)
        else:
            content = export_srt(cues, voice_style=voice_style)
    elif normalized == "md":
        content = export_markdown(
            segment_items,
            speaker_aliases=speaker_aliases,
            title=job.title,
        )
    else:
        job_meta = {
            "id": job.id,
            "title": job.title,
            "mode": job.mode,
            "language_requested": job.language_requested,
            "language_detected": job.language_detected,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
        }
        content = export_json(
            job_meta,
            segment_items,
            speaker_aliases=speaker_aliases,
        )

    atomic_write_text(dest, content, paths.tmp)
    return dest


def render_job_outputs(
    paths: DataPaths,
    job: JobRecord,
    segments: Sequence[SegmentRecord],
    speakers: Sequence[Any],
    formats: Iterable[str] | None = None,
    *,
    invalidate_cache: bool = True,
) -> list[Path]:
    """Invalidate stale cached files (optional) and render requested formats (defaults to ['json'])."""
    if invalidate_cache:
        invalidate_job_outputs(paths, job.id)

    target_formats = list(formats) if formats is not None else ["json"]
    written: list[Path] = []
    for fmt in target_formats:
        normalized = str(fmt).strip().lower().lstrip(".")
        if normalized not in VALID_OUTPUT_FORMATS:
            logger.warning("Ignoring unsupported export format %r for job %s", fmt, job.id)
            continue
        written.append(ensure_job_output(paths, job, segments, speakers, normalized))
    return written
