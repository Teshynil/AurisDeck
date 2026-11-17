"""Subtitle, Markdown, JSON, and batch ZIP exporters for AurisDeck."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aurisdeck.export.cue_segmenter import (
    SegmentItem,
    SubtitleCue,
    _join_word_tokens,
    resolve_speaker_name,
)
from aurisdeck.storage.paths import sanitize_filename

_VALID_VOICE_STYLES = {"v_tag", "prefix", "none"}
_DETERMINISTIC_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def _split_ms(ms: int) -> tuple[int, int, int, int]:
    clamped = max(0, int(ms))
    hours, remainder = divmod(clamped, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return hours, minutes, seconds, millis


def format_vtt_time(ms: int) -> str:
    """Format milliseconds as a WebVTT timestamp (HH:MM:SS.mmm)."""
    hours, minutes, seconds, millis = _split_ms(ms)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def format_srt_time(ms: int) -> str:
    """Format milliseconds as a SubRip timestamp (HH:MM:SS,mmm)."""
    hours, minutes, seconds, millis = _split_ms(ms)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def escape_vtt_text(text: str) -> str:
    """Escape characters with special meaning in W3C WebVTT cue payloads."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _format_markdown_time(ms: int) -> str:
    hours, minutes, seconds, _ = _split_ms(ms)
    if hours > 0:
        return f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
    return f"[{minutes:02d}:{seconds:02d}]"


def _clean_speaker_label(speaker_name: str | None) -> str | None:
    if not speaker_name:
        return None
    cleaned = " ".join(speaker_name.split())
    return cleaned or None


def export_vtt(
    cues: Sequence[SubtitleCue],
    language: str | None = None,
    voice_style: str = "v_tag",
) -> str:
    """Render subtitle cues as a W3C-compliant WebVTT document."""
    if voice_style not in _VALID_VOICE_STYLES:
        raise ValueError(f"Unsupported voice_style: {voice_style!r}")

    header_lines = ["WEBVTT", "Kind: captions"]
    if language and language.strip():
        header_lines.append(f"Language: {language.strip()}")

    blocks: list[str] = ["\n".join(header_lines)]

    for cue in cues:
        timing = f"{format_vtt_time(cue.start_ms)} --> {format_vtt_time(cue.end_ms)}"
        escaped_body = "\n".join(escape_vtt_text(line) for line in cue.lines)
        speaker = _clean_speaker_label(cue.speaker_name)

        if speaker and voice_style == "v_tag":
            payload = f"<v {escape_vtt_text(speaker)}>{escaped_body}</v>"
        elif speaker and voice_style == "prefix":
            payload = f"[{escape_vtt_text(speaker)}]: {escaped_body}"
        else:
            payload = escaped_body

        blocks.append(f"{timing}\n{payload}")

    return "\n\n".join(blocks) + "\n"


def export_srt(
    cues: Sequence[SubtitleCue],
    voice_style: str = "prefix",
) -> str:
    """Render subtitle cues as a SubRip (.srt) document."""
    if voice_style not in _VALID_VOICE_STYLES:
        raise ValueError(f"Unsupported voice_style: {voice_style!r}")

    if not cues:
        return ""

    blocks: list[str] = []
    for idx, cue in enumerate(cues, start=1):
        timing = f"{format_srt_time(cue.start_ms)} --> {format_srt_time(cue.end_ms)}"
        body = "\n".join(cue.lines)
        speaker = _clean_speaker_label(cue.speaker_name)

        if speaker and voice_style in {"prefix", "v_tag"}:
            payload = f"[{speaker}]: {body}"
        else:
            payload = body

        blocks.append(f"{idx}\n{timing}\n{payload}")

    return "\n\n".join(blocks) + "\n"


def _extract_segment_text(segment: SegmentItem) -> str:
    raw = segment.text if segment.text.strip() else segment.raw_text
    if not raw.strip() and segment.words:
        raw = _join_word_tokens(segment.words)
    return " ".join(raw.split())


def _group_speaker_turns(
    segments: Sequence[SegmentItem],
    speaker_aliases: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    ordered = sorted(segments, key=lambda seg: (seg.start_ms, seg.seq, seg.end_ms))
    turns: list[dict[str, Any]] = []

    for segment in ordered:
        text = _extract_segment_text(segment)
        if not text:
            continue

        speaker_name = resolve_speaker_name(segment.speaker_key, speaker_aliases)
        start_ms = max(0, int(segment.start_ms))
        end_ms = max(start_ms + 1, int(segment.end_ms))

        if turns and turns[-1]["speaker_name"] == speaker_name:
            current = turns[-1]
            current["end_ms"] = max(current["end_ms"], end_ms)
            current["parts"].append(text)
            current["segment_seqs"].append(segment.seq)
        else:
            turns.append(
                {
                    "speaker_key": segment.speaker_key,
                    "speaker_name": speaker_name,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "parts": [text],
                    "segment_seqs": [segment.seq],
                }
            )

    for turn in turns:
        turn["text"] = " ".join(turn.pop("parts"))

    return turns


def export_markdown(
    segments: Sequence[SegmentItem],
    speaker_aliases: Mapping[str, str] | None = None,
    title: str = "",
) -> str:
    """Export consecutive same-speaker segments as clean Markdown paragraphs."""
    sections: list[str] = []
    if title and title.strip():
        sections.append(f"# {title.strip()}")

    turns = _group_speaker_turns(segments, speaker_aliases)
    for turn in turns:
        ts = _format_markdown_time(turn["start_ms"])
        speaker_name = turn["speaker_name"]
        if speaker_name:
            sections.append(f"{ts} **{speaker_name}:** {turn['text']}")
        else:
            sections.append(f"{ts} {turn['text']}")

    if not sections:
        return ""
    return "\n\n".join(sections) + "\n"


def export_json(
    job_meta: Mapping[str, Any],
    segments: Sequence[SegmentItem],
    speaker_aliases: Mapping[str, str] | None = None,
) -> str:
    """Export job metadata, speaker-grouped turns, and segment details as deterministic JSON."""
    ordered = sorted(segments, key=lambda seg: (seg.start_ms, seg.seq, seg.end_ms))
    turns = _group_speaker_turns(ordered, speaker_aliases)

    serialized_segments = [
        {
            "seq": seg.seq,
            "start_ms": max(0, int(seg.start_ms)),
            "end_ms": max(max(0, int(seg.start_ms)) + 1, int(seg.end_ms)),
            "speaker_key": seg.speaker_key,
            "speaker_name": resolve_speaker_name(seg.speaker_key, speaker_aliases),
            "raw_text": seg.raw_text,
            "text": seg.text,
            "words": [
                {
                    "start_ms": max(0, int(word.start_ms)),
                    "end_ms": max(max(0, int(word.start_ms)) + 1, int(word.end_ms)),
                    "text": word.text,
                }
                for word in seg.words
            ],
        }
        for seg in ordered
    ]

    payload = {
        "schema_version": 1,
        "job": dict(job_meta),
        "speakers": dict(speaker_aliases or {}),
        "turns": turns,
        "segments": serialized_segments,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _deduplicate_archive_name(filename: str, used_names: set[str]) -> str:
    safe_name = sanitize_filename(filename)
    if safe_name.casefold() not in used_names:
        used_names.add(safe_name.casefold())
        return safe_name

    parsed = Path(safe_name)
    stem = parsed.stem
    suffix = parsed.suffix
    counter = 2
    while True:
        candidate = f"{stem} ({counter}){suffix}"
        if candidate.casefold() not in used_names:
            used_names.add(candidate.casefold())
            return candidate
        counter += 1


def build_batch_zip(
    output_files: Sequence[Path],
    dest_zip: Path,
    tmp_dir: Path,
) -> Path:
    """Package output files into a deterministic ZIP archive, deduplicating names."""
    dest_zip = Path(dest_zip)
    tmp_dir = Path(tmp_dir)
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=tmp_dir,
            prefix=f".{dest_zip.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)

        with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in output_files:
                src = Path(file_path)
                if not src.is_file():
                    raise FileNotFoundError(f"Output file does not exist: {src}")

                arcname = _deduplicate_archive_name(src.name, used_names)
                zinfo = zipfile.ZipInfo(filename=arcname, date_time=_DETERMINISTIC_ZIP_DATE)
                zinfo.compress_type = zipfile.ZIP_DEFLATED
                zinfo.external_attr = 0o644 << 16
                with src.open("rb") as src_handle, zf.open(zinfo, mode="w") as dst_handle:
                    shutil.copyfileobj(src_handle, dst_handle)

        os.replace(tmp_path, dest_zip)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return dest_zip
