"""Parsers for multitrack metadata files (Craig info.txt and Audacity .aup XML)."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_TRACK_LINE_RE = re.compile(r"^track\s+(\d+)\s*[:\-]\s*(.+)$", re.IGNORECASE)
_FILE_SPEAKER_RE = re.compile(r"^([^:\r\n]+?\.[A-Za-z0-9]{2,5})\s*:\s*(.+)$")
_NUMBERED_NAME_RE = re.compile(r"^(\d+)\s*[-_]\s*([^\s:][^\r\n]*)$")
_TRACK_PREFIX_RE = re.compile(r"^(?:track[_\-\s]*)?(\d+)(?:[_\-\s\.]|$)", re.IGNORECASE)
_UNSAFE_XML_RE = re.compile(r"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)
_MEDIA_EXT_HINT_RE = re.compile(
    r"\.(?:wav|flac|mp3|ogg|opus|m4a|aac|mp4|webm|mkv)$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class TrackHint:
    """Speaker and timeline offset metadata extracted for a multitrack file."""

    filename: str
    speaker_name: str
    offset_ms: int = 0


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _parse_offset_ms(raw_offset: str | None) -> int:
    if raw_offset is None:
        return 0
    try:
        seconds = float(raw_offset.strip())
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(seconds) or seconds <= 0:
        return 0
    return round(seconds * 1000)


def _extract_track_number(name: str) -> str | None:
    stem = Path(name).stem if "." in name else name
    match = _TRACK_PREFIX_RE.match(stem.strip())
    if not match:
        return None
    return str(int(match.group(1)))


class CraigInfoParser:
    """Parse Craig multitrack recording `info.txt` files into TrackHint mappings."""

    @classmethod
    def parse(cls, info_text: str | None) -> dict[str, TrackHint]:
        """Extract track hints from Craig `info.txt` content without failing on unknown lines."""
        if not info_text or not info_text.strip():
            return {}

        hints: dict[str, TrackHint] = {}
        for raw_line in info_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            track_match = _TRACK_LINE_RE.match(line)
            if track_match:
                track_num = str(int(track_match.group(1)))
                speaker = track_match.group(2).strip()
                if speaker:
                    hints[track_num] = TrackHint(
                        filename=track_num,
                        speaker_name=speaker,
                        offset_ms=0,
                    )
                continue

            file_match = _FILE_SPEAKER_RE.match(line)
            if file_match:
                filename = file_match.group(1).strip()
                speaker = file_match.group(2).strip()
                if filename and speaker:
                    hints[filename] = TrackHint(
                        filename=filename,
                        speaker_name=speaker,
                        offset_ms=0,
                    )
                continue

            numbered_match = _NUMBERED_NAME_RE.match(line)
            if numbered_match:
                track_num = str(int(numbered_match.group(1)))
                rest = numbered_match.group(2).strip()
                if rest:
                    speaker = Path(rest).stem if _MEDIA_EXT_HINT_RE.search(rest) else rest
                    hints[line] = TrackHint(
                        filename=line,
                        speaker_name=speaker,
                        offset_ms=0,
                    )
                    hints.setdefault(
                        track_num,
                        TrackHint(
                            filename=track_num,
                            speaker_name=speaker,
                            offset_ms=0,
                        ),
                    )

        return hints


class AudacityAupParser:
    """Safely parse Audacity `.aup` XML files into TrackHint mappings."""

    @classmethod
    def parse(cls, aup_text: str | None) -> dict[str, TrackHint]:
        """Extract wavetrack names and first waveclip offsets from `.aup` XML."""
        if not aup_text or not aup_text.strip():
            return {}

        if _UNSAFE_XML_RE.search(aup_text):
            raise ValueError("Unsafe XML declarations (DOCTYPE/ENTITY) are not allowed in .aup")

        try:
            root = ET.fromstring(aup_text)
        except ET.ParseError:
            return {}

        hints: dict[str, TrackHint] = {}
        for elem in root.iter():
            if _xml_local_name(elem.tag) != "wavetrack":
                continue

            track_name = (elem.attrib.get("name") or "").strip()
            first_clip: ET.Element | None = None
            for child in elem.iter():
                if child is not elem and _xml_local_name(child.tag) == "waveclip":
                    first_clip = child
                    break

            offset_ms = (
                _parse_offset_ms(first_clip.attrib.get("offset"))
                if first_clip is not None
                else 0
            )
            clip_filename = (
                (first_clip.attrib.get("filename") or "").strip()
                if first_clip is not None
                else ""
            )

            if track_name:
                speaker = (
                    Path(track_name).stem
                    if _MEDIA_EXT_HINT_RE.search(track_name)
                    else track_name
                )
                hints[track_name] = TrackHint(
                    filename=track_name,
                    speaker_name=speaker or track_name,
                    offset_ms=offset_ms,
                )

            if clip_filename:
                speaker_for_clip = (
                    hints[track_name].speaker_name
                    if track_name and track_name in hints
                    else Path(clip_filename).stem
                )
                hints[clip_filename] = TrackHint(
                    filename=clip_filename,
                    speaker_name=speaker_for_clip,
                    offset_ms=offset_ms,
                )

        return hints


def parse_track_hints(
    info_text: str | None = None,
    aup_text: str | None = None,
) -> dict[str, TrackHint]:
    """Parse Craig `info.txt` and Audacity `.aup` metadata into a combined hint map."""
    craig_hints = CraigInfoParser.parse(info_text)
    aup_hints = AudacityAupParser.parse(aup_text)

    if not aup_hints:
        return craig_hints
    if not craig_hints:
        return aup_hints

    merged: dict[str, TrackHint] = dict(craig_hints)
    craig_lower = {k.lower(): v for k, v in craig_hints.items()}

    for key, aup_hint in aup_hints.items():
        speaker_name = aup_hint.speaker_name
        existing = craig_hints.get(key) or craig_lower.get(key.lower())
        if existing is not None:
            speaker_name = existing.speaker_name
        else:
            track_num = _extract_track_number(key)
            if track_num and track_num in craig_hints and _NUMBERED_NAME_RE.match(key):
                speaker_name = craig_hints[track_num].speaker_name

        merged[key] = TrackHint(
            filename=aup_hint.filename,
            speaker_name=speaker_name,
            offset_ms=aup_hint.offset_ms,
        )

    return merged


def resolve_track_hint(
    filename: str,
    hints: Mapping[str, TrackHint] | None = None,
    *,
    original_filename: str | None = None,
) -> TrackHint:
    """Resolve a TrackHint for `filename`, falling back to stem name and 0 ms offset."""
    if not hints:
        return TrackHint(filename=filename, speaker_name=Path(filename).stem, offset_ms=0)

    hints_lower = {key.lower(): hint for key, hint in hints.items()}
    names_to_check = [filename]
    if original_filename and original_filename != filename:
        names_to_check.append(original_filename)

    matched_direct: TrackHint | None = None
    for candidate in names_to_check:
        stem = Path(candidate).stem
        for key in (candidate, stem):
            if key in hints:
                matched_direct = hints[key]
                break
            lowered = key.lower()
            if lowered in hints_lower:
                matched_direct = hints_lower[lowered]
                break
        if matched_direct is not None:
            break

    matched_track: TrackHint | None = None
    for candidate in names_to_check:
        track_num = _extract_track_number(candidate)
        if track_num and track_num in hints:
            matched_track = hints[track_num]
            break

    if matched_direct is not None:
        speaker_name = matched_direct.speaker_name
        if (
            matched_track is not None
            and matched_direct.speaker_name == Path(matched_direct.filename).stem
            and _NUMBERED_NAME_RE.match(matched_direct.filename)
        ):
            speaker_name = matched_track.speaker_name
        return TrackHint(
            filename=filename,
            speaker_name=speaker_name,
            offset_ms=matched_direct.offset_ms,
        )

    if matched_track is not None:
        return TrackHint(
            filename=filename,
            speaker_name=matched_track.speaker_name,
            offset_ms=matched_track.offset_ms,
        )

    return TrackHint(filename=filename, speaker_name=Path(filename).stem, offset_ms=0)
