"""Deterministic post-ASR replacements and subtitle cue segmentation."""

from __future__ import annotations

import re
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(slots=True)
class WordTiming:
    """Single word timing emitted by an ASR backend."""

    start_ms: int
    end_ms: int
    text: str


@dataclass(slots=True)
class SegmentItem:
    """Normalized transcript segment used for cue segmentation and exports."""

    seq: int
    start_ms: int
    end_ms: int
    speaker_key: str | None
    raw_text: str
    text: str
    words: list[WordTiming] = field(default_factory=list)


@dataclass(slots=True)
class SubtitleCue:
    """Single subtitle cue ready for WebVTT or SubRip serialization."""

    index: int
    start_ms: int
    end_ms: int
    speaker_name: str | None
    lines: list[str]


def resolve_speaker_name(
    speaker_key: str | None,
    speaker_aliases: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve a speaker key to its display alias or fallback key."""
    if not speaker_key:
        return None
    if speaker_aliases and speaker_key in speaker_aliases:
        alias = speaker_aliases[speaker_key].strip()
        if alias:
            return alias
    cleaned = speaker_key.strip()
    return cleaned or None


def apply_replacements(
    text: str,
    rules: Mapping[str, str],
    case_sensitive: bool = False,
) -> str:
    """Apply deterministic post-ASR term replacements without cascading matches."""
    if not text or not rules:
        return text

    ordered_keys = sorted((key for key in rules if key), key=len, reverse=True)
    if not ordered_keys:
        return text

    patterns: list[str] = []
    lookup: dict[str, str] = {}
    for key in ordered_keys:
        prefix = r"\b" if key[0].isalnum() else ""
        suffix = r"\b" if key[-1].isalnum() else ""
        patterns.append(f"{prefix}{re.escape(key)}{suffix}")
        lookup_key = key if case_sensitive else key.lower()
        lookup.setdefault(lookup_key, rules[key])

    flags = 0 if case_sensitive else re.IGNORECASE
    compiled = re.compile("|".join(patterns), flags=flags)

    def _replace(match: re.Match[str]) -> str:
        matched = match.group(0)
        key = matched if case_sensitive else matched.lower()
        return lookup[key]

    return compiled.sub(_replace, text)


def _wrap_text(text: str, max_chars_per_line: int) -> list[str]:
    """Wrap normalized text into lines up to max_chars_per_line."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    return textwrap.wrap(
        normalized,
        width=max_chars_per_line,
        break_long_words=True,
        break_on_hyphens=True,
    )


def _join_word_tokens(words: Sequence[WordTiming]) -> str:
    """Join word tokens while respecting Whisper-style leading/trailing whitespace."""
    valid = [word for word in words if word.text and word.text.strip()]
    if not valid:
        return ""

    has_leading_ws = any(word.text[:1].isspace() for word in valid)
    has_trailing_ws = any(word.text[-1:].isspace() for word in valid)

    parts: list[str] = []
    for idx, word in enumerate(valid):
        clean = word.text.strip()
        if idx == 0:
            parts.append(clean)
            continue

        if has_leading_ws:
            needs_space = word.text[:1].isspace()
        elif has_trailing_ws:
            needs_space = valid[idx - 1].text[-1:].isspace()
        else:
            needs_space = True

        parts.append(f" {clean}" if needs_space else clean)

    return "".join(parts)


def _should_use_words(segment: SegmentItem) -> bool:
    """Return True when segment word timings match the active segment text."""
    if not segment.words:
        return False
    joined = _join_word_tokens(segment.words)
    if not joined:
        return False
    if not segment.text or segment.text == segment.raw_text:
        return True
    return re.sub(r"\s+", "", joined) == re.sub(r"\s+", "", segment.text)


def _segment_cues_from_words(
    segment: SegmentItem,
    speaker_name: str | None,
    max_chars_per_line: int,
    max_lines_per_cue: int,
    max_cue_ms: int,
) -> list[tuple[int, int, str | None, list[str]]]:
    """Group word timings into cues respecting line limits and max_cue_ms."""
    valid_words = [word for word in segment.words if word.text and word.text.strip()]
    if not valid_words:
        return []

    raw_cues: list[tuple[int, int, str | None, list[str]]] = []
    current_words: list[WordTiming] = []
    current_lines: list[str] = []
    cue_start_ms = 0
    cue_end_ms = 0
    prev_subcue_end_ms = max(0, segment.start_ms)

    for word in valid_words:
        w_start = max(0, int(word.start_ms))
        w_end = max(w_start + 1, int(word.end_ms))

        if not current_words:
            cue_start_ms = max(prev_subcue_end_ms, w_start)
            cue_end_ms = max(cue_start_ms + 1, w_end)
            current_words = [word]
            current_lines = _wrap_text(_join_word_tokens(current_words), max_chars_per_line)
            continue

        candidate_words = [*current_words, word]
        candidate_end_ms = max(cue_end_ms, w_end)
        candidate_lines = _wrap_text(_join_word_tokens(candidate_words), max_chars_per_line)

        exceeds_lines = len(candidate_lines) > max_lines_per_cue
        exceeds_duration = (candidate_end_ms - cue_start_ms) > max_cue_ms

        if exceeds_lines or exceeds_duration:
            final_end_ms = max(cue_start_ms + 1, cue_end_ms)
            raw_cues.append((cue_start_ms, final_end_ms, speaker_name, current_lines))
            prev_subcue_end_ms = final_end_ms

            cue_start_ms = max(prev_subcue_end_ms, w_start)
            cue_end_ms = max(cue_start_ms + 1, w_end)
            current_words = [word]
            current_lines = _wrap_text(_join_word_tokens(current_words), max_chars_per_line)
        else:
            current_words = candidate_words
            current_lines = candidate_lines
            cue_end_ms = candidate_end_ms

    if current_words and current_lines:
        final_end_ms = max(cue_start_ms + 1, cue_end_ms)
        raw_cues.append((cue_start_ms, final_end_ms, speaker_name, current_lines))

    return raw_cues


def _allocate_proportional_chunks(
    chunks: list[list[str]],
    seg_start: int,
    seg_end: int,
) -> list[tuple[int, int, list[str]]]:
    """Distribute [seg_start, seg_end] across line chunks by character count."""
    count = len(chunks)
    if count == 0:
        return []

    seg_start = max(0, seg_start)
    seg_end = max(seg_start + 1, seg_end)
    total_duration = seg_end - seg_start

    weights = [max(1, sum(len(line) for line in chunk)) for chunk in chunks]
    total_weight = sum(weights)

    allocated: list[tuple[int, int, list[str]]] = []
    prev_end = seg_start
    cum_weight = 0

    for idx, chunk in enumerate(chunks):
        chunk_start = seg_start if idx == 0 else prev_end
        if idx == count - 1:
            chunk_end = max(chunk_start + 1, seg_end)
        else:
            cum_weight += weights[idx]
            target_end = seg_start + round(total_duration * cum_weight / total_weight)
            remaining = count - 1 - idx
            if total_duration >= count:
                chunk_end = max(chunk_start + 1, min(target_end, seg_end - remaining))
            else:
                chunk_end = chunk_start + 1
        allocated.append((chunk_start, chunk_end, chunk))
        prev_end = chunk_end

    return allocated


def _split_chunk_for_duration(
    chunk: list[str],
    chunk_duration_ms: int,
    max_cue_ms: int,
) -> list[list[str]]:
    """Split a multi-line chunk into smaller line groups when duration exceeds max_cue_ms."""
    if len(chunk) <= 1 or chunk_duration_ms <= max_cue_ms:
        return [chunk]

    total_chars = max(1, sum(len(line) for line in chunk))
    groups: list[list[str]] = []
    current_group: list[str] = []
    current_chars = 0

    for line in chunk:
        line_chars = max(1, len(line))
        projected_ms = chunk_duration_ms * (current_chars + line_chars) / total_chars
        if current_group and projected_ms > max_cue_ms:
            groups.append(current_group)
            current_group = [line]
            current_chars = line_chars
        else:
            current_group.append(line)
            current_chars += line_chars

    if current_group:
        groups.append(current_group)
    return groups


def _segment_cues_from_text(
    segment: SegmentItem,
    speaker_name: str | None,
    max_chars_per_line: int,
    max_lines_per_cue: int,
    max_cue_ms: int,
) -> list[tuple[int, int, str | None, list[str]]]:
    """Split segment text with textwrap and distribute timestamps proportionally."""
    source_text = segment.text if segment.text.strip() else segment.raw_text
    lines = _wrap_text(source_text, max_chars_per_line)
    if not lines:
        return []

    initial_chunks = [
        lines[idx : idx + max_lines_per_cue]
        for idx in range(0, len(lines), max_lines_per_cue)
    ]

    seg_start = max(0, int(segment.start_ms))
    seg_end = max(seg_start + 1, int(segment.end_ms))
    initial_allocated = _allocate_proportional_chunks(initial_chunks, seg_start, seg_end)

    refined_chunks: list[list[str]] = []
    for c_start, c_end, chunk in initial_allocated:
        refined_chunks.extend(_split_chunk_for_duration(chunk, c_end - c_start, max_cue_ms))

    final_allocated = _allocate_proportional_chunks(refined_chunks, seg_start, seg_end)
    return [(c_start, c_end, speaker_name, chunk) for c_start, c_end, chunk in final_allocated]


def segment_to_cues(
    segments: Sequence[SegmentItem],
    speaker_aliases: Mapping[str, str] | None = None,
    max_chars_per_line: int = 68,
    max_lines_per_cue: int = 2,
    max_cue_seconds: float = 6.0,
) -> list[SubtitleCue]:
    """Convert transcript segments into chronologically ordered SubtitleCue objects."""
    if not segments:
        return []

    safe_max_chars = max(1, int(max_chars_per_line))
    safe_max_lines = max(1, int(max_lines_per_cue))
    safe_max_cue_ms = max(1, round(max_cue_seconds * 1000))

    ordered_segments = sorted(segments, key=lambda seg: (seg.start_ms, seg.seq, seg.end_ms))

    cues: list[SubtitleCue] = []
    prev_start_ms = 0

    for segment in ordered_segments:
        speaker_name = resolve_speaker_name(segment.speaker_key, speaker_aliases)
        if _should_use_words(segment):
            raw_cues = _segment_cues_from_words(
                segment,
                speaker_name,
                safe_max_chars,
                safe_max_lines,
                safe_max_cue_ms,
            )
        else:
            raw_cues = _segment_cues_from_text(
                segment,
                speaker_name,
                safe_max_chars,
                safe_max_lines,
                safe_max_cue_ms,
            )

        for start_ms, end_ms, cue_speaker, lines in raw_cues:
            clamped_start = max(prev_start_ms, max(0, start_ms))
            clamped_end = max(clamped_start + 1, end_ms)
            cues.append(
                SubtitleCue(
                    index=len(cues) + 1,
                    start_ms=clamped_start,
                    end_ms=clamped_end,
                    speaker_name=cue_speaker,
                    lines=lines,
                )
            )
            prev_start_ms = clamped_start

    return cues
