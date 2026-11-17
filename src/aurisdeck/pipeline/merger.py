"""Transcript merging across diarization turns and multitrack recordings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from aurisdeck.backends.base import ASRResult, ASRWord, SpeakerTurn
from aurisdeck.export.cue_segmenter import SegmentItem, WordTiming, apply_replacements

_DEFAULT_SPEAKER_KEY = "SPEAKER_00"


def _resolve_speaker_for_interval(
    start_ms: int,
    end_ms: int,
    turns: Sequence[SpeakerTurn],
) -> str:
    """Return the speaker key with max overlap, or nearest midpoint if no overlap."""
    item_mid = (start_ms + end_ms) / 2.0

    best_overlap = 0
    best_overlap_dist = float("inf")
    best_overlap_speaker: str | None = None

    nearest_dist = float("inf")
    nearest_speaker = turns[0].speaker_key

    for turn in turns:
        turn_mid = (turn.start_ms + turn.end_ms) / 2.0
        dist = abs(item_mid - turn_mid)

        overlap = min(end_ms, turn.end_ms) - max(start_ms, turn.start_ms)
        if overlap > 0 and (
            overlap > best_overlap
            or (overlap == best_overlap and dist < best_overlap_dist)
        ):
            best_overlap = overlap
            best_overlap_dist = dist
            best_overlap_speaker = turn.speaker_key

        if dist < nearest_dist:
            nearest_dist = dist
            nearest_speaker = turn.speaker_key

    return best_overlap_speaker if best_overlap_speaker is not None else nearest_speaker


def _join_words_cleanly(words: Sequence[ASRWord]) -> str:
    """Join word texts cleanly with single spaces and no leading/trailing padding."""
    return " ".join(cleaned for word in words if (cleaned := word.text.strip()))


def _build_segment_item(
    seq: int,
    start_ms: int,
    end_ms: int,
    speaker_key: str | None,
    raw_text: str,
    words: list[WordTiming],
    replacements: Mapping[str, str],
    case_sensitive: bool,
) -> SegmentItem:
    text = apply_replacements(raw_text, replacements, case_sensitive=case_sensitive)
    return SegmentItem(
        seq=seq,
        start_ms=start_ms,
        end_ms=end_ms,
        speaker_key=speaker_key,
        raw_text=raw_text,
        text=text,
        words=words,
    )


def _split_words_by_speaker(
    words: Sequence[ASRWord],
    turns: Sequence[SpeakerTurn],
    replacements: Mapping[str, str],
    case_sensitive: bool,
    start_seq: int,
) -> list[SegmentItem]:
    """Group contiguous words sharing the same resolved speaker into SegmentItems."""
    groups: list[tuple[str | None, list[ASRWord]]] = []
    for word in words:
        speaker_key = _resolve_speaker_for_interval(word.start_ms, word.end_ms, turns)
        if groups and groups[-1][0] == speaker_key:
            groups[-1][1].append(word)
        else:
            groups.append((speaker_key, [word]))

    items: list[SegmentItem] = []
    for idx, (speaker_key, group) in enumerate(groups):
        items.append(
            _build_segment_item(
                seq=start_seq + idx,
                start_ms=min(w.start_ms for w in group),
                end_ms=max(w.end_ms for w in group),
                speaker_key=speaker_key,
                raw_text=_join_words_cleanly(group),
                words=[WordTiming(w.start_ms, w.end_ms, w.text) for w in group],
                replacements=replacements,
                case_sensitive=case_sensitive,
            )
        )
    return items


class TranscriptMerger:
    """Merges ASR results with diarization turns or multitrack offsets."""

    @staticmethod
    def merge_single(
        asr: ASRResult,
        turns: list[SpeakerTurn] | None,
        replacements: Mapping[str, str],
        case_sensitive: bool = False,
    ) -> list[SegmentItem]:
        """Merge a single-track ASRResult with optional speaker turns."""
        items: list[SegmentItem] = []

        if not turns:
            for seg in asr.segments:
                words = [WordTiming(w.start_ms, w.end_ms, w.text) for w in seg.words]
                items.append(
                    _build_segment_item(
                        seq=len(items),
                        start_ms=seg.start_ms,
                        end_ms=seg.end_ms,
                        speaker_key=_DEFAULT_SPEAKER_KEY,
                        raw_text=seg.text,
                        words=words,
                        replacements=replacements,
                        case_sensitive=case_sensitive,
                    )
                )
            return items

        for seg in asr.segments:
            valid_words = [word for word in seg.words if word.text and word.text.strip()]
            if not valid_words:
                speaker_key = _resolve_speaker_for_interval(seg.start_ms, seg.end_ms, turns)
                items.append(
                    _build_segment_item(
                        seq=len(items),
                        start_ms=seg.start_ms,
                        end_ms=seg.end_ms,
                        speaker_key=speaker_key,
                        raw_text=seg.text,
                        words=[],
                        replacements=replacements,
                        case_sensitive=case_sensitive,
                    )
                )
                continue

            items.extend(
                _split_words_by_speaker(
                    words=valid_words,
                    turns=turns,
                    replacements=replacements,
                    case_sensitive=case_sensitive,
                    start_seq=len(items),
                )
            )

        return items

    @staticmethod
    def merge_multitrack(
        tracks: list[tuple[str, str, int, ASRResult]],
        replacements: Mapping[str, str],
        case_sensitive: bool = False,
    ) -> list[SegmentItem]:
        """Merge multiple per-speaker tracks applying offsets and chronological sorting."""
        items: list[SegmentItem] = []

        for _track_id, speaker_key, offset_ms, asr_result in tracks:
            for seg in asr_result.segments:
                shifted_words = [
                    WordTiming(
                        start_ms=w.start_ms + offset_ms,
                        end_ms=w.end_ms + offset_ms,
                        text=w.text,
                    )
                    for w in seg.words
                ]
                items.append(
                    _build_segment_item(
                        seq=0,
                        start_ms=seg.start_ms + offset_ms,
                        end_ms=seg.end_ms + offset_ms,
                        speaker_key=speaker_key,
                        raw_text=seg.text,
                        words=shifted_words,
                        replacements=replacements,
                        case_sensitive=case_sensitive,
                    )
                )

        items.sort(key=lambda item: (item.start_ms, item.end_ms))
        for idx, item in enumerate(items):
            item.seq = idx

        return items
