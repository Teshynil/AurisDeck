"""Tests for aurisdeck.export (CueSegmenter and exporters)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from aurisdeck.export import (
    SegmentItem,
    SubtitleCue,
    WordTiming,
    apply_replacements,
    build_batch_zip,
    escape_vtt_text,
    export_json,
    export_markdown,
    export_srt,
    export_vtt,
    format_srt_time,
    format_vtt_time,
    segment_to_cues,
)


def test_time_formatting_vtt_and_srt() -> None:
    assert format_vtt_time(62_340) == "00:01:02.340"
    assert format_srt_time(62_340) == "00:01:02,340"
    assert format_vtt_time(3_661_005) == "01:01:01.005"
    assert format_srt_time(3_661_005) == "01:01:01,005"
    assert format_vtt_time(-50) == "00:00:00.000"
    assert format_srt_time(-50) == "00:00:00,000"


def test_escape_vtt_text() -> None:
    assert escape_vtt_text("C++ <int> & bash") == "C++ &lt;int&gt; &amp; bash"
    assert escape_vtt_text("a & b < c > d") == "a &amp; b &lt; c &gt; d"


def test_apply_replacements_boundaries_order_and_case() -> None:
    rules = {
        "C++": "CPP",
        "C": "Clang",
        "San Francisco": "SF",
        "Francisco": "Frank",
        "ai": "IA",
    }
    text = "Wait for ai in San Francisco with Francisco using C++ and C."
    replaced = apply_replacements(text, rules)
    assert replaced == "Wait for IA in SF with Frank using CPP and Clang."

    assert apply_replacements("AI and ai", {"ai": "IA"}, case_sensitive=True) == "AI and IA"
    assert apply_replacements("a b", {"a": "b", "b": "c"}) == "b c"
    assert apply_replacements("no changes", {}) == "no changes"


def test_segment_to_cues_with_words_wrapping_and_duration() -> None:
    words = [
        WordTiming(0, 800, "Primera"),
        WordTiming(800, 1600, "línea"),
        WordTiming(1600, 2400, "bastante"),
        WordTiming(2400, 3200, "larga"),
        WordTiming(3200, 4000, "para"),
        WordTiming(4000, 4800, "probar"),
        WordTiming(4800, 5600, "el"),
        WordTiming(5600, 6400, "corte"),
        WordTiming(6400, 7200, "automático"),
        WordTiming(7200, 8000, "entre"),
        WordTiming(8000, 8800, "subtítulos"),
        WordTiming(8800, 9600, "consecutivos."),
    ]
    segment = SegmentItem(
        seq=1,
        start_ms=0,
        end_ms=9600,
        speaker_key="SPEAKER_00",
        raw_text=" ".join(w.text for w in words),
        text=" ".join(w.text for w in words),
        words=words,
    )

    cues = segment_to_cues(
        [segment],
        speaker_aliases={"SPEAKER_00": "Alice <Admin> & Host"},
        max_chars_per_line=42,
        max_lines_per_cue=2,
        max_cue_seconds=6.0,
    )

    assert len(cues) >= 2
    prev_start = 0
    for idx, cue in enumerate(cues, start=1):
        assert cue.index == idx
        assert cue.start_ms < cue.end_ms
        assert cue.start_ms >= prev_start
        assert cue.end_ms - cue.start_ms <= 6000
        assert len(cue.lines) <= 2
        assert all(len(line) <= 42 for line in cue.lines)
        assert cue.speaker_name == "Alice <Admin> & Host"
        prev_start = cue.start_ms


def test_segment_to_cues_without_words_proportional_and_wrapping() -> None:
    text = (
        "Esta es una frase suficientemente extensa para dividirse en varias "
        "líneas respetando exactamente cuarenta y dos caracteres por línea."
    )
    segment = SegmentItem(
        seq=1,
        start_ms=1000,
        end_ms=11_000,
        speaker_key="SPEAKER_01",
        raw_text=text,
        text=text,
        words=[],
    )

    cues = segment_to_cues(
        [segment],
        speaker_aliases={"SPEAKER_01": "Bob"},
        max_chars_per_line=42,
        max_lines_per_cue=2,
        max_cue_seconds=6.0,
    )

    assert len(cues) >= 2
    assert cues[0].start_ms == 1000
    assert cues[-1].end_ms == 11_000
    for cue in cues:
        assert cue.start_ms < cue.end_ms
        assert len(cue.lines) <= 2
        assert all(len(line) <= 42 for line in cue.lines)
        assert cue.end_ms - cue.start_ms <= 6000

    reconstructed = " ".join(" ".join(cue.lines) for cue in cues)
    assert reconstructed == text


def test_segment_to_cues_whisper_spacing_and_post_asr_replacement_preserved() -> None:
    words = [
        WordTiming(0, 400, " Hola"),
        WordTiming(400, 800, " mundo"),
        WordTiming(800, 900, ","),
        WordTiming(900, 1400, " Python"),
        WordTiming(1400, 1500, "!"),
    ]
    seg_unchanged = SegmentItem(
        seq=1,
        start_ms=0,
        end_ms=1500,
        speaker_key=None,
        raw_text="Hola mundo, Python!",
        text="Hola mundo, Python!",
        words=words,
    )
    seg_replaced = SegmentItem(
        seq=2,
        start_ms=1500,
        end_ms=3000,
        speaker_key=None,
        raw_text="see plus plus",
        text="C++",
        words=[
            WordTiming(1500, 2000, "see"),
            WordTiming(2000, 2500, "plus"),
            WordTiming(2500, 3000, "plus"),
        ],
    )

    cues = segment_to_cues([seg_replaced, seg_unchanged], max_chars_per_line=42, max_lines_per_cue=2)
    assert len(cues) == 2
    assert cues[0].lines == ["Hola mundo, Python!"]
    assert cues[0].start_ms == 0
    assert cues[1].lines == ["C++"]
    assert cues[1].start_ms == 1500


def test_export_vtt_styles_and_escaping_after_wrap() -> None:
    # Unescaped length is 38 chars (<= 42), whereas escaped length is 48 chars (> 42).
    # Wrapping MUST happen before escaping, and <v ...> MUST NOT count toward max_chars_per_line.
    raw_line = "Use C++ <int> & bash for safe scripts!"
    assert len(raw_line) == 38

    seg = SegmentItem(
        seq=1,
        start_ms=1_230,
        end_ms=62_340,
        speaker_key="SPK_1",
        raw_text=raw_line,
        text=raw_line,
    )
    cues = segment_to_cues(
        [seg],
        speaker_aliases={"SPK_1": "Nombre Muy Largo <Lead> & Co"},
        max_chars_per_line=42,
        max_lines_per_cue=2,
    )
    assert len(cues) == 1
    assert cues[0].lines == [raw_line]

    vtt_vtag = export_vtt(cues, language="es", voice_style="v_tag")
    assert vtt_vtag.startswith("WEBVTT\nKind: captions\nLanguage: es\n\n")
    assert "00:00:01.230 --> 00:01:02.340\n" in vtt_vtag
    assert (
        "<v Nombre Muy Largo &lt;Lead&gt; &amp; Co>Use C++ &lt;int&gt; &amp; bash for safe scripts!</v>"
        in vtt_vtag
    )

    vtt_prefix = export_vtt(cues, voice_style="prefix")
    assert (
        "[Nombre Muy Largo &lt;Lead&gt; &amp; Co]: Use C++ &lt;int&gt; &amp; bash for safe scripts!"
        in vtt_prefix
    )

    vtt_none = export_vtt(cues, voice_style="none")
    assert "<v " not in vtt_none
    assert "[Nombre" not in vtt_none
    assert "Use C++ &lt;int&gt; &amp; bash for safe scripts!" in vtt_none

    solo_cue = SubtitleCue(index=1, start_ms=0, end_ms=1000, speaker_name=None, lines=["Solo"])
    vtt_solo = export_vtt([solo_cue], voice_style="v_tag")
    assert "<v " not in vtt_solo
    assert "00:00:00.000 --> 00:00:01.000\nSolo\n" in vtt_solo

    with pytest.raises(ValueError):
        export_vtt(cues, voice_style="invalid")


def test_export_srt_modes() -> None:
    cues = [
        SubtitleCue(
            index=1,
            start_ms=1_000,
            end_ms=62_340,
            speaker_name="Elena",
            lines=["Usamos C++ <int> & bash", "en producción."],
        )
    ]

    srt_prefix = export_srt(cues, voice_style="prefix")
    assert srt_prefix == (
        "1\n"
        "00:00:01,000 --> 00:01:02,340\n"
        "[Elena]: Usamos C++ <int> & bash\n"
        "en producción.\n"
    )

    srt_none = export_srt(cues, voice_style="none")
    assert "[Elena]:" not in srt_none
    assert "Usamos C++ <int> & bash\nen producción.\n" in srt_none


def test_export_markdown_and_json_group_consecutive_speakers() -> None:
    segments = [
        SegmentItem(1, 5_000, 10_000, "SPK_0", "Hola", "Hola"),
        SegmentItem(2, 10_500, 15_000, "SPK_0", "¿Cómo estás?", "¿Cómo estás?"),
        SegmentItem(3, 65_000, 70_000, "SPK_1", "Muy bien.", "Muy bien."),
        SegmentItem(4, 3_665_000, 3_670_000, "SPK_0", "Seguimos.", "Seguimos."),
    ]
    aliases = {"SPK_0": "Ana", "SPK_1": "Luis"}

    md = export_markdown(segments, aliases, title="Reunión Semanal")
    assert md == (
        "# Reunión Semanal\n\n"
        "[00:05] **Ana:** Hola ¿Cómo estás?\n\n"
        "[01:05] **Luis:** Muy bien.\n\n"
        "[01:01:05] **Ana:** Seguimos.\n"
    )

    raw_json = export_json({"id": "job-1", "title": "Reunión Semanal"}, segments, aliases)
    parsed = json.loads(raw_json)
    assert parsed["schema_version"] == 1
    assert parsed["job"]["id"] == "job-1"
    assert len(parsed["turns"]) == 3
    assert parsed["turns"][0]["speaker_name"] == "Ana"
    assert parsed["turns"][0]["text"] == "Hola ¿Cómo estás?"
    assert parsed["turns"][0]["segment_seqs"] == [1, 2]
    assert len(parsed["segments"]) == 4


def test_build_batch_zip_deduplicates_filenames_and_cleans_tmp(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_c = tmp_path / "c"
    dir_a.mkdir()
    dir_b.mkdir()
    dir_c.mkdir()

    f1 = dir_a / "archivo.vtt"
    f2 = dir_b / "archivo.vtt"
    f3 = dir_c / "archivo.vtt"
    f4 = dir_a / "resumen.md"

    f1.write_text("vtt-1", encoding="utf-8")
    f2.write_text("vtt-2", encoding="utf-8")
    f3.write_text("vtt-3", encoding="utf-8")
    f4.write_text("md-1", encoding="utf-8")

    tmp_dir = tmp_path / "tmp"
    dest_zip = tmp_path / "outputs" / "batch.zip"

    result = build_batch_zip([f1, f2, f3, f4], dest_zip, tmp_dir)
    assert result == dest_zip
    assert dest_zip.is_file()
    assert list(tmp_dir.iterdir()) == []

    with zipfile.ZipFile(dest_zip, "r") as zf:
        assert zf.namelist() == [
            "archivo.vtt",
            "archivo (2).vtt",
            "archivo (3).vtt",
            "resumen.md",
        ]
        assert zf.read("archivo.vtt").decode("utf-8") == "vtt-1"
        assert zf.read("archivo (2).vtt").decode("utf-8") == "vtt-2"
        assert zf.read("archivo (3).vtt").decode("utf-8") == "vtt-3"
        assert zf.read("resumen.md").decode("utf-8") == "md-1"

    dest_zip_2 = tmp_path / "outputs" / "batch2.zip"
    build_batch_zip([f1, f2, f3, f4], dest_zip_2, tmp_dir)
    assert dest_zip.read_bytes() == dest_zip_2.read_bytes()
