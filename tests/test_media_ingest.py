"""Tests for media probing/extraction, .aup/info.txt parsers, and safe ZIP ingestion."""

from __future__ import annotations

import stat
import subprocess
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from aurisdeck.media import (
    SUPPORTED_MEDIA_EXTENSIONS,
    AudacityAupParser,
    CraigInfoParser,
    MediaInvalidError,
    TrackHint,
    ZipSlipError,
    extract_wav_16k,
    parse_track_hints,
    probe_duration_ms,
    unpack_multitrack_zip,
)
from aurisdeck.media.ffmpeg import mix_multitrack_audio


def _write_zip(zip_path: Path, entries: list[tuple[str, bytes]]) -> Path:
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return zip_path


def test_zip_slip_attack_rejected_and_writes_nothing_outside_dest(tmp_path: Path) -> None:
    dest_dir = tmp_path / "dest"
    escaped_target = tmp_path / "evil.wav"
    zip_path = _write_zip(
        tmp_path / "attack.zip",
        [
            ("good.wav", b"RIFFgood"),
            ("../../evil.wav", b"RIFFevil"),
        ],
    )

    with pytest.raises(ZipSlipError) as exc_info:
        unpack_multitrack_zip(zip_path, dest_dir)

    assert isinstance(exc_info.value, ValueError)
    assert isinstance(exc_info.value, MediaInvalidError)
    assert exc_info.value.error_code == "ZIP_SLIP"
    assert not escaped_target.exists()
    assert not (dest_dir / "good.wav").exists()


@pytest.mark.parametrize(
    "bad_member",
    [
        "/etc/evil.wav",
        r"..\..\evil.wav",
        "C:/evil.wav",
        "sub/../../evil.wav",
    ],
)
def test_zip_slip_variants_rejected(tmp_path: Path, bad_member: str) -> None:
    zip_path = _write_zip(tmp_path / "bad.zip", [(bad_member, b"data")])
    with pytest.raises(ZipSlipError):
        unpack_multitrack_zip(zip_path, tmp_path / "out")


def test_zip_symlink_entry_rejected(tmp_path: Path) -> None:
    zip_path = tmp_path / "symlink.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        info = zipfile.ZipInfo("link.wav")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/etc/passwd")

    with pytest.raises(ZipSlipError):
        unpack_multitrack_zip(zip_path, tmp_path / "out")


def test_zip_ignores_macos_metadata_and_unsupported_files(tmp_path: Path) -> None:
    zip_path = _write_zip(
        tmp_path / "macos.zip",
        [
            ("__MACOSX/._1.wav", b"junk"),
            ("__MACOSX/1.wav", b"junk"),
            ("tracks/._speaker.flac", b"junk"),
            (".DS_Store", b"junk"),
            ("tracks/.DS_Store", b"junk"),
            ("tracks/readme.txt", b"not audio"),
            ("tracks/alice.wav", b"RIFFalice"),
        ],
    )

    dest_dir = tmp_path / "extracted"
    result = unpack_multitrack_zip(zip_path, dest_dir)

    assert len(result) == 1
    extracted_path, hint = result[0]
    assert extracted_path == dest_dir.resolve() / "alice.wav"
    assert extracted_path.read_bytes() == b"RIFFalice"
    assert hint == TrackHint(filename="alice.wav", speaker_name="alice", offset_ms=0)


def test_zip_deduplicates_identical_filenames_across_subdirs(tmp_path: Path) -> None:
    zip_path = _write_zip(
        tmp_path / "dups.zip",
        [
            ("a/track.wav", b"audio-a"),
            ("b/track.wav", b"audio-b"),
            ("c/track.wav", b"audio-c"),
        ],
    )

    dest_dir = tmp_path / "extracted"
    result = unpack_multitrack_zip(zip_path, dest_dir)

    paths = [path for path, _ in result]
    assert [p.name for p in paths] == ["track.wav", "track_2.wav", "track_3.wav"]
    assert {p.read_bytes() for p in paths} == {b"audio-a", b"audio-b", b"audio-c"}


def test_zip_unpacks_skipping_excluded_members(tmp_path: Path) -> None:
    zip_path = _write_zip(
        tmp_path / "excluded.zip",
        [
            ("a/track.wav", b"audio-a"),
            ("b/track.wav", b"audio-b"),
            ("c/bot.flac", b"bot-audio"),
        ],
    )
    dest_dir = tmp_path / "out_excluded"
    result = unpack_multitrack_zip(
        zip_path,
        dest_dir,
        excluded_members=["a/track.wav", "bot.flac"],
    )

    paths = [path for path, _ in result]
    assert [p.name for p in paths] == ["track.wav"]
    assert paths[0].read_bytes() == b"audio-b"
    assert not (dest_dir / "bot.flac").exists()


def test_zip_unpacks_with_case_insensitive_aup_and_info_txt(tmp_path: Path) -> None:
    aup_xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<project>"
        '  <wavetrack name="1-user">'
        '    <waveclip offset="3.5"/>'
        '    <waveclip offset="12.0"/>'
        "  </wavetrack>"
        '  <wavetrack name="Bob">'
        '    <waveclip offset="1.25"/>'
        "  </wavetrack>"
        "</project>"
    )
    info_txt = "Recording: Session 12\nDate: 2026-11-17\nTrack 1: Alice\n"
    zip_path = _write_zip(
        tmp_path / "multitrack.zip",
        [
            ("Session/INFO.TXT", info_txt.encode("utf-8")),
            ("Session/PROJECT.AUP", aup_xml.encode("utf-8")),
            ("Session/1-user.flac", b"flac-1"),
            ("Session/Bob.ogg", b"ogg-2"),
            ("Session/extra.mp3", b"mp3-3"),
        ],
    )

    dest_dir = tmp_path / "out"
    result = unpack_multitrack_zip(zip_path, dest_dir)

    by_name = {path.name: hint for path, hint in result}
    assert set(by_name) == {"1-user.flac", "Bob.ogg", "extra.mp3"}
    assert by_name["1-user.flac"] == TrackHint(
        filename="1-user.flac",
        speaker_name="Alice",
        offset_ms=3500,
    )
    assert by_name["Bob.ogg"] == TrackHint(
        filename="Bob.ogg",
        speaker_name="Bob",
        offset_ms=1250,
    )
    assert by_name["extra.mp3"] == TrackHint(
        filename="extra.mp3",
        speaker_name="extra",
        offset_ms=0,
    )
    assert not (dest_dir / "INFO.TXT").exists()
    assert not (dest_dir / "PROJECT.AUP").exists()


def test_aup_parser_extracts_first_waveclip_and_handles_namespaces() -> None:
    aup_xml = (
        '<project xmlns="http://audacity.sourceforge.net/xml/">'
        '  <wavetrack name="Alice">'
        '    <waveclip offset="3.5"/>'
        '    <waveclip offset="9.0"/>'
        "  </wavetrack>"
        '  <wavetrack name="Narrator"/>'
        "</project>"
    )
    hints = AudacityAupParser.parse(aup_xml)
    assert hints["Alice"] == TrackHint(filename="Alice", speaker_name="Alice", offset_ms=3500)
    assert hints["Narrator"] == TrackHint(
        filename="Narrator",
        speaker_name="Narrator",
        offset_ms=0,
    )


@pytest.mark.parametrize(
    "unsafe_xml",
    [
        '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><project/>',
        '<!doctype project PUBLIC "-//audacity//DTD//EN" "http://example.com/a.dtd"><project/>',
        '<project><!ENTITY test "1"></project>',
    ],
)
def test_aup_parser_rejects_doctype_and_entity(unsafe_xml: str) -> None:
    with pytest.raises(ValueError, match="DOCTYPE|ENTITY"):
        AudacityAupParser.parse(unsafe_xml)


def test_craig_info_parser_patterns_and_empty_fallback() -> None:
    assert CraigInfoParser.parse("") == {}
    assert CraigInfoParser.parse("Recording: Foo\nRandom line without mapping\n") == {}

    info_text = (
        "Recording: RPG Session\n"
        "Track 1: Alice\n"
        "Track 2 - Bob#4321\n"
        "3-charlie.flac: Charlie\n"
        "4-dave"
    )
    parsed = CraigInfoParser.parse(info_text)
    assert parsed["1"].speaker_name == "Alice"
    assert parsed["2"].speaker_name == "Bob#4321"
    assert parsed["3-charlie.flac"].speaker_name == "Charlie"
    assert parsed["4-dave"].speaker_name == "dave"
    assert parse_track_hints(info_text=None, aup_text=None) == {}


def test_supported_media_extensions_contains_expected_formats() -> None:
    assert {
        ".wav",
        ".flac",
        ".mp3",
        ".ogg",
        ".opus",
        ".m4a",
        ".aac",
        ".mp4",
        ".mov",
        ".webm",
        ".mkv",
    } == set(SUPPORTED_MEDIA_EXTENSIONS)


def test_probe_duration_ms_success_and_no_shell(tmp_path: Path) -> None:
    media_file = tmp_path / "sample.wav"

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs.get("shell") is not True
        if "-select_streams" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="audio\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="12.3456\n", stderr="")

    with patch("subprocess.run", side_effect=fake_run) as mock_run:
        duration_ms = probe_duration_ms(media_file)

    assert duration_ms == 12346
    assert mock_run.call_count == 2
    first_cmd = mock_run.call_args_list[0].args[0]
    second_cmd = mock_run.call_args_list[1].args[0]
    assert first_cmd[0] == "ffprobe"
    assert "-select_streams" in first_cmd and "a:0" in first_cmd
    assert second_cmd == [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_file),
    ]


def test_probe_duration_ms_errors_on_missing_audio_or_invalid_duration(tmp_path: Path) -> None:
    media_file = tmp_path / "video_only.mp4"

    # Case 1: no audio stream
    with patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(["ffprobe"], 0, stdout="", stderr=""),
    ):
        with pytest.raises(MediaInvalidError) as exc_info:
            probe_duration_ms(media_file)
        assert exc_info.value.error_code == "MEDIA_INVALID"

    # Case 2: ffprobe failure
    with patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(["ffprobe"], 1, stdout="", stderr="corrupt file"),
    ):
        with pytest.raises(MediaInvalidError) as exc_info:
            probe_duration_ms(media_file)
        assert "corrupt file" in exc_info.value.detail

    # Case 3: non-positive duration
    def zero_duration(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "-select_streams" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="audio\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="0.0\n", stderr="")

    with patch("subprocess.run", side_effect=zero_duration), pytest.raises(MediaInvalidError):
        probe_duration_ms(media_file)


def test_extract_wav_16k_commands_loudnorm_and_error(tmp_path: Path) -> None:
    src = tmp_path / "input.mp4"
    dst = tmp_path / "nested" / "out.wav"

    with patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(["ffmpeg"], 0, stdout="", stderr=""),
    ) as mock_run:
        extract_wav_16k(src, dst, normalize_audio=False)

    assert dst.parent.is_dir()
    cmd_no_norm = mock_run.call_args.args[0]
    assert mock_run.call_args.kwargs.get("shell") is not True
    assert cmd_no_norm == [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]

    with patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(["ffmpeg"], 0, stdout="", stderr=""),
    ) as mock_run:
        extract_wav_16k(src, dst, normalize_audio=True)

    cmd_norm = mock_run.call_args.args[0]
    assert "-af" in cmd_norm and "loudnorm" in cmd_norm
    assert cmd_norm[-1] == str(dst)

    with patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(["ffmpeg"], 1, stdout="", stderr="Invalid data"),
    ):
        with pytest.raises(MediaInvalidError) as exc_info:
            extract_wav_16k(src, dst)
        assert exc_info.value.error_code == "MEDIA_INVALID"
        assert "Invalid data" in exc_info.value.detail


def test_mix_multitrack_audio_applies_offsets_and_writes_atomically(tmp_path: Path) -> None:
    t1 = tmp_path / "t1.flac"
    t2 = tmp_path / "t2.flac"
    t1.write_bytes(b"a")
    t2.write_bytes(b"b")
    dst = tmp_path / "out" / "mix.m4a"

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs.get("shell") is not True
        tmp_out = Path(cmd[-1])
        tmp_out.write_bytes(b"fake-m4a")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=fake_run) as mock_run:
        mix_multitrack_audio([(t1, 0), (t2, 350)], dst, normalize_audio=True)

    assert dst.is_file()
    assert dst.read_bytes() == b"fake-m4a"
    cmd = mock_run.call_args.args[0]
    assert cmd[0] == "ffmpeg"
    fc_idx = cmd.index("-filter_complex")
    filter_complex = cmd[fc_idx + 1]
    assert "[1:a]adelay=350:all=1[a1]" in filter_complex
    assert "[0:a][a1]amix=inputs=2:duration=longest:normalize=0,loudnorm[aout]" in filter_complex
    assert "-c:a" in cmd and "aac" in cmd

    with pytest.raises(MediaInvalidError):
        mix_multitrack_audio([], dst)
