"""Tests for aurisdeck.config and aurisdeck.storage.paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from aurisdeck.config import Settings
from aurisdeck.storage.paths import (
    DataPaths,
    atomic_write_bytes,
    atomic_write_text,
    is_ignored_archive_member,
    sanitize_filename,
)


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "AURISDECK_DATA_DIR",
        "AURISDECK_AUTH_MODE",
        "AURISDECK_AUTH_PASSWORD",
        "AURISDECK_BASE_PATH",
        "AURISDECK_MODEL_IDLE_UNLOAD_SECONDS",
        "AURISDECK_CPU_THREADS",
        "HF_ENDPOINT",
        "HF_TOKEN",
        "AURISDECK_HF_ENDPOINT",
        "AURISDECK_HF_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)

    settings = Settings(_env_file=None)
    assert settings.data_dir == Path("./data")
    assert settings.auth_mode == "none"
    assert settings.auth_password is None
    assert settings.base_path == ""
    assert settings.model_idle_unload_seconds == 120
    assert settings.cpu_threads == 0
    assert settings.hf_endpoint is None
    assert not hasattr(settings, "hf_token")


def test_settings_defaults_and_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AURISDECK_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AURISDECK_AUTH_MODE", "password")
    monkeypatch.setenv("AURISDECK_AUTH_PASSWORD", "secret")
    monkeypatch.setenv("AURISDECK_BASE_PATH", "/auris/")
    monkeypatch.setenv("AURISDECK_MODEL_IDLE_UNLOAD_SECONDS", "45")
    monkeypatch.setenv("AURISDECK_CPU_THREADS", "4")
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    settings = Settings(_env_file=None)
    assert settings.data_dir == tmp_path
    assert settings.auth_mode == "password"
    assert settings.auth_password == "secret"
    assert settings.base_path == "/auris"
    assert settings.model_idle_unload_seconds == 45
    assert settings.cpu_threads == 4
    assert settings.hf_endpoint == "https://hf-mirror.example"
    assert not hasattr(settings, "hf_token")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        ("/", ""),
        ("/aurisdeck/", "/aurisdeck"),
        ("aurisdeck/", "/aurisdeck"),
        ("/a/b/", "/a/b"),
    ],
)
def test_settings_base_path_normalization(raw: str, expected: str) -> None:
    settings = Settings(_env_file=None, base_path=raw)
    assert settings.base_path == expected


def test_data_paths_ensure_dirs_and_job_subdirs(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path / "data")
    paths.ensure_dirs()
    # Calling twice must stay idempotent
    paths.ensure_dirs()

    expected_dirs = [
        paths.profiles,
        paths.glossaries,
        paths.replacements,
        paths.locales,
        paths.media,
        paths.checkpoints,
        paths.outputs,
        paths.tmp,
        paths.models,
    ]
    for directory in expected_dirs:
        assert directory.is_dir()

    assert paths.db_path == tmp_path / "data" / "aurisdeck.db"
    assert paths.job_media_dir("job-123") == paths.media / "job-123"
    assert paths.job_checkpoints_dir("job-123") == paths.checkpoints / "job-123"
    assert paths.job_outputs_dir("job-123") == paths.outputs / "job-123"
    assert paths.job_tmp_dir("job-123") == paths.tmp / "job-123"

    with pytest.raises(ValueError):
        paths.job_media_dir("../escape")


def test_sanitize_filename_preserves_intermediate_dots_and_blocks_traversal() -> None:
    assert sanitize_filename("mi.archivo.v2.mp3") == "mi.archivo.v2.mp3"

    sanitized_traversal = sanitize_filename("../../etc/passwd")
    assert sanitized_traversal == "passwd"
    assert ".." not in sanitized_traversal
    assert "/" not in sanitized_traversal

    assert sanitize_filename(r"..\..\secret\track..v1.wav") == "track.v1.wav"
    assert sanitize_filename("audio\x00:test?.m4a") == "audio__test_.m4a"
    assert sanitize_filename("") == "archivo"
    assert sanitize_filename("..") == "archivo"
    assert sanitize_filename("...") == "archivo"


@pytest.mark.parametrize(
    ("member", "ignored"),
    [
        ("__MACOSX/._track.wav", True),
        ("__MACOSX/track.wav", True),
        ("nested/__MACOSX/foo.mp3", True),
        ("tracks/._speaker1.flac", True),
        (".DS_Store", True),
        ("tracks/.DS_Store", True),
        ("tracks/speaker1.wav", False),
        ("tracks/my.audio.v2.mp3", False),
    ],
)
def test_is_ignored_archive_member(member: str, ignored: bool) -> None:
    assert is_ignored_archive_member(member) is ignored


def test_atomic_write_text_and_bytes(tmp_path: Path) -> None:
    tmp_dir = tmp_path / "tmp"
    dest_txt = tmp_path / "outputs" / "sample.vtt"
    dest_bin = tmp_path / "outputs" / "sample.bin"

    atomic_write_text(dest_txt, "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHola\n", tmp_dir)
    assert dest_txt.read_text(encoding="utf-8").startswith("WEBVTT")

    atomic_write_bytes(dest_bin, b"\x00\x01\x02", tmp_dir)
    assert dest_bin.read_bytes() == b"\x00\x01\x02"
    assert list(tmp_dir.iterdir()) == []
