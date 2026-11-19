"""Tests for cached model inspection, pull/update, deletion, and admin endpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from aurisdeck.api.app import create_app
from aurisdeck.config import Settings
from aurisdeck.storage.models_cache import delete_cached_model, list_cached_models


def _create_fake_hf_repo(models_dir: Path, repo_id: str, size_bytes: int = 1024) -> Path:
    folder_name = f"models--{repo_id.replace('/', '--')}"
    repo_dir = models_dir / folder_name
    commit = hashlib.sha1(repo_id.encode("utf-8")).hexdigest()
    blob_hash = hashlib.sha256(repo_id.encode("utf-8")).hexdigest()
    blobs_dir = repo_dir / "blobs"
    snapshots_dir = repo_dir / "snapshots" / commit
    refs_dir = repo_dir / "refs"
    blobs_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)

    blob_file = blobs_dir / blob_hash
    blob_file.write_bytes(b"x" * size_bytes)
    (snapshots_dir / "model.bin").symlink_to(blob_file)
    (refs_dir / "main").write_text(commit, encoding="utf-8")
    return repo_dir


def test_list_and_delete_cached_models(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    assert list_cached_models(models_dir) == {"models": [], "total_size_bytes": 0}

    _create_fake_hf_repo(models_dir, "mobiuslabsgmbh/faster-whisper-large-v3-turbo", 2048)
    _create_fake_hf_repo(models_dir, "pyannote/speaker-diarization-3.1", 512)

    listed = list_cached_models(models_dir)
    assert listed["total_size_bytes"] == 2560
    assert len(listed["models"]) == 2
    assert listed["models"][0]["display_name"] == "large-v3-turbo"
    assert listed["models"][0]["category"] == "asr"
    assert listed["models"][1]["category"] == "diarization"

    deleted = delete_cached_model("pyannote/speaker-diarization-3.1", models_dir)
    assert deleted["deleted"] is True
    assert deleted["freed_bytes"] == 512
    assert len(deleted["models"]) == 1

    with pytest.raises(FileNotFoundError):
        delete_cached_model("pyannote/speaker-diarization-3.1", models_dir)


class _StubCoordinator:
    def __init__(self, busy: bool = False) -> None:
        self.busy = busy
        self.unloaded = False

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def list_devices(self) -> list[Any]:
        return []

    def get_device_snapshots(self) -> list[dict[str, Any]]:
        return [{"id": "cuda:0", "state": "busy" if self.busy else "idle"}]

    def unload_all_models(self) -> None:
        self.unloaded = True


def test_system_models_endpoints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    _create_fake_hf_repo(models_dir, "Systran/faster-whisper-small", 1024)

    coord = _StubCoordinator(busy=False)
    settings = Settings(data_dir=tmp_path, auth_mode="none")
    app = create_app(settings, coordinator=coord)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "aurisdeck.storage.models_cache.pull_or_update_model",
        lambda model, mdir, **kw: {
            "model": model,
            "path": str(mdir),
            "models": [],
            "total_size_bytes": 1024,
        },
    )

    with TestClient(app) as client:
        resp = client.get("/api/v1/system/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_size_bytes"] == 1024
        assert body["models"][0]["display_name"] == "small"

        pull_resp = client.post("/api/v1/system/models/pull", json={"model": "tiny"})
        assert pull_resp.status_code == 200
        assert pull_resp.json()["model"] == "tiny"

        # Busy guard returns 409 MODEL_IN_USE
        coord.busy = True
        conflict_resp = client.delete(
            "/api/v1/system/models",
            params={"repo_id": "Systran/faster-whisper-small"},
        )
        assert conflict_resp.status_code == 409
        assert conflict_resp.json()["error_code"] == "MODEL_IN_USE"

        # Idle allows deletion and unloads models first
        coord.busy = False
        del_resp = client.delete(
            "/api/v1/system/models",
            params={"repo_id": "Systran/faster-whisper-small"},
        )
        assert del_resp.status_code == 200
        assert del_resp.json()["deleted"] is True
        assert coord.unloaded is True
