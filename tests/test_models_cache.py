"""Tests for cached model inspection, pull/update, deletion, and admin endpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from aurisdeck.api.app import create_app
from aurisdeck.config import Settings
from aurisdeck.storage.models_cache import (
    delete_cached_model,
    list_cached_models,
    pull_or_update_model,
)


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


def test_pull_or_update_model_validation_and_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    models_dir = tmp_path / "models"

    with pytest.raises(ValueError, match="Invalid Whisper model size"):
        pull_or_update_model("nonexistent-model-alias", models_dir)

    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(repo_id: str, **kwargs: Any) -> str:
        calls.append({"repo_id": repo_id, **kwargs})
        tqdm_cls = kwargs.get("tqdm_class")
        if tqdm_cls is not None:
            # Transfer bar must NOT double-count total_bytes
            t_bar = tqdm_cls(
                total=1000,
                unit="B",
                desc="Downloading bytes",
                name="huggingface_hub.snapshot_download.transfer",
                disable=True,
            )
            r_bar = tqdm_cls(
                total=1000,
                unit="B",
                desc="Reconstructing",
                name="huggingface_hub.snapshot_download",
                disable=True,
            )
            r_bar.update(500)
            r_bar.update(500)
            r_bar.close()
            t_bar.close()
        return str(models_dir / "done")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    events: list[dict[str, Any]] = []
    pull_or_update_model("small", models_dir, progress_cb=events.append)
    assert calls[0]["repo_id"] == "Systran/faster-whisper-small"
    assert "allow_patterns" in calls[0]
    assert events[-1]["percent"] == 100
    assert events[-1]["total_bytes"] == 1000
    assert events[-1]["downloaded_bytes"] == 1000

    pull_or_update_model("pyannote/speaker-diarization-3.1", models_dir)
    assert calls[1]["repo_id"] == "pyannote/speaker-diarization-3.1"
    assert "allow_patterns" not in calls[1]


def test_pull_endpoint_emits_sse_progress_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    coord = _StubCoordinator(busy=False)
    settings = Settings(data_dir=tmp_path, auth_mode="none")
    app = create_app(settings, coordinator=coord)  # type: ignore[arg-type]

    def fake_pull(model: str, mdir: Path, **kw: Any) -> dict[str, Any]:
        cb = kw.get("progress_cb")
        if cb is not None:
            cb(
                {
                    "model": model,
                    "status": "downloading",
                    "percent": 50,
                    "downloaded_bytes": 500,
                    "total_bytes": 1000,
                }
            )
        return {"model": model, "path": str(mdir), "models": [], "total_size_bytes": 0}

    monkeypatch.setattr("aurisdeck.storage.models_cache.pull_or_update_model", fake_pull)

    with TestClient(app) as client:
        sub_q = app.state.event_bus.subscribe()
        resp = client.post("/api/v1/system/models/pull", json={"model": "tiny"})
        assert resp.status_code == 200

        published = []
        while not sub_q.empty():
            published.append(sub_q.get_nowait())
        app.state.event_bus.unsubscribe(sub_q)

        statuses = [ev["payload"]["status"] for ev in published if ev and ev["type"] == "model.pull.progress"]
        assert statuses == ["checking", "downloading", "completed"]
