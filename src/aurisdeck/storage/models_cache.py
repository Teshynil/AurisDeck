"""Helpers to inspect, update/pull, and delete cached Hugging Face / Whisper models."""

from __future__ import annotations

import os
import shutil
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_WHISPER_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]


def _classify_repo(repo_id: str) -> tuple[str, str]:
    """Return (display_name, category) for a cached Hugging Face repo_id."""
    lower = repo_id.lower()
    if "faster-whisper-" in lower:
        short = repo_id.split("faster-whisper-", 1)[1]
        return short, "asr"
    if "pyannote" in lower or "diarization" in lower or "segmentation" in lower:
        return repo_id.split("/")[-1], "diarization"
    return repo_id, "other"


def list_cached_models(models_dir: Path) -> dict[str, Any]:
    """Scan *models_dir* and return metadata + disk usage for cached models."""
    if not models_dir.is_dir():
        return {"models": [], "total_size_bytes": 0}

    from huggingface_hub import scan_cache_dir

    try:
        cache_info = scan_cache_dir(cache_dir=models_dir)
    except (OSError, ValueError):
        return {"models": [], "total_size_bytes": 0}

    models: list[dict[str, Any]] = []
    for repo in cache_info.repos:
        display_name, category = _classify_repo(repo.repo_id)
        revisions = sorted(
            repo.revisions,
            key=lambda rev: rev.last_modified,
            reverse=True,
        )
        latest_rev = revisions[0] if revisions else None
        last_modified_iso = (
            datetime.fromtimestamp(repo.last_modified, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            if repo.last_modified
            else None
        )
        models.append(
            {
                "repo_id": repo.repo_id,
                "repo_type": repo.repo_type,
                "display_name": display_name,
                "category": category,
                "size_on_disk_bytes": int(repo.size_on_disk),
                "nb_files": int(repo.nb_files),
                "revisions_count": len(revisions),
                "latest_commit": latest_rev.commit_hash[:12] if latest_rev else None,
                "last_modified": last_modified_iso,
            }
        )

    models.sort(key=lambda item: item["size_on_disk_bytes"], reverse=True)
    return {
        "models": models,
        "total_size_bytes": int(cache_info.size_on_disk),
    }


def _make_snapshot_tqdm_class(
    model_label: str,
    progress_cb: Callable[[dict[str, Any]], None],
) -> type[Any]:
    """Create a huggingface_hub tqdm subclass combining transfer + reconstruct progress."""
    from huggingface_hub.utils import tqdm as hf_tqdm

    lock = threading.Lock()
    state: dict[str, Any] = {
        "total_bytes": 0,
        "reconstruct_n": 0,
        "transfer_n": 0,
        "last_emit": 0.0,
        "last_pct": -1,
    }

    class _SnapshotProgressTqdm(hf_tqdm):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            unit = kwargs.get("unit")
            desc = str(kwargs.get("desc") or "")
            name = kwargs.get("name")
            self._is_reconstruct = unit == "B" and (
                name == "huggingface_hub.snapshot_download" or desc.startswith("Reconstructing")
            )
            self._is_transfer = unit == "B" and (
                name == "huggingface_hub.snapshot_download.transfer"
                or desc.startswith("Downloading bytes")
            )
            init_total = int(kwargs.get("total") or 0)
            init_n = int(kwargs.get("initial") or 0)
            super().__init__(*args, **kwargs)
            if self._is_reconstruct or self._is_transfer:
                with lock:
                    if self._is_reconstruct:
                        state["total_bytes"] = max(state["total_bytes"], init_total)
                        state["reconstruct_n"] += init_n
                    elif self._is_transfer:
                        state["transfer_n"] += init_n
                self._maybe_emit(force=True)

        def _maybe_emit(self, *, force: bool = False) -> None:
            with lock:
                total = state["total_bytes"]
                if total <= 0:
                    return
                downloaded = min(total, max(state["transfer_n"], state["reconstruct_n"]))
                pct = min(99, max(0, round((downloaded / total) * 100)))
                now = time.monotonic()
                if not force and pct == state["last_pct"] and (now - state["last_emit"]) < 0.20:
                    return
                if not force and pct == state["last_pct"] and downloaded == 0:
                    return
                state["last_pct"] = pct
                state["last_emit"] = now
            progress_cb(
                {
                    "model": model_label,
                    "status": "downloading",
                    "percent": pct,
                    "downloaded_bytes": downloaded,
                    "total_bytes": total,
                }
            )

        def refresh(self, *args: Any, **kwargs: Any) -> Any:
            res = super().refresh(*args, **kwargs)
            if self._is_reconstruct:
                cur_total = int(getattr(self, "total", 0) or 0)
                if cur_total > 0:
                    with lock:
                        state["total_bytes"] = max(state["total_bytes"], cur_total)
                    self._maybe_emit()
            return res

        def update(self, n: float | None = 1) -> bool | None:
            res = super().update(n)
            if (self._is_reconstruct or self._is_transfer) and n:
                delta = int(n)
                with lock:
                    if self._is_reconstruct:
                        cur_total = int(getattr(self, "total", 0) or 0)
                        state["total_bytes"] = max(state["total_bytes"], cur_total)
                        state["reconstruct_n"] += delta
                    else:
                        state["transfer_n"] += delta
                self._maybe_emit()
            return res

        def close(self) -> None:
            if self._is_reconstruct or self._is_transfer:
                self._maybe_emit(force=True)
            super().close()

    return _SnapshotProgressTqdm


def pull_or_update_model(
    model: str,
    models_dir: Path,
    *,
    hf_token: str | None = None,
    hf_endpoint: str | None = None,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Download or update a Whisper alias or Hugging Face repo_id into *models_dir*."""
    target = model.strip()
    if not target:
        raise ValueError("Model name or repo_id must not be empty")

    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(models_dir)
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    from aurisdeck.backends.hf_verify import configure_hf_http_client

    configure_hf_http_client()

    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {
        "cache_dir": str(models_dir),
        "token": hf_token or None,
        "local_files_only": False,
    }

    if "/" in target:
        repo_id = target
    else:
        from faster_whisper.utils import _MODELS

        if target not in _MODELS:
            raise ValueError(
                f"Invalid Whisper model size '{target}', expected one of: {', '.join(_MODELS)}"
            )
        repo_id = _MODELS[target]
        kwargs["allow_patterns"] = _WHISPER_ALLOW_PATTERNS

    if progress_cb is not None:
        kwargs["tqdm_class"] = _make_snapshot_tqdm_class(target, progress_cb)

    downloaded_path = snapshot_download(repo_id=repo_id, **kwargs)

    inventory = list_cached_models(models_dir)
    return {
        "model": target,
        "path": str(downloaded_path),
        **inventory,
    }


def delete_cached_model(repo_id: str, models_dir: Path) -> dict[str, Any]:
    """Delete all cached revisions for *repo_id* from *models_dir*."""
    target = repo_id.strip()
    if not target:
        raise ValueError("repo_id must not be empty")
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Model {target!r} not found in cache")

    from huggingface_hub import scan_cache_dir

    cache_info = scan_cache_dir(cache_dir=models_dir)
    matched = next((repo for repo in cache_info.repos if repo.repo_id == target), None)
    if matched is None or not matched.revisions:
        raise FileNotFoundError(f"Model {target!r} not found in cache")

    revision_hashes = [rev.commit_hash for rev in matched.revisions]
    freed_bytes = int(matched.size_on_disk)
    if revision_hashes:
        strategy = cache_info.delete_revisions(*revision_hashes)
        strategy.execute()
    if matched.repo_path.is_dir():
        shutil.rmtree(matched.repo_path, ignore_errors=True)

    inventory = list_cached_models(models_dir)
    return {
        "deleted": True,
        "repo_id": target,
        "freed_bytes": freed_bytes,
        **inventory,
    }
