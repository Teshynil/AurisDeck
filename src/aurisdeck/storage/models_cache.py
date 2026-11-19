"""Helpers to inspect, update/pull, and delete cached Hugging Face / Whisper models."""

from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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


def pull_or_update_model(
    model: str,
    models_dir: Path,
    *,
    hf_token: str | None = None,
    hf_endpoint: str | None = None,
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

    if "/" in target:
        from huggingface_hub import snapshot_download

        downloaded_path = snapshot_download(
            repo_id=target,
            cache_dir=str(models_dir),
            token=hf_token or None,
            local_files_only=False,
        )
    else:
        from faster_whisper.utils import download_model

        kwargs: dict[str, Any] = {
            "cache_dir": str(models_dir),
            "local_files_only": False,
        }
        if hf_token:
            kwargs["use_auth_token"] = hf_token
        downloaded_path = download_model(target, **kwargs)

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
