"""Lightweight Hugging Face token and gated diarization model access verification."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

DIARIZATION_REQUIRED_REPOS: tuple[str, ...] = (
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
)

DEFAULT_HF_CONNECT_TIMEOUT = 5.0
DEFAULT_HF_READ_TIMEOUT = 30.0
DEFAULT_HF_WRITE_TIMEOUT = 30.0
DEFAULT_HF_POOL_TIMEOUT = 10.0

_HF_CLIENT_LOCK = threading.Lock()
_HF_CLIENT_CONFIGURED = False


def configure_hf_http_client(
    *,
    connect_timeout: float = DEFAULT_HF_CONNECT_TIMEOUT,
    read_timeout: float = DEFAULT_HF_READ_TIMEOUT,
    write_timeout: float = DEFAULT_HF_WRITE_TIMEOUT,
    pool_timeout: float = DEFAULT_HF_POOL_TIMEOUT,
    force: bool = False,
) -> bool:
    """Register a finite-timeout httpx2 client factory for huggingface_hub."""
    global _HF_CLIENT_CONFIGURED
    with _HF_CLIENT_LOCK:
        if _HF_CLIENT_CONFIGURED and not force:
            return False
        try:
            import httpx2
            from huggingface_hub import set_client_factory
            from huggingface_hub.utils._http import hf_request_event_hook
        except ImportError:
            return False

        timeout = httpx2.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=write_timeout,
            pool=pool_timeout,
        )

        def _client_factory() -> httpx2.Client:
            return httpx2.Client(
                event_hooks={"request": [hf_request_event_hook]},
                follow_redirects=True,
                timeout=timeout,
            )

        set_client_factory(_client_factory)
        _HF_CLIENT_CONFIGURED = True
        return True


@dataclass(frozen=True, slots=True)
class HfVerifyResult:
    """Structured result of verifying a Hugging Face token without downloading weights."""

    valid: bool
    status: str  # "ok" | "missing_token" | "invalid_token" | "gated_terms_required" | "network_error"
    missing_models: list[str] = field(default_factory=list)
    username: str | None = None
    detail: str | None = None


def verify_hf_token(
    token: str | None,
    *,
    hf_endpoint: str | None = None,
    required_repos: tuple[str, ...] = DIARIZATION_REQUIRED_REPOS,
    api_client: Any | None = None,
) -> HfVerifyResult:
    """Verify token authenticity via whoami and gated repo metadata access."""
    cleaned = token.strip() if token else ""
    if not cleaned:
        return HfVerifyResult(
            valid=False,
            status="missing_token",
            detail="HF_TOKEN is empty",
        )

    if api_client is None:
        from huggingface_hub import HfApi

        configure_hf_http_client()
        kwargs: dict[str, Any] = {"token": cleaned}
        if hf_endpoint:
            kwargs["endpoint"] = hf_endpoint
        api = HfApi(**kwargs)
    else:
        api = api_client

    try:
        who = api.whoami(token=cleaned)
        username = who.get("name") if isinstance(who, dict) else None
    except Exception as exc:  # noqa: BLE001
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        exc_name = type(exc).__name__.lower()
        if status_code == 401 or "401" in str(exc) or "unauthorized" in exc_name:
            return HfVerifyResult(
                valid=False,
                status="invalid_token",
                detail="Invalid or revoked Hugging Face token",
            )
        return HfVerifyResult(
            valid=False,
            status="network_error",
            detail=str(exc),
        )

    missing_models: list[str] = []
    for repo_id in required_repos:
        try:
            api.model_info(repo_id=repo_id, token=cleaned, files_metadata=False)
        except Exception as exc:  # noqa: BLE001
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            exc_name = type(exc).__name__.lower()
            if status_code in (401, 403) or "gated" in exc_name or "403" in str(exc):
                missing_models.append(repo_id)
            else:
                return HfVerifyResult(
                    valid=False,
                    status="network_error",
                    username=username,
                    detail=str(exc),
                )

    if missing_models:
        return HfVerifyResult(
            valid=False,
            status="gated_terms_required",
            missing_models=missing_models,
            username=username,
            detail="License terms must be accepted for required diarization models",
        )

    return HfVerifyResult(
        valid=True,
        status="ok",
        username=username,
    )
