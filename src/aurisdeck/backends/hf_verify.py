"""Lightweight Hugging Face token and gated diarization model access verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DIARIZATION_REQUIRED_REPOS: tuple[str, ...] = (
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
)


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
