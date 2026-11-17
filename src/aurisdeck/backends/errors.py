"""Typed backend exceptions mapped to AurisDeck i18n error codes."""

from __future__ import annotations

from typing import Any


class AurisDeckBackendError(Exception):
    """Base exception for ASR and diarization backend failures."""

    def __init__(
        self,
        error_code: str,
        error_params: dict[str, Any] | None = None,
        detail: str = "",
    ) -> None:
        self.error_code = error_code
        self.error_params: dict[str, Any] = dict(error_params) if error_params is not None else {}
        self.detail = detail
        super().__init__(detail or error_code)


class MissingCredentialsError(AurisDeckBackendError):
    """Raised when required credentials (e.g., HuggingFace token) are missing."""

    def __init__(
        self,
        error_code: str = "DIARIZATION_HF_TOKEN_MISSING",
        error_params: dict[str, Any] | None = None,
        detail: str = "",
    ) -> None:
        super().__init__(error_code=error_code, error_params=error_params, detail=detail)


class GatedModelTermsError(AurisDeckBackendError):
    """Raised when a gated model requires accepting license terms."""

    def __init__(
        self,
        error_code: str = "DIARIZATION_GATED_TERMS",
        error_params: dict[str, Any] | None = None,
        detail: str = "",
    ) -> None:
        super().__init__(error_code=error_code, error_params=error_params, detail=detail)


class GpuOomError(AurisDeckBackendError):
    """Raised when GPU memory is exhausted while loading or running a model."""

    def __init__(
        self,
        error_code: str = "GPU_OOM",
        error_params: dict[str, Any] | None = None,
        detail: str = "",
    ) -> None:
        super().__init__(error_code=error_code, error_params=error_params, detail=detail)


class ModelLoadError(AurisDeckBackendError):
    """Raised when loading a model or its runtime dependencies fails."""

    def __init__(
        self,
        error_code: str = "MODEL_LOAD_FAILED",
        error_params: dict[str, Any] | None = None,
        detail: str = "",
    ) -> None:
        super().__init__(error_code=error_code, error_params=error_params, detail=detail)


class JobCancelledError(AurisDeckBackendError):
    """Raised when a running job observes a cancellation request."""

    def __init__(
        self,
        error_code: str = "JOB_CANCELLED",
        error_params: dict[str, Any] | None = None,
        detail: str = "",
    ) -> None:
        super().__init__(error_code=error_code, error_params=error_params, detail=detail)
