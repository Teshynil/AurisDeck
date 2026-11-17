"""Backend contracts, DTOs, and typed errors for ASR and diarization."""

from aurisdeck.backends.base import (
    ASRBackend,
    ASRCapabilities,
    ASRContext,
    ASRResult,
    ASRSegment,
    ASRWord,
    DiarizerBackend,
    SpeakerTurn,
)
from aurisdeck.backends.errors import (
    AurisDeckBackendError,
    GatedModelTermsError,
    GpuOomError,
    JobCancelledError,
    MissingCredentialsError,
    ModelLoadError,
)

__all__ = [
    "ASRBackend",
    "ASRCapabilities",
    "ASRContext",
    "ASRResult",
    "ASRSegment",
    "ASRWord",
    "AurisDeckBackendError",
    "DiarizerBackend",
    "GatedModelTermsError",
    "GpuOomError",
    "JobCancelledError",
    "MissingCredentialsError",
    "ModelLoadError",
    "SpeakerTurn",
]
