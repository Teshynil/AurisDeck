"""ML-agnostic DTOs and protocols for ASR and diarization backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass(slots=True)
class ASRCapabilities:
    """Feature flags and audio preferences declared by an ASR backend."""

    supports_hotwords: bool
    supports_initial_prompt: bool
    supports_word_timestamps: bool
    preferred_audio_format: Literal["wav_16k", "compressed"] = "wav_16k"
    max_file_bytes: int | None = None


@dataclass(slots=True)
class ASRWord:
    """Single word timing emitted by an ASR backend."""

    start_ms: int
    end_ms: int
    text: str
    prob: float = 1.0


@dataclass(slots=True)
class ASRSegment:
    """Single transcription segment containing optional word-level timings."""

    start_ms: int
    end_ms: int
    text: str
    words: list[ASRWord] = field(default_factory=list)


@dataclass(slots=True)
class ASRResult:
    """Complete ASR output for a single audio track."""

    language_detected: str | None
    duration_ms: int
    segments: list[ASRSegment] = field(default_factory=list)


@dataclass(slots=True)
class SpeakerTurn:
    """Single diarization speaker interval in milliseconds."""

    start_ms: int
    end_ms: int
    speaker_key: str


@dataclass(slots=True)
class ASRContext:
    """Runtime decoding and contextual biasing parameters passed to ASR."""

    language: str | None = None
    initial_prompt: str = ""
    hotwords: list[str] = field(default_factory=list)
    vad_filter: bool = True
    word_timestamps: bool = True
    beam_size: int = 5


@runtime_checkable
class ASRBackend(Protocol):
    """Protocol implemented by concrete speech-to-text backends."""

    capabilities: ASRCapabilities

    def ensure_loaded(self, *args: Any, **kwargs: Any) -> None:
        """Load the model into memory/VRAM if not already ready."""
        ...

    def transcribe(
        self,
        audio_path: Path,
        context: ASRContext,
        cancel_check: Callable[[], bool] | None = None,
        progress_cb: Callable[[int, int, float], None] | None = None,
        status_cb: Callable[[str], None] | None = None,
    ) -> ASRResult:
        """Transcribe an audio file using the provided ASRContext."""
        ...

    def unload(self) -> None:
        """Unload model references and release device memory."""
        ...


@runtime_checkable
class DiarizerBackend(Protocol):
    """Protocol implemented by concrete speaker diarization backends."""

    def ensure_loaded(self, *args: Any, **kwargs: Any) -> None:
        """Load the diarization model into memory/VRAM if not already ready."""
        ...

    def diarize(
        self,
        audio_path: Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[SpeakerTurn]:
        """Compute speaker activity turns for an audio file."""
        ...

    def unload(self) -> None:
        """Unload diarization model references and release device memory."""
        ...
