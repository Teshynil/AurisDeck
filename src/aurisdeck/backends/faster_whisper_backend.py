"""Concrete ASRBackend implementation backed by faster-whisper (CTranslate2)."""

from __future__ import annotations

import gc
import logging
import os
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aurisdeck.backends.base import (
    ASRCapabilities,
    ASRContext,
    ASRResult,
    ASRSegment,
    ASRWord,
)
from aurisdeck.backends.errors import GpuOomError, JobCancelledError, ModelLoadError

logger = logging.getLogger(__name__)


def _is_oom_error(exc: BaseException) -> bool:
    """Return True when an exception message indicates device/host out-of-memory."""
    return "out of memory" in str(exc).lower()


def _to_ms(seconds: float) -> int:
    """Convert a timestamp in seconds to rounded integer milliseconds."""
    return round(float(seconds) * 1000)


def _load_audio_input(audio_path: Path) -> Any:
    """Decode 16 kHz mono 16-bit PCM WAVs directly to float32 numpy array, bypassing PyAV."""
    if audio_path.is_file() and audio_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as wf:
                if wf.getnchannels() == 1 and wf.getframerate() == 16000 and wf.getsampwidth() == 2:
                    frames = wf.readframes(wf.getnframes())
                    import numpy as np

                    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        except (wave.Error, OSError, ImportError):
            pass
    return str(audio_path)


class FasterWhisperBackend:
    """ASR backend adapter using lazy-imported ``faster_whisper.WhisperModel``."""

    capabilities: ASRCapabilities = ASRCapabilities(
        supports_hotwords=True,
        supports_initial_prompt=True,
        supports_word_timestamps=True,
        preferred_audio_format="wav_16k",
    )

    def __init__(self) -> None:
        self._model: Any = None
        self._loaded_key: tuple[str, str, str] | None = None

    def ensure_loaded(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        models_dir: Path | str,
        cpu_threads: int = 0,
        hf_endpoint: str | None = None,
        status_cb: Callable[[str], None] | None = None,
    ) -> None:
        """Load the requested Whisper model lazily if not already resident."""
        target_key = (model_name, device, compute_type)
        if self._model is not None and self._loaded_key == target_key:
            return

        if self._model is not None:
            self.unload()

        os.environ["HF_HOME"] = str(models_dir)
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint

        if status_cb is not None:
            status_cb("loading_model")

        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                download_root=str(models_dir),
            )
            self._loaded_key = target_key
        except Exception as exc:
            self._model = None
            self._loaded_key = None
            if _is_oom_error(exc):
                raise GpuOomError(detail=str(exc)) from exc
            raise ModelLoadError(detail=str(exc)) from exc

    def transcribe(
        self,
        audio_path: Path,
        context: ASRContext,
        cancel_check: Callable[[], bool] | None = None,
        progress_cb: Callable[[int, int, float], None] | None = None,
    ) -> ASRResult:
        """Transcribe an audio file using the currently loaded Whisper model."""
        if self._model is None:
            raise ModelLoadError(detail="ASR model not loaded")

        language = None if context.language in (None, "", "auto") else context.language
        kwargs: dict[str, Any] = {
            "language": language,
            "beam_size": context.beam_size,
            "vad_filter": context.vad_filter,
            "word_timestamps": context.word_timestamps,
        }
        if context.initial_prompt.strip():
            kwargs["initial_prompt"] = context.initial_prompt

        cleaned_hotwords = [word.strip() for word in context.hotwords if word.strip()]
        if cleaned_hotwords:
            kwargs["hotwords"] = ", ".join(cleaned_hotwords)

        segments: list[ASRSegment] = []
        try:
            if cancel_check is not None and cancel_check():
                raise JobCancelledError()

            audio_input = _load_audio_input(audio_path)
            segments_gen, info = self._model.transcribe(audio_input, **kwargs)
            total_ms = _to_ms(float(getattr(info, "duration", 0.0) or 0.0))
            t0 = time.monotonic()

            for seg in segments_gen:
                seg_start_ms = _to_ms(seg.start)
                seg_end_ms = _to_ms(seg.end)
                raw_words = getattr(seg, "words", None) or []
                words: list[ASRWord] = []
                for word in raw_words:
                    prob_raw = getattr(word, "probability", 1.0)
                    prob = 1.0 if prob_raw is None else float(prob_raw)
                    words.append(
                        ASRWord(
                            start_ms=_to_ms(word.start),
                            end_ms=_to_ms(word.end),
                            text=word.word,
                            prob=prob,
                        )
                    )

                segments.append(
                    ASRSegment(
                        start_ms=seg_start_ms,
                        end_ms=seg_end_ms,
                        text=str(getattr(seg, "text", "")).strip(),
                        words=words,
                    )
                )

                elapsed = max(time.monotonic() - t0, 1e-6)
                processed_ms = seg_end_ms
                rtfx = (processed_ms / 1000.0) / elapsed
                if progress_cb is not None:
                    progress_cb(processed_ms, total_ms, rtfx)

                if cancel_check is not None and cancel_check():
                    raise JobCancelledError()

            return ASRResult(
                language_detected=getattr(info, "language", None),
                duration_ms=total_ms,
                segments=segments,
            )
        except JobCancelledError:
            raise
        except Exception as exc:
            if _is_oom_error(exc):
                raise GpuOomError(detail=str(exc)) from exc
            raise

    def unload(self) -> None:
        """Release the loaded model reference and clear CUDA cache if available."""
        self._model = None
        self._loaded_key = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, OSError, RuntimeError, AttributeError) as exc:
            logger.debug("Skipping CUDA cache cleanup during unload: %s", exc)
