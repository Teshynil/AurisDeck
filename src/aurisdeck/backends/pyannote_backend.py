"""On-demand speaker diarization backend powered by pyannote.audio."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Any

from aurisdeck.backends.base import SpeakerTurn
from aurisdeck.backends.errors import (
    AurisDeckBackendError,
    GatedModelTermsError,
    GpuOomError,
    MissingCredentialsError,
    ModelLoadError,
)


def _has_usable_offline_cache(models_dir: Path, model_id: str) -> bool:
    """Return True if models_dir or model_id already holds a usable local snapshot."""
    direct_path = Path(model_id)
    if direct_path.is_dir() and (direct_path / "config.yaml").is_file():
        return True

    repo_dir_name = f"models--{model_id.replace('/', '--')}"
    candidate_roots = (
        models_dir / repo_dir_name,
        models_dir / "hub" / repo_dir_name,
    )
    for root in candidate_roots:
        snapshots_dir = root / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        for snapshot in snapshots_dir.iterdir():
            if not snapshot.is_dir():
                continue
            if any(snapshot.iterdir()):
                return True
    return False


def _map_load_exception(exc: Exception) -> AurisDeckBackendError:
    """Translate pyannote/HuggingFace/CUDA exceptions into typed backend errors."""
    messages: list[str] = []
    type_names: list[str] = []
    status_codes: set[int] = set()
    visited: set[int] = set()

    current: BaseException | None = exc
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        messages.append(str(current).lower())
        type_names.append(type(current).__name__.lower())
        response = getattr(current, "response", None)
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            status_codes.add(status_code)
        current = current.__cause__ or current.__context__

    combined_text = " ".join(messages)
    combined_types = " ".join(type_names)

    if (
        401 in status_codes
        or 403 in status_codes
        or "401" in combined_text
        or "403" in combined_text
        or "gated" in combined_text
        or "accept the user conditions" in combined_text
        or "gatedrepoerror" in combined_types
    ):
        return GatedModelTermsError(detail=str(exc))

    if "out of memory" in combined_text or "outofmemoryerror" in combined_types:
        return GpuOomError(detail=str(exc))

    return ModelLoadError(detail=str(exc))


class PyannoteDiarizerBackend:
    """Lazy-loading DiarizerBackend adapter for pyannote/speaker-diarization-3.1."""

    def __init__(self) -> None:
        self._pipeline: Any = None
        self._device: str | None = None
        self._model_id: str | None = None

    def ensure_loaded(
        self,
        model_id: str = "pyannote/speaker-diarization-3.1",
        device: str = "cuda",
        models_dir: Path | str = ".",
        hf_token: str | None = None,
        hf_endpoint: str | None = None,
    ) -> None:
        """Load the pyannote pipeline on demand and move it to the target device."""
        if (
            self._pipeline is not None
            and self._model_id == model_id
            and self._device == device
        ):
            return

        if self._pipeline is not None:
            self.unload()

        resolved_models_dir = Path(models_dir)
        os.environ["HF_HOME"] = str(resolved_models_dir)
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint

        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise ModelLoadError(
                error_code="DIARIZATION_DEPENDENCY_MISSING",
                detail=str(exc),
            ) from exc

        normalized_token = hf_token.strip() if hf_token and hf_token.strip() else None
        if not normalized_token and not _has_usable_offline_cache(resolved_models_dir, model_id):
            raise MissingCredentialsError()

        try:
            try:
                pipeline = Pipeline.from_pretrained(
                    model_id,
                    token=normalized_token,
                    cache_dir=str(resolved_models_dir),
                )
            except TypeError as exc:
                if "token" not in str(exc):
                    raise
                pipeline = Pipeline.from_pretrained(
                    model_id,
                    use_auth_token=normalized_token,
                    cache_dir=str(resolved_models_dir),
                )

            if pipeline is None:
                raise GatedModelTermsError(
                    detail=f"Failed to load gated or inaccessible model '{model_id}'"
                )

            pipeline.to(torch.device(device))
        except AurisDeckBackendError:
            raise
        except Exception as exc:
            raise _map_load_exception(exc) from exc

        self._pipeline = pipeline
        self._device = device
        self._model_id = model_id

    def diarize(
        self,
        audio_path: Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[SpeakerTurn]:
        """Run speaker diarization on an audio file and return millisecond turns."""
        if self._pipeline is None:
            raise ModelLoadError(detail="Diarization pipeline not loaded")

        kwargs: dict[str, Any] = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        try:
            output = self._pipeline(str(audio_path), **kwargs)
        except AurisDeckBackendError:
            raise
        except Exception as exc:
            mapped = _map_load_exception(exc)
            if isinstance(mapped, GpuOomError):
                raise mapped from exc
            raise

        annotation = getattr(output, "speaker_diarization", output)
        turns: list[SpeakerTurn] = []
        for turn, _track, speaker in annotation.itertracks(yield_label=True):
            turns.append(
                SpeakerTurn(
                    start_ms=round(float(turn.start) * 1000),
                    end_ms=round(float(turn.end) * 1000),
                    speaker_key=str(speaker),
                )
            )
        return turns

    def unload(self) -> None:
        """Release pipeline references and clear CUDA cache if torch is loaded."""
        self._pipeline = None
        self._device = None
        self._model_id = None
        gc.collect()

        torch_module = sys.modules.get("torch")
        if torch_module is not None:
            cuda = getattr(torch_module, "cuda", None)
            if cuda is not None and cuda.is_available():
                cuda.empty_cache()
