"""Unit tests for FasterWhisperBackend using mocked WhisperModel and torch."""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from aurisdeck.backends import faster_whisper_backend
from aurisdeck.backends.base import ASRBackend, ASRContext
from aurisdeck.backends.errors import GpuOomError, JobCancelledError, ModelLoadError
from aurisdeck.backends.faster_whisper_backend import FasterWhisperBackend


def _install_fake_faster_whisper(monkeypatch: pytest.MonkeyPatch, model_cls: Any) -> None:
    """Register a fake faster_whisper module in sys.modules for lazy import tests."""
    fake_module = ModuleType("faster_whisper")
    fake_module.WhisperModel = model_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)


def test_no_top_level_ml_imports() -> None:
    """faster_whisper and torch must never be imported at module top-level."""
    module_path = Path(faster_whisper_backend.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))

    forbidden = {"faster_whisper", "torch"}
    top_level_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level_imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            top_level_imports.add(node.module.split(".")[0])

    assert top_level_imports.isdisjoint(forbidden)


def test_protocol_and_capabilities() -> None:
    """FasterWhisperBackend satisfies ASRBackend protocol and advertises expected flags."""
    backend = FasterWhisperBackend()
    assert isinstance(backend, ASRBackend)
    assert backend.capabilities.supports_hotwords is True
    assert backend.capabilities.supports_initial_prompt is True
    assert backend.capabilities.supports_word_timestamps is True
    assert backend.capabilities.preferred_audio_format == "wav_16k"


def test_ensure_loaded_sets_env_reuses_and_unloads_on_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ensure_loaded configures HF env, reuses matching model, and unloads on switch."""
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    init_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class FakeWhisperModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            init_calls.append((args, kwargs))

    _install_fake_faster_whisper(monkeypatch, FakeWhisperModel)

    statuses: list[str] = []
    backend = FasterWhisperBackend()
    models_dir = tmp_path / "models"

    backend.ensure_loaded(
        model_name="large-v3-turbo",
        device="cuda",
        compute_type="float16",
        models_dir=models_dir,
        cpu_threads=4,
        hf_endpoint="https://hf-mirror.example",
        status_cb=statuses.append,
    )

    assert statuses == ["loading_model"]
    assert len(init_calls) == 1
    assert init_calls[0][0] == ("large-v3-turbo",)
    assert init_calls[0][1] == {
        "device": "cuda",
        "compute_type": "float16",
        "cpu_threads": 4,
        "download_root": str(models_dir),
        "local_files_only": True,
    }
    assert os.environ["HF_HOME"] == str(models_dir)
    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.example"

    # Reusing identical (model_name, device, compute_type) does not reload.
    backend.ensure_loaded(
        model_name="large-v3-turbo",
        device="cuda",
        compute_type="float16",
        models_dir=models_dir,
        status_cb=statuses.append,
    )
    assert len(init_calls) == 1
    assert statuses == ["loading_model"]

    # Changing model triggers unload() before instantiating the new model.
    unload_calls = 0
    original_unload = backend.unload

    def tracked_unload() -> None:
        nonlocal unload_calls
        unload_calls += 1
        original_unload()

    monkeypatch.setattr(backend, "unload", tracked_unload)

    backend.ensure_loaded(
        model_name="medium",
        device="cuda",
        compute_type="float16",
        models_dir=models_dir,
        status_cb=statuses.append,
    )
    assert unload_calls == 1
    assert len(init_calls) == 2
    assert init_calls[1][0] == ("medium",)
    assert statuses == ["loading_model", "loading_model"]


def test_ensure_loaded_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ensure_loaded maps OOM to GpuOomError and other failures to ModelLoadError."""

    class OomModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("CUDA out of memory while allocating weights")

    _install_fake_faster_whisper(monkeypatch, OomModel)
    backend = FasterWhisperBackend()
    with pytest.raises(GpuOomError):
        backend.ensure_loaded("large-v3", "cuda", "float16", tmp_path)

    class BrokenModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ValueError("corrupt model archive")

    _install_fake_faster_whisper(monkeypatch, BrokenModel)
    with pytest.raises(ModelLoadError):
        backend.ensure_loaded("large-v3", "cuda", "float16", tmp_path)


def test_transcribe_requires_loaded_model(tmp_path: Path) -> None:
    """Calling transcribe before ensure_loaded raises ModelLoadError."""
    backend = FasterWhisperBackend()
    with pytest.raises(ModelLoadError):
        backend.transcribe(tmp_path / "audio.wav", ASRContext())


def test_transcribe_context_kwargs_and_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """transcribe normalizes language/hotwords/prompt and reports segment progress."""
    transcribe_calls: list[tuple[str, dict[str, Any]]] = []

    class FakeModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def transcribe(self, path: str, **kwargs: Any) -> tuple[Any, Any]:
            transcribe_calls.append((path, kwargs))
            seg1 = SimpleNamespace(
                start=0.0,
                end=1.25,
                text=" Hola AurisDeck ",
                words=[
                    SimpleNamespace(start=0.0, end=0.5, word=" Hola", probability=0.95),
                    SimpleNamespace(start=0.5, end=1.25, word=" AurisDeck", probability=None),
                ],
            )
            seg2 = SimpleNamespace(
                start=1.25,
                end=2.5,
                text=" prueba ",
                words=None,
            )
            info = SimpleNamespace(language="es", duration=2.5)
            return iter([seg1, seg2]), info

    _install_fake_faster_whisper(monkeypatch, FakeModel)
    backend = FasterWhisperBackend()
    backend.ensure_loaded("small", "cpu", "int8", tmp_path)

    progress_events: list[tuple[int, int, float]] = []
    audio_path = tmp_path / "sample.wav"

    # Case 1: language="auto", empty initial_prompt and blank hotwords omitted.
    result = backend.transcribe(
        audio_path,
        ASRContext(
            language="auto",
            initial_prompt="   ",
            hotwords=["", "   "],
            vad_filter=True,
            word_timestamps=True,
            beam_size=5,
        ),
        progress_cb=lambda proc, total, rtfx: progress_events.append((proc, total, rtfx)),
    )

    _, kwargs_auto = transcribe_calls[0]
    assert kwargs_auto["language"] is None
    assert "initial_prompt" not in kwargs_auto
    assert "hotwords" not in kwargs_auto
    assert kwargs_auto["vad_filter"] is True
    assert kwargs_auto["word_timestamps"] is True
    assert kwargs_auto["beam_size"] == 5

    assert result.language_detected == "es"
    assert result.duration_ms == 2500
    assert len(result.segments) == 2
    assert result.segments[0].start_ms == 0
    assert result.segments[0].end_ms == 1250
    assert result.segments[0].text == "Hola AurisDeck"
    assert len(result.segments[0].words) == 2
    assert result.segments[0].words[0].prob == pytest.approx(0.95)
    assert result.segments[0].words[1].prob == pytest.approx(1.0)
    assert result.segments[1].words == []

    assert len(progress_events) == 2
    assert progress_events[0][0] == 1250
    assert progress_events[0][1] == 2500
    assert progress_events[0][2] > 0.0
    assert progress_events[1][0] == 2500
    assert progress_events[1][1] == 2500
    assert progress_events[1][2] > 0.0

    # Case 2: explicit language, non-empty prompt and hotwords cleaned and joined.
    backend.transcribe(
        audio_path,
        ASRContext(
            language="es",
            initial_prompt="Términos técnicos",
            hotwords=["  AurisDeck ", "", " Pyannote  "],
        ),
    )
    _, kwargs_custom = transcribe_calls[1]
    assert kwargs_custom["language"] == "es"
    assert kwargs_custom["initial_prompt"] == "Términos técnicos"
    assert kwargs_custom["hotwords"] == "AurisDeck, Pyannote"


def test_transcribe_cooperative_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """cancel_check returning True halts generator consumption and raises JobCancelledError."""
    consumed: list[int] = []

    def segment_stream() -> Any:
        consumed.append(1)
        yield SimpleNamespace(start=0.0, end=1.0, text="one", words=[])
        consumed.append(2)
        yield SimpleNamespace(start=1.0, end=2.0, text="two", words=[])

    class FakeModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def transcribe(self, path: str, **kwargs: Any) -> tuple[Any, Any]:
            return segment_stream(), SimpleNamespace(language="en", duration=2.0)

    _install_fake_faster_whisper(monkeypatch, FakeModel)
    backend = FasterWhisperBackend()
    backend.ensure_loaded("tiny", "cpu", "int8", tmp_path)

    # Cancel right after first segment is processed -> second segment is never pulled.
    with pytest.raises(JobCancelledError):
        backend.transcribe(
            tmp_path / "audio.wav",
            ASRContext(),
            cancel_check=lambda: len(consumed) >= 1,
        )
    assert consumed == [1]

    # Cancel immediately before calling model.transcribe.
    with pytest.raises(JobCancelledError):
        backend.transcribe(
            tmp_path / "audio.wav",
            ASRContext(),
            cancel_check=lambda: True,
        )


def test_transcribe_oom_raises_gpu_oom_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CUDA out-of-memory during transcribe is mapped to GpuOomError."""

    class FakeModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def transcribe(self, path: str, **kwargs: Any) -> tuple[Any, Any]:
            raise RuntimeError("CUDA failed with error out of memory")

    _install_fake_faster_whisper(monkeypatch, FakeModel)
    backend = FasterWhisperBackend()
    backend.ensure_loaded("large-v3", "cuda", "float16", tmp_path)

    with pytest.raises(GpuOomError):
        backend.transcribe(tmp_path / "audio.wav", ASRContext())


def test_unload_clears_model_and_cuda_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """unload clears model reference and empties CUDA cache when torch CUDA is available."""

    class FakeModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    _install_fake_faster_whisper(monkeypatch, FakeModel)

    empty_cache_calls = 0

    def fake_empty_cache() -> None:
        nonlocal empty_cache_calls
        empty_cache_calls += 1

    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: True,
        empty_cache=fake_empty_cache,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    backend = FasterWhisperBackend()
    backend.ensure_loaded("tiny", "cuda", "float16", tmp_path)
    assert backend._model is not None

    backend.unload()
    assert backend._model is None
    assert backend._loaded_key is None
    assert empty_cache_calls == 1

    # Also verify unload succeeds gracefully if torch is missing.
    monkeypatch.setitem(sys.modules, "torch", None)  # type: ignore[arg-type]
    backend.unload()


def test_transcribe_emits_intermediate_stage_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """transcribe() emits loading_audio, analyzing_speech, and transcribing in order."""
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"RIFF")
    monkeypatch.setattr(
        "aurisdeck.backends.faster_whisper_backend._load_audio_input",
        lambda _path: [0.0] * 160,
    )

    class FakeModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, Any]:
            seg = SimpleNamespace(start=0.0, end=0.5, text="hola", words=[])
            return iter([seg]), SimpleNamespace(duration=1.0, language="es")

    _install_fake_faster_whisper(monkeypatch, FakeModel)

    backend = FasterWhisperBackend()
    backend.ensure_loaded("tiny", "cpu", "int8", tmp_path)

    stages: list[str] = []
    res = backend.transcribe(wav_path, ASRContext(), status_cb=stages.append)
    assert len(res.segments) == 1
    assert stages == ["loading_audio", "analyzing_speech", "transcribing"]
