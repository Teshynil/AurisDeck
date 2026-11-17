"""Tests for isolated device discovery and on-demand PyannoteDiarizerBackend."""

from __future__ import annotations

import ast
import builtins
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from aurisdeck.backends.base import DiarizerBackend, SpeakerTurn
from aurisdeck.backends.errors import (
    GatedModelTermsError,
    GpuOomError,
    MissingCredentialsError,
    ModelLoadError,
)
from aurisdeck.backends.pyannote_backend import PyannoteDiarizerBackend
from aurisdeck.workers import DeviceInfo, discover_devices, resolve_allowed_devices

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "aurisdeck"


def _make_discovered(num_gpus: int) -> list[DeviceInfo]:
    devices = [
        DeviceInfo(
            id=f"cuda:{idx}",
            kind="cuda",
            index=idx,
            name=f"Mock GPU {idx}",
            vram_total_mb=16384,
            vram_free_mb=15000,
        )
        for idx in range(num_gpus)
    ]
    devices.append(
        DeviceInfo(
            id="cpu",
            kind="cpu",
            index=None,
            name="CPU (8 threads)",
        )
    )
    return devices


def test_resolve_allowed_devices_zero_gpus() -> None:
    discovered = _make_discovered(0)

    assert resolve_allowed_devices("auto", discovered) == ["cpu"]
    assert resolve_allowed_devices("auto", discovered, cpu_fallback=False) == []
    assert resolve_allowed_devices("cpu", discovered) == ["cpu"]
    assert resolve_allowed_devices([0], discovered, cpu_fallback=True) == ["cpu"]
    assert resolve_allowed_devices([0], discovered, cpu_fallback=False) == []


def test_resolve_allowed_devices_two_gpus() -> None:
    discovered = _make_discovered(2)

    assert resolve_allowed_devices("auto", discovered) == ["cuda:0", "cuda:1"]
    assert resolve_allowed_devices([1], discovered) == ["cuda:1"]
    assert resolve_allowed_devices([1, 0], discovered) == ["cuda:1", "cuda:0"]
    assert resolve_allowed_devices("cpu", discovered) == ["cpu"]
    assert resolve_allowed_devices([9], discovered, cpu_fallback=True) == ["cpu"]
    assert resolve_allowed_devices([9], discovered, cpu_fallback=False) == []


def test_discover_devices_two_gpus_mocked_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [
        {
            "id": "cuda:0",
            "index": 0,
            "name": "NVIDIA RTX 4090",
            "vram_total_mb": 24564,
            "vram_free_mb": 24000,
        },
        {
            "id": "cuda:1",
            "index": 1,
            "name": "NVIDIA RTX 3090",
            "vram_total_mb": 24576,
            "vram_free_mb": 23000,
        },
    ]
    recorded_calls: list[tuple[Any, dict[str, Any]]] = []

    def fake_run(cmd: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        recorded_calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    devices = discover_devices()

    assert len(recorded_calls) == 1
    cmd, kwargs = recorded_calls[0]
    assert cmd[0] == sys.executable
    assert cmd[1] == "-c"
    assert kwargs.get("timeout") == 10
    assert not kwargs.get("shell", False)

    assert len(devices) == 3
    assert devices[0] == DeviceInfo(
        id="cuda:0",
        kind="cuda",
        index=0,
        name="NVIDIA RTX 4090",
        vram_total_mb=24564,
        vram_free_mb=24000,
    )
    assert devices[1] == DeviceInfo(
        id="cuda:1",
        kind="cuda",
        index=1,
        name="NVIDIA RTX 3090",
        vram_total_mb=24576,
        vram_free_mb=23000,
    )
    assert devices[2].id == "cpu"
    assert devices[2].kind == "cpu"
    assert devices[2].index is None


@pytest.mark.parametrize(
    "probe_result",
    [
        subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="nvidia-smi failed"),
        subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr=""),
        subprocess.CompletedProcess(args=[], returncode=0, stdout="not-json", stderr=""),
        subprocess.TimeoutExpired(cmd=["python", "-c"], timeout=10),
    ],
)
def test_discover_devices_failure_or_empty_returns_cpu_only(
    monkeypatch: pytest.MonkeyPatch,
    probe_result: subprocess.CompletedProcess[str] | Exception,
) -> None:
    def fake_run(cmd: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if isinstance(probe_result, Exception):
            raise probe_result
        return probe_result

    monkeypatch.setattr(subprocess, "run", fake_run)

    devices = discover_devices()

    assert len(devices) == 1
    assert devices[0].id == "cpu"
    assert devices[0].kind == "cpu"
    assert devices[0].index is None


def _install_fake_pyannote_and_torch(
    monkeypatch: pytest.MonkeyPatch,
    from_pretrained_impl: Any,
) -> tuple[MagicMock, MagicMock]:
    fake_cuda = MagicMock()
    fake_cuda.is_available.return_value = True
    fake_torch = ModuleType("torch")
    fake_torch.device = lambda dev: f"torch.device({dev})"  # type: ignore[attr-defined]
    fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]

    class FakePipeline:
        from_pretrained = staticmethod(from_pretrained_impl)

    fake_pyannote = ModuleType("pyannote")
    fake_pyannote_audio = ModuleType("pyannote.audio")
    fake_pyannote_audio.Pipeline = FakePipeline  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pyannote", fake_pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote_audio)
    return fake_cuda, MagicMock(return_value=FakePipeline)


def test_pyannote_backend_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delitem(sys.modules, "pyannote.audio", raising=False)
    monkeypatch.delitem(sys.modules, "pyannote", raising=False)
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pyannote.audio" or name.startswith("pyannote"):
            raise ImportError("No module named 'pyannote'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    backend = PyannoteDiarizerBackend()
    assert isinstance(backend, DiarizerBackend)

    with pytest.raises(ModelLoadError) as exc_info:
        backend.ensure_loaded(models_dir=tmp_path, hf_token="hf_test")

    assert exc_info.value.error_code == "DIARIZATION_DEPENDENCY_MISSING"


def test_pyannote_backend_missing_token_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_pyannote_and_torch(monkeypatch, MagicMock())
    backend = PyannoteDiarizerBackend()

    with pytest.raises(MissingCredentialsError) as exc_info:
        backend.ensure_loaded(models_dir=tmp_path, hf_token=None)

    assert exc_info.value.error_code == "DIARIZATION_HF_TOKEN_MISSING"


def test_pyannote_backend_allows_offline_cache_without_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot_dir = tmp_path / "models--pyannote--speaker-diarization-3.1" / "snapshots" / "rev1"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.yaml").write_text("pipeline: {}\n", encoding="utf-8")

    fake_pipeline = MagicMock()
    from_pretrained = MagicMock(return_value=fake_pipeline)
    _install_fake_pyannote_and_torch(monkeypatch, from_pretrained)

    backend = PyannoteDiarizerBackend()
    backend.ensure_loaded(
        device="cpu",
        models_dir=tmp_path,
        hf_token=None,
        hf_endpoint="https://hf-mirror.example",
    )

    from_pretrained.assert_called_once_with(
        "pyannote/speaker-diarization-3.1",
        token=None,
        cache_dir=str(tmp_path),
    )
    fake_pipeline.to.assert_called_once_with("torch.device(cpu)")


@pytest.mark.parametrize(
    "error_message",
    [
        "403 Client Error: Forbidden for url",
        "401 Unauthorized: Invalid credentials",
        "Cannot access gated repo for url",
        "Please visit the model page to accept the user conditions",
    ],
)
def test_pyannote_backend_gated_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error_message: str,
) -> None:
    def raise_gated(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(error_message)

    _install_fake_pyannote_and_torch(monkeypatch, raise_gated)
    backend = PyannoteDiarizerBackend()

    with pytest.raises(GatedModelTermsError) as exc_info:
        backend.ensure_loaded(models_dir=tmp_path, hf_token="hf_secret")

    assert exc_info.value.error_code == "DIARIZATION_GATED_TERMS"
    assert error_message in exc_info.value.detail


def test_pyannote_backend_oom_and_generic_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def raise_oom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    _install_fake_pyannote_and_torch(monkeypatch, raise_oom)
    backend = PyannoteDiarizerBackend()

    with pytest.raises(GpuOomError) as oom_exc:
        backend.ensure_loaded(models_dir=tmp_path, hf_token="hf_secret")
    assert oom_exc.value.error_code == "GPU_OOM"

    def raise_other(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("Corrupted checkpoint file")

    _install_fake_pyannote_and_torch(monkeypatch, raise_other)
    with pytest.raises(ModelLoadError) as load_exc:
        backend.ensure_loaded(models_dir=tmp_path, hf_token="hf_secret")
    assert load_exc.value.error_code == "MODEL_LOAD_FAILED"


def test_pyannote_backend_use_auth_token_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    fake_pipeline = MagicMock()

    def legacy_from_pretrained(model_id: str, **kwargs: Any) -> Any:
        calls.append(kwargs)
        if "token" in kwargs:
            raise TypeError("from_pretrained() got an unexpected keyword argument 'token'")
        return fake_pipeline

    _install_fake_pyannote_and_torch(monkeypatch, legacy_from_pretrained)
    backend = PyannoteDiarizerBackend()
    backend.ensure_loaded(device="cuda:0", models_dir=tmp_path, hf_token="hf_legacy")

    assert calls == [
        {"token": "hf_legacy", "cache_dir": str(tmp_path)},
        {"use_auth_token": "hf_legacy", "cache_dir": str(tmp_path)},
    ]
    fake_pipeline.to.assert_called_once_with("torch.device(cuda:0)")


@dataclass
class _FakeSegment:
    start: float
    end: float


class _FakeAnnotation:
    def __init__(self, tracks: list[tuple[_FakeSegment, str, str]]) -> None:
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False) -> list[tuple[_FakeSegment, str, str]]:
        assert yield_label is True
        return self._tracks


class _FakeDiarizeOutput:
    def __init__(self, annotation: _FakeAnnotation) -> None:
        self.speaker_diarization = annotation


def test_pyannote_backend_diarize_and_unload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = PyannoteDiarizerBackend()
    with pytest.raises(ModelLoadError):
        backend.diarize(tmp_path / "audio.wav")

    annotation = _FakeAnnotation(
        [
            (_FakeSegment(0.1234, 1.5678), "A", "SPEAKER_00"),
            (_FakeSegment(2.0, 4.25), "B", "SPEAKER_01"),
        ]
    )
    fake_pipeline = MagicMock(side_effect=[annotation, _FakeDiarizeOutput(annotation)])
    fake_cuda, _ = _install_fake_pyannote_and_torch(
        monkeypatch, MagicMock(return_value=fake_pipeline)
    )

    backend.ensure_loaded(device="cuda:0", models_dir=tmp_path, hf_token="hf_ok")

    audio_file = tmp_path / "sample.wav"
    turns = backend.diarize(audio_file, min_speakers=2, max_speakers=None)
    assert turns == [
        SpeakerTurn(start_ms=123, end_ms=1568, speaker_key="SPEAKER_00"),
        SpeakerTurn(start_ms=2000, end_ms=4250, speaker_key="SPEAKER_01"),
    ]
    fake_pipeline.assert_called_with(str(audio_file), min_speakers=2)

    turns_v4 = backend.diarize(audio_file, min_speakers=None, max_speakers=4)
    assert turns_v4 == turns
    fake_pipeline.assert_called_with(str(audio_file), max_speakers=4)

    backend.unload()
    fake_cuda.empty_cache.assert_called_once()

    with pytest.raises(ModelLoadError):
        backend.diarize(audio_file)


def test_ast_no_top_level_ml_imports() -> None:
    devices_source = (_SRC_ROOT / "workers" / "devices.py").read_text(encoding="utf-8")
    devices_tree = ast.parse(devices_source)
    for node in ast.walk(devices_tree):
        if isinstance(node, ast.Import):
            imported = {alias.name.split(".")[0] for alias in node.names}
            assert not (imported & {"torch", "ctranslate2", "pyannote"})
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            assert root not in {"torch", "ctranslate2", "pyannote"}

    pyannote_source = (_SRC_ROOT / "backends" / "pyannote_backend.py").read_text(encoding="utf-8")
    pyannote_tree = ast.parse(pyannote_source)
    for node in pyannote_tree.body:
        if isinstance(node, ast.Import):
            imported = {alias.name.split(".")[0] for alias in node.names}
            assert not (imported & {"torch", "pyannote"})
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            assert root not in {"torch", "pyannote"}
