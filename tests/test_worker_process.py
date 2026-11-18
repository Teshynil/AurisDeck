"""Tests for EventBus, pickleable worker IPC messages, and isolated worker_process loop."""

from __future__ import annotations

import ast
import asyncio
import os
import pickle
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from aurisdeck.backends.base import ASRContext, ASRResult, ASRSegment, SpeakerTurn
from aurisdeck.backends.errors import GpuOomError, JobCancelledError
from aurisdeck.workers.events import (
    EventBus,
    WorkerAsrDoneEvent,
    WorkerDiarizeDoneEvent,
    WorkerDiarizeTask,
    WorkerErrorEvent,
    WorkerProgressEvent,
    WorkerShutdownCommand,
    WorkerStageEvent,
    WorkerTranscribeTask,
    WorkerUnloadCommand,
    WorkerUnloadedEvent,
)
from aurisdeck.workers.worker_process import run_worker_loop

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "aurisdeck"


class FakeAsrBackend:
    """In-memory fake ASR backend recording lifecycle calls and parameters."""

    def __init__(self, call_log: list[tuple[str, Any]]) -> None:
        self.call_log = call_log
        self.loaded = False
        self.raise_on_transcribe: Exception | None = None
        self.flip_cancel_on_transcribe: tuple[dict[str, bool], str] | None = None

    def ensure_loaded(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        models_dir: Path | str,
        cpu_threads: int = 0,
        hf_endpoint: str | None = None,
        status_cb: Any = None,
    ) -> None:
        self.call_log.append(("asr.ensure_loaded", {"model_name": model_name, "device": device}))
        if status_cb is not None:
            status_cb("loading_model")
        self.loaded = True

    def transcribe(
        self,
        audio_path: Path,
        context: ASRContext,
        cancel_check: Any = None,
        progress_cb: Any = None,
    ) -> ASRResult:
        self.call_log.append(("asr.transcribe", str(audio_path)))
        if progress_cb is not None:
            progress_cb(500, 1000, 2.5)
        if self.flip_cancel_on_transcribe is not None:
            flag_dict, job_id = self.flip_cancel_on_transcribe
            flag_dict[job_id] = True
        if cancel_check is not None and cancel_check():
            raise JobCancelledError(detail="cancelled mid-stream")
        if self.raise_on_transcribe is not None:
            raise self.raise_on_transcribe
        return ASRResult(
            language_detected=context.language or "es",
            duration_ms=1000,
            segments=[ASRSegment(start_ms=0, end_ms=1000, text="hola mundo")],
        )

    def unload(self) -> None:
        self.call_log.append(("asr.unload", None))
        self.loaded = False


class FakeDiarizerBackend:
    """In-memory fake Diarizer backend recording lifecycle calls and parameters."""

    def __init__(self, call_log: list[tuple[str, Any]]) -> None:
        self.call_log = call_log
        self.loaded = False

    def ensure_loaded(
        self,
        model_id: str,
        device: str,
        models_dir: Path | str,
        hf_token: str | None = None,
        hf_endpoint: str | None = None,
    ) -> None:
        self.call_log.append(("diarizer.ensure_loaded", {"model_id": model_id, "device": device}))
        self.loaded = True

    def diarize(
        self,
        audio_path: Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[SpeakerTurn]:
        self.call_log.append(("diarizer.diarize", str(audio_path)))
        return [SpeakerTurn(start_ms=0, end_ms=1000, speaker_key="SPEAKER_00")]

    def unload(self) -> None:
        self.call_log.append(("diarizer.unload", None))
        self.loaded = False


def _make_transcribe_task(job_id: str = "job-1", track_id: str = "trk-1") -> WorkerTranscribeTask:
    return WorkerTranscribeTask(
        job_id=job_id,
        track_id=track_id,
        audio_path="/tmp/fake.wav",
        model_name="large-v3-turbo",
        compute_type="float16",
        models_dir="/tmp/models",
        context=ASRContext(language="es"),
    )


def _make_diarize_task(job_id: str = "job-1") -> WorkerDiarizeTask:
    return WorkerDiarizeTask(
        job_id=job_id,
        audio_path="/tmp/fake.wav",
        model_id="pyannote/speaker-diarization-3.1",
        models_dir="/tmp/models",
    )


def _drain_events(event_q: queue.Queue[Any]) -> list[Any]:
    events: list[Any] = []
    while True:
        try:
            events.append(event_q.get_nowait())
        except queue.Empty:
            return events


def test_event_bus_subscribe_publish_unsubscribe() -> None:
    async def scenario() -> None:
        bus = EventBus()
        q1 = bus.subscribe()
        q2 = bus.subscribe()

        published = bus.publish("job.stage", job_id="job-42", payload={"stage": "transcribing"})
        assert published["v"] == 1
        assert published["type"] == "job.stage"
        assert published["job_id"] == "job-42"
        assert published["payload"] == {"stage": "transcribing"}
        parsed_ts = datetime.fromisoformat(published["ts"])
        assert parsed_ts.tzinfo is not None

        assert q1.get_nowait() == published
        assert q2.get_nowait() == published

        bus.unsubscribe(q2)
        bus.unsubscribe(q2)  # idempotent

        second = bus.publish("job.done", job_id="job-42")
        assert q1.get_nowait() == second
        assert q2.empty()

    asyncio.run(scenario())


def test_event_bus_slow_consumer_drops_oldest() -> None:
    async def scenario() -> None:
        bus = EventBus()
        q = bus.subscribe(maxsize=2)

        bus.publish("e1", payload={"seq": 1})
        bus.publish("e2", payload={"seq": 2})
        bus.publish("e3", payload={"seq": 3})

        items = [q.get_nowait(), q.get_nowait()]
        assert [item["type"] for item in items] == ["e2", "e3"]
        assert q.empty()

    asyncio.run(scenario())


def test_event_bus_threadsafe_cross_thread_publish() -> None:
    async def scenario() -> None:
        bus = EventBus()
        q = bus.subscribe()

        def worker() -> None:
            bus.publish("thread.event", job_id="j-thread", payload={"ok": True})

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=2.0)

        msg = await asyncio.wait_for(q.get(), timeout=2.0)
        assert msg["v"] == 1
        assert msg["type"] == "thread.event"
        assert msg["job_id"] == "j-thread"
        assert msg["payload"] == {"ok": True}

    asyncio.run(scenario())


def test_ipc_dataclasses_pickle_roundtrip() -> None:
    messages: list[Any] = [
        _make_transcribe_task(),
        _make_diarize_task(),
        WorkerUnloadCommand(),
        WorkerShutdownCommand(),
        WorkerStageEvent("cuda:0", "job-1", "trk-1", "transcribing"),
        WorkerProgressEvent("cuda:0", "job-1", "trk-1", 500, 1000, 2.5),
        WorkerAsrDoneEvent("cuda:0", "job-1", "trk-1", ASRResult("es", 1000, [])),
        WorkerDiarizeDoneEvent("cuda:0", "job-1", [SpeakerTurn(0, 1000, "SPEAKER_00")]),
        WorkerErrorEvent("cuda:0", "job-1", None, "GPU_OOM", {"vram": 8}, "oom"),
        WorkerUnloadedEvent("cuda:0", "idle_timeout"),
    ]
    for msg in messages:
        restored = pickle.loads(pickle.dumps(msg))
        assert restored == msg


def test_worker_process_has_no_sqlite_or_eager_ml_imports() -> None:
    worker_source = (_SRC_ROOT / "workers" / "worker_process.py").read_text(encoding="utf-8")
    tree = ast.parse(worker_source)

    forbidden = {"sqlite3", "aurisdeck.db"}
    top_level_forbidden = {
        "torch",
        "ctranslate2",
        "faster_whisper",
        "pyannote",
        "aurisdeck.backends.faster_whisper_backend",
        "aurisdeck.backends.pyannote_backend",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module not in forbidden
            assert not node.module.startswith("aurisdeck.db")

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in top_level_forbidden
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module not in top_level_forbidden
            assert node.module.split(".")[0] not in {"torch", "ctranslate2", "faster_whisper", "pyannote"}


def test_worker_cuda_visible_devices_and_setsid_for_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    setsid_calls = 0

    def fake_setsid() -> int:
        nonlocal setsid_calls
        setsid_calls += 1
        raise OSError("already session leader")

    monkeypatch.setattr(os, "setsid", fake_setsid)

    observed_cuda_env: list[str | None] = []
    call_log: list[tuple[str, Any]] = []

    def fake_factory() -> tuple[FakeAsrBackend, FakeDiarizerBackend]:
        observed_cuda_env.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
        return FakeAsrBackend(call_log), FakeDiarizerBackend(call_log)

    task_q: queue.Queue[Any] = queue.Queue()
    event_q: queue.Queue[Any] = queue.Queue()
    task_q.put(_make_transcribe_task())
    task_q.put(WorkerShutdownCommand())

    run_worker_loop(
        device_id="cuda:1",
        cuda_visible_index=1,
        task_q=task_q,
        event_q=event_q,
        cancel_flag_dict={},
        idle_unload_seconds=120.0,
        backend_factory=fake_factory,
    )

    assert setsid_calls == 1
    assert observed_cuda_env == ["1"]
    assert call_log[0] == ("asr.ensure_loaded", {"model_name": "large-v3-turbo", "device": "cuda"})


def test_worker_cuda_visible_devices_for_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    observed_cuda_env: list[str | None] = []
    call_log: list[tuple[str, Any]] = []

    def fake_factory() -> tuple[FakeAsrBackend, FakeDiarizerBackend]:
        observed_cuda_env.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
        return FakeAsrBackend(call_log), FakeDiarizerBackend(call_log)

    task_q: queue.Queue[Any] = queue.Queue()
    event_q: queue.Queue[Any] = queue.Queue()
    task_q.put(_make_transcribe_task())
    task_q.put(WorkerShutdownCommand())

    run_worker_loop(
        device_id="cpu",
        cuda_visible_index=None,
        task_q=task_q,
        event_q=event_q,
        cancel_flag_dict={},
        idle_unload_seconds=120.0,
        backend_factory=fake_factory,
    )

    assert observed_cuda_env == [""]
    assert call_log[0] == ("asr.ensure_loaded", {"model_name": "large-v3-turbo", "device": "cpu"})


def test_worker_mutual_unload_asr_and_diarizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "setsid", lambda: 1234)
    call_log: list[tuple[str, Any]] = []
    task_q: queue.Queue[Any] = queue.Queue()
    event_q: queue.Queue[Any] = queue.Queue()

    task_q.put(_make_transcribe_task(job_id="j1", track_id="t1"))
    task_q.put(_make_diarize_task(job_id="j1"))
    task_q.put(_make_transcribe_task(job_id="j2", track_id="t2"))
    task_q.put(WorkerShutdownCommand())

    run_worker_loop(
        device_id="cuda:0",
        cuda_visible_index=0,
        task_q=task_q,
        event_q=event_q,
        cancel_flag_dict={},
        idle_unload_seconds=120.0,
        backend_factory=lambda: (FakeAsrBackend(call_log), FakeDiarizerBackend(call_log)),
    )

    assert [name for name, _ in call_log] == [
        "asr.ensure_loaded",
        "asr.transcribe",
        "asr.unload",
        "diarizer.ensure_loaded",
        "diarizer.diarize",
        "diarizer.unload",
        "asr.ensure_loaded",
        "asr.transcribe",
        "asr.unload",
    ]

    events = _drain_events(event_q)
    assert any(isinstance(ev, WorkerStageEvent) and ev.stage == "loading_model" for ev in events)
    assert any(isinstance(ev, WorkerStageEvent) and ev.stage == "transcribing" for ev in events)
    assert any(isinstance(ev, WorkerProgressEvent) and ev.rtfx == 2.5 for ev in events)
    assert any(isinstance(ev, WorkerAsrDoneEvent) and ev.job_id == "j1" for ev in events)
    assert any(isinstance(ev, WorkerStageEvent) and ev.stage == "diarizing" for ev in events)
    assert any(isinstance(ev, WorkerDiarizeDoneEvent) and ev.job_id == "j1" for ev in events)


def test_worker_manual_unload_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "setsid", lambda: 1234)
    call_log: list[tuple[str, Any]] = []
    task_q: queue.Queue[Any] = queue.Queue()
    event_q: queue.Queue[Any] = queue.Queue()

    task_q.put(_make_transcribe_task())
    task_q.put(WorkerUnloadCommand())
    task_q.put(WorkerShutdownCommand())

    run_worker_loop(
        device_id="cuda:0",
        cuda_visible_index=0,
        task_q=task_q,
        event_q=event_q,
        cancel_flag_dict={},
        idle_unload_seconds=120.0,
        backend_factory=lambda: (FakeAsrBackend(call_log), FakeDiarizerBackend(call_log)),
    )

    assert [name for name, _ in call_log] == [
        "asr.ensure_loaded",
        "asr.transcribe",
        "asr.unload",
    ]
    events = _drain_events(event_q)
    unloaded = [ev for ev in events if isinstance(ev, WorkerUnloadedEvent)]
    assert unloaded == [WorkerUnloadedEvent(device_id="cuda:0", reason="manual")]


def test_worker_idle_timeout_unloads_and_emits_event(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "setsid", lambda: 1234)
    call_log: list[tuple[str, Any]] = []
    task_q: queue.Queue[Any] = queue.Queue()
    event_q: queue.Queue[Any] = queue.Queue()

    worker_thread = threading.Thread(
        target=run_worker_loop,
        kwargs={
            "device_id": "cuda:0",
            "cuda_visible_index": 0,
            "task_q": task_q,
            "event_q": event_q,
            "cancel_flag_dict": {},
            "idle_unload_seconds": 0.15,
            "backend_factory": lambda: (FakeAsrBackend(call_log), FakeDiarizerBackend(call_log)),
        },
        daemon=True,
    )
    worker_thread.start()

    try:
        task_q.put(_make_transcribe_task())
        seen_unloaded: WorkerUnloadedEvent | None = None
        for _ in range(40):
            try:
                ev = event_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(ev, WorkerUnloadedEvent):
                seen_unloaded = ev
                break

        assert seen_unloaded == WorkerUnloadedEvent(device_id="cuda:0", reason="idle_timeout")
        assert ("asr.unload", None) in call_log
    finally:
        task_q.put(WorkerShutdownCommand())
        worker_thread.join(timeout=2.0)
        assert not worker_thread.is_alive()


def test_worker_cancellation_before_and_during_task(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "setsid", lambda: 1234)
    call_log: list[tuple[str, Any]] = []
    cancel_flags: dict[str, bool] = {"job-pre": True}
    asr = FakeAsrBackend(call_log)
    asr.flip_cancel_on_transcribe = (cancel_flags, "job-mid")

    task_q: queue.Queue[Any] = queue.Queue()
    event_q: queue.Queue[Any] = queue.Queue()
    task_q.put(_make_transcribe_task(job_id="job-pre", track_id="t1"))
    task_q.put(_make_diarize_task(job_id="job-pre"))
    task_q.put(_make_transcribe_task(job_id="job-mid", track_id="t2"))
    task_q.put(WorkerShutdownCommand())

    run_worker_loop(
        device_id="cpu",
        cuda_visible_index=None,
        task_q=task_q,
        event_q=event_q,
        cancel_flag_dict=cancel_flags,
        idle_unload_seconds=120.0,
        backend_factory=lambda: (asr, FakeDiarizerBackend(call_log)),
    )

    errors = [ev for ev in _drain_events(event_q) if isinstance(ev, WorkerErrorEvent)]
    assert len(errors) == 3
    assert all(err.error_code == "JOB_CANCELLED" for err in errors)
    assert errors[0].job_id == "job-pre"
    assert errors[1].job_id == "job-pre"
    assert errors[2].job_id == "job-mid"
    # Pre-cancelled tasks must not load models; only job-mid loaded ASR.
    assert [name for name, _ in call_log].count("asr.ensure_loaded") == 1


def test_worker_error_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "setsid", lambda: 1234)
    call_log: list[tuple[str, Any]] = []
    asr = FakeAsrBackend(call_log)
    asr.raise_on_transcribe = GpuOomError(error_params={"vram": 8}, detail="oom")

    task_q: queue.Queue[Any] = queue.Queue()
    event_q: queue.Queue[Any] = queue.Queue()
    task_q.put(_make_transcribe_task(job_id="job-oom"))
    task_q.put(WorkerShutdownCommand())

    run_worker_loop(
        device_id="cuda:0",
        cuda_visible_index=0,
        task_q=task_q,
        event_q=event_q,
        cancel_flag_dict={},
        backend_factory=lambda: (asr, FakeDiarizerBackend(call_log)),
    )

    errors = [ev for ev in _drain_events(event_q) if isinstance(ev, WorkerErrorEvent)]
    assert errors == [
        WorkerErrorEvent(
            device_id="cuda:0",
            job_id="job-oom",
            track_id="trk-1",
            error_code="GPU_OOM",
            error_params={"vram": 8},
            detail="oom",
        )
    ]

    # Unexpected non-backend exception maps to INTERNAL_ERROR
    asr2 = FakeAsrBackend([])
    asr2.raise_on_transcribe = RuntimeError("unexpected boom")
    task_q2: queue.Queue[Any] = queue.Queue()
    event_q2: queue.Queue[Any] = queue.Queue()
    task_q2.put(_make_transcribe_task(job_id="job-boom"))
    task_q2.put(WorkerShutdownCommand())

    run_worker_loop(
        device_id="cpu",
        cuda_visible_index=None,
        task_q=task_q2,
        event_q=event_q2,
        cancel_flag_dict={},
        backend_factory=lambda: (asr2, FakeDiarizerBackend([])),
    )

    errors2 = [ev for ev in _drain_events(event_q2) if isinstance(ev, WorkerErrorEvent)]
    assert errors2 == [
        WorkerErrorEvent(
            device_id="cpu",
            job_id="job-boom",
            track_id="trk-1",
            error_code="INTERNAL_ERROR",
            error_params={},
            detail="unexpected boom",
        )
    ]


def test_worker_logs_traceback_to_task_log_and_bubbles_log_path(tmp_path: Path) -> None:
    worker_log = tmp_path / "logs" / "worker-cpu.log"
    task_log = tmp_path / "outputs" / "job-log-1" / "logs" / "asr-trk-1.log"

    asr = FakeAsrBackend([])
    asr.raise_on_transcribe = RuntimeError("kaboom inside backend")

    task = _make_transcribe_task(job_id="job-log-1")
    task.log_path = str(task_log)
    task.log_level = "DEBUG"

    task_q: queue.Queue[Any] = queue.Queue()
    event_q: queue.Queue[Any] = queue.Queue()
    task_q.put(task)
    task_q.put(WorkerShutdownCommand())

    run_worker_loop(
        device_id="cpu",
        cuda_visible_index=None,
        task_q=task_q,
        event_q=event_q,
        cancel_flag_dict={},
        backend_factory=lambda: (asr, FakeDiarizerBackend([])),
        worker_log_path=str(worker_log),
        log_level="INFO",
    )

    errors = [ev for ev in _drain_events(event_q) if isinstance(ev, WorkerErrorEvent)]
    assert len(errors) == 1
    assert errors[0].log_path == str(task_log)
    assert errors[0].detail == "kaboom inside backend"

    assert worker_log.is_file()
    assert "Worker started device_id=cpu" in worker_log.read_text(encoding="utf-8")

    assert task_log.is_file()
    task_text = task_log.read_text(encoding="utf-8")
    assert "Starting ASR task job=job-log-1 track=trk-1" in task_text
    assert "Traceback (most recent call last):" in task_text
    assert "RuntimeError: kaboom inside backend" in task_text
