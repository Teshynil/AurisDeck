"""Internal asyncio/thread-safe EventBus and pickleable worker IPC messages."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from aurisdeck.backends.base import ASRContext, ASRResult, SpeakerTurn


def _deliver_nowait(queue: asyncio.Queue[dict[str, Any]], event: dict[str, Any]) -> None:
    """Enqueue an event without blocking, dropping the oldest item if full."""
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        queue.put_nowait(event)
    except asyncio.QueueFull:
        pass


class EventBus:
    """Thread-safe pub-sub bus delivering versioned event envelopes to asyncio queues."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[
            asyncio.Queue[dict[str, Any]], asyncio.AbstractEventLoop | None
        ] = {}

    def subscribe(self, maxsize: int = 256) -> asyncio.Queue[dict[str, Any]]:
        """Register a new subscriber queue bound to the current running loop if any."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        with self._lock:
            self._subscribers[queue] = loop
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a subscriber queue from the bus."""
        with self._lock:
            self._subscribers.pop(queue, None)

    def publish(
        self,
        event_type: str,
        job_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Publish a versioned v1 event envelope to all subscribed consumers."""
        envelope: dict[str, Any] = {
            "v": 1,
            "type": event_type,
            "ts": datetime.now(UTC).isoformat(),
            "job_id": job_id,
            "payload": dict(payload) if payload is not None else {},
        }

        with self._lock:
            subscribers = list(self._subscribers.items())

        try:
            current_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        for queue, sub_loop in subscribers:
            target_loop = sub_loop or getattr(queue, "_loop", None)
            if target_loop is not None and not target_loop.is_closed():
                if current_loop is target_loop:
                    _deliver_nowait(queue, envelope)
                else:
                    try:
                        target_loop.call_soon_threadsafe(_deliver_nowait, queue, envelope)
                    except RuntimeError:
                        # Target loop closed between check and dispatch; ignore dead subscriber.
                        pass
            else:
                _deliver_nowait(queue, envelope)

        return envelope


# ---------------------------------------------------------------------------
# Coordinator -> Worker IPC tasks and commands
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WorkerTranscribeTask:
    """Task requesting ASR transcription of a single track on a worker."""

    job_id: str
    track_id: str
    audio_path: str
    model_name: str
    compute_type: str
    models_dir: str
    context: ASRContext
    cpu_threads: int = 0
    hf_endpoint: str | None = None


@dataclass(slots=True)
class WorkerDiarizeTask:
    """Task requesting speaker diarization of an audio file on a worker."""

    job_id: str
    audio_path: str
    model_id: str
    models_dir: str
    hf_token: str | None = None
    hf_endpoint: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


@dataclass(slots=True)
class WorkerUnloadCommand:
    """Command requesting immediate unload of all models resident in the worker."""


@dataclass(slots=True)
class WorkerShutdownCommand:
    """Command requesting clean model unload and worker loop termination."""


# ---------------------------------------------------------------------------
# Worker -> Coordinator IPC events
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WorkerStageEvent:
    """Worker notification indicating transition to a new execution sub-stage."""

    device_id: str
    job_id: str
    track_id: str | None
    stage: str


@dataclass(slots=True)
class WorkerProgressEvent:
    """Worker progress update emitted during ASR segment decoding."""

    device_id: str
    job_id: str
    track_id: str
    processed_ms: int
    total_ms: int
    rtfx: float


@dataclass(slots=True)
class WorkerAsrDoneEvent:
    """Worker result event carrying the completed ASRResult for a track."""

    device_id: str
    job_id: str
    track_id: str
    result: ASRResult


@dataclass(slots=True)
class WorkerDiarizeDoneEvent:
    """Worker result event carrying the completed speaker diarization turns."""

    device_id: str
    job_id: str
    turns: list[SpeakerTurn]


@dataclass(slots=True)
class WorkerErrorEvent:
    """Worker failure or cancellation event mapped to an AurisDeck error_code."""

    device_id: str
    job_id: str
    track_id: str | None
    error_code: str
    error_params: dict[str, Any] = field(default_factory=dict)
    detail: str = ""


@dataclass(slots=True)
class WorkerUnloadedEvent:
    """Worker notification emitted when resident models are unloaded."""

    device_id: str
    reason: str
