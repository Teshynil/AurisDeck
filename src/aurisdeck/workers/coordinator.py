"""Single-writer Coordinator managing Dual-GPU/CPU spawn workers, checkpoints, and exports."""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import queue
import shutil
import signal
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from aurisdeck.backends.base import ASRContext, ASRResult, ASRSegment, ASRWord, SpeakerTurn
from aurisdeck.config import Settings
from aurisdeck.db.recovery import run_startup_recovery
from aurisdeck.db.repositories import JobRecord, JobRepo, SegmentRecord, SegmentRepo
from aurisdeck.export.job_outputs import render_job_outputs
from aurisdeck.logging_setup import task_file_handler
from aurisdeck.media.ffmpeg import MediaInvalidError, extract_wav_16k, mix_multitrack_audio
from aurisdeck.media.ingest import SUPPORTED_MEDIA_EXTENSIONS
from aurisdeck.pipeline.merger import TranscriptMerger
from aurisdeck.storage.paths import DataPaths, atomic_write_text, sanitize_filename
from aurisdeck.storage.stores import GlossaryStore, ReplacementStore
from aurisdeck.workers.devices import DeviceInfo, discover_devices, resolve_allowed_devices
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

logger = logging.getLogger(__name__)

_SHUTDOWN_JOIN_TIMEOUT_S = 3.0
_CANCEL_GRACE_TIMEOUT_S = 2.0


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _serialize_asr_result(result: ASRResult) -> str:
    payload = {
        "schema_version": 1,
        "language_detected": result.language_detected,
        "duration_ms": int(result.duration_ms),
        "segments": [
            {
                "start_ms": int(seg.start_ms),
                "end_ms": int(seg.end_ms),
                "text": seg.text,
                "words": [
                    {
                        "start_ms": int(w.start_ms),
                        "end_ms": int(w.end_ms),
                        "text": w.text,
                        "prob": float(w.prob),
                    }
                    for w in seg.words
                ],
            }
            for seg in result.segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _deserialize_asr_result(raw_text: str) -> ASRResult:
    data = json.loads(raw_text)
    segments = [
        ASRSegment(
            start_ms=int(seg["start_ms"]),
            end_ms=int(seg["end_ms"]),
            text=str(seg["text"]),
            words=[
                ASRWord(
                    start_ms=int(w["start_ms"]),
                    end_ms=int(w["end_ms"]),
                    text=str(w["text"]),
                    prob=float(w.get("prob", 1.0)),
                )
                for w in seg.get("words", [])
            ],
        )
        for seg in data.get("segments", [])
    ]
    return ASRResult(
        language_detected=data.get("language_detected"),
        duration_ms=int(data.get("duration_ms", 0)),
        segments=segments,
    )


def _output_stem(job: JobRecord) -> str:
    raw_title = (job.title or "").strip()
    if not raw_title and job.tracks:
        raw_title = Path(sanitize_filename(job.tracks[0].original_filename)).stem
    if not raw_title:
        return job.id
    safe = sanitize_filename(raw_title)
    if Path(safe).suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS:
        return Path(safe).stem or job.id
    return safe


def _terminate_process_group(proc: Any) -> None:
    """Forcefully terminate a spawned worker's process group and wait for exit."""
    pid = proc.pid
    if pid is not None:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except (OSError, ValueError) as exc:
                logger.debug("Fallback proc.kill failed for pid=%s: %s", pid, exc)
    try:
        proc.join(timeout=2.0)
    except (OSError, ValueError) as exc:
        logger.debug("Process join failed during termination: %s", exc)


@dataclass(slots=True)
class _WorkerHandle:
    device_id: str
    cuda_visible_index: int | None
    proc: Any
    task_q: Any
    busy: bool = False
    current_job_id: str | None = None
    current_track_id: str | None = None
    current_kind: Literal["transcribe", "diarize"] | None = None


@dataclass(slots=True)
class _PendingTask:
    job_id: str
    track_id: str | None
    kind: Literal["transcribe", "diarize"]
    allowed_devices: list[str]
    audio_path: str
    asr_context: ASRContext | None = None


@dataclass(slots=True)
class _PreprocessDone:
    job_id: str
    wav_paths: dict[str, str]


@dataclass(slots=True)
class _PreprocessError:
    job_id: str
    error_code: str
    detail: str


@dataclass(slots=True)
class _PreprocessCancelled:
    job_id: str


@dataclass(slots=True)
class _DiarizeReady:
    job_id: str
    wav_path: str


@dataclass(slots=True)
class _DiarizePrepError:
    job_id: str
    error_code: str
    detail: str


@dataclass(slots=True)
class _DispatchWakeup:
    pass


@dataclass(slots=True)
class _StopSignal:
    pass


class Coordinator:
    """Single-writer orchestrator for AurisDeck jobs, worker processes, and exports."""

    def __init__(
        self,
        settings: Settings,
        paths: DataPaths,
        job_repo: JobRepo,
        segment_repo: SegmentRepo,
        event_bus: EventBus,
        devices: list[DeviceInfo] | None = None,
        backend_factory: Callable[[], tuple[Any, Any]] | None = None,
    ) -> None:
        self._settings = settings
        self._paths = paths
        self._job_repo = job_repo
        self._segment_repo = segment_repo
        self._event_bus = event_bus
        self._devices: list[DeviceInfo] | None = list(devices) if devices is not None else None
        self._backend_factory = backend_factory

        self._mp_ctx = multiprocessing.get_context("spawn")
        self._manager: Any = None
        self._cancel_flag_dict: Any = {}
        self._event_q: Any = None
        self._inbox_q: queue.Queue[Any] = queue.Queue()

        self._workers: dict[str, _WorkerHandle] = {}
        self._pending_tasks: deque[_PendingTask] = deque()
        self._state_lock = threading.RLock()
        self._db_lock = threading.RLock()
        self._worker_done_cond = threading.Condition(self._state_lock)

        self._stop_event = threading.Event()
        self._started = False
        self._coord_thread: threading.Thread | None = None
        self._relay_thread: threading.Thread | None = None

    def list_devices(self) -> list[DeviceInfo]:
        """Return discovered or injected compute devices."""
        if self._devices is None:
            self._devices = discover_devices()
        return list(self._devices)

    def start(self) -> None:
        """Run startup recovery, spawn device workers, and start the coordinator loop."""
        if self._started:
            return

        self._paths.ensure_dirs()
        self._run_recovery()
        devices = self.list_devices()

        self._stop_event.clear()
        self._inbox_q = queue.Queue()
        self._manager = self._mp_ctx.Manager()
        self._cancel_flag_dict = self._manager.dict()
        self._event_q = self._mp_ctx.Queue()

        cuda_devices = [dev for dev in devices if dev.kind == "cuda"]
        with self._state_lock:
            self._workers.clear()
            self._pending_tasks.clear()
            if cuda_devices:
                for dev in cuda_devices:
                    self._workers[dev.id] = self._spawn_worker(dev.id, dev.index)
            else:
                self._workers["cpu"] = self._spawn_worker("cpu", None)

        self._relay_thread = threading.Thread(
            target=self._relay_worker_events,
            daemon=True,
            name="aurisdeck-event-relay",
        )
        self._coord_thread = threading.Thread(
            target=self._coordinator_loop,
            daemon=True,
            name="aurisdeck-coordinator",
        )
        self._relay_thread.start()
        self._coord_thread.start()
        self._started = True

    def stop(self) -> None:
        """Gracefully shut down workers, coordinator threads, and IPC resources."""
        if not self._started:
            return

        self._stop_event.set()

        with self._state_lock:
            workers = list(self._workers.values())
            self._workers.clear()
            self._pending_tasks.clear()

        for worker in workers:
            try:
                worker.task_q.put(WorkerShutdownCommand())
            except (OSError, ValueError, queue.Full) as exc:
                logger.debug("Failed sending shutdown command to %s: %s", worker.device_id, exc)

        for worker in workers:
            try:
                worker.proc.join(timeout=_SHUTDOWN_JOIN_TIMEOUT_S)
            except (OSError, ValueError) as exc:
                logger.debug("Worker %s join failed on stop: %s", worker.device_id, exc)
            if worker.proc.is_alive():
                _terminate_process_group(worker.proc)
            self._close_queue(worker.task_q)

        self._inbox_q.put(_StopSignal())

        if self._coord_thread is not None:
            self._coord_thread.join(timeout=5.0)
            self._coord_thread = None
        if self._relay_thread is not None:
            self._relay_thread.join(timeout=5.0)
            self._relay_thread = None

        self._close_queue(self._event_q)
        self._event_q = None

        if self._manager is not None:
            try:
                self._manager.shutdown()
            except (OSError, EOFError) as exc:
                logger.debug("Manager shutdown failed: %s", exc)
            self._manager = None
        self._cancel_flag_dict = {}
        self._started = False

    def unload_all_models(self) -> None:
        """Request all active workers to unload resident ASR/diarization models."""
        with self._state_lock:
            workers = list(self._workers.values())
        for worker in workers:
            worker.task_q.put(WorkerUnloadCommand())

    def enqueue_job(self, job_id: str) -> None:
        """Validate, enter preprocessing, and schedule pending tracks for a job."""
        if not self._started:
            raise RuntimeError("Coordinator must be started before enqueuing jobs")

        self._clear_cancel_flag(job_id)

        with self._db_lock:
            job = self._job_repo.get_job(job_id)
            if job is None:
                raise ValueError(f"Job not found: {job_id}")

            if not job.tracks:
                self._job_repo.update_job_status(
                    job_id,
                    "failed",
                    error_code="MEDIA_INVALID",
                    error_detail="Job has no tracks",
                )
                self._event_bus.publish(
                    "job.error",
                    job_id=job_id,
                    payload={"error_code": "MEDIA_INVALID", "detail": "Job has no tracks"},
                )
                return

            diar_enabled = job.mode == "single" and bool(
                job.effective_config.get("diarization", {}).get("enabled", False)
            )
            diar_status = "queued" if diar_enabled else "none"
            self._job_repo.update_job_status(
                job_id,
                "preprocessing",
                diarization_status=diar_status,
                error_code=None,
                error_params=None,
                error_detail=None,
                completed_at=None,
            )

        self._event_bus.publish(
            "job.stage",
            job_id=job_id,
            payload={"status": "preprocessing", "stage": "preprocessing"},
        )
        threading.Thread(
            target=self._preprocess_worker,
            args=(job_id,),
            daemon=True,
            name=f"preprocess-{job_id}",
        ).start()

    def trigger_diarize(self, job_id: str) -> None:
        """Trigger post-hoc diarization on a completed/completed_partial job without re-running ASR."""
        if not self._started:
            raise RuntimeError("Coordinator must be started before triggering diarization")

        self._clear_cancel_flag(job_id)

        with self._db_lock:
            job = self._job_repo.get_job(job_id)
            if job is None:
                raise ValueError(f"Job not found: {job_id}")
            if job.status not in {"completed", "completed_partial"}:
                raise ValueError(
                    f"Job {job_id} in status {job.status!r} cannot trigger diarization"
                )
            if job.mode != "single" or not job.tracks:
                raise ValueError(f"Job {job_id} mode {job.mode!r} does not support diarization")
            if not all(track.asr_status == "completed" for track in job.tracks):
                raise ValueError(f"Job {job_id} does not have completed ASR for all tracks")

            self._job_repo.update_job_status(
                job_id,
                "diarizing",
                diarization_status="queued",
                error_code=None,
                error_params=None,
                error_detail=None,
            )

        self._event_bus.publish(
            "job.stage",
            job_id=job_id,
            payload={"status": "diarizing", "stage": "diarizing"},
        )
        threading.Thread(
            target=self._prepare_diarize_worker,
            args=(job_id,),
            daemon=True,
            name=f"diarize-prep-{job_id}",
        ).start()

    def retry_job(self, job_id: str) -> None:
        """Re-enqueue a failed, partial, or cancelled job preserving completed ASR checkpoints."""
        self._clear_cancel_flag(job_id)
        self.enqueue_job(job_id)

    def cancel_job(self, job_id: str) -> None:
        """Cancel a queued or running job, killing stubborn workers after a 2s grace period."""
        with self._db_lock:
            job = self._job_repo.get_job(job_id)
            if job is None:
                raise ValueError(f"Job not found: {job_id}")
            if job.status in {"completed", "failed", "cancelled"}:
                return
            self._cancel_flag_dict[job_id] = True
            self._job_repo.update_job_status(job_id, "cancelling")

        self._event_bus.publish(
            "job.stage",
            job_id=job_id,
            payload={"status": "cancelling", "stage": "cancelling"},
        )

        with self._state_lock:
            self._remove_pending_tasks_locked(job_id)
            deadline = time.monotonic() + _CANCEL_GRACE_TIMEOUT_S
            while any(
                worker.busy and worker.current_job_id == job_id
                for worker in self._workers.values()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._worker_done_cond.wait(timeout=remaining)

            stubborn = [
                worker
                for worker in self._workers.values()
                if worker.busy and worker.current_job_id == job_id
            ]
            for worker in stubborn:
                self._kill_and_respawn_worker_locked(worker)

        with self._db_lock:
            self._job_repo.update_job_status(job_id, "cancelled")

        self._cleanup_job_tmp(job_id)
        self._event_bus.publish(
            "job.cancelled",
            job_id=job_id,
            payload={"status": "cancelled"},
        )
        self._inbox_q.put(_DispatchWakeup())

    def reexport_job(self, job_id: str) -> list[Path]:
        """Re-render all configured output files purely on CPU from SQLite data."""
        with self._db_lock:
            job = self._job_repo.get_job(job_id)
            if job is None:
                raise ValueError(f"Job not found: {job_id}")
            segments = self._segment_repo.list_segments(job_id)
            speakers = self._segment_repo.list_speakers(job_id)

        return self._export_job_outputs(job, segments, speakers)

    # ------------------------------------------------------------------
    # Internal lifecycle, relay, and coordinator loop
    # ------------------------------------------------------------------

    def _run_recovery(self) -> None:
        db_source = self._job_repo._db_path
        if callable(db_source):
            conn = db_source()
            try:
                run_startup_recovery(conn, self._paths.tmp)
            finally:
                conn.close()
        else:
            run_startup_recovery(db_source, self._paths.tmp)

    def _spawn_worker(self, device_id: str, cuda_visible_index: int | None) -> _WorkerHandle:
        task_q = self._mp_ctx.Queue()
        proc = self._mp_ctx.Process(
            target=run_worker_loop,
            args=(
                device_id,
                cuda_visible_index,
                task_q,
                self._event_q,
                self._cancel_flag_dict,
                float(self._settings.model_idle_unload_seconds),
                self._backend_factory,
                str(self._paths.worker_log_path(device_id)),
                str(self._settings.log_level),
                int(self._settings.log_max_bytes),
                int(self._settings.log_backup_count),
            ),
            daemon=True,
            name=f"aurisdeck-worker-{device_id}",
        )
        proc.start()
        return _WorkerHandle(
            device_id=device_id,
            cuda_visible_index=cuda_visible_index,
            proc=proc,
            task_q=task_q,
        )

    def _kill_and_respawn_worker_locked(self, worker: _WorkerHandle) -> None:
        _terminate_process_group(worker.proc)
        self._close_queue(worker.task_q)
        self._workers[worker.device_id] = self._spawn_worker(
            worker.device_id,
            worker.cuda_visible_index,
        )

    def _ensure_cpu_worker_locked(self) -> None:
        if "cpu" not in self._workers:
            self._workers["cpu"] = self._spawn_worker("cpu", None)

    @staticmethod
    def _close_queue(q: Any) -> None:
        if q is None:
            return
        try:
            q.close()
            q.join_thread()
        except (OSError, ValueError) as exc:
            logger.debug("Queue cleanup error: %s", exc)

    def _clear_cancel_flag(self, job_id: str) -> None:
        try:
            self._cancel_flag_dict.pop(job_id, None)
        except (OSError, EOFError, KeyError) as exc:
            logger.debug("Failed clearing cancel flag for %s: %s", job_id, exc)

    def _is_cancelled(self, job_id: str) -> bool:
        try:
            return bool(self._cancel_flag_dict.get(job_id, False))
        except Exception:  # noqa: BLE001
            return False

    def _remove_pending_tasks_locked(self, job_id: str) -> None:
        self._pending_tasks = deque(
            task for task in self._pending_tasks if task.job_id != job_id
        )

    def _cleanup_job_tmp(self, job_id: str) -> None:
        shutil.rmtree(self._paths.job_tmp_dir(job_id), ignore_errors=True)

    def _relay_worker_events(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self._event_q.get(timeout=0.1)
            except queue.Empty:
                continue
            except (EOFError, OSError, ValueError):
                break
            self._inbox_q.put(event)

    def _coordinator_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._inbox_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if isinstance(item, _StopSignal):
                break

            try:
                self._handle_inbox_item(item)
            except Exception:
                logger.exception("Unhandled error in coordinator loop handling %r", item)

    def _handle_inbox_item(self, item: Any) -> None:
        if isinstance(item, _DispatchWakeup):
            with self._state_lock:
                self._dispatch_pending_locked()
            return

        if isinstance(item, _PreprocessDone):
            self._on_preprocess_done(item)
            return

        if isinstance(item, _PreprocessError):
            self._on_preprocess_error(item)
            return

        if isinstance(item, _PreprocessCancelled):
            self._on_job_cancelled_cleanup(item.job_id)
            return

        if isinstance(item, _DiarizeReady):
            self._on_diarize_ready(item)
            return

        if isinstance(item, _DiarizePrepError):
            self._on_diarize_prep_error(item)
            return

        if isinstance(item, WorkerStageEvent):
            self._event_bus.publish(
                "job.stage",
                job_id=item.job_id,
                payload={
                    "device_id": item.device_id,
                    "track_id": item.track_id,
                    "stage": item.stage,
                },
            )
            return

        if isinstance(item, WorkerProgressEvent):
            self._event_bus.publish(
                "job.progress",
                job_id=item.job_id,
                payload={
                    "device_id": item.device_id,
                    "track_id": item.track_id,
                    "processed_ms": item.processed_ms,
                    "total_ms": item.total_ms,
                    "rtfx": item.rtfx,
                },
            )
            return

        if isinstance(item, WorkerUnloadedEvent):
            self._event_bus.publish(
                "system.worker_unloaded",
                job_id=None,
                payload={"device_id": item.device_id, "reason": item.reason},
            )
            return

        if isinstance(item, WorkerAsrDoneEvent):
            self._on_asr_done(item)
            return

        if isinstance(item, WorkerDiarizeDoneEvent):
            self._on_diarize_done(item)
            return

        if isinstance(item, WorkerErrorEvent):
            self._on_worker_error(item)
            return

    # ------------------------------------------------------------------
    # Preprocessing workers (run off the coordinator thread)
    # ------------------------------------------------------------------

    def _preprocess_worker(self, job_id: str) -> None:
        with self._db_lock:
            job = self._job_repo.get_job(job_id)

        if job is None or self._is_cancelled(job_id):
            self._inbox_q.put(_PreprocessCancelled(job_id=job_id))
            return

        cfg = job.effective_config
        normalize_audio = bool(cfg.get("preprocessing", {}).get("normalize_audio", False))
        diar_enabled = job.mode == "single" and bool(
            cfg.get("diarization", {}).get("enabled", False)
        )

        tmp_dir = self._paths.job_tmp_dir(job_id)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        wav_paths: dict[str, str] = {}

        for track in job.tracks:
            if self._is_cancelled(job_id) or self._stop_event.is_set():
                self._inbox_q.put(_PreprocessCancelled(job_id=job_id))
                return

            dst_wav = tmp_dir / f"{track.id}.wav"
            needs_wav = (track.asr_status != "completed") or (
                diar_enabled and not dst_wav.is_file()
            )
            if needs_wav:
                src_path = Path(track.media_relpath)
                if not src_path.is_absolute():
                    src_path = self._paths.root / src_path
                try:
                    extract_wav_16k(src_path, dst_wav, normalize_audio=normalize_audio)
                except MediaInvalidError as exc:
                    self._inbox_q.put(
                        _PreprocessError(job_id=job_id, error_code=exc.error_code, detail=exc.detail)
                    )
                    return
                except Exception as exc:  # noqa: BLE001
                    self._inbox_q.put(
                        _PreprocessError(job_id=job_id, error_code="INTERNAL_ERROR", detail=str(exc))
                    )
                    return

            wav_paths[track.id] = str(dst_wav)

        self._inbox_q.put(_PreprocessDone(job_id=job_id, wav_paths=wav_paths))

    def _prepare_diarize_worker(self, job_id: str) -> None:
        with self._db_lock:
            job = self._job_repo.get_job(job_id)

        if job is None or not job.tracks or self._is_cancelled(job_id):
            self._inbox_q.put(_PreprocessCancelled(job_id=job_id))
            return

        track = job.tracks[0]
        tmp_dir = self._paths.job_tmp_dir(job_id)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        dst_wav = tmp_dir / f"{track.id}.wav"

        if not dst_wav.is_file():
            normalize_audio = bool(
                job.effective_config.get("preprocessing", {}).get("normalize_audio", False)
            )
            src_path = Path(track.media_relpath)
            if not src_path.is_absolute():
                src_path = self._paths.root / src_path
            try:
                extract_wav_16k(src_path, dst_wav, normalize_audio=normalize_audio)
            except MediaInvalidError as exc:
                self._inbox_q.put(
                    _DiarizePrepError(job_id=job_id, error_code=exc.error_code, detail=exc.detail)
                )
                return
            except Exception as exc:  # noqa: BLE001
                self._inbox_q.put(
                    _DiarizePrepError(job_id=job_id, error_code="INTERNAL_ERROR", detail=str(exc))
                )
                return

        self._inbox_q.put(_DiarizeReady(job_id=job_id, wav_path=str(dst_wav)))

    # ------------------------------------------------------------------
    # Coordinator event handlers (single-writer thread)
    # ------------------------------------------------------------------

    def _resolve_job_devices(self, cfg: Mapping[str, Any]) -> list[str]:
        hw = cfg.get("hardware", {})
        gpus_spec = hw.get("gpus", "auto")
        cpu_fallback = bool(hw.get("cpu_fallback", True))
        return resolve_allowed_devices(gpus_spec, self.list_devices(), cpu_fallback=cpu_fallback)

    def _on_preprocess_done(self, item: _PreprocessDone) -> None:
        with self._db_lock:
            job = self._job_repo.get_job(item.job_id)
            if job is None:
                return
            if job.status in {"cancelling", "cancelled"} or self._is_cancelled(item.job_id):
                self._job_repo.update_job_status(item.job_id, "cancelled")
                self._cleanup_job_tmp(item.job_id)
                self._event_bus.publish("job.cancelled", job_id=item.job_id, payload={"status": "cancelled"})
                return

            allowed_devices = self._resolve_job_devices(job.effective_config)
            if not allowed_devices:
                self._job_repo.update_job_status(
                    item.job_id,
                    "failed",
                    error_code="NO_COMPATIBLE_DEVICE",
                    error_detail="No compatible compute device available for job hardware config",
                )
                self._cleanup_job_tmp(item.job_id)
                self._event_bus.publish(
                    "job.error",
                    job_id=item.job_id,
                    payload={"error_code": "NO_COMPATIBLE_DEVICE"},
                )
                return

            pending_tracks = [t for t in job.tracks if t.asr_status != "completed"]
            if not pending_tracks:
                self._proceed_post_asr_locked(job, item.wav_paths, allowed_devices)
                return

            self._job_repo.update_job_status(item.job_id, "transcribing")

        self._event_bus.publish(
            "job.stage",
            job_id=item.job_id,
            payload={"status": "transcribing", "stage": "transcribing"},
        )

        asr_ctx = self._build_asr_context(job)
        with self._state_lock:
            if "cpu" in allowed_devices:
                self._ensure_cpu_worker_locked()
            for track in pending_tracks:
                self._pending_tasks.append(
                    _PendingTask(
                        job_id=job.id,
                        track_id=track.id,
                        kind="transcribe",
                        allowed_devices=allowed_devices,
                        audio_path=item.wav_paths[track.id],
                        asr_context=asr_ctx,
                    )
                )
            self._dispatch_pending_locked()

    def _on_preprocess_error(self, item: _PreprocessError) -> None:
        with self._db_lock:
            job = self._job_repo.get_job(item.job_id)
            if job is None:
                return
            if job.status in {"cancelling", "cancelled"} or self._is_cancelled(item.job_id):
                self._job_repo.update_job_status(item.job_id, "cancelled")
                self._cleanup_job_tmp(item.job_id)
                return
            self._job_repo.update_job_status(
                item.job_id,
                "failed",
                error_code=item.error_code,
                error_detail=item.detail,
            )

        self._cleanup_job_tmp(item.job_id)
        self._event_bus.publish(
            "job.error",
            job_id=item.job_id,
            payload={"error_code": item.error_code, "detail": item.detail},
        )

    def _on_job_cancelled_cleanup(self, job_id: str) -> None:
        with self._state_lock:
            self._remove_pending_tasks_locked(job_id)
            still_busy = any(
                w.busy and w.current_job_id == job_id for w in self._workers.values()
            )
        if not still_busy:
            with self._db_lock:
                job = self._job_repo.get_job(job_id)
                if job is not None and job.status != "cancelled":
                    self._job_repo.update_job_status(job_id, "cancelled")
            self._cleanup_job_tmp(job_id)

    def _on_diarize_ready(self, item: _DiarizeReady) -> None:
        with self._db_lock:
            job = self._job_repo.get_job(item.job_id)
            if job is None:
                return
            if job.status in {"cancelling", "cancelled"} or self._is_cancelled(item.job_id):
                self._job_repo.update_job_status(item.job_id, "cancelled")
                self._cleanup_job_tmp(item.job_id)
                return
            allowed_devices = self._resolve_job_devices(job.effective_config)
            if not allowed_devices:
                self._finalize_merge_and_export(
                    job,
                    turns=None,
                    diarization_failed=True,
                    error_code="NO_COMPATIBLE_DEVICE",
                    error_detail="No compatible compute device available for diarization",
                )
                return

        with self._state_lock:
            if "cpu" in allowed_devices:
                self._ensure_cpu_worker_locked()
            self._pending_tasks.append(
                _PendingTask(
                    job_id=item.job_id,
                    track_id=None,
                    kind="diarize",
                    allowed_devices=allowed_devices,
                    audio_path=item.wav_path,
                )
            )
            self._dispatch_pending_locked()

    def _on_diarize_prep_error(self, item: _DiarizePrepError) -> None:
        with self._db_lock:
            job = self._job_repo.get_job(item.job_id)
            if job is None:
                return
            if all(t.asr_status == "completed" for t in job.tracks):
                self._finalize_merge_and_export(
                    job,
                    turns=None,
                    diarization_failed=True,
                    error_code=item.error_code,
                    error_detail=item.detail,
                )
            else:
                self._job_repo.update_job_status(
                    item.job_id,
                    "failed",
                    diarization_status="failed",
                    error_code=item.error_code,
                    error_detail=item.detail,
                )
                self._cleanup_job_tmp(item.job_id)
                self._event_bus.publish(
                    "job.error",
                    job_id=item.job_id,
                    payload={"error_code": item.error_code, "detail": item.detail},
                )

    def _mark_worker_idle(
        self,
        device_id: str,
        job_id: str,
        track_id: str | None,
    ) -> None:
        with self._state_lock:
            worker = self._workers.get(device_id)
            if (
                worker is not None
                and worker.busy
                and worker.current_job_id == job_id
                and worker.current_track_id == track_id
            ):
                worker.busy = False
                worker.current_job_id = None
                worker.current_track_id = None
                worker.current_kind = None
                self._worker_done_cond.notify_all()

    def _on_asr_done(self, event: WorkerAsrDoneEvent) -> None:
        self._mark_worker_idle(event.device_id, event.job_id, event.track_id)

        with self._db_lock:
            job = self._job_repo.get_job(event.job_id)
            if job is None:
                with self._state_lock:
                    self._dispatch_pending_locked()
                return

            if job.status in {"cancelling", "cancelled"} or self._is_cancelled(event.job_id):
                self._on_job_cancelled_cleanup(event.job_id)
                with self._state_lock:
                    self._dispatch_pending_locked()
                return

            if job.status == "failed":
                with self._state_lock:
                    self._dispatch_pending_locked()
                return

            ckpt_path = self._paths.job_checkpoints_dir(event.job_id) / f"{event.track_id}.asr.json"
            atomic_write_text(ckpt_path, _serialize_asr_result(event.result), self._paths.tmp)
            relpath = ckpt_path.relative_to(self._paths.root).as_posix()

            duration_ms = event.result.duration_ms if event.result.duration_ms > 0 else None
            self._job_repo.update_track_asr(
                event.track_id,
                "completed",
                relpath,
                duration_ms=duration_ms,
            )

            if event.result.language_detected and not job.language_detected:
                self._job_repo.update_job_status(
                    event.job_id,
                    job.status,
                    language_detected=event.result.language_detected,
                )

            refreshed = self._job_repo.get_job(event.job_id)
            assert refreshed is not None

            if all(track.asr_status == "completed" for track in refreshed.tracks):
                allowed_devices = self._resolve_job_devices(refreshed.effective_config)
                wav_paths = {
                    t.id: str(self._paths.job_tmp_dir(refreshed.id) / f"{t.id}.wav")
                    for t in refreshed.tracks
                }
                self._proceed_post_asr_locked(refreshed, wav_paths, allowed_devices)

        with self._state_lock:
            self._dispatch_pending_locked()

    def _proceed_post_asr_locked(
        self,
        job: JobRecord,
        wav_paths: dict[str, str],
        allowed_devices: list[str],
    ) -> None:
        diar_cfg = job.effective_config.get("diarization", {})
        if job.mode == "single" and bool(diar_cfg.get("enabled", False)):
            if not allowed_devices:
                self._finalize_merge_and_export(
                    job,
                    turns=None,
                    diarization_failed=True,
                    error_code="NO_COMPATIBLE_DEVICE",
                    error_detail="No compatible compute device available for diarization",
                )
                return

            self._job_repo.update_job_status(job.id, "diarizing", diarization_status="queued")
            self._event_bus.publish(
                "job.stage",
                job_id=job.id,
                payload={"status": "diarizing", "stage": "diarizing"},
            )
            track_id = job.tracks[0].id
            with self._state_lock:
                if "cpu" in allowed_devices:
                    self._ensure_cpu_worker_locked()
                self._pending_tasks.append(
                    _PendingTask(
                        job_id=job.id,
                        track_id=None,
                        kind="diarize",
                        allowed_devices=allowed_devices,
                        audio_path=wav_paths[track_id],
                    )
                )
                self._dispatch_pending_locked()
            return

        self._finalize_merge_and_export(job, turns=None, diarization_failed=False)

    def _on_diarize_done(self, event: WorkerDiarizeDoneEvent) -> None:
        self._mark_worker_idle(event.device_id, event.job_id, None)

        with self._db_lock:
            job = self._job_repo.get_job(event.job_id)
            if job is None:
                with self._state_lock:
                    self._dispatch_pending_locked()
                return

            if job.status in {"cancelling", "cancelled"} or self._is_cancelled(event.job_id):
                self._on_job_cancelled_cleanup(event.job_id)
                with self._state_lock:
                    self._dispatch_pending_locked()
                return

            self._finalize_merge_and_export(
                job,
                turns=event.turns,
                diarization_failed=False,
                diarization_completed=True,
            )

        with self._state_lock:
            self._dispatch_pending_locked()

    def _on_worker_error(self, event: WorkerErrorEvent) -> None:
        self._mark_worker_idle(event.device_id, event.job_id, event.track_id)

        with self._db_lock:
            job = self._job_repo.get_job(event.job_id)
            if job is None:
                with self._state_lock:
                    self._dispatch_pending_locked()
                return

            if (
                job.status in {"cancelling", "cancelled"}
                or self._is_cancelled(event.job_id)
                or event.error_code == "JOB_CANCELLED"
            ):
                with self._state_lock:
                    self._remove_pending_tasks_locked(event.job_id)
                self._on_job_cancelled_cleanup(event.job_id)
                with self._state_lock:
                    self._dispatch_pending_locked()
                return

            is_diarizing_error = job.status == "diarizing" or event.track_id is None
            log_ref = self._format_log_ref(event.log_path)
            effective_detail = (
                f"{event.detail} (log: {log_ref})" if log_ref and event.detail else (event.detail or (f"See log: {log_ref}" if log_ref else ""))
            )
            if is_diarizing_error and all(t.asr_status == "completed" for t in job.tracks):
                logger.warning(
                    "Job %s diarization failed on %s (%s): %s | log=%s",
                    event.job_id,
                    event.device_id,
                    event.error_code,
                    event.detail,
                    log_ref or "-",
                )
                self._finalize_merge_and_export(
                    job,
                    turns=None,
                    diarization_failed=True,
                    error_code=event.error_code,
                    error_params=event.error_params,
                    error_detail=effective_detail,
                )
            else:
                logger.error(
                    "Job %s failed on %s track=%s (%s): %s | log=%s",
                    event.job_id,
                    event.device_id,
                    event.track_id or "-",
                    event.error_code,
                    event.detail,
                    log_ref or "-",
                )
                with self._state_lock:
                    self._remove_pending_tasks_locked(event.job_id)
                    self._cancel_flag_dict[event.job_id] = True
                    sibling_busy = any(
                        w.busy and w.current_job_id == event.job_id for w in self._workers.values()
                    )

                self._job_repo.update_job_status(
                    event.job_id,
                    "failed",
                    diarization_status="failed" if is_diarizing_error else None,
                    error_code=event.error_code,
                    error_params=event.error_params,
                    error_detail=effective_detail,
                )
                if not sibling_busy:
                    self._cleanup_job_tmp(event.job_id)
                self._event_bus.publish(
                    "job.error",
                    job_id=event.job_id,
                    payload={
                        "error_code": event.error_code,
                        "error_params": event.error_params,
                        "detail": effective_detail,
                        "log_path": log_ref,
                    },
                )

        with self._state_lock:
            self._dispatch_pending_locked()

    # ------------------------------------------------------------------
    # Task scheduling
    # ------------------------------------------------------------------

    def _dispatch_pending_locked(self) -> None:
        if self._stop_event.is_set() or not self._pending_tasks:
            return

        remaining: deque[_PendingTask] = deque()
        while self._pending_tasks:
            task = self._pending_tasks.popleft()
            if self._is_cancelled(task.job_id):
                continue

            chosen_worker: _WorkerHandle | None = None
            for dev_id in task.allowed_devices:
                candidate = self._workers.get(dev_id)
                if candidate is not None and not candidate.busy:
                    chosen_worker = candidate
                    break

            if chosen_worker is None:
                remaining.append(task)
                continue

            with self._db_lock:
                job = self._job_repo.get_job(task.job_id)
            if job is None or job.status in {"cancelling", "cancelled", "failed"}:
                continue

            chosen_worker.busy = True
            chosen_worker.current_job_id = task.job_id
            chosen_worker.current_track_id = task.track_id
            chosen_worker.current_kind = task.kind

            worker_task = self._build_worker_task(job, task, chosen_worker.device_id)
            if task.kind == "diarize":
                with self._db_lock:
                    self._job_repo.update_job_status(
                        job.id,
                        "diarizing",
                        diarization_status="running",
                    )
            chosen_worker.task_q.put(worker_task)

        self._pending_tasks = remaining

    def _build_worker_task(
        self,
        job: JobRecord,
        task: _PendingTask,
        device_id: str,
    ) -> WorkerTranscribeTask | WorkerDiarizeTask:
        cfg = job.effective_config
        if task.kind == "transcribe":
            asr_cfg = cfg.get("asr", {})
            hw_cfg = cfg.get("hardware", {})
            compute_type = (
                str(hw_cfg.get("cpu_compute_type", "int8"))
                if device_id == "cpu"
                else str(asr_cfg.get("compute_type", "float16"))
            )
            cpu_threads = int(hw_cfg.get("cpu_threads", 0) or self._settings.cpu_threads)
            assert task.track_id is not None
            assert task.asr_context is not None
            return WorkerTranscribeTask(
                job_id=job.id,
                track_id=task.track_id,
                audio_path=task.audio_path,
                model_name=str(asr_cfg.get("model", "large-v3-turbo")),
                compute_type=compute_type,
                models_dir=str(self._paths.models),
                context=task.asr_context,
                cpu_threads=cpu_threads,
                hf_endpoint=self._settings.hf_endpoint,
                log_path=str(self._paths.job_logs_dir(job.id) / f"asr-{task.track_id}.log"),
                log_level=str(self._settings.log_level),
            )

        diar_cfg = cfg.get("diarization", {})
        model_id = str(diar_cfg.get("model_id") or "pyannote/speaker-diarization-3.1")
        return WorkerDiarizeTask(
            job_id=job.id,
            audio_path=task.audio_path,
            model_id=model_id,
            models_dir=str(self._paths.models),
            hf_token=self._settings.hf_token,
            hf_endpoint=self._settings.hf_endpoint,
            min_speakers=diar_cfg.get("min_speakers"),
            max_speakers=diar_cfg.get("max_speakers"),
            log_path=str(self._paths.job_logs_dir(job.id) / "diarize.log"),
            log_level=str(self._settings.log_level),
        )

    def _format_log_ref(self, log_path: str | None) -> str | None:
        if not log_path:
            return None
        path = Path(log_path)
        try:
            return str(path.resolve().relative_to(self._paths.root.resolve()))
        except ValueError:
            return str(path)

    # ------------------------------------------------------------------
    # Merging, exporting, and context resolution
    # ------------------------------------------------------------------

    def _build_asr_context(self, job: JobRecord) -> ASRContext:
        cfg = job.effective_config
        asr_cfg = cfg.get("asr", {})
        ctx_cfg = cfg.get("effective_context") or cfg.get("context") or {}

        initial_prompt = str(ctx_cfg.get("initial_prompt", ""))
        hotwords_raw: list[str] = [
            str(w).strip() for w in ctx_cfg.get("hotwords", []) if str(w).strip()
        ]
        if (glossary_ids := ctx_cfg.get("glossary_ids")) and self._paths.glossaries.exists():
            store = GlossaryStore(self._paths)
            for glossary_id in glossary_ids:
                glossary = store.get(str(glossary_id))
                if glossary is not None:
                    hotwords_raw.extend(glossary.terms)

        deduped_hotwords: list[str] = []
        seen: set[str] = set()
        for word in hotwords_raw:
            cleaned = word.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                deduped_hotwords.append(cleaned)

        language = (
            job.language_requested
            if job.language_requested not in (None, "")
            else asr_cfg.get("language")
        )

        return ASRContext(
            language=language,
            initial_prompt=initial_prompt,
            hotwords=deduped_hotwords,
            vad_filter=bool(asr_cfg.get("vad_filter", True)),
            word_timestamps=bool(asr_cfg.get("word_timestamps", True)),
            beam_size=int(asr_cfg.get("beam_size", 5)),
        )

    def _resolve_replacements(self, cfg: Mapping[str, Any]) -> tuple[dict[str, str], bool]:
        ctx_cfg = cfg.get("effective_context") or cfg.get("context") or {}
        if not ctx_cfg.get("apply_replacements", True):
            return {}, False

        replacements: dict[str, str] = {}
        case_sensitive_flags: list[bool] = []

        if isinstance(ctx_cfg.get("replacements"), Mapping):
            replacements.update({str(k): str(v) for k, v in ctx_cfg["replacements"].items()})
            case_sensitive_flags.append(bool(ctx_cfg.get("case_sensitive", False)))
        else:
            if (replacement_ids := ctx_cfg.get("replacement_ids")) and self._paths.replacements.exists():
                store = ReplacementStore(self._paths)
                for dict_id in replacement_ids:
                    rdict = store.get(str(dict_id))
                    if rdict is not None:
                        replacements.update(rdict.rules)
                        case_sensitive_flags.append(rdict.case_sensitive)
            if isinstance(ctx_cfg.get("inline_replacements"), Mapping):
                replacements.update(
                    {str(k): str(v) for k, v in ctx_cfg["inline_replacements"].items()}
                )

        return replacements, any(case_sensitive_flags)

    def _load_track_checkpoint(self, job_id: str, track_id: str, relpath: str | None) -> ASRResult:
        if relpath:
            ckpt_path = Path(relpath)
            if not ckpt_path.is_absolute():
                ckpt_path = self._paths.root / ckpt_path
        else:
            ckpt_path = self._paths.job_checkpoints_dir(job_id) / f"{track_id}.asr.json"
        return _deserialize_asr_result(ckpt_path.read_text(encoding="utf-8"))

    def _finalize_merge_and_export(
        self,
        job: JobRecord,
        turns: list[SpeakerTurn] | None,
        *,
        diarization_failed: bool,
        diarization_completed: bool = False,
        error_code: str | None = None,
        error_params: Mapping[str, Any] | None = None,
        error_detail: str | None = None,
    ) -> None:
        self._job_repo.update_job_status(job.id, "merging")
        self._event_bus.publish(
            "job.stage",
            job_id=job.id,
            payload={"status": "merging", "stage": "merging"},
        )

        loaded = [
            (track, self._load_track_checkpoint(job.id, track.id, track.asr_checkpoint_relpath))
            for track in job.tracks
        ]
        replacements, case_sensitive = self._resolve_replacements(job.effective_config)
        existing_aliases = {sp.speaker_key: sp.display_name for sp in job.speakers}

        if job.mode == "single":
            primary_track, primary_asr = loaded[0]
            merged = TranscriptMerger.merge_single(
                primary_asr,
                turns=turns,
                replacements=replacements,
                case_sensitive=case_sensitive,
            )
            segment_rows = [
                {
                    "seq": item.seq,
                    "track_id": primary_track.id,
                    "start_ms": item.start_ms,
                    "end_ms": item.end_ms,
                    "speaker_key": item.speaker_key or "SPEAKER_00",
                    "raw_text": item.raw_text,
                    "text": item.text,
                    "words": item.words,
                }
                for item in merged
            ]
            seen_speakers: list[str] = []
            for item in merged:
                key = item.speaker_key or "SPEAKER_00"
                if key not in seen_speakers:
                    seen_speakers.append(key)
            if not seen_speakers:
                seen_speakers = [primary_track.speaker_key or "SPEAKER_00"]
            speaker_rows = [
                {
                    "speaker_key": key,
                    "display_name": existing_aliases.get(key, key),
                    "track_id": primary_track.id if len(seen_speakers) == 1 else None,
                }
                for key in seen_speakers
            ]
        else:
            track_tuples = [
                (track.id, track.speaker_key, track.offset_ms, asr_res)
                for track, asr_res in loaded
            ]
            merged = TranscriptMerger.merge_multitrack(
                track_tuples,
                replacements=replacements,
                case_sensitive=case_sensitive,
            )
            track_id_by_speaker = {track.speaker_key: track.id for track, _ in loaded}
            segment_rows = [
                {
                    "seq": item.seq,
                    "track_id": track_id_by_speaker.get(item.speaker_key or ""),
                    "start_ms": item.start_ms,
                    "end_ms": item.end_ms,
                    "speaker_key": item.speaker_key or "",
                    "raw_text": item.raw_text,
                    "text": item.text,
                    "words": item.words,
                }
                for item in merged
            ]
            speaker_rows = [
                {
                    "speaker_key": track.speaker_key,
                    "display_name": existing_aliases.get(track.speaker_key, track.speaker_key),
                    "track_id": track.id,
                }
                for track, _ in loaded
            ]

        saved_segments = self._segment_repo.replace_segments(
            job.id,
            segment_rows,
            preserve_edited=True,
        )
        saved_speakers = self._segment_repo.upsert_speakers(job.id, speaker_rows)

        self._job_repo.update_job_status(job.id, "exporting")
        self._event_bus.publish(
            "job.stage",
            job_id=job.id,
            payload={"status": "exporting", "stage": "exporting"},
        )

        refreshed = self._job_repo.get_job(job.id)
        assert refreshed is not None
        self._export_job_outputs(refreshed, saved_segments, saved_speakers)
        self._maybe_generate_multitrack_mix(refreshed)

        self._cleanup_job_tmp(job.id)
        completed_at = _utc_now()

        if diarization_failed:
            self._job_repo.update_job_status(
                job.id,
                "completed_partial",
                diarization_status="failed",
                error_code=error_code,
                error_params=error_params,
                error_detail=error_detail,
                completed_at=completed_at,
            )
            self._event_bus.publish(
                "job.completed_partial",
                job_id=job.id,
                payload={
                    "status": "completed_partial",
                    "diarization_status": "failed",
                    "error_code": error_code,
                    "error_params": dict(error_params) if error_params else {},
                    "detail": error_detail or "",
                },
            )
            return

        next_diar = "completed" if diarization_completed else refreshed.diarization_status
        self._job_repo.update_job_status(
            job.id,
            "completed",
            diarization_status=next_diar,
            error_code=None,
            error_params=None,
            error_detail=None,
            completed_at=completed_at,
        )
        self._event_bus.publish(
            "job.completed",
            job_id=job.id,
            payload={"status": "completed", "diarization_status": next_diar},
        )

    def _export_job_outputs(
        self,
        job: JobRecord,
        segments: list[SegmentRecord],
        speakers: list[Any],
    ) -> list[Path]:
        output_cfg = job.effective_config.get("output", {})
        explicit_formats = output_cfg.get("formats") if isinstance(output_cfg, dict) else None
        target_formats = list(explicit_formats) if explicit_formats else ["json"]
        return render_job_outputs(
            self._paths,
            job,
            segments,
            speakers,
            formats=target_formats,
            invalidate_cache=True,
        )

    def _maybe_generate_multitrack_mix(self, job: JobRecord) -> Path | None:
        if job.mode != "multitrack" or len(job.tracks) < 2 or not job.media_retained:
            return None
        preprocessing = job.effective_config.get("preprocessing", {})
        if not isinstance(preprocessing, dict) or not preprocessing.get("generate_mix"):
            return None

        mix_path = self._paths.job_media_dir(job.id) / "mix.m4a"
        if mix_path.is_file():
            return mix_path

        track_inputs: list[tuple[Path, int]] = []
        for track in sorted(job.tracks, key=lambda t: t.track_index):
            raw_path = Path(track.media_relpath)
            resolved = raw_path if raw_path.is_absolute() else (self._paths.root / raw_path)
            if not resolved.is_file():
                logger.warning("Skipping mix generation for job %s; missing track %s", job.id, resolved)
                return None
            track_inputs.append((resolved, int(track.offset_ms or 0)))

        normalize_audio = bool(preprocessing.get("normalize_audio", False))
        try:
            mix_multitrack_audio(track_inputs, mix_path, normalize_audio=normalize_audio)
        except MediaInvalidError as exc:
            logger.warning("Failed generating multitrack mix for job %s: %s", job.id, exc)
            return None
        return mix_path
