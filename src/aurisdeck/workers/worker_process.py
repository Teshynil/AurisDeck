"""Isolated worker loop executed in a spawned process with single-GPU visibility."""

from __future__ import annotations

import logging
import os
import queue
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aurisdeck.backends.errors import AurisDeckBackendError, JobCancelledError
from aurisdeck.logging_setup import configure_worker_logging, task_file_handler
from aurisdeck.workers.events import (
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

logger = logging.getLogger(__name__)


def default_backend_factory() -> tuple[Any, Any]:
    """Lazily import and instantiate default ASR and diarization backends."""
    from aurisdeck.backends.faster_whisper_backend import FasterWhisperBackend
    from aurisdeck.backends.pyannote_backend import PyannoteDiarizerBackend

    return FasterWhisperBackend(), PyannoteDiarizerBackend()


def _configure_worker_process(device_id: str, cuda_visible_index: int | None) -> str:
    """Create a dedicated POSIX session and lock CUDA_VISIBLE_DEVICES before ML imports."""
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass

    if device_id == "cpu" or cuda_visible_index is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_index)
    return "cuda"


def _is_cancelled(cancel_flag_dict: Any, job_id: str) -> bool:
    """Return True if cancellation has been requested for the given job_id."""
    return bool(cancel_flag_dict and cancel_flag_dict.get(job_id, False))


def _emit_error(
    event_q: Any,
    device_id: str,
    job_id: str,
    track_id: str | None,
    exc: BaseException,
    *,
    log_path: str | None = None,
) -> None:
    """Translate a backend or unexpected exception into a WorkerErrorEvent."""
    if isinstance(exc, AurisDeckBackendError):
        event_q.put(
            WorkerErrorEvent(
                device_id=device_id,
                job_id=job_id,
                track_id=track_id,
                error_code=exc.error_code,
                error_params=dict(exc.error_params),
                detail=exc.detail,
                log_path=log_path,
            )
        )
        return

    event_q.put(
        WorkerErrorEvent(
            device_id=device_id,
            job_id=job_id,
            track_id=track_id,
            error_code="INTERNAL_ERROR",
            error_params={},
            detail=str(exc),
            log_path=log_path,
        )
    )


def run_worker_loop(
    device_id: str,
    cuda_visible_index: int | None,
    task_q: Any,
    event_q: Any,
    cancel_flag_dict: Any,
    idle_unload_seconds: float = 120.0,
    backend_factory: Callable[[], tuple[Any, Any]] | None = None,
    worker_log_path: str | None = None,
    log_level: str = "INFO",
    log_max_bytes: int = 10 * 1024 * 1024,
    log_backup_count: int = 5,
) -> None:
    """Run the isolated worker loop processing ASR, diarization, and unload commands."""
    target_device = _configure_worker_process(device_id, cuda_visible_index)
    configure_worker_logging(
        worker_log_path,
        level=log_level,
        max_bytes=log_max_bytes,
        backup_count=log_backup_count,
    )
    logger.info(
        "Worker started device_id=%s target_device=%s cuda_visible=%s",
        device_id,
        target_device,
        cuda_visible_index,
    )

    factory = backend_factory or default_backend_factory
    asr_backend, diarizer_backend = factory()

    asr_loaded = False
    diarizer_loaded = False
    last_activity = time.monotonic()

    def unload_asr() -> None:
        nonlocal asr_loaded
        if asr_loaded:
            asr_backend.unload()
            asr_loaded = False

    def unload_diarizer() -> None:
        nonlocal diarizer_loaded
        if diarizer_loaded:
            diarizer_backend.unload()
            diarizer_loaded = False

    def unload_all() -> None:
        unload_asr()
        unload_diarizer()

    while True:
        if (asr_loaded or diarizer_loaded) and idle_unload_seconds > 0:
            poll_timeout = min(0.2, max(0.01, idle_unload_seconds / 2))
        else:
            poll_timeout = 0.5

        try:
            task = task_q.get(timeout=poll_timeout)
        except queue.Empty:
            if (
                (asr_loaded or diarizer_loaded)
                and idle_unload_seconds > 0
                and (time.monotonic() - last_activity >= idle_unload_seconds)
            ):
                unload_all()
                event_q.put(WorkerUnloadedEvent(device_id=device_id, reason="idle_timeout"))
            continue

        if isinstance(task, WorkerShutdownCommand):
            unload_all()
            break

        if isinstance(task, WorkerUnloadCommand):
            unload_all()
            event_q.put(WorkerUnloadedEvent(device_id=device_id, reason="manual"))
            continue

        if isinstance(task, WorkerTranscribeTask):
            if _is_cancelled(cancel_flag_dict, task.job_id):
                _emit_error(
                    event_q,
                    device_id,
                    task.job_id,
                    task.track_id,
                    JobCancelledError(detail="Job cancelled"),
                )
                continue

            unload_diarizer()
            with task_file_handler(task.log_path, level=task.log_level):
                logger.info(
                    "Starting ASR task job=%s track=%s device=%s model=%s compute_type=%s",
                    task.job_id,
                    task.track_id,
                    device_id,
                    task.model_name,
                    task.compute_type,
                )
                try:
                    asr_backend.ensure_loaded(
                        model_name=task.model_name,
                        device=target_device,
                        compute_type=task.compute_type,
                        models_dir=task.models_dir,
                        cpu_threads=task.cpu_threads,
                        hf_endpoint=task.hf_endpoint,
                        status_cb=lambda st, jid=task.job_id, tid=task.track_id: event_q.put(
                            WorkerStageEvent(device_id, jid, tid, st)
                        ),
                    )
                    asr_loaded = True
                    event_q.put(WorkerStageEvent(device_id, task.job_id, task.track_id, "transcribing"))
                    result = asr_backend.transcribe(
                        Path(task.audio_path),
                        task.context,
                        cancel_check=lambda jid=task.job_id: _is_cancelled(cancel_flag_dict, jid),
                        progress_cb=lambda p, t, r, jid=task.job_id, tid=task.track_id: event_q.put(
                            WorkerProgressEvent(device_id, jid, tid, p, t, r)
                        ),
                    )
                    logger.info(
                        "Completed ASR task job=%s track=%s segments=%d duration_ms=%d",
                        task.job_id,
                        task.track_id,
                        len(result.segments),
                        result.duration_ms,
                    )
                    event_q.put(WorkerAsrDoneEvent(device_id, task.job_id, task.track_id, result))
                except Exception as exc:
                    logger.exception(
                        "ASR task failed on %s for job=%s track=%s",
                        device_id,
                        task.job_id,
                        task.track_id,
                    )
                    _emit_error(
                        event_q,
                        device_id,
                        task.job_id,
                        task.track_id,
                        exc,
                        log_path=task.log_path,
                    )
                finally:
                    last_activity = time.monotonic()
            continue

        if isinstance(task, WorkerDiarizeTask):
            if _is_cancelled(cancel_flag_dict, task.job_id):
                _emit_error(
                    event_q,
                    device_id,
                    task.job_id,
                    None,
                    JobCancelledError(detail="Job cancelled"),
                )
                continue

            unload_asr()
            with task_file_handler(task.log_path, level=task.log_level):
                logger.info(
                    "Starting diarize task job=%s device=%s model=%s",
                    task.job_id,
                    device_id,
                    task.model_id,
                )
                try:
                    diarizer_backend.ensure_loaded(
                        model_id=task.model_id,
                        device=target_device,
                        models_dir=task.models_dir,
                        hf_token=task.hf_token,
                        hf_endpoint=task.hf_endpoint,
                    )
                    diarizer_loaded = True
                    if _is_cancelled(cancel_flag_dict, task.job_id):
                        raise JobCancelledError(detail="Job cancelled")
                    event_q.put(WorkerStageEvent(device_id, task.job_id, None, "diarizing"))
                    turns = diarizer_backend.diarize(
                        Path(task.audio_path),
                        min_speakers=task.min_speakers,
                        max_speakers=task.max_speakers,
                    )
                    if _is_cancelled(cancel_flag_dict, task.job_id):
                        raise JobCancelledError(detail="Job cancelled")
                    logger.info(
                        "Completed diarize task job=%s turns=%d",
                        task.job_id,
                        len(turns),
                    )
                    event_q.put(WorkerDiarizeDoneEvent(device_id, task.job_id, turns))
                except Exception as exc:
                    logger.exception(
                        "Diarize task failed on %s for job=%s",
                        device_id,
                        task.job_id,
                    )
                    _emit_error(
                        event_q,
                        device_id,
                        task.job_id,
                        None,
                        exc,
                        log_path=task.log_path,
                    )
                finally:
                    last_activity = time.monotonic()
            continue

        logger.warning("Ignoring unknown worker task on %s: %r", device_id, task)
