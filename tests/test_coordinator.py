"""End-to-end tests for the Single-Writer Coordinator with spawn worker processes."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from aurisdeck.backends.base import ASRContext, ASRResult, ASRSegment, ASRWord, SpeakerTurn
from aurisdeck.backends.errors import JobCancelledError, MissingCredentialsError
from aurisdeck.config import Settings
from aurisdeck.db import run_migrations
from aurisdeck.db.repositories import JobRepo, SegmentRepo
from aurisdeck.media.ffmpeg import MediaInvalidError
from aurisdeck.storage.paths import DataPaths
from aurisdeck.workers.coordinator import Coordinator
from aurisdeck.workers.devices import DeviceInfo
from aurisdeck.workers.events import EventBus


def _append_log(log_file: Path | None, entry: dict[str, Any]) -> None:
    if log_file is None:
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def _read_log(log_file: Path) -> list[dict[str, Any]]:
    if not log_file.is_file():
        return []
    return [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class FakeAsrBackend:
    """Pickle-friendly fake ASR backend executed inside spawned worker processes."""

    def __init__(
        self,
        *,
        log_file: Path | None = None,
        asr_delay_s: float = 0.0,
        stubborn_marker: Path | None = None,
    ) -> None:
        self._log_file = log_file
        self._asr_delay_s = asr_delay_s
        self._stubborn_marker = stubborn_marker
        self._device = "cpu"

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
        self._device = device
        if status_cb is not None:
            status_cb("loading_model")
        _append_log(
            self._log_file,
            {
                "event": "asr.ensure_loaded",
                "device": device,
                "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "model_name": model_name,
                "compute_type": compute_type,
                "pid": os.getpid(),
            },
        )

    def transcribe(
        self,
        audio_path: Path,
        context: ASRContext,
        cancel_check: Any = None,
        progress_cb: Any = None,
    ) -> ASRResult:
        _append_log(
            self._log_file,
            {
                "event": "asr.transcribe",
                "device": self._device,
                "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "audio_name": audio_path.name,
                "pid": os.getpid(),
            },
        )
        if progress_cb is not None:
            progress_cb(500, 1000, 2.0)

        if self._stubborn_marker is not None and self._stubborn_marker.exists():
            # Simulate a C++/CUDA call stuck ignoring Python cancel checks.
            time.sleep(self._asr_delay_s)
        elif self._asr_delay_s > 0:
            deadline = time.monotonic() + self._asr_delay_s
            while time.monotonic() < deadline:
                if cancel_check is not None and cancel_check():
                    raise JobCancelledError(detail="cancelled during fake transcribe")
                time.sleep(0.05)

        if cancel_check is not None and cancel_check():
            raise JobCancelledError(detail="cancelled")

        stem = audio_path.stem
        if stem == "trk-b":
            return ASRResult(
                language_detected=context.language or "es",
                duration_ms=1000,
                segments=[
                    ASRSegment(
                        start_ms=100,
                        end_ms=700,
                        text="Segunda pista",
                        words=[
                            ASRWord(100, 350, "Segunda"),
                            ASRWord(360, 700, "pista"),
                        ],
                    )
                ],
            )

        if stem == "trk-a":
            return ASRResult(
                language_detected=context.language or "es",
                duration_ms=1000,
                segments=[
                    ASRSegment(
                        start_ms=200,
                        end_ms=900,
                        text="Primera pista",
                        words=[
                            ASRWord(200, 500, "Primera"),
                            ASRWord(510, 900, "pista"),
                        ],
                    )
                ],
            )

        return ASRResult(
            language_detected=context.language or "es",
            duration_ms=2000,
            segments=[
                ASRSegment(
                    start_ms=0,
                    end_ms=900,
                    text="Hola desde githab",
                    words=[
                        ASRWord(0, 250, "Hola"),
                        ASRWord(260, 500, "desde"),
                        ASRWord(510, 900, "githab"),
                    ],
                ),
                ASRSegment(
                    start_ms=1000,
                    end_ms=1900,
                    text="Todo funciona bien",
                    words=[
                        ASRWord(1000, 1250, "Todo"),
                        ASRWord(1260, 1600, "funciona"),
                        ASRWord(1610, 1900, "bien"),
                    ],
                ),
            ],
        )

    def unload(self) -> None:
        _append_log(self._log_file, {"event": "asr.unload", "pid": os.getpid()})


class FakeDiarizerBackend:
    """Pickle-friendly fake Diarizer backend executed inside spawned worker processes."""

    def __init__(
        self,
        *,
        log_file: Path | None = None,
        require_hf_token: bool = False,
    ) -> None:
        self._log_file = log_file
        self._require_hf_token = require_hf_token

    def ensure_loaded(
        self,
        model_id: str,
        device: str,
        models_dir: Path | str,
        hf_token: str | None = None,
        hf_endpoint: str | None = None,
    ) -> None:
        _append_log(
            self._log_file,
            {
                "event": "diarizer.ensure_loaded",
                "device": device,
                "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "has_token": bool(hf_token),
                "pid": os.getpid(),
            },
        )
        if self._require_hf_token and not hf_token:
            raise MissingCredentialsError(detail="HF_TOKEN is required for test diarizer")

    def diarize(
        self,
        audio_path: Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[SpeakerTurn]:
        _append_log(
            self._log_file,
            {
                "event": "diarizer.diarize",
                "audio_name": audio_path.name,
                "pid": os.getpid(),
            },
        )
        return [
            SpeakerTurn(start_ms=0, end_ms=950, speaker_key="SPEAKER_00"),
            SpeakerTurn(start_ms=950, end_ms=2000, speaker_key="SPEAKER_01"),
        ]

    def unload(self) -> None:
        _append_log(self._log_file, {"event": "diarizer.unload", "pid": os.getpid()})


@dataclass(slots=True)
class FakeBackendFactory:
    """Top-level pickleable factory passed to Coordinator for spawn worker processes."""

    log_file: Path | None = None
    asr_delay_s: float = 0.0
    require_hf_token: bool = False
    stubborn_marker: Path | None = None

    def __call__(self) -> tuple[FakeAsrBackend, FakeDiarizerBackend]:
        return (
            FakeAsrBackend(
                log_file=self.log_file,
                asr_delay_s=self.asr_delay_s,
                stubborn_marker=self.stubborn_marker,
            ),
            FakeDiarizerBackend(
                log_file=self.log_file,
                require_hf_token=self.require_hf_token,
            ),
        )


def _fake_extract_wav_16k(
    src: Path | str,
    dst_wav: Path | str,
    normalize_audio: bool = False,
) -> None:
    dst = Path(dst_wav)
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_path = Path(src)
    if not src_path.is_file():
        raise MediaInvalidError(detail=f"Missing source media: {src_path}")
    dst.write_bytes(src_path.read_bytes())


def _failing_extract_wav_16k(
    src: Path | str,
    dst_wav: Path | str,
    normalize_audio: bool = False,
) -> None:
    raise MediaInvalidError(detail="Corrupt or unsupported media stream")


def _wait_for_status(
    job_repo: JobRepo,
    job_id: str,
    target_statuses: set[str],
    timeout_s: float = 10.0,
) -> Any:
    deadline = time.monotonic() + timeout_s
    last_status = None
    while time.monotonic() < deadline:
        job = job_repo.get_job(job_id)
        if job is not None:
            last_status = job.status
            if job.status in target_statuses:
                return job
        time.sleep(0.05)
    raise AssertionError(
        f"Job {job_id} did not reach {target_statuses} within {timeout_s}s (last={last_status!r})"
    )


def _drain_bus_queue(q: asyncio.Queue[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    while True:
        try:
            items.append(q.get_nowait())
        except asyncio.QueueEmpty:
            return items


def _setup_env(tmp_path: Path) -> tuple[Settings, DataPaths, JobRepo, SegmentRepo, EventBus]:
    data_root = tmp_path / "data"
    paths = DataPaths(data_root)
    paths.ensure_dirs()
    run_migrations(paths.db_path)
    settings = Settings(data_dir=data_root, model_idle_unload_seconds=60)
    job_repo = JobRepo(paths.db_path)
    segment_repo = SegmentRepo(paths.db_path)
    event_bus = EventBus()
    return settings, paths, job_repo, segment_repo, event_bus


def test_single_flow_completes_and_cleans_tmp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aurisdeck.workers.coordinator.extract_wav_16k",
        _fake_extract_wav_16k,
    )
    settings, paths, job_repo, segment_repo, event_bus = _setup_env(tmp_path)
    bus_q = event_bus.subscribe()

    log_file = tmp_path / "worker.jsonl"
    devices = [
        DeviceInfo(id="cuda:0", kind="cuda", index=0, name="Fake GPU 0"),
        DeviceInfo(id="cpu", kind="cpu", index=None, name="CPU"),
    ]

    job = job_repo.create_job(
        title="Session 1.0",
        mode="single",
        effective_config={
            "asr": {"model": "large-v3-turbo", "language": "es", "compute_type": "float16"},
            "diarization": {"enabled": False},
            "context": {
                "apply_replacements": True,
                "inline_replacements": {"githab": "GitHub"},
            },
            "hardware": {"gpus": "auto", "cpu_fallback": True},
            "output": {
                "formats": ["vtt", "srt", "md", "json"],
                "subtitles": {"voice_style": "v_tag"},
            },
        },
        job_id="job-single-1",
    )
    media_file = paths.job_media_dir(job.id) / "trk-1_audio.wav"
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"RIFFfake")
    job_repo.add_track(
        job.id,
        0,
        "audio.wav",
        media_file.relative_to(paths.root).as_posix(),
        "SPEAKER_00",
        track_id="trk-1",
    )

    coord = Coordinator(
        settings=settings,
        paths=paths,
        job_repo=job_repo,
        segment_repo=segment_repo,
        event_bus=event_bus,
        devices=devices,
        backend_factory=FakeBackendFactory(log_file=log_file),
    )
    coord.start()
    try:
        coord.enqueue_job(job.id)
        done = _wait_for_status(job_repo, job.id, {"completed", "failed"})
        assert done.status == "completed"
        assert done.diarization_status == "none"
        assert done.language_detected == "es"
        assert done.completed_at is not None

        ckpt_file = paths.job_checkpoints_dir(job.id) / "trk-1.asr.json"
        assert ckpt_file.is_file()
        ckpt_data = json.loads(ckpt_file.read_text(encoding="utf-8"))
        assert ckpt_data["segments"][0]["text"] == "Hola desde githab"

        out_dir = paths.job_outputs_dir(job.id)
        vtt_path = out_dir / "Session 1.0.vtt"
        assert vtt_path.is_file()
        assert (out_dir / "Session 1.0.srt").is_file()
        assert (out_dir / "Session 1.0.md").is_file()
        assert (out_dir / "Session 1.0.json").is_file()

        vtt_text = vtt_path.read_text(encoding="utf-8")
        assert "WEBVTT" in vtt_text
        assert "Hola desde GitHub" in vtt_text

        assert not paths.job_tmp_dir(job.id).exists()

        events = _drain_bus_queue(bus_q)
        event_types = [ev["type"] for ev in events]
        assert "job.completed" in event_types

        logs = _read_log(log_file)
        asr_runs = [entry for entry in logs if entry["event"] == "asr.transcribe"]
        assert len(asr_runs) == 1
        assert asr_runs[0]["device"] == "cuda"
        assert asr_runs[0]["cuda_visible"] == "0"
    finally:
        coord.stop()
        event_bus.unsubscribe(bus_q)


def test_multitrack_dual_gpu_no_cpu_competition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aurisdeck.workers.coordinator.extract_wav_16k",
        _fake_extract_wav_16k,
    )
    settings, paths, job_repo, segment_repo, event_bus = _setup_env(tmp_path)
    log_file = tmp_path / "dual_gpu.jsonl"

    devices = [
        DeviceInfo(id="cuda:0", kind="cuda", index=0, name="Fake GPU 0"),
        DeviceInfo(id="cuda:1", kind="cuda", index=1, name="Fake GPU 1"),
        DeviceInfo(id="cpu", kind="cpu", index=None, name="CPU"),
    ]

    job = job_repo.create_job(
        title="Multitrack Session",
        mode="multitrack",
        effective_config={
            "asr": {"model": "large-v3-turbo", "language": "es", "compute_type": "float16"},
            "diarization": {"enabled": False},
            "hardware": {"gpus": "auto", "cpu_fallback": True},
            "output": {"formats": ["vtt", "md"], "subtitles": {"voice_style": "v_tag"}},
        },
        job_id="job-multi-1",
    )

    for idx, (tid, fname, speaker, offset) in enumerate(
        [
            ("trk-a", "alice.ogg", "Alice", 0),
            ("trk-b", "bob.ogg", "Bob", 500),
        ]
    ):
        media_file = paths.job_media_dir(job.id) / f"{tid}_{fname}"
        media_file.parent.mkdir(parents=True, exist_ok=True)
        media_file.write_bytes(b"OGGfake")
        job_repo.add_track(
            job.id,
            idx,
            fname,
            media_file.relative_to(paths.root).as_posix(),
            speaker,
            track_id=tid,
            offset_ms=offset,
        )

    coord = Coordinator(
        settings=settings,
        paths=paths,
        job_repo=job_repo,
        segment_repo=segment_repo,
        event_bus=event_bus,
        devices=devices,
        backend_factory=FakeBackendFactory(log_file=log_file, asr_delay_s=0.3),
    )
    coord.start()
    try:
        assert set(coord._workers.keys()) == {"cuda:0", "cuda:1"}
        coord.enqueue_job(job.id)
        done = _wait_for_status(job_repo, job.id, {"completed", "failed"})
        assert done.status == "completed"
        assert "cpu" not in coord._workers

        logs = _read_log(log_file)
        asr_runs = [entry for entry in logs if entry["event"] == "asr.transcribe"]
        assert len(asr_runs) == 2
        assert {entry["device"] for entry in asr_runs} == {"cuda"}
        assert {entry["cuda_visible"] for entry in asr_runs} == {"0", "1"}

        segs = segment_repo.list_segments(job.id)
        assert len(segs) == 2
        assert segs[0].speaker_key == "Alice"
        assert segs[0].start_ms == 200
        assert segs[1].speaker_key == "Bob"
        assert segs[1].start_ms == 600  # 100ms + 500ms offset

        vtt_text = (paths.job_outputs_dir(job.id) / "Multitrack Session.vtt").read_text(
            encoding="utf-8"
        )
        assert "<v Alice>Primera pista</v>" in vtt_text
        assert "<v Bob>Segunda pista</v>" in vtt_text
        assert not paths.job_tmp_dir(job.id).exists()
    finally:
        coord.stop()


def test_diarization_failure_partial_then_trigger_diarize_without_rerunning_asr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aurisdeck.workers.coordinator.extract_wav_16k",
        _fake_extract_wav_16k,
    )
    settings, paths, job_repo, segment_repo, event_bus = _setup_env(tmp_path)
    settings.hf_token = None
    log_file = tmp_path / "partial_diarize.jsonl"

    devices = [DeviceInfo(id="cpu", kind="cpu", index=None, name="CPU")]
    job = job_repo.create_job(
        title="Podcast",
        mode="single",
        effective_config={
            "asr": {"model": "tiny", "language": "es", "compute_type": "int8"},
            "diarization": {"enabled": True},
            "hardware": {"gpus": "auto", "cpu_fallback": True},
            "output": {"formats": ["vtt"], "subtitles": {"voice_style": "v_tag"}},
        },
        job_id="job-partial-1",
    )
    media_file = paths.job_media_dir(job.id) / "trk-1_podcast.wav"
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"RIFFfake")
    job_repo.add_track(
        job.id,
        0,
        "podcast.wav",
        media_file.relative_to(paths.root).as_posix(),
        "SPEAKER_00",
        track_id="trk-1",
    )

    coord = Coordinator(
        settings=settings,
        paths=paths,
        job_repo=job_repo,
        segment_repo=segment_repo,
        event_bus=event_bus,
        devices=devices,
        backend_factory=FakeBackendFactory(log_file=log_file, require_hf_token=True),
    )
    coord.start()
    try:
        coord.enqueue_job(job.id)
        partial = _wait_for_status(job_repo, job.id, {"completed_partial", "failed"})
        assert partial.status == "completed_partial"
        assert partial.diarization_status == "failed"
        assert partial.error_code == "DIARIZATION_HF_TOKEN_MISSING"
        assert partial.tracks[0].asr_status == "completed"

        ckpt_file = paths.job_checkpoints_dir(job.id) / "trk-1.asr.json"
        assert ckpt_file.is_file()
        vtt_file = paths.job_outputs_dir(job.id) / "Podcast.vtt"
        assert vtt_file.is_file()
        assert "Hola desde githab" in vtt_file.read_text(encoding="utf-8")
        assert not paths.job_tmp_dir(job.id).exists()

        # Now configure HF_TOKEN and trigger diarization post-hoc.
        settings.hf_token = "hf_test_valid_token"
        coord.trigger_diarize(job.id)
        completed = _wait_for_status(job_repo, job.id, {"completed", "completed_partial", "failed"})
        assert completed.status == "completed"
        assert completed.diarization_status == "completed"
        assert completed.error_code is None

        logs = _read_log(log_file)
        asr_runs = [entry for entry in logs if entry["event"] == "asr.transcribe"]
        diarize_runs = [entry for entry in logs if entry["event"] == "diarizer.diarize"]
        assert len(asr_runs) == 1
        assert len(diarize_runs) == 1

        updated_vtt = vtt_file.read_text(encoding="utf-8")
        assert "<v SPEAKER_00>" in updated_vtt
        assert "<v SPEAKER_01>" in updated_vtt
        assert not paths.job_tmp_dir(job.id).exists()
    finally:
        coord.stop()


def test_reexport_job_after_speaker_rename_is_fast_cpu_only(tmp_path: Path) -> None:
    settings, paths, job_repo, segment_repo, event_bus = _setup_env(tmp_path)

    job = job_repo.create_job(
        title="Interview",
        mode="single",
        status="completed",
        diarization_status="completed",
        language_detected="es",
        effective_config={
            "output": {
                "formats": ["vtt", "md"],
                "subtitles": {"voice_style": "v_tag"},
            }
        },
        job_id="job-reexport-1",
    )
    job_repo.add_track(
        job.id,
        0,
        "interview.wav",
        "media/job-reexport-1/trk-1_interview.wav",
        "SPEAKER_00",
        track_id="trk-1",
        asr_status="completed",
    )
    segment_repo.replace_segments(
        job.id,
        [
            {
                "seq": 0,
                "track_id": "trk-1",
                "start_ms": 0,
                "end_ms": 1200,
                "speaker_key": "SPEAKER_00",
                "raw_text": "Buenos días",
                "text": "Buenos días",
                "words": [
                    {"start_ms": 0, "end_ms": 600, "text": "Buenos"},
                    {"start_ms": 610, "end_ms": 1200, "text": "días"},
                ],
            }
        ],
    )
    segment_repo.upsert_speakers(
        job.id,
        [{"speaker_key": "SPEAKER_00", "display_name": "SPEAKER_00", "track_id": "trk-1"}],
    )

    # Note: Coordinator is NOT even started (no worker queue / GPU needed).
    coord = Coordinator(
        settings=settings,
        paths=paths,
        job_repo=job_repo,
        segment_repo=segment_repo,
        event_bus=event_bus,
    )

    segment_repo.rename_speaker(job.id, "SPEAKER_00", "Dra. Elena Vance")

    t0 = time.perf_counter()
    written = coord.reexport_job(job.id)
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0
    assert len(written) == 2
    vtt_path = paths.job_outputs_dir(job.id) / "Interview.vtt"
    md_path = paths.job_outputs_dir(job.id) / "Interview.md"
    assert vtt_path in written and md_path in written
    assert "<v Dra. Elena Vance>Buenos días</v>" in vtt_path.read_text(encoding="utf-8")
    assert "**Dra. Elena Vance:** Buenos días" in md_path.read_text(encoding="utf-8")


def test_cancel_job_kills_stubborn_worker_and_respawns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aurisdeck.workers.coordinator.extract_wav_16k",
        _fake_extract_wav_16k,
    )
    settings, paths, job_repo, segment_repo, event_bus = _setup_env(tmp_path)
    log_file = tmp_path / "cancel.jsonl"
    stubborn_marker = tmp_path / "stubborn.flag"
    stubborn_marker.write_text("1", encoding="utf-8")

    devices = [DeviceInfo(id="cuda:0", kind="cuda", index=0, name="Fake GPU 0")]

    def _make_job(jid: str) -> str:
        job_repo.create_job(
            title=jid,
            mode="single",
            effective_config={
                "asr": {"model": "tiny", "language": "es", "compute_type": "float16"},
                "diarization": {"enabled": False},
                "hardware": {"gpus": "auto"},
                "output": {"formats": ["vtt"]},
            },
            job_id=jid,
        )
        media_file = paths.job_media_dir(jid) / "trk-1_audio.wav"
        media_file.parent.mkdir(parents=True, exist_ok=True)
        media_file.write_bytes(b"RIFFfake")
        job_repo.add_track(
            jid,
            0,
            "audio.wav",
            media_file.relative_to(paths.root).as_posix(),
            "SPEAKER_00",
            track_id=f"{jid}-trk",
        )
        return jid

    j1 = _make_job("job-cancel-1")
    j2 = _make_job("job-after-cancel-2")

    coord = Coordinator(
        settings=settings,
        paths=paths,
        job_repo=job_repo,
        segment_repo=segment_repo,
        event_bus=event_bus,
        devices=devices,
        backend_factory=FakeBackendFactory(
            log_file=log_file,
            asr_delay_s=6.0,
            stubborn_marker=stubborn_marker,
        ),
    )
    coord.start()
    try:
        initial_pid = coord._workers["cuda:0"].proc.pid
        coord.enqueue_job(j1)
        _wait_for_status(job_repo, j1, {"transcribing"})

        # Wait until worker actually enters asr.transcribe.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if any(e["event"] == "asr.transcribe" for e in _read_log(log_file)):
                break
            time.sleep(0.05)

        coord.cancel_job(j1)
        cancelled = job_repo.get_job(j1)
        assert cancelled is not None
        assert cancelled.status == "cancelled"
        assert not paths.job_tmp_dir(j1).exists()

        respawned_pid = coord._workers["cuda:0"].proc.pid
        assert respawned_pid is not None
        assert respawned_pid != initial_pid
        assert coord._workers["cuda:0"].proc.is_alive()

        # Verify respawned worker handles a normal subsequent job cleanly.
        stubborn_marker.unlink(missing_ok=True)
        coord.enqueue_job(j2)
        done2 = _wait_for_status(job_repo, j2, {"completed", "failed"})
        assert done2.status == "completed"
    finally:
        coord.stop()


def test_startup_recovery_and_media_invalid_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aurisdeck.workers.coordinator.extract_wav_16k",
        _failing_extract_wav_16k,
    )
    settings, paths, job_repo, segment_repo, event_bus = _setup_env(tmp_path)

    # Create orphan tmp directory and an interrupted transcribing job before start().
    orphan_tmp = paths.job_tmp_dir("orphan-1")
    orphan_tmp.mkdir(parents=True, exist_ok=True)
    (orphan_tmp / "leftover.wav").write_bytes(b"abc")

    job_repo.create_job(
        title="Zombie",
        mode="single",
        status="transcribing",
        effective_config={},
        job_id="zombie-1",
    )

    bad_job = job_repo.create_job(
        title="Bad Media",
        mode="single",
        effective_config={
            "asr": {"model": "tiny"},
            "hardware": {"gpus": "cpu"},
            "output": {"formats": ["vtt"]},
        },
        job_id="bad-media-1",
    )
    media_file = paths.job_media_dir(bad_job.id) / "trk-1_bad.mp4"
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"corrupt")
    job_repo.add_track(
        bad_job.id,
        0,
        "bad.mp4",
        media_file.relative_to(paths.root).as_posix(),
        "SPEAKER_00",
        track_id="trk-bad",
    )

    coord = Coordinator(
        settings=settings,
        paths=paths,
        job_repo=job_repo,
        segment_repo=segment_repo,
        event_bus=event_bus,
        devices=[DeviceInfo(id="cpu", kind="cpu", index=None, name="CPU")],
        backend_factory=FakeBackendFactory(),
    )
    coord.start()
    try:
        assert not orphan_tmp.exists()
        recovered = job_repo.get_job("zombie-1")
        assert recovered is not None
        assert recovered.status == "failed"
        assert recovered.error_code == "SERVER_INTERRUPTED"

        coord.enqueue_job(bad_job.id)
        failed = _wait_for_status(job_repo, bad_job.id, {"failed", "completed"})
        assert failed.status == "failed"
        assert failed.error_code == "MEDIA_INVALID"
        assert not paths.job_tmp_dir(bad_job.id).exists()
    finally:
        coord.stop()
