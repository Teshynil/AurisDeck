"""Synthetic end-to-end integration tests for AurisDeck HTTP API + Coordinator + ffmpeg."""

from __future__ import annotations

import io
import json
import os
import subprocess
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aurisdeck.api import create_app
from aurisdeck.api.sse import SSEBroker
from aurisdeck.backends.base import (
    ASRCapabilities,
    ASRContext,
    ASRResult,
    ASRSegment,
    ASRWord,
    SpeakerTurn,
)
from aurisdeck.backends.errors import MissingCredentialsError
from aurisdeck.config import Settings
from aurisdeck.workers.devices import DeviceInfo


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


class FakeASRBackend:
    """Pickleable synthetic ASR backend executed inside spawned Coordinator workers."""

    capabilities = ASRCapabilities(
        supports_hotwords=True,
        supports_initial_prompt=True,
        supports_word_timestamps=True,
    )

    def __init__(self, *, log_file: Path | None = None) -> None:
        self._log_file = log_file

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
        if status_cb is not None:
            status_cb("loading_model")
        _append_log(
            self._log_file,
            {
                "event": "asr.ensure_loaded",
                "model_name": model_name,
                "device": device,
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
        assert audio_path.is_file()
        assert audio_path.stat().st_size > 44  # Valid WAV beyond 44-byte RIFF header

        _append_log(
            self._log_file,
            {
                "event": "asr.transcribe",
                "audio_name": audio_path.name,
                "language": context.language,
                "pid": os.getpid(),
            },
        )

        if progress_cb is not None:
            progress_cb(1000, 2000, 4.0)
            progress_cb(2000, 2000, 4.5)

        return ASRResult(
            language_detected=context.language or "es",
            duration_ms=2000,
            segments=[
                ASRSegment(
                    start_ms=100,
                    end_ms=900,
                    text="Hola mundo",
                    words=[
                        ASRWord(100, 450, "Hola"),
                        ASRWord(460, 900, "mundo"),
                    ],
                ),
                ASRSegment(
                    start_ms=1000,
                    end_ms=1850,
                    text="Audio verificado",
                    words=[
                        ASRWord(1000, 1350, "Audio"),
                        ASRWord(1360, 1850, "verificado"),
                    ],
                ),
            ],
        )

    def unload(self) -> None:
        _append_log(self._log_file, {"event": "asr.unload", "pid": os.getpid()})


class FakeDiarizerBackend:
    """Pickleable synthetic diarization backend executed inside spawned workers."""

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
                "model_id": model_id,
                "device": device,
                "has_token": bool(hf_token),
                "pid": os.getpid(),
            },
        )
        if self._require_hf_token and not hf_token:
            raise MissingCredentialsError(detail="HF_TOKEN is required for test diarization")

    def diarize(
        self,
        audio_path: Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[SpeakerTurn]:
        assert audio_path.is_file()
        _append_log(
            self._log_file,
            {
                "event": "diarizer.diarize",
                "audio_name": audio_path.name,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
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
    """Top-level pickleable backend factory wired into the real Coordinator."""

    log_file: Path | None = None
    require_hf_token: bool = False

    def __call__(self) -> tuple[FakeASRBackend, FakeDiarizerBackend]:
        return (
            FakeASRBackend(log_file=self.log_file),
            FakeDiarizerBackend(
                log_file=self.log_file,
                require_hf_token=self.require_hf_token,
            ),
        )


def _generate_sine_wav(dest: Path, *, frequency: int = 440, duration_s: int = 2) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:duration={duration_s}",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )
    return dest


def _make_zip_bytes(entries: dict[str, bytes | str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in entries.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            zf.writestr(name, data)
    return buf.getvalue()


def _build_e2e_app(
    data_dir: Path,
    *,
    hf_token: str | None = None,
    log_file: Path | None = None,
    require_hf_token: bool = False,
) -> FastAPI:
    settings = Settings(
        data_dir=data_dir,
        auth_mode="none",
        hf_token=hf_token,
        model_idle_unload_seconds=60,
    )
    devices = [DeviceInfo(id="cpu", kind="cpu", index=None, name="CPU (4 threads)")]
    app = create_app(settings, devices=devices)
    app.state.coordinator._backend_factory = FakeBackendFactory(
        log_file=log_file,
        require_hf_token=require_hf_token,
    )
    app.state.sse_broker = SSEBroker(app.state.event_bus, heartbeat_seconds=0.05)
    return app


def _poll_job_until(
    client: TestClient,
    job_id: str,
    target_statuses: set[str],
    *,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        resp = client.get(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200
        last_payload = resp.json()
        if last_payload["status"] in target_statuses:
            return last_payload
        time.sleep(0.05)
    raise AssertionError(
        f"Job {job_id} did not reach {target_statuses} within {timeout_s}s "
        f"(last={last_payload})"
    )


def test_e2e_single_upload_sse_speakers_patch_and_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    wav_path = _generate_sine_wav(tmp_path / "inputs" / "meeting.v1.wav", frequency=440, duration_s=2)
    log_file = tmp_path / "worker_single.jsonl"
    app = _build_e2e_app(tmp_path / "data", log_file=log_file)

    captured_events: list[tuple[str, dict[str, Any]]] = []
    sse_ready = threading.Event()
    sse_stop = threading.Event()
    sse_errors: list[BaseException] = []

    with TestClient(app) as client:

        def _consume_sse() -> None:
            try:
                with client.stream("GET", "/api/v1/events") as stream_resp:
                    assert stream_resp.status_code == 200
                    current_event: str | None = None
                    data_lines: list[str] = []
                    for raw_line in stream_resp.iter_lines():
                        if sse_stop.is_set():
                            break
                        line = raw_line.rstrip("\r")
                        if line == ": keepalive":
                            sse_ready.set()
                            continue
                        if line == "":
                            if data_lines:
                                payload = json.loads("\n".join(data_lines))
                                ev_type = str(payload.get("type") or current_event or "message")
                                captured_events.append((ev_type, payload))
                                current_event = None
                                data_lines = []
                                if ev_type == "job.completed":
                                    break
                            continue
                        field_name, sep, value = line.partition(":")
                        if not sep:
                            continue
                        value = value.removeprefix(" ")
                        if field_name == "event":
                            current_event = value
                        elif field_name == "data":
                            data_lines.append(value)
            except BaseException as exc:  # noqa: BLE001
                sse_errors.append(exc)
                sse_ready.set()

        listener = threading.Thread(target=_consume_sse, daemon=True)
        listener.start()
        assert sse_ready.wait(timeout=5.0), "SSE stream did not connect in time"

        with wav_path.open("rb") as audio_handle:
            create_resp = client.post(
                "/api/v1/jobs",
                files=[("files", ("meeting.v1.wav", audio_handle, "audio/wav"))],
                data={
                    "mode": "single",
                    "profile_id": "es-tecnico",
                    "overrides_json": json.dumps(
                        {
                            "diarize": False,
                            "formats": ["vtt", "srt", "md", "json"],
                        }
                    ),
                },
            )
        assert create_resp.status_code == 201, create_resp.text
        created = create_resp.json()
        job_id = created["id"]
        assert created["title"] == "meeting.v1"

        listener.join(timeout=15.0)
        sse_stop.set()
        assert not sse_errors, f"SSE listener failed: {sse_errors}"
        assert not listener.is_alive(), "SSE listener did not finish after job.completed"

        job_events = [ev for ev in captured_events if ev[1].get("job_id") == job_id]
        event_types = [ev_type for ev_type, _ in job_events]
        assert "job.stage" in event_types
        assert "job.progress" in event_types
        assert "job.completed" in event_types

        progress_payloads = [
            payload["payload"] for ev_type, payload in job_events if ev_type == "job.progress"
        ]
        assert any(p.get("processed_ms") == 2000 and p.get("total_ms") == 2000 for p in progress_payloads)

        detail = client.get(f"/api/v1/jobs/{job_id}").json()
        assert detail["status"] == "completed"
        assert detail["available_formats"] == ["vtt", "srt", "md", "json"]

        # Rename speaker and verify instant CPU re-export across all 4 formats.
        patch_resp = client.patch(
            f"/api/v1/jobs/{job_id}/speakers",
            json={"aliases": {"SPEAKER_00": "Elena"}},
        )
        assert patch_resp.status_code == 200
        patch_data = patch_resp.json()
        assert patch_data["speakers"][0]["display_name"] == "Elena"
        assert set(patch_data["available_formats"]) == {"vtt", "srt", "md", "json"}

        vtt_resp = client.get(f"/api/v1/jobs/{job_id}/download/vtt")
        assert vtt_resp.status_code == 200
        assert 'filename="meeting.v1.vtt"' in vtt_resp.headers["content-disposition"]
        assert "WEBVTT" in vtt_resp.text
        assert "<v Elena>Hola mundo</v>" in vtt_resp.text
        assert "<v Elena>Audio verificado</v>" in vtt_resp.text

        srt_resp = client.get(f"/api/v1/jobs/{job_id}/download/srt")
        assert srt_resp.status_code == 200
        assert 'filename="meeting.v1.srt"' in srt_resp.headers["content-disposition"]
        assert "[Elena]: Hola mundo" in srt_resp.text

        md_resp = client.get(f"/api/v1/jobs/{job_id}/download/md")
        assert md_resp.status_code == 200
        assert 'filename="meeting.v1.md"' in md_resp.headers["content-disposition"]
        assert "**Elena:** Hola mundo Audio verificado" in md_resp.text

        json_resp = client.get(f"/api/v1/jobs/{job_id}/download/json")
        assert json_resp.status_code == 200
        assert 'filename="meeting.v1.json"' in json_resp.headers["content-disposition"]
        exported_json = json_resp.json()
        assert exported_json["speakers"]["SPEAKER_00"] == "Elena"
        assert exported_json["turns"][0]["speaker_name"] == "Elena"
        assert len(exported_json["segments"]) == 2


def test_e2e_multitrack_zip_with_offsets_preserves_chronological_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    alice_wav = _generate_sine_wav(tmp_path / "inputs" / "1-alice.wav", frequency=440, duration_s=2)
    bob_wav = _generate_sine_wav(tmp_path / "inputs" / "2-bob.wav", frequency=660, duration_s=2)

    info_txt = "Track 1: Alice\nTrack 2: Bob\n"
    aup_xml = """<?xml version="1.0" standalone="no" ?>
    <project>
      <wavetrack name="1-alice.wav">
        <waveclip offset="1.500" />
      </wavetrack>
      <wavetrack name="2-bob.wav">
        <waveclip offset="0.0" />
      </wavetrack>
    </project>
    """
    archive_bytes = _make_zip_bytes(
        {
            "1-alice.wav": alice_wav.read_bytes(),
            "2-bob.wav": bob_wav.read_bytes(),
            "info.txt": info_txt,
            "session.aup": aup_xml,
        }
    )

    app = _build_e2e_app(tmp_path / "data")
    with TestClient(app) as client:
        inspect_resp = client.post(
            "/api/v1/jobs/inspect-zip",
            files={"file": ("session.zip", archive_bytes, "application/zip")},
        )
        assert inspect_resp.status_code == 200
        inspected = {item["filename"]: item for item in inspect_resp.json()}
        assert inspected["1-alice.wav"] == {
            "filename": "1-alice.wav",
            "speaker_name": "Alice",
            "offset_ms": 1500,
        }
        assert inspected["2-bob.wav"] == {
            "filename": "2-bob.wav",
            "speaker_name": "Bob",
            "offset_ms": 0,
        }

        create_resp = client.post(
            "/api/v1/jobs",
            files=[("files", ("session.zip", archive_bytes, "application/zip"))],
            data={
                "mode": "multitrack",
                "profile_id": "multipista-sync",
                "overrides_json": json.dumps({"formats": ["vtt"]}),
            },
        )
        assert create_resp.status_code == 201, create_resp.text
        created = create_resp.json()
        job_id = created["id"]
        tracks_by_name = {t["original_filename"]: t for t in created["tracks"]}
        assert tracks_by_name["1-alice.wav"]["offset_ms"] == 1500
        assert tracks_by_name["2-bob.wav"]["offset_ms"] == 0

        completed = _poll_job_until(client, job_id, {"completed", "failed"})
        assert completed["status"] == "completed"

        vtt_resp = client.get(f"/api/v1/jobs/{job_id}/download/vtt")
        assert vtt_resp.status_code == 200
        vtt_text = vtt_resp.text

        # Bob starts at 0ms offset (00:00:00.100 -> 00:00:01.850)
        # Alice has a +1500ms offset (00:00:01.600 -> 00:00:03.350)
        bob_first_pos = vtt_text.index("00:00:00.100 --> 00:00:00.900\n<v Bob>Hola mundo</v>")
        bob_second_pos = vtt_text.index("00:00:01.000 --> 00:00:01.850\n<v Bob>Audio verificado</v>")
        alice_first_pos = vtt_text.index("00:00:01.600 --> 00:00:02.400\n<v Alice>Hola mundo</v>")
        alice_second_pos = vtt_text.index(
            "00:00:02.500 --> 00:00:03.350\n<v Alice>Audio verificado</v>"
        )
        assert bob_first_pos < bob_second_pos < alice_first_pos < alice_second_pos


def test_e2e_diarization_failure_partial_then_post_hoc_diarize_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    wav_path = _generate_sine_wav(tmp_path / "inputs" / "podcast.wav", frequency=440, duration_s=2)
    log_file = tmp_path / "worker_diarize.jsonl"
    app = _build_e2e_app(
        tmp_path / "data",
        hf_token=None,
        log_file=log_file,
        require_hf_token=True,
    )

    with TestClient(app) as client:
        with wav_path.open("rb") as audio_handle:
            create_resp = client.post(
                "/api/v1/jobs",
                files=[("files", ("podcast.wav", audio_handle, "audio/wav"))],
                data={
                    "mode": "single",
                    "profile_id": "es-tecnico",
                    "overrides_json": json.dumps({"diarize": True, "formats": ["vtt"]}),
                },
            )
        assert create_resp.status_code == 201, create_resp.text
        job_id = create_resp.json()["id"]

        partial = _poll_job_until(client, job_id, {"completed_partial", "failed"})
        assert partial["status"] == "completed_partial"
        assert partial["diarization_status"] == "failed"
        assert partial["error_code"] == "DIARIZATION_HF_TOKEN_MISSING"

        partial_vtt = client.get(f"/api/v1/jobs/{job_id}/download/vtt")
        assert partial_vtt.status_code == 200
        assert "WEBVTT" in partial_vtt.text
        assert "Hola mundo" in partial_vtt.text

        # Configure HF_TOKEN and trigger post-hoc diarization without re-running ASR.
        app.state.settings.hf_token = "hf_test_valid_token"
        diarize_resp = client.post(
            f"/api/v1/jobs/{job_id}/diarize",
            json={"min_speakers": 2, "max_speakers": 2},
        )
        assert diarize_resp.status_code == 200, diarize_resp.text
        assert diarize_resp.json()["status"] == "diarizing"

        completed = _poll_job_until(client, job_id, {"completed", "completed_partial", "failed"})
        assert completed["status"] == "completed"
        assert completed["diarization_status"] == "completed"
        assert completed["error_code"] is None

        final_vtt = client.get(f"/api/v1/jobs/{job_id}/download/vtt")
        assert final_vtt.status_code == 200
        assert "<v SPEAKER_00>Hola mundo</v>" in final_vtt.text
        assert "<v SPEAKER_01>Audio verificado</v>" in final_vtt.text

        logs = _read_log(log_file)
        asr_runs = [entry for entry in logs if entry["event"] == "asr.transcribe"]
        diarize_runs = [entry for entry in logs if entry["event"] == "diarizer.diarize"]
        assert len(asr_runs) == 1
        assert len(diarize_runs) == 1


def test_e2e_account_settings_locale_en_persists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    data_dir = tmp_path / "data"

    app1 = _build_e2e_app(data_dir)
    with TestClient(app1) as client:
        initial = client.get("/api/v1/me")
        assert initial.status_code == 200
        assert initial.json()["preferences"]["locale"] == "es"

        patched = client.patch("/api/v1/me", json={"locale": "en"})
        assert patched.status_code == 200
        assert patched.json()["preferences"]["locale"] == "en"

        fetched = client.get("/api/v1/me")
        assert fetched.status_code == 200
        assert fetched.json()["preferences"]["locale"] == "en"

    # Re-create the app against the same data_dir to confirm SQLite persistence.
    app2 = _build_e2e_app(data_dir)
    with TestClient(app2) as client:
        persisted = client.get("/api/v1/me")
        assert persisted.status_code == 200
        assert persisted.json()["preferences"]["locale"] == "en"
