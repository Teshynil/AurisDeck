"""Tests for the AurisDeck Click + Rich CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

import aurisdeck.cli as cli_module
from aurisdeck.auth.service import hash_token, verify_secret
from aurisdeck.db.repositories import JobRepo, UserRepo
from aurisdeck.storage.paths import DataPaths, atomic_write_text


def _sse_frame(seq: int, event_type: str, job_id: str, payload: dict[str, Any]) -> bytes:
    envelope = {
        "v": 1,
        "type": event_type,
        "ts": "2026-11-17T12:00:00Z",
        "job_id": job_id,
        "payload": payload,
    }
    encoded = json.dumps(envelope, separators=(",", ":"))
    return f"id: {seq}\nevent: {event_type}\ndata: {encoded}\n\n".encode()


def test_cli_help_commands() -> None:
    runner = CliRunner()
    commands = [
        ["--help"],
        ["serve", "--help"],
        ["transcribe", "--help"],
        ["profiles", "--help"],
        ["profiles", "list", "--help"],
        ["auth", "--help"],
        ["auth", "reset-password", "--help"],
        ["auth", "create-token", "--help"],
        ["pull-model", "--help"],
    ]
    for argv in commands:
        result = runner.invoke(cli_module.main, argv)
        assert result.exit_code == 0, f"Failed for {argv}: {result.output}"
        assert "Usage:" in result.output


def test_cli_module_entrypoint_help() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "aurisdeck.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0
    assert "AurisDeck CLI" in proc.stdout


def test_profiles_list_seeds_and_displays_defaults(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli_module.main, ["profiles", "list", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    for profile_id in ("es-tecnico", "subtitulos-video", "multipista-sync", "en-general"):
        assert profile_id in result.output
    assert (tmp_path / "profiles" / "es-tecnico.json").is_file()


def test_auth_reset_password_persists_scrypt_hash(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli_module.main,
        [
            "auth",
            "reset-password",
            "--username",
            "default",
            "--password",
            "S3cretPass!",
            "--data-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Password updated" in result.output

    user = UserRepo(DataPaths(tmp_path).db_path).get_by_username("default")
    assert user is not None
    assert user.password_hash is not None
    assert verify_secret("S3cretPass!", user.password_hash)
    assert not verify_secret("wrong", user.password_hash)


def test_auth_reset_password_interactive_and_missing_user(tmp_path: Path) -> None:
    runner = CliRunner()
    interactive = runner.invoke(
        cli_module.main,
        ["auth", "reset-password", "--data-dir", str(tmp_path)],
        input="PromptPass123\nPromptPass123\n",
    )
    assert interactive.exit_code == 0, interactive.output

    user = UserRepo(DataPaths(tmp_path).db_path).get_by_username("default")
    assert user is not None and user.password_hash is not None
    assert verify_secret("PromptPass123", user.password_hash)

    missing = runner.invoke(
        cli_module.main,
        [
            "auth",
            "reset-password",
            "--username",
            "nobody",
            "--password",
            "abc",
            "--data-dir",
            str(tmp_path),
        ],
    )
    assert missing.exit_code != 0
    assert "AUTH_USER_NOT_FOUND" in missing.output


def test_auth_create_token_prints_bearer_token_and_stores_hash(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli_module.main,
        ["auth", "create-token", "--username", "default", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output

    token_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert len(token_lines) == 1
    token = token_lines[0]
    assert token.startswith("adk_")

    user = UserRepo(DataPaths(tmp_path).db_path).find_by_api_token_hash(hash_token(token))
    assert user is not None
    assert user.username == "default"


def test_transcribe_server_success_sse_and_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_file = tmp_path / "sample.audio.wav"
    audio_file.write_bytes(b"RIFFdummywav")
    out_dir = tmp_path / "exported"

    seen_requests: list[tuple[str, str, str | None]] = []
    job_id = "job_12345"

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(
            (request.method, request.url.path, request.headers.get("authorization"))
        )
        if request.method == "POST" and request.url.path == "/api/v1/jobs":
            body_bytes = request.read()
            assert b"es-tecnico" in body_bytes
            assert b"AurisDeck" in body_bytes
            return httpx.Response(
                201,
                json={
                    "id": job_id,
                    "title": "sample.audio",
                    "status": "queued",
                    "jobs": [{"id": job_id, "title": "sample.audio", "status": "queued"}],
                },
            )
        if request.method == "GET" and request.url.path == "/api/v1/events":
            sse_bytes = b"".join(
                [
                    b": keepalive\n\n",
                    _sse_frame(
                        1,
                        "job.stage",
                        job_id,
                        {"device_id": "cuda:0", "stage": "transcribing"},
                    ),
                    _sse_frame(
                        2,
                        "job.progress",
                        job_id,
                        {
                            "device_id": "cuda:0",
                            "track_id": "trk_1",
                            "processed_ms": 5000,
                            "total_ms": 10000,
                            "rtfx": 14.5,
                        },
                    ),
                    _sse_frame(
                        3,
                        "job.completed",
                        job_id,
                        {"status": "completed", "diarization_status": "none"},
                    ),
                ]
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse_bytes,
            )
        if request.method == "GET" and request.url.path == f"/api/v1/jobs/{job_id}/download/vtt":
            return httpx.Response(
                200,
                headers={
                    "content-type": "text/vtt; charset=utf-8",
                    "content-disposition": 'attachment; filename="sample.audio.vtt"',
                },
                content=b"WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHola AurisDeck\n",
            )
        if request.method == "GET" and request.url.path == f"/api/v1/jobs/{job_id}/download/md":
            return httpx.Response(
                200,
                headers={
                    "content-type": "text/markdown; charset=utf-8",
                    "content-disposition": 'attachment; filename="sample.audio.md"',
                },
                content=b"# sample.audio\n\nHola AurisDeck\n",
            )
        return httpx.Response(404, json={"error_code": "NOT_FOUND", "detail": "unexpected path"})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(
        cli_module.httpx,
        "Client",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.main,
        [
            "transcribe",
            str(audio_file),
            "--profile",
            "es-tecnico",
            "--hotwords",
            "AurisDeck,Whisper",
            "--format",
            "vtt,md",
            "--out-dir",
            str(out_dir),
            "--server",
            "http://aurisdeck.local:8080",
            "--token",
            "adk_test_token",
        ],
    )
    assert result.exit_code == 0, result.output
    vtt_out = out_dir / "sample.audio.vtt"
    md_out = out_dir / "sample.audio.md"
    assert vtt_out.is_file()
    assert md_out.is_file()
    assert "Hola AurisDeck" in vtt_out.read_text(encoding="utf-8")
    assert all(auth == "Bearer adk_test_token" for _, _, auth in seen_requests)
    assert ("GET", "/api/v1/events", "Bearer adk_test_token") in seen_requests


def test_transcribe_server_failure_exits_nonzero_with_error_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_file = tmp_path / "broken.wav"
    audio_file.write_bytes(b"RIFFbroken")
    job_id = "job_fail_1"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/jobs":
            return httpx.Response(
                201,
                json={"jobs": [{"id": job_id, "title": "broken", "status": "queued"}]},
            )
        if request.method == "GET" and request.url.path == "/api/v1/events":
            sse_bytes = _sse_frame(
                1,
                "job.error",
                job_id,
                {"error_code": "CUDA_OOM", "detail": "Out of memory on cuda:0"},
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse_bytes,
            )
        return httpx.Response(500)

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(
        cli_module.httpx,
        "Client",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.main,
        [
            "transcribe",
            str(audio_file),
            "--server",
            "http://localhost:8080",
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "CUDA_OOM" in result.output


def test_transcribe_server_fallback_polling_and_batch_zip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a_wav = tmp_path / "a.wav"
    b_wav = tmp_path / "b.wav"
    a_wav.write_bytes(b"a")
    b_wav.write_bytes(b"b")
    out_dir = tmp_path / "batch_out"
    batch_id = "batch_999"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/jobs":
            return httpx.Response(
                201,
                json={
                    "batch_id": batch_id,
                    "jobs": [
                        {"id": "j1", "title": "a", "status": "queued"},
                        {"id": "j2", "title": "b", "status": "queued"},
                    ],
                },
            )
        if request.method == "GET" and request.url.path == "/api/v1/events":
            # Stream closes early before terminal event to exercise GET fallback polling.
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse_frame(1, "job.stage", "j1", {"stage": "transcribing"}),
            )
        if request.method == "GET" and request.url.path == "/api/v1/jobs/j1":
            return httpx.Response(200, json={"id": "j1", "status": "completed"})
        if request.method == "GET" and request.url.path == "/api/v1/jobs/j2":
            return httpx.Response(200, json={"id": "j2", "status": "completed"})
        if (
            request.method == "GET"
            and request.url.path == f"/api/v1/jobs/batches/{batch_id}/download.zip"
        ):
            assert request.url.params.get("fmt") == "vtt"
            return httpx.Response(
                200,
                headers={
                    "content-type": "application/zip",
                    "content-disposition": f'attachment; filename="{batch_id}_vtt.zip"',
                },
                content=b"PK\x03\x04fakezip",
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(
        cli_module.httpx,
        "Client",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_module.main,
        [
            "transcribe",
            str(a_wav),
            str(b_wav),
            "--mode",
            "batch",
            "--server",
            "http://localhost:8080",
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    zip_file = out_dir / f"{batch_id}_vtt.zip"
    assert zip_file.is_file()
    assert zip_file.read_bytes() == b"PK\x03\x04fakezip"


def test_transcribe_local_writes_output_next_to_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "recordings"
    input_dir.mkdir()
    audio_file = input_dir / "session.wav"
    audio_file.write_bytes(b"RIFFlocal")
    data_dir = tmp_path / "data"

    class FakeCoordinator:
        def __init__(
            self,
            *,
            settings: Any,
            paths: DataPaths,
            job_repo: JobRepo,
            segment_repo: Any,
            event_bus: Any,
        ) -> None:
            self.paths = paths
            self.job_repo = job_repo
            self.event_bus = event_bus
            self.started = False
            self.stopped = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

        def enqueue_job(self, job_id: str) -> None:
            assert self.started
            job = self.job_repo.get_job(job_id)
            assert job is not None
            assert (self.paths.root / job.tracks[0].media_relpath).is_file()
            out_file = self.paths.job_outputs_dir(job_id) / "session.vtt"
            atomic_write_text(
                out_file,
                "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nLocal ok\n",
                self.paths.tmp,
            )
            self.job_repo.update_job_status(job_id, "completed")
            self.event_bus.publish(
                "job.progress",
                job_id=job_id,
                payload={"device_id": "cpu", "processed_ms": 1000, "total_ms": 1000, "rtfx": 5.0},
            )
            self.event_bus.publish(
                "job.completed",
                job_id=job_id,
                payload={"status": "completed"},
            )

    monkeypatch.setattr(cli_module, "Coordinator", FakeCoordinator)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.main,
        [
            "transcribe",
            str(audio_file),
            "--data-dir",
            str(data_dir),
            "--gpus",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    expected_vtt = input_dir / "session.vtt"
    assert expected_vtt.is_file()
    assert "Local ok" in expected_vtt.read_text(encoding="utf-8")


def test_pull_model_sets_hf_env_and_downloads_to_data_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import faster_whisper.utils as fw_utils

    captured: dict[str, Any] = {}

    def fake_download_model(size_or_id: str, **kwargs: Any) -> str:
        captured["size_or_id"] = size_or_id
        captured["kwargs"] = kwargs
        captured["hf_home"] = cli_module.os.environ.get("HF_HOME")
        captured["hf_endpoint"] = cli_module.os.environ.get("HF_ENDPOINT")
        return str(tmp_path / "models" / size_or_id)

    monkeypatch.setattr(fw_utils, "download_model", fake_download_model)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.main,
        [
            "pull-model",
            "tiny",
            "--hf-endpoint",
            "https://hf-mirror.example.com",
            "--data-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["size_or_id"] == "tiny"
    assert captured["kwargs"]["cache_dir"] == str(tmp_path / "models")
    assert captured["hf_home"] == str(tmp_path / "models")
    assert captured["hf_endpoint"] == "https://hf-mirror.example.com"
