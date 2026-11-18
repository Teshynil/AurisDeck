"""Integration tests for /api/v1/jobs REST endpoints."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aurisdeck.api import create_app
from aurisdeck.api.routes_jobs import router as jobs_router
from aurisdeck.config import Settings
from aurisdeck.db.connection import open_connection
from aurisdeck.db.repositories import JobRepo, SegmentRepo
from aurisdeck.domain.models import GlossarySchema
from aurisdeck.export.cue_segmenter import SegmentItem, WordTiming, segment_to_cues
from aurisdeck.export.exporters import export_json, export_markdown, export_srt, export_vtt
from aurisdeck.storage.paths import DataPaths, atomic_write_text, sanitize_filename
from aurisdeck.storage.stores import GlossaryStore
from aurisdeck.workers.devices import DeviceInfo


class StubCoordinator:
    """In-process Coordinator stub for API tests without worker subprocesses."""

    def __init__(self, paths: DataPaths, job_repo: JobRepo, segment_repo: SegmentRepo) -> None:
        self.paths = paths
        self.job_repo = job_repo
        self.segment_repo = segment_repo
        self.started = False
        self.stopped = False
        self.enqueued: list[str] = []
        self.cancelled: list[str] = []
        self.retried: list[str] = []
        self.diarized: list[str] = []
        self.reexported: list[str] = []
        self._devices = [
            DeviceInfo(id="cuda:0", kind="cuda", index=0, name="Fake GPU 0"),
            DeviceInfo(id="cpu", kind="cpu", index=None, name="CPU (4 threads)"),
        ]

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def list_devices(self) -> list[DeviceInfo]:
        return list(self._devices)

    def unload_all_models(self) -> None:
        return None

    def enqueue_job(self, job_id: str) -> None:
        self.enqueued.append(job_id)

    def cancel_job(self, job_id: str) -> None:
        self.cancelled.append(job_id)
        self.job_repo.update_job_status(job_id, "cancelled")

    def retry_job(self, job_id: str) -> None:
        self.retried.append(job_id)
        self.job_repo.update_job_status(job_id, "queued")

    def trigger_diarize(self, job_id: str) -> None:
        job = self.job_repo.get_job(job_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")
        if job.status not in {"completed", "completed_partial"}:
            raise ValueError(f"Job {job_id} in status {job.status!r} cannot trigger diarization")
        self.diarized.append(job_id)
        self.job_repo.update_job_status(job_id, "diarizing", diarization_status="queued")

    def reexport_job(self, job_id: str) -> list[Path]:
        self.reexported.append(job_id)
        job = self.job_repo.get_job(job_id)
        assert job is not None
        segments = self.segment_repo.list_segments(job_id)
        speakers = self.segment_repo.list_speakers(job_id)
        speaker_aliases = {sp.speaker_key: sp.display_name for sp in speakers}

        items = [
            SegmentItem(
                seq=seg.seq,
                start_ms=seg.start_ms,
                end_ms=seg.end_ms,
                speaker_key=seg.speaker_key or None,
                raw_text=seg.raw_text,
                text=seg.text,
                words=[
                    WordTiming(
                        start_ms=int(w.get("start_ms", 0)),
                        end_ms=int(w.get("end_ms", 0)),
                        text=str(w.get("text", "")),
                    )
                    for w in seg.words
                ],
            )
            for seg in segments
        ]
        output_cfg = job.effective_config.get("output", {})
        formats = list(output_cfg.get("formats", ["vtt"]))
        sub_cfg = output_cfg.get("subtitles", {})
        voice_style = str(sub_cfg.get("voice_style", "v_tag"))
        cues = segment_to_cues(
            items,
            speaker_aliases=speaker_aliases,
            max_chars_per_line=int(sub_cfg.get("max_chars_per_line", 68)),
            max_lines_per_cue=int(sub_cfg.get("max_lines_per_cue", 2)),
            max_cue_seconds=float(sub_cfg.get("max_cue_seconds", 6.0)),
        )
        out_dir = self.paths.job_outputs_dir(job_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = sanitize_filename(job.title) if job.title else job.id
        written: list[Path] = []
        for fmt in formats:
            if fmt == "vtt":
                dest = out_dir / f"{stem}.vtt"
                content = export_vtt(
                    cues,
                    language=job.language_detected or job.language_requested,
                    voice_style=voice_style,
                )
            elif fmt == "srt":
                dest = out_dir / f"{stem}.srt"
                content = export_srt(cues, voice_style=voice_style)
            elif fmt == "md":
                dest = out_dir / f"{stem}.md"
                content = export_markdown(items, speaker_aliases=speaker_aliases, title=job.title)
            elif fmt == "json":
                dest = out_dir / f"{stem}.json"
                content = export_json({"id": job.id, "title": job.title}, items, speaker_aliases)
            else:
                continue
            atomic_write_text(dest, content, self.paths.tmp)
            written.append(dest)
        return written


def _build_jobs_test_app(
    tmp_path: Path,
    *,
    auth_mode: str = "none",
    auth_password: str = "",
) -> tuple[FastAPI, StubCoordinator]:
    settings = Settings(
        data_dir=tmp_path,
        auth_mode=auth_mode,  # type: ignore[arg-type]
        auth_password=auth_password,
    )
    paths = DataPaths(tmp_path)
    stub = StubCoordinator(paths, JobRepo(paths.db_path), SegmentRepo(paths.db_path))
    app = create_app(settings, coordinator=stub)  # type: ignore[arg-type]
    registered_paths = {getattr(route, "path", None) for route in app.routes}
    if "/api/v1/jobs/inspect-zip" not in registered_paths:
        app.include_router(jobs_router)
    return app, stub


def _make_zip_bytes(entries: dict[str, bytes | str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in entries.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            zf.writestr(name, data)
    return buf.getvalue()


def test_jobs_require_auth_when_password_enabled(tmp_path: Path) -> None:
    app, _ = _build_jobs_test_app(tmp_path, auth_mode="password", auth_password="secret")
    with TestClient(app) as client:
        resp = client.get("/api/v1/jobs")
        assert resp.status_code == 401
        assert resp.json()["error_code"] == "AUTH_REQUIRED"

        login_resp = client.post("/api/v1/auth/login", json={"password": "secret"})
        assert login_resp.status_code == 200
        assert client.get("/api/v1/jobs").status_code == 200


def test_inspect_zip_multitrack_and_tmp_cleanup(tmp_path: Path) -> None:
    app, _ = _build_jobs_test_app(tmp_path)
    archive_bytes = _make_zip_bytes(
        {
            "1-alice.ogg": b"OGG_ALICE",
            "2-bob.ogg": b"OGG_BOB",
            "session.aup": "<project/>",
            "info.txt": "Track 1: Alicia\nTrack 2: Roberto\n",
            "__MACOSX/._1-alice.ogg": b"junk",
            ".DS_Store": b"junk",
        }
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/jobs/inspect-zip",
            files={"file": ("recording.zip", archive_bytes, "application/zip")},
        )
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 2
        by_file = {item["filename"]: item for item in items}
        assert by_file["1-alice.ogg"] == {
            "filename": "1-alice.ogg",
            "speaker_name": "1-alice",
            "offset_ms": 0,
        }
        assert by_file["2-bob.ogg"] == {
            "filename": "2-bob.ogg",
            "speaker_name": "2-bob",
            "offset_ms": 0,
        }

        # Verify tmp/ has no leftover inspect_* directories
        tmp_dir = DataPaths(tmp_path).tmp
        assert list(tmp_dir.iterdir()) == []


def test_inspect_zip_rejects_zip_slip_and_invalid_archive(tmp_path: Path) -> None:
    app, _ = _build_jobs_test_app(tmp_path)
    evil_zip = _make_zip_bytes({"../escape.wav": b"BAD"})

    with TestClient(app) as client:
        slip_resp = client.post(
            "/api/v1/jobs/inspect-zip",
            files={"file": ("evil.zip", evil_zip, "application/zip")},
        )
        assert slip_resp.status_code == 400
        assert slip_resp.json()["error_code"] == "INVALID_ARCHIVE"

        bad_resp = client.post(
            "/api/v1/jobs/inspect-zip",
            files={"file": ("corrupt.zip", b"not-a-zip", "application/zip")},
        )
        assert bad_resp.status_code == 400
        assert bad_resp.json()["error_code"] == "INVALID_ARCHIVE"

        assert list(DataPaths(tmp_path).tmp.iterdir()) == []


def test_create_single_job_overrides_snapshot_and_range_streaming(tmp_path: Path) -> None:
    app, stub = _build_jobs_test_app(tmp_path)
    paths = DataPaths(tmp_path)

    with TestClient(app) as client:
        # Missing profile returns 404 PROFILE_NOT_FOUND
        missing_prof = client.post(
            "/api/v1/jobs",
            files=[("files", ("reunion.v2.wav", b"0123456789ABCDEF", "audio/wav"))],
            data={"mode": "single", "profile_id": "non-existent"},
        )
        assert missing_prof.status_code == 404
        assert missing_prof.json()["error_code"] == "PROFILE_NOT_FOUND"

        GlossaryStore(paths).save(
            GlossarySchema(id="devops", name="DevOps", terms=["Kubernetes", "Istio"])
        )

        overrides = {
            "language": "en",
            "diarize": True,
            "hotwords": ["AurisDeck", "Kubernetes"],
            "initial_prompt": "Architecture review",
            "glossary_ids": ["devops"],
            "gpus": [0],
            "subtitles": {"voice_style": "prefix", "max_chars_per_line": 50},
        }
        create_resp = client.post(
            "/api/v1/jobs",
            files=[("files", ("reunion.v2.wav", b"0123456789ABCDEF", "audio/wav"))],
            data={
                "mode": "single",
                "profile_id": "es-tecnico",
                "overrides_json": json.dumps(overrides),
            },
        )
        assert create_resp.status_code == 201
        payload = create_resp.json()
        job_id = payload["id"]
        assert payload["title"] == "reunion.v2"
        assert payload["mode"] == "single"
        assert payload["language_requested"] == "en"
        assert payload["effective_config"]["asr"]["language"] == "en"
        assert payload["effective_config"]["diarization"]["enabled"] is True
        assert payload["effective_config"]["hardware"]["gpus"] == [0]
        assert payload["effective_config"]["output"]["subtitles"]["voice_style"] == "prefix"
        assert "Istio" in payload["effective_config"]["effective_context"]["hotwords"]
        assert stub.enqueued == [job_id]

        assert len(payload["tracks"]) == 1
        track = payload["tracks"][0]
        assert track["original_filename"] == "reunion.v2.wav"
        assert (paths.root / track["media_relpath"]).is_file()

        # Range request returns 206 Partial Content
        range_resp = client.get(
            f"/api/v1/jobs/{job_id}/media/{track['id']}",
            headers={"Range": "bytes=0-9"},
        )
        assert range_resp.status_code == 206
        assert range_resp.content == b"0123456789"
        assert range_resp.headers["content-range"].startswith("bytes 0-9/16")

        # Full media request
        full_resp = client.get(f"/api/v1/jobs/{job_id}/media/{track['id']}")
        assert full_resp.status_code == 200
        assert full_resp.content == b"0123456789ABCDEF"

        # Path traversal guard on media_relpath
        conn = open_connection(paths.db_path)
        try:
            with conn:
                conn.execute(
                    "UPDATE tracks SET media_relpath = ? WHERE id = ?",
                    ("../../etc/passwd", track["id"]),
                )
        finally:
            conn.close()
        traversal_resp = client.get(f"/api/v1/jobs/{job_id}/media/{track['id']}")
        assert traversal_resp.status_code == 403
        assert traversal_resp.json()["error_code"] == "FORBIDDEN"


def test_create_multitrack_zip_and_loose_files_with_aliases_and_offsets(tmp_path: Path) -> None:
    app, stub = _build_jobs_test_app(tmp_path)
    zip_bytes = _make_zip_bytes(
        {
            "1-alice.ogg": b"ALICE_BYTES",
            "2-bob.ogg": b"BOB_BYTES",
            "info.txt": "Track 1: Alice\nTrack 2: Bob\n",
        }
    )

    with TestClient(app) as client:
        resp_zip = client.post(
            "/api/v1/jobs",
            files=[("files", ("session.zip", zip_bytes, "application/zip"))],
            data={
                "mode": "multitrack",
                "title": "D&D Session 12",
                "overrides_json": json.dumps(
                    {
                        "speaker_aliases": {"1-alice.ogg": "Alicia", "2-bob.ogg": "Roberto"},
                        "track_offsets": {"2-bob.ogg": 2400},
                    }
                ),
            },
        )
        assert resp_zip.status_code == 201
        job = resp_zip.json()
        assert job["mode"] == "multitrack"
        assert len(job["tracks"]) == 2
        tracks_by_name = {t["original_filename"]: t for t in job["tracks"]}
        assert tracks_by_name["1-alice.ogg"]["offset_ms"] == 0
        assert tracks_by_name["2-bob.ogg"]["offset_ms"] == 2400

        speakers_by_key = {s["speaker_key"]: s["display_name"] for s in job["speakers"]}
        assert speakers_by_key[tracks_by_name["1-alice.ogg"]["speaker_key"]] == "Alicia"
        assert speakers_by_key[tracks_by_name["2-bob.ogg"]["speaker_key"]] == "Roberto"
        assert job["id"] in stub.enqueued

        # Loose files in multitrack mode
        resp_loose = client.post(
            "/api/v1/jobs",
            files=[
                ("files", ("carlos.m4a", b"CARLOS", "audio/mp4")),
                ("files", ("elena.m4a", b"ELENA", "audio/mp4")),
            ],
            data={
                "mode": "multitrack",
                "overrides_json": json.dumps(
                    {
                        "speaker_aliases": {"carlos.m4a": "Carlos DM"},
                        "track_offsets": {"elena.m4a": 500},
                    }
                ),
            },
        )
        assert resp_loose.status_code == 201
        loose_job = resp_loose.json()
        assert len(loose_job["tracks"]) == 2
        loose_tracks = {t["original_filename"]: t for t in loose_job["tracks"]}
        assert loose_tracks["elena.m4a"]["offset_ms"] == 500
        loose_speakers = {s["speaker_key"]: s["display_name"] for s in loose_job["speakers"]}
        assert loose_speakers[loose_tracks["carlos.m4a"]["speaker_key"]] == "Carlos DM"
        assert loose_speakers[loose_tracks["elena.m4a"]["speaker_key"]] == "elena"


def test_batch_create_list_filter_and_download_zip(tmp_path: Path) -> None:
    app, stub = _build_jobs_test_app(tmp_path)
    paths = DataPaths(tmp_path)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/jobs",
            files=[
                ("files", ("ep1.v1.wav", b"EP1", "audio/wav")),
                ("files", ("ep2.v1.wav", b"EP2", "audio/wav")),
                ("files", ("ep3.failed.wav", b"EP3", "audio/wav")),
            ],
            data={"mode": "batch"},
        )
        assert resp.status_code == 201
        data = resp.json()
        batch_id = data["batch_id"]
        jobs = data["jobs"]
        assert len(jobs) == 3
        assert len(stub.enqueued) == 3

        # Mark first two jobs completed with .vtt outputs, third failed
        job_repo = JobRepo(paths.db_path)
        for job_item in jobs[:2]:
            jid = job_item["id"]
            job_repo.update_job_status(jid, "completed")
            out_file = paths.job_outputs_dir(jid) / f"{job_item['title']}.vtt"
            atomic_write_text(out_file, f"WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n{jid}\n", paths.tmp)
        job_repo.update_job_status(jobs[2]["id"], "failed")

        # Filter list endpoint by batch_id and status
        list_resp = client.get(f"/api/v1/jobs?batch_id={batch_id}&status=completed&limit=10&offset=0")
        assert list_resp.status_code == 200
        assert len(list_resp.json()["jobs"]) == 2

        # Download batch ZIP
        zip_resp = client.get(f"/api/v1/jobs/batches/{batch_id}/download.zip?fmt=vtt")
        assert zip_resp.status_code == 200
        assert zip_resp.headers["content-type"] == "application/zip"
        with zipfile.ZipFile(io.BytesIO(zip_resp.content), "r") as zf:
            names = sorted(zf.namelist())
            assert names == ["ep1.v1.vtt", "ep2.v1.vtt"]


def test_speakers_patch_reexport_and_download_preserves_dots(tmp_path: Path) -> None:
    app, stub = _build_jobs_test_app(tmp_path)
    paths = DataPaths(tmp_path)

    with TestClient(app) as client:
        create_resp = client.post(
            "/api/v1/jobs",
            files=[("files", ("reunion.v2.wav", b"AUDIO", "audio/wav"))],
            data={"mode": "single"},
        )
        assert create_resp.status_code == 201
        job_id = create_resp.json()["id"]

        seg_repo = SegmentRepo(paths.db_path)
        seg_repo.upsert_speakers(
            job_id,
            [{"speaker_key": "SPEAKER_00", "display_name": "SPEAKER_00", "track_id": None}],
        )
        seg_repo.replace_segments(
            job_id,
            [
                {
                    "seq": 0,
                    "start_ms": 0,
                    "end_ms": 2000,
                    "speaker_key": "SPEAKER_00",
                    "raw_text": "Hola equipo",
                    "text": "Hola equipo",
                    "words": [
                        {"start_ms": 0, "end_ms": 800, "text": "Hola"},
                        {"start_ms": 900, "end_ms": 2000, "text": "equipo"},
                    ],
                }
            ],
        )
        JobRepo(paths.db_path).update_job_status(job_id, "completed")
        stub.reexport_job(job_id)

        # Rename speaker -> reexport runs and updated speakers/segments returned
        patch_resp = client.patch(
            f"/api/v1/jobs/{job_id}/speakers",
            json={"aliases": {"SPEAKER_00": "Ana"}},
        )
        assert patch_resp.status_code == 200
        patch_data = patch_resp.json()
        assert patch_data["speakers"][0]["display_name"] == "Ana"
        assert len(patch_data["segments"]) == 1
        assert len(patch_data["available_formats"]) >= 1

        # Download VTT on demand (generates and caches) preserving dot in reunion.v2.vtt
        vtt_resp = client.get(f"/api/v1/jobs/{job_id}/download/vtt")
        assert vtt_resp.status_code == 200
        assert 'filename="reunion.v2.vtt"' in vtt_resp.headers["content-disposition"]
        assert "<v Ana>Hola equipo</v>" in vtt_resp.text

        # Re-export with voice_style=prefix and formats=["vtt", "srt"]
        exp_resp = client.post(
            f"/api/v1/jobs/{job_id}/export",
            json={"voice_style": "prefix", "formats": ["vtt", "srt"]},
        )
        assert exp_resp.status_code == 200
        exp_data = exp_resp.json()
        assert exp_data["files"] == ["reunion.v2.vtt", "reunion.v2.srt"]
        assert {"vtt", "srt"}.issubset(set(exp_data["available_formats"]))

        srt_resp = client.get(f"/api/v1/jobs/{job_id}/download/srt")
        assert srt_resp.status_code == 200
        assert 'filename="reunion.v2.srt"' in srt_resp.headers["content-disposition"]
        assert "[Ana]: Hola equipo" in srt_resp.text

        vtt_prefix_resp = client.get(f"/api/v1/jobs/{job_id}/download/vtt")
        assert vtt_prefix_resp.status_code == 200
        assert "[Ana]: Hola equipo" in vtt_prefix_resp.text

        # Invalid download format returns 400 VALIDATION_ERROR
        bad_fmt = client.get(f"/api/v1/jobs/{job_id}/download/docx")
        assert bad_fmt.status_code == 400
        assert bad_fmt.json()["error_code"] == "VALIDATION_ERROR"

        # Job detail includes segments and available_formats
        detail_resp = client.get(f"/api/v1/jobs/{job_id}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert {"vtt", "srt"}.issubset(set(detail["available_formats"]))
        assert detail["segments"][0]["text"] == "Hola equipo"


def test_cancel_retry_diarize_delete_and_404_job_not_found(tmp_path: Path) -> None:
    app, stub = _build_jobs_test_app(tmp_path)
    paths = DataPaths(tmp_path)

    with TestClient(app) as client:
        # 404 checks for nonexistent job
        for method, url, kwargs in (
            ("get", "/api/v1/jobs/missing-id", {}),
            ("post", "/api/v1/jobs/missing-id/cancel", {}),
            ("post", "/api/v1/jobs/missing-id/retry", {}),
            ("post", "/api/v1/jobs/missing-id/diarize", {}),
            ("patch", "/api/v1/jobs/missing-id/speakers", {"json": {"aliases": {"S0": "A"}}}),
            ("post", "/api/v1/jobs/missing-id/export", {}),
            ("get", "/api/v1/jobs/missing-id/media/t1", {}),
            ("get", "/api/v1/jobs/missing-id/download/vtt", {}),
            ("delete", "/api/v1/jobs/missing-id", {}),
        ):
            resp = getattr(client, method)(url, **kwargs)
            assert resp.status_code == 404, (method, url, resp.text)
            assert resp.json()["error_code"] == "JOB_NOT_FOUND"

        # Create job
        create_resp = client.post(
            "/api/v1/jobs",
            files=[("files", ("sample.wav", b"AUDIO", "audio/wav"))],
            data={"mode": "single"},
        )
        assert create_resp.status_code == 201
        job_id = create_resp.json()["id"]

        # Nonexistent track returns 404
        bad_track = client.get(f"/api/v1/jobs/{job_id}/media/nonexistent-track")
        assert bad_track.status_code == 404

        # Diarize while queued raises 400 VALIDATION_ERROR from Coordinator check
        bad_diarize = client.post(f"/api/v1/jobs/{job_id}/diarize", json={"min_speakers": 2})
        assert bad_diarize.status_code == 400
        assert bad_diarize.json()["error_code"] == "VALIDATION_ERROR"

        # Cancel
        cancel_resp = client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancelled"
        assert job_id in stub.cancelled

        # Retry without body
        retry_plain = client.post(f"/api/v1/jobs/{job_id}/retry")
        assert retry_plain.status_code == 200
        assert retry_plain.json()["status"] == "queued"

        # Retry with overrides
        retry_resp = client.post(
            f"/api/v1/jobs/{job_id}/retry",
            json={"language": "fr", "hotwords": ["Bonjour"]},
        )
        assert retry_resp.status_code == 200
        assert retry_resp.json()["language_requested"] == "fr"
        assert retry_resp.json()["effective_config"]["effective_context"]["hotwords"] == ["Bonjour"]
        assert stub.retried == [job_id, job_id]

        # Mark completed and trigger diarize with speaker bounds
        JobRepo(paths.db_path).update_job_status(job_id, "completed")
        diarize_resp = client.post(
            f"/api/v1/jobs/{job_id}/diarize",
            json={"min_speakers": 2, "max_speakers": 4},
        )
        assert diarize_resp.status_code == 200
        assert diarize_resp.json()["status"] == "diarizing"
        assert diarize_resp.json()["effective_config"]["diarization"]["min_speakers"] == 2
        assert diarize_resp.json()["effective_config"]["diarization"]["max_speakers"] == 4
        assert job_id in stub.diarized

        # Populate checkpoints/outputs/tmp directories to verify DELETE removes all 4 dirs & DB row
        for directory in (
            paths.job_checkpoints_dir(job_id),
            paths.job_outputs_dir(job_id),
            paths.job_tmp_dir(job_id),
        ):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "artifact.txt").write_text("temp", encoding="utf-8")

        stub.cancelled.clear()
        del_resp = client.delete(f"/api/v1/jobs/{job_id}")
        assert del_resp.status_code == 200
        assert del_resp.json() == {"deleted": True, "id": job_id}
        # Job was active ("diarizing"), so cancel_job must have been called first
        assert stub.cancelled == [job_id]
        assert JobRepo(paths.db_path).get_job(job_id) is None
        assert not paths.job_media_dir(job_id).exists()
        assert not paths.job_checkpoints_dir(job_id).exists()
        assert not paths.job_outputs_dir(job_id).exists()
        assert not paths.job_tmp_dir(job_id).exists()


def test_multitrack_mix_endpoints_generate_and_stream(tmp_path: Path) -> None:
    app, _stub = _build_jobs_test_app(tmp_path)
    paths = DataPaths(tmp_path)
    with TestClient(app) as client:
        # Create a multitrack job with 2 tracks
        create_resp = client.post(
            "/api/v1/jobs",
            data={
                "mode": "multitrack",
                "profile_id": "multipista-sync",
                "overrides_json": json.dumps({"track_offsets": {"0": 0, "1": 250}}),
            },
            files=[
                ("files", ("alice.flac", b"track-a", "audio/flac")),
                ("files", ("bob.flac", b"track-b", "audio/flac")),
            ],
        )
        assert create_resp.status_code == 201
        job_data = create_resp.json()["job"]
        job_id = job_data["id"]
        assert job_data["has_mix"] is False
        assert job_data["effective_config"]["preprocessing"]["generate_mix"] is True

        # Streaming mix before it exists -> 404
        missing_mix = client.get(f"/api/v1/jobs/{job_id}/media/mix")
        assert missing_mix.status_code == 404

        def fake_mix(tracks: list[tuple[Path, int]], dst: Path | str, normalize_audio: bool = False) -> None:
            assert len(tracks) == 2
            assert tracks[0][1] == 0
            assert tracks[1][1] == 250
            Path(dst).write_bytes(b"ftypM4A-fake-mix-bytes")

        with patch("aurisdeck.api.routes_jobs.mix_multitrack_audio", side_effect=fake_mix):
            gen_resp = client.post(f"/api/v1/jobs/{job_id}/mix")
            assert gen_resp.status_code == 200
            assert gen_resp.json()["has_mix"] is True

        # Detail endpoint reports has_mix=True
        detail_resp = client.get(f"/api/v1/jobs/{job_id}")
        assert detail_resp.status_code == 200
        assert detail_resp.json()["has_mix"] is True

        # Stream mix with Range support
        stream_resp = client.get(
            f"/api/v1/jobs/{job_id}/media/mix",
            headers={"Range": "bytes=0-6"},
        )
        assert stream_resp.status_code == 206
        assert stream_resp.content == b"ftypM4A"

        # Single-track job rejects /mix with 422
        single_resp = client.post(
            "/api/v1/jobs",
            data={"mode": "single", "profile_id": "es-tecnico"},
            files=[("files", ("solo.wav", b"solo-bytes", "audio/wav"))],
        )
        assert single_resp.status_code == 201
        single_id = single_resp.json()["job"]["id"]
        bad_mix = client.post(f"/api/v1/jobs/{single_id}/mix")
        assert bad_mix.status_code == 422


def test_generic_zip_in_single_and_batch_modes(tmp_path: Path) -> None:
    app, stub = _build_jobs_test_app(tmp_path)
    paths = DataPaths(tmp_path)

    # Build a ZIP with 2 audio files inside a subfolder
    multi_zip_buf = io.BytesIO()
    with zipfile.ZipFile(multi_zip_buf, "w") as zf:
        zf.writestr("folder/track1.wav", b"wav-1")
        zf.writestr("folder/track2.mp3", b"mp3-2")
    multi_zip_bytes = multi_zip_buf.getvalue()

    # Build a ZIP with 1 audio file
    single_zip_buf = io.BytesIO()
    with zipfile.ZipFile(single_zip_buf, "w") as zf:
        zf.writestr("nested/only.m4a", b"m4a-1")
    single_zip_bytes = single_zip_buf.getvalue()

    with TestClient(app) as client:
        # 1. Batch mode with ZIP containing 2 files -> 2 independent batch jobs
        batch_resp = client.post(
            "/api/v1/jobs",
            data={"mode": "batch", "profile_id": "es-tecnico"},
            files=[("files", ("bundle.zip", multi_zip_bytes, "application/zip"))],
        )
        assert batch_resp.status_code == 201
        batch_data = batch_resp.json()
        assert len(batch_data["jobs"]) == 2
        filenames = sorted(j["tracks"][0]["original_filename"] for j in batch_data["jobs"])
        assert filenames == ["track1.wav", "track2.mp3"]

        # 2. Single mode with ZIP containing 1 file -> succeeds
        single_ok = client.post(
            "/api/v1/jobs",
            data={"mode": "single", "profile_id": "es-tecnico"},
            files=[("files", ("one.zip", single_zip_bytes, "application/zip"))],
        )
        assert single_ok.status_code == 201
        assert single_ok.json()["job"]["tracks"][0]["original_filename"] == "only.m4a"
        assert (paths.root / single_ok.json()["job"]["tracks"][0]["media_relpath"]).is_file()

        # 3. Single mode with ZIP containing 2 files -> 422
        single_bad = client.post(
            "/api/v1/jobs",
            data={"mode": "single", "profile_id": "es-tecnico"},
            files=[("files", ("two.zip", multi_zip_bytes, "application/zip"))],
        )
        assert single_bad.status_code == 422
        assert len(stub.enqueued) == 3


def test_zip_excluded_tracks_in_multitrack_batch_and_single_modes(tmp_path: Path) -> None:
    app, _stub = _build_jobs_test_app(tmp_path)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("session/alice.flac", b"alice-audio")
        zf.writestr("session/bot.ogg", b"bot-audio")
        zf.writestr("session/bob.wav", b"bob-audio")
    zip_bytes = zip_buf.getvalue()

    with TestClient(app) as client:
        # 1. Multitrack mode excluding bot.ogg -> 2 tracks with aligned aliases/offsets
        mt_resp = client.post(
            "/api/v1/jobs",
            data={
                "mode": "multitrack",
                "profile_id": "multipista-sync",
                "overrides_json": json.dumps(
                    {
                        "excluded_tracks": ["session/bot.ogg"],
                        "speaker_aliases": {"0": "Alicia", "1": "Roberto"},
                        "track_offsets": {"0": 150, "1": 300},
                    }
                ),
            },
            files=[("files", ("session.zip", zip_bytes, "application/zip"))],
        )
        assert mt_resp.status_code == 201
        mt_tracks = mt_resp.json()["job"]["tracks"]
        assert [t["original_filename"] for t in mt_tracks] == ["alice.flac", "bob.wav"]
        assert [t["offset_ms"] for t in mt_tracks] == [150, 300]
        mt_speakers = {s["speaker_key"]: s["display_name"] for s in mt_resp.json()["job"]["speakers"]}
        assert mt_speakers == {"SPEAKER_00": "Alicia", "SPEAKER_01": "Roberto"}

        # 2. Batch mode excluding session/bot.ogg -> 2 independent batch jobs
        batch_resp = client.post(
            "/api/v1/jobs",
            data={
                "mode": "batch",
                "profile_id": "es-tecnico",
                "overrides_json": json.dumps({"excluded_tracks": ["session/bot.ogg"]}),
            },
            files=[("files", ("session.zip", zip_bytes, "application/zip"))],
        )
        assert batch_resp.status_code == 201
        batch_files = sorted(j["tracks"][0]["original_filename"] for j in batch_resp.json()["jobs"])
        assert batch_files == ["alice.flac", "bob.wav"]

        # 3. Single mode excluding 2 of 3 tracks -> succeeds as single job
        single_resp = client.post(
            "/api/v1/jobs",
            data={
                "mode": "single",
                "profile_id": "es-tecnico",
                "overrides_json": json.dumps(
                    {"excluded_tracks": ["session/bot.ogg", "session/bob.wav"]}
                ),
            },
            files=[("files", ("session.zip", zip_bytes, "application/zip"))],
        )
        assert single_resp.status_code == 201
        assert single_resp.json()["job"]["tracks"][0]["original_filename"] == "alice.flac"


def test_zip_ignores_aup_and_job_logs_endpoint(tmp_path: Path) -> None:
    app, _coordinator = _build_jobs_test_app(tmp_path)
    paths = DataPaths(tmp_path)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("project.aup", '<?xml version="1.0"?><!DOCTYPE foo><project/>')
        zf.writestr("track1.wav", b"RIFF" + b"\x00" * 36)

    with TestClient(app) as client:
        ok = client.post(
            "/api/v1/jobs/inspect-zip",
            files=[("file", ("archive.zip", zip_buf.getvalue(), "application/zip"))],
        )
        assert ok.status_code == 200
        assert ok.json() == [{"filename": "track1.wav", "speaker_name": "track1", "offset_ms": 0}]

        bad = client.post(
            "/api/v1/jobs/inspect-zip",
            files=[("file", ("bad.zip", b"not-a-zip", "application/zip"))],
        )
        assert bad.status_code == 400
        assert bad.json()["error_code"] == "INVALID_ARCHIVE"

        main_log = paths.logs / "aurisdeck.log"
        assert main_log.is_file()
        assert "INVALID_ARCHIVE" in main_log.read_text(encoding="utf-8")

        created = client.post(
            "/api/v1/jobs",
            data={"mode": "single"},
            files=[("files", ("audio.wav", b"RIFF" + b"\x00" * 36, "audio/wav"))],
        )
        assert created.status_code == 201
        job_id = created.json()["job"]["id"]

        empty_logs = client.get(f"/api/v1/jobs/{job_id}/logs")
        assert empty_logs.status_code == 200
        assert empty_logs.json() == {"job_id": job_id, "files": []}

        log_dir = paths.job_logs_dir(job_id)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "asr-trk1.log").write_text("line 1\nline 2\n", encoding="utf-8")

        listed = client.get(f"/api/v1/jobs/{job_id}/logs")
        assert listed.status_code == 200
        assert listed.json()["files"][0]["name"] == "asr-trk1.log"

        fetched = client.get(f"/api/v1/jobs/{job_id}/logs", params={"file": "asr-trk1.log"})
        assert fetched.status_code == 200
        assert fetched.json()["content"] == "line 1\nline 2\n"

        traversal = client.get(f"/api/v1/jobs/{job_id}/logs", params={"file": "../secret.txt"})
        assert traversal.status_code == 400
