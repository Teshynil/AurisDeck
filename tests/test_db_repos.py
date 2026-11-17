"""Tests for SQLite repositories (UserRepo, JobRepo, SegmentRepo) and startup recovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from aurisdeck.db import open_connection, run_migrations
from aurisdeck.db.recovery import run_startup_recovery
from aurisdeck.db.repositories import JobRepo, SegmentRepo, UserRepo
from aurisdeck.export.cue_segmenter import SegmentItem, WordTiming


def _init_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "aurisdeck.db"
    run_migrations(db_path)
    return db_path


def test_user_repo_crud_preferences_password_and_api_token(tmp_path: Path) -> None:
    db_path = _init_db(tmp_path)
    repo = UserRepo(db_path)

    by_id = repo.get_by_id("default")
    by_name = repo.get_by_username("default")
    assert by_id is not None
    assert by_name == by_id
    assert by_id.role == "admin"
    assert by_id.preferences["locale"] == "es"
    assert repo.get_by_id("nonexistent") is None
    assert repo.get_by_username("nonexistent") is None

    new_prefs = {
        "locale": "en",
        "ui_mode": "advanced",
        "default_profile_id": "en-general",
        "player": {"playback_rate": 1.25, "auto_scroll": False},
    }
    updated = repo.update_preferences("default", new_prefs)
    assert updated is not None
    assert updated.preferences == new_prefs
    assert repo.update_preferences("missing", new_prefs) is None

    assert repo.set_password_hash("default", "scrypt$salt$digest") is True
    assert repo.set_password_hash("missing", "hash") is False
    reloaded = repo.get_by_id("default")
    assert reloaded is not None
    assert reloaded.password_hash == "scrypt$salt$digest"

    assert repo.set_api_token_hash("default", "sha256-token-hash") is True
    assert repo.set_api_token_hash("missing", "sha256-token-hash") is False
    found = repo.find_by_api_token_hash("sha256-token-hash")
    assert found is not None
    assert found.id == "default"
    assert repo.find_by_api_token_hash("wrong-hash") is None
    assert repo.find_by_api_token_hash("") is None

    assert repo.set_api_token_hash("default", None) is True
    assert repo.find_by_api_token_hash("sha256-token-hash") is None


def test_job_repo_crud_filters_updates_and_cascade_delete(tmp_path: Path) -> None:
    db_path = _init_db(tmp_path)
    job_repo = JobRepo(lambda: open_connection(db_path))
    seg_repo = SegmentRepo(db_path)

    job1 = job_repo.create_job(
        job_id="job-1",
        title="Episodio 1",
        mode="multitrack",
        effective_config={"asr": {"model": "large-v3-turbo"}},
        batch_id="batch-a",
        language_requested="es",
        created_at="2026-01-01T10:00:00Z",
    )
    job2 = job_repo.create_job(
        title="Nota rapida",
        mode="single",
        effective_config={"asr": {"model": "base"}},
        status="completed",
        batch_id="batch-b",
        created_at="2026-01-01T11:00:00Z",
    )
    assert job1.id == "job-1"
    assert len(job2.id) == 32

    trk2 = job_repo.add_track(
        "job-1",
        track_index=1,
        original_filename="b.ogg",
        media_relpath="media/job-1/trk-2_b.ogg",
        speaker_key="SPEAKER_01",
        track_id="trk-2",
        offset_ms=250,
    )
    trk1 = job_repo.add_track(
        "job-1",
        track_index=0,
        original_filename="a.ogg",
        media_relpath="media/job-1/trk-1_a.ogg",
        speaker_key="SPEAKER_00",
        track_id="trk-1",
    )
    assert trk1.track_index == 0
    assert trk2.offset_ms == 250

    seg_repo.upsert_speakers(
        "job-1",
        [
            {"speaker_key": "SPEAKER_01", "display_name": "Luis", "track_id": "trk-2"},
            {"speaker_key": "SPEAKER_00", "display_name": "Ana", "track_id": "trk-1"},
        ],
    )

    loaded = job_repo.get_job("job-1")
    assert loaded is not None
    assert [t.id for t in loaded.tracks] == ["trk-1", "trk-2"]
    assert [s.speaker_key for s in loaded.speakers] == ["SPEAKER_00", "SPEAKER_01"]
    assert job_repo.get_job("missing-job") is None

    # Update track ASR and job status while preserving omitted fields
    updated_track = job_repo.update_track_asr(
        "trk-1",
        asr_status="completed",
        checkpoint_relpath="checkpoints/job-1/trk-1.asr.json",
        duration_ms=42_500,
    )
    assert updated_track is not None
    assert updated_track.asr_status == "completed"
    assert updated_track.duration_ms == 42_500
    assert job_repo.update_track_asr("missing-trk", "completed", None) is None

    err_job = job_repo.update_job_status(
        "job-1",
        "failed",
        diarization_status="failed",
        error_code="ASR_FAILED",
        error_params={"track": 1},
        error_detail="cuda error",
        language_detected="es",
    )
    assert err_job is not None
    assert err_job.error_params == {"track": 1}
    assert err_job.language_detected == "es"

    # Partial update preserves language_detected and diarization_status
    partial_update = job_repo.update_job_status(
        "job-1",
        "completed",
        completed_at="2026-01-01T12:00:00Z",
    )
    assert partial_update is not None
    assert partial_update.language_detected == "es"
    assert partial_update.diarization_status == "failed"
    assert partial_update.completed_at == "2026-01-01T12:00:00Z"

    # Explicit None clears nullable error columns
    cleared = job_repo.update_job_status(
        "job-1",
        "completed",
        error_code=None,
        error_params=None,
        error_detail=None,
    )
    assert cleared is not None
    assert cleared.error_code is None
    assert cleared.error_params is None
    assert cleared.error_detail is None
    assert job_repo.update_job_status("missing-job", "completed") is None

    # list_jobs filters & pagination
    all_jobs = job_repo.list_jobs()
    assert [j.id for j in all_jobs] == [job2.id, "job-1"]
    assert [j.id for j in job_repo.list_jobs(batch_id="batch-a")] == ["job-1"]
    assert [j.id for j in job_repo.list_jobs(status="completed")] == [job2.id, "job-1"]
    assert [j.id for j in job_repo.list_jobs(owner_user_id="other")] == []
    assert [j.id for j in job_repo.list_jobs(limit=1, offset=1)] == ["job-1"]

    with pytest.raises(ValueError):
        job_repo.list_jobs(limit=-1)
    with pytest.raises(ValueError):
        job_repo.list_jobs(offset=-1)

    # Cascade delete removes tracks, speakers, and segments
    seg_repo.replace_segments(
        "job-1",
        [
            {
                "track_id": "trk-1",
                "seq": 0,
                "start_ms": 0,
                "end_ms": 1000,
                "speaker_key": "SPEAKER_00",
                "raw_text": "Hola",
                "text": "Hola",
            }
        ],
    )
    assert job_repo.delete_job("job-1") is True
    assert job_repo.delete_job("job-1") is False
    assert job_repo.get_job("job-1") is None
    assert seg_repo.list_segments("job-1") == []
    assert seg_repo.list_speakers("job-1") == []


def test_segment_repo_effective_text_preserve_edited_and_speakers(tmp_path: Path) -> None:
    db_path = _init_db(tmp_path)
    conn = open_connection(db_path)
    try:
        job_repo = JobRepo(conn)
        seg_repo = SegmentRepo(conn)

        job_repo.create_job(
            job_id="job-seg",
            title="Test",
            mode="single",
            effective_config={},
        )
        job_repo.add_track(
            "job-seg",
            track_index=0,
            original_filename="clip.wav",
            media_relpath="media/job-seg/trk-1_clip.wav",
            speaker_key="SPEAKER_00",
            track_id="trk-1",
        )

        initial = seg_repo.replace_segments(
            "job-seg",
            [
                SegmentItem(
                    seq=0,
                    start_ms=0,
                    end_ms=1200,
                    speaker_key="SPEAKER_00",
                    raw_text="hola kuernavaca",
                    text="hola Cuernavaca",
                    words=[WordTiming(start_ms=0, end_ms=500, text="hola")],
                ),
                {
                    "track_id": "trk-1",
                    "seq": 1,
                    "start_ms": 1200,
                    "end_ms": 2500,
                    "speaker_key": "SPEAKER_00",
                    "raw_text": "segundo segmento",
                    "text": "Segundo segmento",
                    "edited_text": "Segundo segmento editado",
                    "is_edited": True,
                },
            ],
        )
        assert len(initial) == 2
        assert initial[0].raw_text == "hola kuernavaca"
        assert initial[0].text == "hola Cuernavaca"
        assert initial[0].is_edited is False
        assert initial[0].words == [{"start_ms": 0, "end_ms": 500, "text": "hola"}]

        assert initial[1].raw_text == "segundo segmento"
        assert initial[1].base_text == "Segundo segmento"
        assert initial[1].edited_text == "Segundo segmento editado"
        assert initial[1].text == "Segundo segmento editado"
        assert initial[1].is_edited is True

        # Re-replacing with preserve_edited=True keeps manual edit on seq=1
        reapplied = seg_repo.replace_segments(
            "job-seg",
            [
                {
                    "seq": 0,
                    "start_ms": 0,
                    "end_ms": 1200,
                    "speaker_key": "SPEAKER_01",
                    "raw_text": "hola kuernavaca",
                    "text": "Hola Cuernavaca!",
                },
                {
                    "seq": 1,
                    "start_ms": 1250,
                    "end_ms": 2500,
                    "speaker_key": "SPEAKER_01",
                    "raw_text": "segundo segmento",
                    "text": "Segundo segmento actualizado",
                },
            ],
            preserve_edited=True,
        )
        assert reapplied[0].text == "Hola Cuernavaca!"
        assert reapplied[0].is_edited is False
        assert reapplied[1].base_text == "Segundo segmento actualizado"
        assert reapplied[1].text == "Segundo segmento editado"
        assert reapplied[1].is_edited is True

        # Re-replacing with preserve_edited=False discards manual edits
        reset = seg_repo.replace_segments(
            "job-seg",
            [
                {
                    "seq": 1,
                    "start_ms": 1250,
                    "end_ms": 2500,
                    "speaker_key": "SPEAKER_01",
                    "raw_text": "segundo segmento",
                    "text": "Segundo segmento limpio",
                }
            ],
            preserve_edited=False,
        )
        assert len(reset) == 1
        assert reset[0].text == "Segundo segmento limpio"
        assert reset[0].edited_text is None
        assert reset[0].is_edited is False

        # Speakers upsert + rename
        speakers = seg_repo.upsert_speakers(
            "job-seg",
            [
                {"speaker_key": "SPEAKER_00", "display_name": "Hablante 1"},
                {"speaker_key": "SPEAKER_01", "display_name": "Hablante 2", "track_id": "trk-1"},
            ],
        )
        assert [s.display_name for s in speakers] == ["Hablante 1", "Hablante 2"]

        renamed = seg_repo.rename_speaker("job-seg", "SPEAKER_01", "  Narrador  ")
        assert renamed is not None
        assert renamed.display_name == "Narrador"
        assert seg_repo.rename_speaker("job-seg", "SPEAKER_99", "Nadie") is None

        with pytest.raises(ValueError):
            seg_repo.rename_speaker("job-seg", "SPEAKER_01", "   ")
    finally:
        conn.close()


def test_startup_recovery_cleans_tmp_and_reconciles_all_job_states(tmp_path: Path) -> None:
    db_path = _init_db(tmp_path)
    tmp_root = tmp_path / "tmp"

    conn = open_connection(db_path)
    try:
        job_repo = JobRepo(conn)

        # 1. Empty or missing tmp_root should not fail
        summary_empty = run_startup_recovery(conn, tmp_root)
        assert summary_empty == {
            "tmp_dirs_removed": 0,
            "failed": 0,
            "completed_partial": 0,
            "cancelled": 0,
            "total_jobs_recovered": 0,
        }

        tmp_root.mkdir()
        assert run_startup_recovery(conn, tmp_root)["tmp_dirs_removed"] == 0

        # Populate orphan subdirectories and stray files under tmp_root
        (tmp_root / "job-prep").mkdir()
        (tmp_root / "job-prep" / "audio.wav").write_bytes(b"wav")
        (tmp_root / "job-trans").mkdir()
        (tmp_root / ".orphan.tmp").write_text("temp", encoding="utf-8")

        # Job in preprocessing (with 1 completed track checkpoint that must remain untouched)
        job_repo.create_job(
            job_id="job-prep",
            title="Prep",
            mode="single",
            status="preprocessing",
            effective_config={},
        )
        job_repo.add_track(
            "job-prep",
            0,
            "a.wav",
            "media/job-prep/t1.wav",
            "SPEAKER_00",
            track_id="t-prep-1",
            asr_status="completed",
            asr_checkpoint_relpath="checkpoints/job-prep/t-prep-1.asr.json",
        )

        # Job in transcribing (1 completed track, 1 pending track)
        job_repo.create_job(
            job_id="job-trans",
            title="Trans",
            mode="multitrack",
            status="transcribing",
            effective_config={},
        )
        job_repo.add_track(
            "job-trans",
            0,
            "a.wav",
            "media/job-trans/t1.wav",
            "SPEAKER_00",
            track_id="t-trans-1",
            asr_status="completed",
            asr_checkpoint_relpath="checkpoints/job-trans/t-trans-1.asr.json",
        )
        job_repo.add_track(
            "job-trans",
            1,
            "b.wav",
            "media/job-trans/t2.wav",
            "SPEAKER_01",
            track_id="t-trans-2",
            asr_status="running",
        )

        # Job in diarizing with all tracks completed -> completed_partial + DIARIZATION_INTERRUPTED
        job_repo.create_job(
            job_id="job-diar-ok",
            title="Diar OK",
            mode="single",
            status="diarizing",
            diarization_status="running",
            effective_config={},
        )
        job_repo.add_track(
            "job-diar-ok",
            0,
            "c.wav",
            "media/job-diar-ok/t1.wav",
            "SPEAKER_00",
            track_id="t-diar-1",
            asr_status="completed",
            asr_checkpoint_relpath="checkpoints/job-diar-ok/t-diar-1.asr.json",
        )

        # Job in merging with all tracks completed -> completed_partial + SERVER_INTERRUPTED
        job_repo.create_job(
            job_id="job-merge-ok",
            title="Merge OK",
            mode="single",
            status="merging",
            diarization_status="none",
            effective_config={},
        )
        job_repo.add_track(
            "job-merge-ok",
            0,
            "d.wav",
            "media/job-merge-ok/t1.wav",
            "SPEAKER_00",
            track_id="t-merge-1",
            asr_status="completed",
            asr_checkpoint_relpath="checkpoints/job-merge-ok/t-merge-1.asr.json",
        )

        # Job in exporting with incomplete ASR -> failed + SERVER_INTERRUPTED
        job_repo.create_job(
            job_id="job-export-incomplete",
            title="Export Incomplete",
            mode="single",
            status="exporting",
            diarization_status="running",
            effective_config={},
        )
        job_repo.add_track(
            "job-export-incomplete",
            0,
            "e.wav",
            "media/job-export-incomplete/t1.wav",
            "SPEAKER_00",
            track_id="t-exp-1",
            asr_status="pending",
        )

        # Job in cancelling -> cancelled
        job_repo.create_job(
            job_id="job-cancel",
            title="Cancel",
            mode="single",
            status="cancelling",
            effective_config={},
        )

        # Jobs that must remain untouched
        job_repo.create_job(
            job_id="job-queued",
            title="Queued",
            mode="single",
            status="queued",
            effective_config={},
        )
        job_repo.create_job(
            job_id="job-done",
            title="Done",
            mode="single",
            status="completed",
            effective_config={},
        )

        summary = run_startup_recovery(conn, tmp_root)
        assert summary == {
            "tmp_dirs_removed": 2,
            "failed": 3,
            "completed_partial": 2,
            "cancelled": 1,
            "total_jobs_recovered": 6,
        }

        assert tmp_root.is_dir()
        assert list(tmp_root.iterdir()) == []

        prep = job_repo.get_job("job-prep")
        assert prep is not None
        assert prep.status == "failed"
        assert prep.error_code == "SERVER_INTERRUPTED"
        assert prep.tracks[0].asr_status == "completed"
        assert prep.tracks[0].asr_checkpoint_relpath == "checkpoints/job-prep/t-prep-1.asr.json"

        trans = job_repo.get_job("job-trans")
        assert trans is not None
        assert trans.status == "failed"
        assert trans.error_code == "SERVER_INTERRUPTED"
        assert trans.tracks[0].asr_status == "completed"
        assert trans.tracks[0].asr_checkpoint_relpath == "checkpoints/job-trans/t-trans-1.asr.json"

        diar_ok = job_repo.get_job("job-diar-ok")
        assert diar_ok is not None
        assert diar_ok.status == "completed_partial"
        assert diar_ok.diarization_status == "failed"
        assert diar_ok.error_code == "DIARIZATION_INTERRUPTED"

        merge_ok = job_repo.get_job("job-merge-ok")
        assert merge_ok is not None
        assert merge_ok.status == "completed_partial"
        assert merge_ok.diarization_status == "none"
        assert merge_ok.error_code == "SERVER_INTERRUPTED"

        exp_inc = job_repo.get_job("job-export-incomplete")
        assert exp_inc is not None
        assert exp_inc.status == "failed"
        assert exp_inc.diarization_status == "failed"
        assert exp_inc.error_code == "SERVER_INTERRUPTED"

        cancelled = job_repo.get_job("job-cancel")
        assert cancelled is not None
        assert cancelled.status == "cancelled"

        assert job_repo.get_job("job-queued").status == "queued"  # type: ignore[union-attr]
        assert job_repo.get_job("job-done").status == "completed"  # type: ignore[union-attr]
    finally:
        conn.close()
