"""Tests for aurisdeck.db connection PRAGMAs, SQL schema, and migration runner."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from aurisdeck.db import open_connection, run_migrations
from aurisdeck.db.migrations import SQL_DIR


def test_open_connection_sets_required_pragmas_on_every_connection(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "aurisdeck.db"

    conn1 = open_connection(db_path)
    try:
        assert conn1.row_factory is sqlite3.Row
        assert conn1.execute("PRAGMA journal_mode;").fetchone()[0].lower() == "wal"
        assert conn1.execute("PRAGMA synchronous;").fetchone()[0] == 1
        assert conn1.execute("PRAGMA busy_timeout;").fetchone()[0] == 5000
        assert conn1.execute("PRAGMA foreign_keys;").fetchone()[0] == 1
    finally:
        conn1.close()

    # foreign_keys is per-connection in SQLite and must be re-applied on reconnect
    conn2 = open_connection(db_path)
    try:
        assert conn2.execute("PRAGMA journal_mode;").fetchone()[0].lower() == "wal"
        assert conn2.execute("PRAGMA synchronous;").fetchone()[0] == 1
        assert conn2.execute("PRAGMA busy_timeout;").fetchone()[0] == 5000
        assert conn2.execute("PRAGMA foreign_keys;").fetchone()[0] == 1
    finally:
        conn2.close()


def test_run_migrations_is_idempotent_and_seeds_default_user(tmp_path: Path) -> None:
    db_path = tmp_path / "aurisdeck.db"

    first_applied = run_migrations(db_path)
    assert first_applied == ["0001_initial", "0002_system_settings"]
    assert not list(tmp_path.glob("aurisdeck.db.bak.*"))

    second_applied = run_migrations(db_path)
    assert second_applied == []
    assert not list(tmp_path.glob("aurisdeck.db.bak.*"))

    conn = open_connection(db_path)
    try:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {
            "schema_migrations",
            "users",
            "jobs",
            "tracks",
            "transcript_segments",
            "job_speakers",
        }.issubset(tables)

        indexes = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        assert {"idx_jobs_status", "idx_jobs_batch_id", "idx_segments_job_seq"}.issubset(indexes)

        migrations = conn.execute("SELECT version FROM schema_migrations").fetchall()
        assert [row["version"] for row in migrations] == [
            "0001_initial",
            "0002_system_settings",
        ]

        user = conn.execute("SELECT * FROM users WHERE id = 'default'").fetchone()
        assert user is not None
        assert user["username"] == "default"
        assert user["display_name"] == "Default User"
        assert user["role"] == "admin"
        assert user["password_hash"] is None
        assert user["api_token_hash"] is None
        assert json.loads(user["preferences_json"]) == {
            "locale": "es",
            "default_profile_id": "es-tecnico",
            "player": {"playback_rate": 1.0, "auto_scroll": True},
        }
    finally:
        conn.close()


def test_foreign_keys_checks_and_cascade_delete_on_job(tmp_path: Path) -> None:
    db_path = tmp_path / "aurisdeck.db"
    run_migrations(db_path)

    conn = open_connection(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO jobs (id, owner_user_id, title, mode, status, effective_config_json, created_at, updated_at)
                VALUES ('bad-owner', 'missing-user', 'Test', 'single', 'queued', '{}', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
                """
            )

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO jobs (id, title, mode, status, effective_config_json, created_at, updated_at)
                VALUES ('bad-mode', 'Test', 'invalid', 'queued', '{}', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
                """
            )

        with conn:
            conn.execute(
                """
                INSERT INTO jobs (id, title, mode, status, effective_config_json, created_at, updated_at)
                VALUES ('job-1', 'Meeting', 'multitrack', 'queued', '{}', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
                """
            )
            conn.execute(
                """
                INSERT INTO tracks (id, job_id, track_index, original_filename, media_relpath, speaker_key)
                VALUES
                    ('trk-1', 'job-1', 0, 'a.ogg', 'media/job-1/trk-1_a.ogg', 'SPEAKER_00'),
                    ('trk-2', 'job-1', 1, 'b.ogg', 'media/job-1/trk-2_b.ogg', 'SPEAKER_01')
                """
            )
            conn.execute(
                """
                INSERT INTO job_speakers (job_id, speaker_key, display_name, track_id)
                VALUES
                    ('job-1', 'SPEAKER_00', 'Ana', 'trk-1'),
                    ('job-1', 'SPEAKER_01', 'Luis', 'trk-2')
                """
            )
            conn.execute(
                """
                INSERT INTO transcript_segments (job_id, track_id, seq, start_ms, end_ms, speaker_key, raw_text, text)
                VALUES
                    ('job-1', 'trk-1', 0, 0, 1000, 'SPEAKER_00', 'Hola', 'Hola'),
                    ('job-1', 'trk-2', 1, 1000, 2000, 'SPEAKER_01', 'Que tal', 'Que tal'),
                    ('job-1', NULL, 2, 2000, 3000, 'SPEAKER_00', 'Bien', 'Bien')
                """
            )

        # Deleting trk-1 sets speaker track_id to NULL and cascades trk-1 segments
        with conn:
            conn.execute("DELETE FROM tracks WHERE id = 'trk-1'")

        speaker_00 = conn.execute(
            "SELECT track_id FROM job_speakers WHERE job_id = 'job-1' AND speaker_key = 'SPEAKER_00'"
        ).fetchone()
        assert speaker_00 is not None
        assert speaker_00["track_id"] is None
        assert (
            conn.execute("SELECT COUNT(*) FROM transcript_segments WHERE job_id = 'job-1'").fetchone()[0]
            == 2
        )

        # Deleting the job cascades to remaining tracks, transcript_segments, and job_speakers
        with conn:
            conn.execute("DELETE FROM jobs WHERE id = 'job-1'")

        assert conn.execute("SELECT COUNT(*) FROM tracks WHERE job_id = 'job-1'").fetchone()[0] == 0
        assert (
            conn.execute("SELECT COUNT(*) FROM transcript_segments WHERE job_id = 'job-1'").fetchone()[0]
            == 0
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM job_speakers WHERE job_id = 'job-1'").fetchone()[0]
            == 0
        )
    finally:
        conn.close()


def test_run_migrations_creates_backup_before_pending_migration_and_rolls_back_on_error(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "aurisdeck.db"
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir()
    shutil.copy2(SQL_DIR / "0001_initial.sql", sql_dir / "0001_initial.sql")

    assert run_migrations(db_path, sql_dir=sql_dir) == ["0001_initial"]

    conn = open_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO jobs (id, title, mode, status, effective_config_json, created_at, updated_at)
                VALUES ('job-keep', 'Existing', 'single', 'completed', '{}', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
                """
            )
    finally:
        conn.close()

    (sql_dir / "0002_add_notes.sql").write_text(
        "CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT NOT NULL);\n",
        encoding="utf-8",
    )

    assert run_migrations(db_path, sql_dir=sql_dir) == ["0002_add_notes"]

    backup_path = tmp_path / "aurisdeck.db.bak.0001_initial"
    assert backup_path.is_file()

    # Backup preserves pre-migration data (including WAL writes) without 0002 schema
    bak_conn = sqlite3.connect(backup_path)
    bak_conn.row_factory = sqlite3.Row
    try:
        assert bak_conn.execute("SELECT id FROM jobs WHERE id = 'job-keep'").fetchone() is not None
        bak_tables = {
            row["name"]
            for row in bak_conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert "notes" not in bak_tables
    finally:
        bak_conn.close()

    # Failed 0003 migration must roll back completely
    (sql_dir / "0003_broken.sql").write_text(
        "CREATE TABLE should_rollback (id INTEGER PRIMARY KEY);\nINVALID SQL STATEMENT;\n",
        encoding="utf-8",
    )
    with pytest.raises(sqlite3.Error):
        run_migrations(db_path, sql_dir=sql_dir)

    conn = open_connection(db_path)
    try:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert "notes" in tables
        assert "should_rollback" not in tables
        versions = [
            row["version"]
            for row in conn.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
        ]
        assert versions == ["0001_initial", "0002_add_notes"]
    finally:
        conn.close()
