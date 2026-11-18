"""Typed SQLite repositories for users, jobs, tracks, transcript segments, and speakers."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aurisdeck.db.connection import open_connection

type ConnectionSource = Path | str | sqlite3.Connection | Callable[[], sqlite3.Connection]


class _UnsetType:
    """Sentinel distinguishing omitted optional arguments from explicit None."""


_UNSET = _UnsetType()


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 Z format."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _attr_or_key(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _serialize_words(words: Any) -> list[dict[str, Any]]:
    if not words:
        return []
    serialized: list[dict[str, Any]] = []
    for word in words:
        if isinstance(word, Mapping):
            serialized.append(dict(word))
        elif hasattr(word, "model_dump"):
            serialized.append(word.model_dump())
        else:
            serialized.append(
                {
                    "start_ms": int(getattr(word, "start_ms", 0)),
                    "end_ms": int(getattr(word, "end_ms", 0)),
                    "text": str(getattr(word, "text", "")),
                }
            )
    return serialized


@dataclass(slots=True)
class UserRecord:
    """Typed representation of a row in the users table."""

    id: str
    username: str
    display_name: str
    role: str
    password_hash: str | None
    api_token_hash: str | None
    preferences: dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(slots=True)
class TrackRecord:
    """Typed representation of a row in the tracks table."""

    id: str
    job_id: str
    track_index: int
    original_filename: str
    media_relpath: str
    duration_ms: int
    offset_ms: int
    speaker_key: str
    asr_status: str
    asr_checkpoint_relpath: str | None


@dataclass(slots=True)
class SpeakerRecord:
    """Typed representation of a row in the job_speakers table."""

    job_id: str
    speaker_key: str
    display_name: str
    track_id: str | None


@dataclass(slots=True)
class JobRecord:
    """Typed representation of a job with its associated tracks and speakers."""

    id: str
    owner_user_id: str
    visibility: str
    batch_id: str | None
    collection_id: str | None
    title: str
    mode: str
    status: str
    diarization_status: str
    language_requested: str | None
    language_detected: str | None
    source_type: str
    source_fingerprint: str | None
    media_retained: bool
    effective_config: dict[str, Any]
    error_code: str | None
    error_params: dict[str, Any] | None
    error_detail: str | None
    created_at: str
    updated_at: str
    completed_at: str | None
    tracks: list[TrackRecord]
    speakers: list[SpeakerRecord]


@dataclass(slots=True)
class SegmentRecord:
    """Typed representation of a transcript segment with effective text resolved."""

    id: int
    job_id: str
    track_id: str | None
    seq: int
    start_ms: int
    end_ms: int
    speaker_key: str
    raw_text: str
    text: str
    base_text: str
    edited_text: str | None
    is_edited: bool
    words: list[dict[str, Any]]


def _row_to_user(row: sqlite3.Row) -> UserRecord:
    return UserRecord(
        id=row["id"],
        username=row["username"],
        display_name=row["display_name"],
        role=row["role"],
        password_hash=row["password_hash"],
        api_token_hash=row["api_token_hash"],
        preferences=json.loads(row["preferences_json"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_track(row: sqlite3.Row) -> TrackRecord:
    return TrackRecord(
        id=row["id"],
        job_id=row["job_id"],
        track_index=row["track_index"],
        original_filename=row["original_filename"],
        media_relpath=row["media_relpath"],
        duration_ms=row["duration_ms"],
        offset_ms=row["offset_ms"],
        speaker_key=row["speaker_key"],
        asr_status=row["asr_status"],
        asr_checkpoint_relpath=row["asr_checkpoint_relpath"],
    )


def _row_to_speaker(row: sqlite3.Row) -> SpeakerRecord:
    return SpeakerRecord(
        job_id=row["job_id"],
        speaker_key=row["speaker_key"],
        display_name=row["display_name"],
        track_id=row["track_id"],
    )


def _row_to_job(
    row: sqlite3.Row,
    tracks: list[TrackRecord],
    speakers: list[SpeakerRecord],
) -> JobRecord:
    error_params_raw = row["error_params_json"]
    return JobRecord(
        id=row["id"],
        owner_user_id=row["owner_user_id"],
        visibility=row["visibility"],
        batch_id=row["batch_id"],
        collection_id=row["collection_id"],
        title=row["title"],
        mode=row["mode"],
        status=row["status"],
        diarization_status=row["diarization_status"],
        language_requested=row["language_requested"],
        language_detected=row["language_detected"],
        source_type=row["source_type"],
        source_fingerprint=row["source_fingerprint"],
        media_retained=bool(row["media_retained"]),
        effective_config=json.loads(row["effective_config_json"]),
        error_code=row["error_code"],
        error_params=json.loads(error_params_raw) if error_params_raw is not None else None,
        error_detail=row["error_detail"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
        tracks=tracks,
        speakers=speakers,
    )


def _row_to_segment(row: sqlite3.Row) -> SegmentRecord:
    edited_text = row["edited_text"]
    base_text = row["text"]
    effective_text = edited_text or base_text
    return SegmentRecord(
        id=row["id"],
        job_id=row["job_id"],
        track_id=row["track_id"],
        seq=row["seq"],
        start_ms=row["start_ms"],
        end_ms=row["end_ms"],
        speaker_key=row["speaker_key"],
        raw_text=row["raw_text"],
        text=effective_text,
        base_text=base_text,
        edited_text=edited_text,
        is_edited=bool(row["is_edited"]),
        words=json.loads(row["words_json"]),
    )


class _BaseRepo:
    """Shared SQLite connection lifecycle helper enforcing short WAL transactions."""

    def __init__(self, db_path: ConnectionSource) -> None:
        self._db_path = db_path

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        if isinstance(self._db_path, sqlite3.Connection):
            yield self._db_path
            return
        conn = self._db_path() if callable(self._db_path) else open_connection(self._db_path)
        try:
            yield conn
        finally:
            conn.close()


class UserRepo(_BaseRepo):
    """Repository for querying and updating user credentials and preferences."""

    def get_by_id(self, user_id: str) -> UserRecord | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return _row_to_user(row) if row is not None else None

    def get_by_username(self, username: str) -> UserRecord | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            return _row_to_user(row) if row is not None else None

    def update_preferences(
        self,
        user_id: str,
        prefs_dict: Mapping[str, Any],
    ) -> UserRecord | None:
        payload = _dump_json(dict(prefs_dict))
        now = _utc_now()
        with self._connection() as conn, conn:
            cursor = conn.execute(
                "UPDATE users SET preferences_json = ?, updated_at = ? WHERE id = ?",
                (payload, now, user_id),
            )
            if cursor.rowcount == 0:
                return None
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return _row_to_user(row) if row is not None else None

    def set_password_hash(self, user_id: str, password_hash: str | None) -> bool:
        now = _utc_now()
        with self._connection() as conn, conn:
            cursor = conn.execute(
                "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
                (password_hash, now, user_id),
            )
            return cursor.rowcount > 0

    def set_api_token_hash(self, user_id: str, token_hash: str | None) -> bool:
        now = _utc_now()
        with self._connection() as conn, conn:
            cursor = conn.execute(
                "UPDATE users SET api_token_hash = ?, updated_at = ? WHERE id = ?",
                (token_hash, now, user_id),
            )
            return cursor.rowcount > 0

    def find_by_api_token_hash(self, token_hash: str) -> UserRecord | None:
        if not token_hash:
            return None
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE api_token_hash = ?",
                (token_hash,),
            ).fetchone()
            return _row_to_user(row) if row is not None else None

    def setup_initial_admin(
        self,
        username: str,
        password_hash: str,
        *,
        locale: str = "es",
    ) -> UserRecord:
        """Configure the initial admin user during first-run setup."""
        now = _utc_now()
        cleaned_username = username.strip()
        with self._connection() as conn, conn:
            existing_default = conn.execute(
                "SELECT * FROM users WHERE id = 'default'"
            ).fetchone()
            if existing_default is not None:
                prefs = json.loads(existing_default["preferences_json"])
                if not isinstance(prefs, dict):
                    prefs = {}
                prefs["locale"] = locale
                conn.execute(
                    """
                    UPDATE users
                    SET username = ?,
                        display_name = ?,
                        role = 'admin',
                        password_hash = ?,
                        preferences_json = ?,
                        updated_at = ?
                    WHERE id = 'default'
                    """,
                    (
                        cleaned_username,
                        cleaned_username,
                        password_hash,
                        _dump_json(prefs),
                        now,
                    ),
                )
                row = conn.execute("SELECT * FROM users WHERE id = 'default'").fetchone()
                assert row is not None
                return _row_to_user(row)

            prefs_json = _dump_json(
                {
                    "locale": locale,
                    "default_profile_id": "es-tecnico" if locale == "es" else "en-general",
                    "player": {"playback_rate": 1.0, "auto_scroll": True},
                }
            )
            conn.execute(
                """
                INSERT INTO users (
                    id, username, display_name, role, password_hash,
                    api_token_hash, preferences_json, created_at, updated_at
                )
                VALUES ('default', ?, ?, 'admin', ?, NULL, ?, ?, ?)
                """,
                (cleaned_username, cleaned_username, password_hash, prefs_json, now, now),
            )
            row = conn.execute("SELECT * FROM users WHERE id = 'default'").fetchone()
            assert row is not None
            return _row_to_user(row)


class SystemSettingsRepo(_BaseRepo):
    """Repository for global system settings stored exclusively in SQLite."""

    KEY_SETUP_COMPLETED = "setup_completed"
    KEY_HF_TOKEN = "hf_token"
    KEY_AUTH_MODE = "auth_mode"

    def get(self, key: str) -> str | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT value FROM system_settings WHERE key = ?",
                (key,),
            ).fetchone()
            return row["value"] if row is not None else None

    def set(self, key: str, value: str | None) -> None:
        now = _utc_now()
        with self._connection() as conn, conn:
            if value is None:
                conn.execute("DELETE FROM system_settings WHERE key = ?", (key,))
            else:
                conn.execute(
                    """
                    INSERT INTO system_settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                    """,
                    (key, value, now),
                )

    def is_setup_completed(self) -> bool:
        return self.get(self.KEY_SETUP_COMPLETED) == "1"

    def set_setup_completed(self, completed: bool = True) -> None:
        self.set(self.KEY_SETUP_COMPLETED, "1" if completed else "0")

    def get_hf_token(self) -> str | None:
        raw = self.get(self.KEY_HF_TOKEN)
        if raw is None:
            return None
        cleaned = raw.strip()
        return cleaned or None

    def set_hf_token(self, token: str | None) -> None:
        cleaned = token.strip() if token else None
        self.set(self.KEY_HF_TOKEN, cleaned)

    def get_auth_mode(self) -> str | None:
        return self.get(self.KEY_AUTH_MODE)

    def set_auth_mode(self, mode: str) -> None:
        self.set(self.KEY_AUTH_MODE, mode)


class JobRepo(_BaseRepo):
    """Repository for managing transcription jobs and their tracks."""

    def create_job(
        self,
        title: str,
        mode: str,
        effective_config: Mapping[str, Any] | str,
        *,
        job_id: str | None = None,
        status: str = "queued",
        diarization_status: str = "none",
        owner_user_id: str = "default",
        visibility: str = "shared",
        batch_id: str | None = None,
        collection_id: str | None = None,
        language_requested: str | None = None,
        language_detected: str | None = None,
        source_type: str = "upload",
        source_fingerprint: str | None = None,
        media_retained: bool = True,
        error_code: str | None = None,
        error_params: Mapping[str, Any] | None = None,
        error_detail: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        completed_at: str | None = None,
    ) -> JobRecord:
        assigned_id = job_id or uuid.uuid4().hex
        now = _utc_now()
        created = created_at or now
        updated = updated_at or created
        config_json = (
            effective_config
            if isinstance(effective_config, str)
            else _dump_json(dict(effective_config))
        )
        error_params_json = (
            _dump_json(dict(error_params)) if error_params is not None else None
        )

        with self._connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        id, owner_user_id, visibility, batch_id, collection_id,
                        title, mode, status, diarization_status,
                        language_requested, language_detected,
                        source_type, source_fingerprint, media_retained,
                        effective_config_json, error_code, error_params_json, error_detail,
                        created_at, updated_at, completed_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        assigned_id,
                        owner_user_id,
                        visibility,
                        batch_id,
                        collection_id,
                        title,
                        mode,
                        status,
                        diarization_status,
                        language_requested,
                        language_detected,
                        source_type,
                        source_fingerprint,
                        int(bool(media_retained)),
                        config_json,
                        error_code,
                        error_params_json,
                        error_detail,
                        created,
                        updated,
                        completed_at,
                    ),
                )
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (assigned_id,)).fetchone()
            assert row is not None
            return _row_to_job(row, tracks=[], speakers=[])

    def add_track(
        self,
        job_id: str,
        track_index: int,
        original_filename: str,
        media_relpath: str,
        speaker_key: str,
        *,
        track_id: str | None = None,
        duration_ms: int = 0,
        offset_ms: int = 0,
        asr_status: str = "pending",
        asr_checkpoint_relpath: str | None = None,
    ) -> TrackRecord:
        assigned_id = track_id or uuid.uuid4().hex
        with self._connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO tracks (
                        id, job_id, track_index, original_filename, media_relpath,
                        duration_ms, offset_ms, speaker_key, asr_status, asr_checkpoint_relpath
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        assigned_id,
                        job_id,
                        track_index,
                        original_filename,
                        media_relpath,
                        duration_ms,
                        offset_ms,
                        speaker_key,
                        asr_status,
                        asr_checkpoint_relpath,
                    ),
                )
            row = conn.execute("SELECT * FROM tracks WHERE id = ?", (assigned_id,)).fetchone()
            assert row is not None
            return _row_to_track(row)

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._connection() as conn:
            job_row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job_row is None:
                return None

            track_rows = conn.execute(
                "SELECT * FROM tracks WHERE job_id = ? ORDER BY track_index ASC, id ASC",
                (job_id,),
            ).fetchall()
            speaker_rows = conn.execute(
                "SELECT * FROM job_speakers WHERE job_id = ? ORDER BY speaker_key ASC",
                (job_id,),
            ).fetchall()
            return _row_to_job(
                job_row,
                tracks=[_row_to_track(row) for row in track_rows],
                speakers=[_row_to_speaker(row) for row in speaker_rows],
            )

    def list_jobs(
        self,
        owner_user_id: str | None = None,
        batch_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[JobRecord]:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        if offset < 0:
            raise ValueError("offset must be >= 0")

        clauses: list[str] = []
        params: list[Any] = []
        if owner_user_id is not None:
            clauses.append("owner_user_id = ?")
            params.append(owner_user_id)
        if batch_id is not None:
            clauses.append("batch_id = ?")
            params.append(batch_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            f"SELECT * FROM jobs {where_sql} "
            "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        with self._connection() as conn:
            job_rows = conn.execute(query, params).fetchall()
            if not job_rows:
                return []

            job_ids = [row["id"] for row in job_rows]
            placeholders = ", ".join("?" for _ in job_ids)

            tracks_by_job: dict[str, list[TrackRecord]] = {jid: [] for jid in job_ids}
            track_rows = conn.execute(
                f"SELECT * FROM tracks WHERE job_id IN ({placeholders}) "
                "ORDER BY track_index ASC, id ASC",
                job_ids,
            ).fetchall()
            for row in track_rows:
                tracks_by_job[row["job_id"]].append(_row_to_track(row))

            speakers_by_job: dict[str, list[SpeakerRecord]] = {jid: [] for jid in job_ids}
            speaker_rows = conn.execute(
                f"SELECT * FROM job_speakers WHERE job_id IN ({placeholders}) "
                "ORDER BY speaker_key ASC",
                job_ids,
            ).fetchall()
            for row in speaker_rows:
                speakers_by_job[row["job_id"]].append(_row_to_speaker(row))

            return [
                _row_to_job(
                    row,
                    tracks=tracks_by_job[row["id"]],
                    speakers=speakers_by_job[row["id"]],
                )
                for row in job_rows
            ]

    def update_job_status(
        self,
        job_id: str,
        status: str,
        diarization_status: str | None | _UnsetType = _UNSET,
        error_code: str | None | _UnsetType = _UNSET,
        error_params: Mapping[str, Any] | None | _UnsetType = _UNSET,
        error_detail: str | None | _UnsetType = _UNSET,
        language_detected: str | None | _UnsetType = _UNSET,
        completed_at: str | None | _UnsetType = _UNSET,
    ) -> JobRecord | None:
        assignments = ["status = ?", "updated_at = ?"]
        params: list[Any] = [status, _utc_now()]

        if not isinstance(diarization_status, _UnsetType) and diarization_status is not None:
            assignments.append("diarization_status = ?")
            params.append(diarization_status)
        if not isinstance(error_code, _UnsetType):
            assignments.append("error_code = ?")
            params.append(error_code)
        if not isinstance(error_params, _UnsetType):
            assignments.append("error_params_json = ?")
            params.append(_dump_json(dict(error_params)) if error_params is not None else None)
        if not isinstance(error_detail, _UnsetType):
            assignments.append("error_detail = ?")
            params.append(error_detail)
        if not isinstance(language_detected, _UnsetType):
            assignments.append("language_detected = ?")
            params.append(language_detected)
        if not isinstance(completed_at, _UnsetType):
            assignments.append("completed_at = ?")
            params.append(completed_at)

        params.append(job_id)

        with self._connection() as conn, conn:
            cursor = conn.execute(
                f"UPDATE jobs SET {', '.join(assignments)} WHERE id = ?",
                params,
            )
            if cursor.rowcount == 0:
                return None
        return self.get_job(job_id)

    def update_track_asr(
        self,
        track_id: str,
        asr_status: str,
        checkpoint_relpath: str | None,
        duration_ms: int | None = None,
    ) -> TrackRecord | None:
        assignments = ["asr_status = ?", "asr_checkpoint_relpath = ?"]
        params: list[Any] = [asr_status, checkpoint_relpath]
        if duration_ms is not None:
            assignments.append("duration_ms = ?")
            params.append(duration_ms)
        params.append(track_id)

        with self._connection() as conn, conn:
            cursor = conn.execute(
                f"UPDATE tracks SET {', '.join(assignments)} WHERE id = ?",
                params,
            )
            if cursor.rowcount == 0:
                return None
            row = conn.execute("SELECT * FROM tracks WHERE id = ?", (track_id,)).fetchone()
            return _row_to_track(row) if row is not None else None

    def delete_job(self, job_id: str) -> bool:
        with self._connection() as conn, conn:
            cursor = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            return cursor.rowcount > 0


class SegmentRepo(_BaseRepo):
    """Repository for transcript segments and per-job speaker aliases."""

    def replace_segments(
        self,
        job_id: str,
        segments: Sequence[Any],
        preserve_edited: bool = True,
    ) -> list[SegmentRecord]:
        rows_to_insert: list[tuple[Any, ...]] = []

        with self._connection() as conn, conn:
            edited_by_span: dict[tuple[str | None, int, int], str] = {}
            edited_by_seq: dict[int, str] = {}
            if preserve_edited:
                existing_edited = conn.execute(
                    """
                        SELECT seq, track_id, start_ms, end_ms, edited_text
                        FROM transcript_segments
                        WHERE job_id = ? AND is_edited = 1 AND edited_text IS NOT NULL
                        """,
                    (job_id,),
                ).fetchall()
                for row in existing_edited:
                    edited_by_span[(row["track_id"], row["start_ms"], row["end_ms"])] = row[
                        "edited_text"
                    ]
                    edited_by_seq[row["seq"]] = row["edited_text"]

            conn.execute("DELETE FROM transcript_segments WHERE job_id = ?", (job_id,))

            used_spans: set[tuple[str | None, int, int]] = set()
            for idx, item in enumerate(segments):
                seq_val = _attr_or_key(item, "seq")
                seq = idx if seq_val is None else int(seq_val)
                track_id = _attr_or_key(item, "track_id")
                start_ms = int(_attr_or_key(item, "start_ms", 0))
                end_ms = int(_attr_or_key(item, "end_ms", 0))
                speaker_key = _attr_or_key(item, "speaker_key") or ""
                raw_text = str(_attr_or_key(item, "raw_text", ""))
                text = str(_attr_or_key(item, "text", raw_text))
                incoming_edited = _attr_or_key(item, "edited_text")
                incoming_is_edited = bool(_attr_or_key(item, "is_edited", False))
                words_json = _dump_json(_serialize_words(_attr_or_key(item, "words")))

                edited_text = incoming_edited
                is_edited = incoming_is_edited or (incoming_edited is not None)

                if preserve_edited and not is_edited:
                    span_key = (track_id, start_ms, end_ms)
                    if span_key in edited_by_span and span_key not in used_spans:
                        edited_text = edited_by_span[span_key]
                        is_edited = True
                        used_spans.add(span_key)
                    elif seq in edited_by_seq:
                        edited_text = edited_by_seq[seq]
                        is_edited = True

                rows_to_insert.append(
                    (
                        job_id,
                        track_id,
                        seq,
                        start_ms,
                        end_ms,
                        speaker_key,
                        raw_text,
                        text,
                        edited_text,
                        int(is_edited),
                        words_json,
                    )
                )

            if rows_to_insert:
                conn.executemany(
                    """
                        INSERT INTO transcript_segments (
                            job_id, track_id, seq, start_ms, end_ms, speaker_key,
                            raw_text, text, edited_text, is_edited, words_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                    rows_to_insert,
                )

        return self.list_segments(job_id)

    def list_segments(self, job_id: str) -> list[SegmentRecord]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM transcript_segments
                WHERE job_id = ?
                ORDER BY seq ASC, start_ms ASC, id ASC
                """,
                (job_id,),
            ).fetchall()
            return [_row_to_segment(row) for row in rows]

    def upsert_speakers(
        self,
        job_id: str,
        speakers: Sequence[Any],
    ) -> list[SpeakerRecord]:
        records: list[tuple[str, str, str, str | None]] = []
        for item in speakers:
            speaker_key = str(_attr_or_key(item, "speaker_key"))
            display_name = str(_attr_or_key(item, "display_name") or speaker_key)
            track_id = _attr_or_key(item, "track_id")
            records.append((job_id, speaker_key, display_name, track_id))

        if records:
            with self._connection() as conn, conn:
                conn.executemany(
                    """
                        INSERT INTO job_speakers (job_id, speaker_key, display_name, track_id)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(job_id, speaker_key) DO UPDATE SET
                            display_name = excluded.display_name,
                            track_id = excluded.track_id
                        """,
                    records,
                )

        return self.list_speakers(job_id)

    def rename_speaker(
        self,
        job_id: str,
        speaker_key: str,
        display_name: str,
    ) -> SpeakerRecord | None:
        cleaned = display_name.strip()
        if not cleaned:
            raise ValueError("display_name cannot be empty")

        with self._connection() as conn, conn:
            cursor = conn.execute(
                """
                    UPDATE job_speakers
                    SET display_name = ?
                    WHERE job_id = ? AND speaker_key = ?
                    """,
                (cleaned, job_id, speaker_key),
            )
            if cursor.rowcount == 0:
                return None
            row = conn.execute(
                """
                    SELECT job_id, speaker_key, display_name, track_id
                    FROM job_speakers
                    WHERE job_id = ? AND speaker_key = ?
                    """,
                (job_id, speaker_key),
            ).fetchone()
            return _row_to_speaker(row) if row is not None else None

    def list_speakers(self, job_id: str) -> list[SpeakerRecord]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT job_id, speaker_key, display_name, track_id
                FROM job_speakers
                WHERE job_id = ?
                ORDER BY speaker_key ASC
                """,
                (job_id,),
            ).fetchall()
            return [_row_to_speaker(row) for row in rows]
