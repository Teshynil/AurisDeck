"""Startup recovery for cleaning orphan tmp directories and reconciling zombie jobs."""

from __future__ import annotations

import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from aurisdeck.db.connection import open_connection


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cleanup_tmp_root(tmp_root: Path) -> int:
    """Remove orphan subdirectories and stray entries inside tmp_root without deleting tmp_root."""
    if not tmp_root.exists() or not tmp_root.is_dir():
        return 0

    entries = list(tmp_root.iterdir())
    if not entries:
        return 0

    removed_dirs = 0
    for entry in entries:
        if entry.is_symlink() or not entry.is_dir():
            entry.unlink(missing_ok=True)
            continue
        shutil.rmtree(entry, ignore_errors=False)
        removed_dirs += 1
    return removed_dirs


def run_startup_recovery(
    conn: sqlite3.Connection | Path | str,
    tmp_root: Path | str,
) -> dict[str, int]:
    """Clean orphan temp directories and reconcile interrupted in-flight jobs on startup."""
    tmp_path = Path(tmp_root)
    tmp_dirs_removed = _cleanup_tmp_root(tmp_path)

    owns_conn = not isinstance(conn, sqlite3.Connection)
    db_conn = open_connection(conn) if owns_conn else conn
    now = _utc_now()

    failed_count = 0
    partial_count = 0
    cancelled_count = 0

    try:
        with db_conn:
            cursor = db_conn.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    error_code = 'SERVER_INTERRUPTED',
                    updated_at = ?
                WHERE status IN ('preprocessing', 'transcribing')
                """,
                (now,),
            )
            failed_count += cursor.rowcount

            in_post_asr = db_conn.execute(
                """
                SELECT id, status, diarization_status
                FROM jobs
                WHERE status IN ('diarizing', 'merging', 'exporting')
                """
            ).fetchall()

            for job_row in in_post_asr:
                job_id = job_row["id"]
                job_status = job_row["status"]
                diar_status = job_row["diarization_status"]

                counts = db_conn.execute(
                    """
                    SELECT
                        COUNT(*) AS total_tracks,
                        COALESCE(
                            SUM(CASE WHEN asr_status = 'completed' THEN 1 ELSE 0 END),
                            0
                        ) AS completed_tracks
                    FROM tracks
                    WHERE job_id = ?
                    """,
                    (job_id,),
                ).fetchone()

                total_tracks = int(counts["total_tracks"])
                completed_tracks = int(counts["completed_tracks"])
                all_asr_completed = total_tracks > 0 and completed_tracks == total_tracks

                if all_asr_completed:
                    if job_status == "diarizing":
                        next_diar = "failed"
                        error_code = "DIARIZATION_INTERRUPTED"
                    else:
                        next_diar = (
                            "failed" if diar_status in {"queued", "running"} else diar_status
                        )
                        error_code = "SERVER_INTERRUPTED"

                    db_conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'completed_partial',
                            diarization_status = ?,
                            error_code = ?,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (next_diar, error_code, now, job_id),
                    )
                    partial_count += 1
                else:
                    next_diar = (
                        "failed" if diar_status in {"queued", "running"} else diar_status
                    )
                    db_conn.execute(
                        """
                        UPDATE jobs
                        SET status = 'failed',
                            diarization_status = ?,
                            error_code = 'SERVER_INTERRUPTED',
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (next_diar, now, job_id),
                    )
                    failed_count += 1

            cancel_cursor = db_conn.execute(
                """
                UPDATE jobs
                SET status = 'cancelled',
                    updated_at = ?
                WHERE status = 'cancelling'
                """,
                (now,),
            )
            cancelled_count += cancel_cursor.rowcount
    finally:
        if owns_conn:
            db_conn.close()

    return {
        "tmp_dirs_removed": tmp_dirs_removed,
        "failed": failed_count,
        "completed_partial": partial_count,
        "cancelled": cancelled_count,
        "total_jobs_recovered": failed_count + partial_count + cancelled_count,
    }
