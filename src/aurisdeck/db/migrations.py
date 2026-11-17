"""Versioned SQL migration runner with automatic pre-migration backups."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from aurisdeck.db.connection import open_connection

SQL_DIR = Path(__file__).parent / "sql"


def _backup_database(conn: sqlite3.Connection, backup_path: Path) -> None:
    """Create a consistent SQLite backup copy (including WAL state)."""
    backup_conn = sqlite3.connect(backup_path)
    try:
        conn.backup(backup_conn)
    finally:
        backup_conn.close()


def run_migrations(db_path: Path | str, *, sql_dir: Path | None = None) -> list[str]:
    """Apply pending SQL migrations in filename order and return applied versions."""
    target_sql_dir = sql_dir if sql_dir is not None else SQL_DIR
    migration_files = sorted(target_sql_dir.glob("*.sql"), key=lambda path: path.name)

    db_path_str = str(db_path)
    is_file_db = db_path_str != ":memory:" and not db_path_str.startswith("file:")

    conn = open_connection(db_path)
    try:
        existing_tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        had_existing_tables = bool(existing_tables)

        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )

        applied_rows = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version ASC"
        ).fetchall()
        applied_versions = [row["version"] for row in applied_rows]
        applied_set = set(applied_versions)

        pending = [path for path in migration_files if path.stem not in applied_set]
        if not pending:
            return []

        if had_existing_tables and is_file_db:
            file_db_path = Path(db_path)
            last_version = applied_versions[-1] if applied_versions else "0000"
            backup_path = file_db_path.with_name(f"{file_db_path.name}.bak.{last_version}")
            _backup_database(conn, backup_path)

        applied_now: list[str] = []
        for sql_file in pending:
            version = sql_file.stem
            sql_text = sql_file.read_text(encoding="utf-8")
            applied_at = datetime.now(UTC).isoformat()
            try:
                conn.executescript(f"BEGIN IMMEDIATE;\n{sql_text}\n")
                conn.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (version, applied_at),
                )
                conn.commit()
            except Exception:
                if conn.in_transaction:
                    conn.rollback()
                raise
            applied_now.append(version)

        return applied_now
    finally:
        conn.close()
