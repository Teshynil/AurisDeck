"""SQLite connection factory enforcing WAL mode and per-connection PRAGMAs."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def open_connection(db_path: Path | str) -> sqlite3.Connection:
    """Open a SQLite connection configured with AurisDeck concurrency and FK PRAGMAs."""
    path_str = str(db_path)
    if path_str != ":memory:" and not path_str.startswith("file:"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path_str, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA foreign_keys = ON;")
    except Exception:
        conn.close()
        raise
    return conn
