-- System-wide settings stored in SQLite (setup state, DB-only HF_TOKEN, auth mode)

CREATE TABLE IF NOT EXISTS system_settings (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT NOT NULL
);
