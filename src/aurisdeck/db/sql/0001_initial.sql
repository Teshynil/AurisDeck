-- Initial AurisDeck SQLite schema (ARCHITECTURE.md §3)

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'admin',
    password_hash TEXT,
    api_token_hash TEXT,
    preferences_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

INSERT OR IGNORE INTO users (
    id,
    username,
    display_name,
    role,
    password_hash,
    api_token_hash,
    preferences_json,
    created_at,
    updated_at
)
VALUES (
    'default',
    'default',
    'Default User',
    'admin',
    NULL,
    NULL,
    '{"locale":"es","default_profile_id":"es-tecnico","player":{"playback_rate":1.0,"auto_scroll":true}}',
    strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
    strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL DEFAULT 'default' REFERENCES users(id),
    visibility TEXT NOT NULL DEFAULT 'shared' CHECK (visibility IN ('private', 'shared')),
    batch_id TEXT,
    collection_id TEXT,
    title TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('single', 'multitrack', 'batch')),
    status TEXT NOT NULL,
    diarization_status TEXT NOT NULL DEFAULT 'none',
    language_requested TEXT,
    language_detected TEXT,
    source_type TEXT NOT NULL DEFAULT 'upload',
    source_fingerprint TEXT,
    media_retained INTEGER NOT NULL DEFAULT 1,
    effective_config_json TEXT NOT NULL,
    error_code TEXT,
    error_params_json TEXT,
    error_detail TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS tracks (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    track_index INTEGER NOT NULL,
    original_filename TEXT NOT NULL,
    media_relpath TEXT NOT NULL,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    offset_ms INTEGER NOT NULL DEFAULT 0,
    speaker_key TEXT NOT NULL,
    asr_status TEXT NOT NULL DEFAULT 'pending',
    asr_checkpoint_relpath TEXT
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    track_id TEXT REFERENCES tracks(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    speaker_key TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    text TEXT NOT NULL,
    edited_text TEXT,
    is_edited INTEGER NOT NULL DEFAULT 0,
    words_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS job_speakers (
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    speaker_key TEXT NOT NULL,
    display_name TEXT NOT NULL,
    track_id TEXT REFERENCES tracks(id) ON DELETE SET NULL,
    PRIMARY KEY (job_id, speaker_key)
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_batch_id ON jobs(batch_id);
CREATE INDEX IF NOT EXISTS idx_segments_job_seq ON transcript_segments(job_id, seq);
