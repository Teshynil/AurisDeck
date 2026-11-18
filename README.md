# AurisDeck

Self-hosted, portable web platform and CLI for multi-GPU audio/video transcription with optional speaker diarization, contextual biasing (*initial prompt*, *hotwords*, glossaries, and post-ASR replacements), reusable JSON profiles, and W3C WebVTT export.

## Features

- **Browser-first & OS-independent UI:** Simple drag-and-drop workflow for everyday use plus an Advanced view for managing JSON profiles, glossaries, replacement dictionaries, and hardware strategies.
- **Three input workflows:**
  - **Single (`single`):** Transcribe one audio or video file (`.wav`, `.flac`, `.mp3`, `.ogg`, `.opus`, `.m4a`, `.aac`, `.mp4`, `.mkv`, `.webm`, `.mov`) with optional speaker diarization.
  - **Multitrack (`multitrack`):** Transcribe separate per-speaker tracks (loose files or `.zip` archives) in parallel across available devices and merge them chronologically.
  - **Batch (`batch`):** Enqueue multiple independent files under a single batch and download all finished transcripts together as a `.zip` bundle.
- **Dynamic hardware scaling:** Automatic CUDA GPU discovery (`0`, `1`, or `N` devices) with isolated per-GPU worker processes and seamless CPU (`int8`) fallback.
- **Context & vocabulary control:** Native `faster-whisper` (`CTranslate2`) `initial_prompt` and `hotwords`, reusable glossaries (`.txt` / `.csv` import), and non-destructive post-ASR replacement rules.
- **Live progress & interactive viewer:** Real-time Server-Sent Events (`/api/v1/events`), synchronized audio/video playback, speaker renaming, and instant CPU-only re-export to `.vtt`, `.srt`, `.md`, and `.json`.
- **Extensible localization:** Built-in English (`en`) and Spanish (`es`) UI locales, extensible by dropping JSON files into `data/locales/`.

---

## Quick Start with Docker Compose

### 1. Configure environment

Copy `.env.example` to `.env` and adjust settings as needed:

```bash
cp .env.example .env
```

In `.env`, set `PUID` and `PGID` to match your host user (`id -u` and `id -g`) so files created under `./data` have your host ownership:

```dotenv
PUID=1000
PGID=1000
AURISDECK_PORT=8080
AURISDECK_AUTH_MODE=none
AURISDECK_AUTH_PASSWORD=
```

### 2. Start with NVIDIA GPU (`docker-compose.yml`)

Requires Docker with the NVIDIA Container Toolkit installed on the host:

```bash
docker compose up -d --build
```

To include optional speaker diarization dependencies (`pyannote.audio`, `torch`, `torchaudio`) inside the image, pass `INSTALL_DIARIZATION=true`:

```bash
INSTALL_DIARIZATION=true docker compose up -d --build
```

### 3. Start in CPU-only mode (`docker-compose.cpu.yml`)

For hosts without an NVIDIA GPU:

```bash
docker compose -f docker-compose.cpu.yml up -d --build
```

### 4. Access the application

Open `http://localhost:8080` in your browser (or check `GET /healthz`). Persistent data, SQLite database, profiles, models (`HF_HOME=/data/models`), and outputs are stored in `./data`.

---

## Quick Start Native with `uv`

### Prerequisites

- **`ffmpeg`** (including `ffprobe`) installed and available on `PATH`
- **`uv`** installed ([astral.sh/uv](https://docs.astral.sh/uv/))

The installer automatically provisions Python 3.12 into an isolated `.venv` using `uv`.

### Install

Base installation (ASR + Web UI + CLI):

```bash
./scripts/install.sh
```

With optional speaker diarization dependencies (`pyannote.audio`, `torch`, `torchaudio`):

```bash
./scripts/install.sh --with-diarization
```

### Run

```bash
source .venv/bin/activate
aurisdeck serve --host 0.0.0.0 --port 8080
```

Or without activating the virtual environment:

```bash
.venv/bin/aurisdeck serve --host 0.0.0.0 --port 8080
```

---

## Input Workflows

### Single (`single`)

Upload one audio or video file. Audio is automatically extracted as 16 kHz mono PCM WAV via `ffmpeg`. Optionally enable speaker diarization (`min_speakers` / `max_speakers`) to assign speaker turns (`SPEAKER_00`, `SPEAKER_01`, ...) and rename speakers in the viewer afterwards.

### Multitrack (`multitrack`)

Upload multiple per-speaker audio tracks or a `.zip` archive containing individual tracks:

- **Agnostic archive ingestion:** `.zip` archives are unpacked directly for supported audio/video tracks (with Zip-Slip protection), deriving initial speaker labels from filenames with `0 ms` default offsets.
- **Manual overrides:** You can inspect `.zip` contents before submission (`POST /api/v1/jobs/inspect-zip`) and override `speaker_aliases` or `track_offsets` per file.
- **Parallel execution:** Independent tracks are scheduled across all allowed CUDA/CPU workers simultaneously and merged chronologically into a single transcript with exact speaker attribution.

### Batch (`batch`)

Select multiple independent media files to create one job per file under a shared `batch_id`. Jobs run across available workers and completed results can be downloaded individually or together as a `.zip` bundle (`/api/v1/jobs/batches/{batch_id}/download.zip?fmt=vtt`).

---

## Context, Vocabulary, and Profiles

### JSON Profiles

Profiles live in `data/profiles/<id>.json` and store full presets for input mode, ASR model (`large-v3-turbo`, `medium`, etc.), language, compute type, VAD, diarization defaults, context, hardware selection, and subtitle formatting.

On first startup, four generic starter profiles are seeded automatically when `data/profiles/` is empty:

- `es-tecnico` — Spanish + Technical Terms (`single`, `v_tag` WebVTT + Markdown)
- `subtitulos-video` — Video Subtitles (`single`, `none` speaker style, compact subtitle line lengths, `.vtt` + `.srt`)
- `multipista-sync` — Synchronized Multitrack (`multitrack`, `.vtt` + `.md` + `.json`)
- `en-general` — English General (`single`, `.vtt` + `.md`)

### Glossaries, `hotwords`, and `initial_prompt`

- **`initial_prompt`:** Guides style, punctuation, and recurring domain terms at the start of decoding.
- **`hotwords`:** Passed directly to `faster-whisper` / CTranslate2 to boost recognition of names, acronyms, and specialized vocabulary.
- **Shared glossaries (`data/glossaries/`):** Reusable term lists (created in the UI, via API, or imported from `.txt` / `.csv` files) that are merged into the effective `hotwords` list for any linked profile or job.

### Post-ASR Replacements

Replacement dictionaries (`data/replacements/`) and per-profile `inline_replacements` apply deterministic text normalizations after ASR (for example correcting common phonetic misspellings) while preserving the original `raw_text` in SQLite and JSON exports.

### Export Formats

Jobs can export to `.vtt` (W3C WebVTT), `.srt`, `.md`, and `.json`. WebVTT and SRT speaker formatting supports three styles via `output.subtitles.voice_style`:

- `v_tag`: `<v Speaker>Text</v>` (WebVTT voice spans)
- `prefix`: `[Speaker]: Text`
- `none`: Plain text without speaker labels

---

## Optional Speaker Diarization (`HF_TOKEN`)

Speaker diarization for `single` mode uses `pyannote/speaker-diarization-3.1`.

1. Install diarization dependencies (`./scripts/install.sh --with-diarization` or `INSTALL_DIARIZATION=true` in Docker).
2. Accept the model conditions on Hugging Face for:
   - `pyannote/speaker-diarization-3.1`
   - `pyannote/segmentation-3.0`
3. Set `HF_TOKEN` in `.env` or your shell environment before starting AurisDeck:

```bash
export HF_TOKEN=hf_your_token_here
```

If ASR succeeds but diarization cannot run (for example due to a missing or unauthorized `HF_TOKEN`), AurisDeck saves the job as `completed_partial` with all ASR checkpoints and downloadable transcripts intact. Once `HF_TOKEN` is configured, you can trigger diarization later (`POST /api/v1/jobs/{id}/diarize` or **Diarize now** in the viewer) without re-running ASR.

---

## Extending UI Languages

AurisDeck ships with built-in `en` and `es` UI locales. To add a new language or override existing labels, place a `<lang>.json` file inside `data/locales/` (for example `data/locales/fr.json`) containing a `_meta` block (`code`, `name`, and `direction` set to `"ltr"` or `"rtl"`):

```json
{
  "_meta": {
    "code": "fr",
    "name": "Français",
    "direction": "ltr"
  },
  "nav": {
    "simple": "Transcrire",
    "library": "Bibliothèque",
    "viewer": "Lecteur",
    "advanced": "Avancé",
    "settings": "Paramètres"
  }
}
```

- Custom files in `data/locales/` are discovered automatically by `GET /api/v1/locales` and appear in Account Settings without rebuilding the application.
- If a file matches an existing built-in code (such as `es.json` or `en.json`), its keys are deep-merged on top of the built-in translations.

---

## CLI Quick Reference

Use `aurisdeck --help` (`-h`) or `aurisdeck <command> --help` for full option details.

### `aurisdeck serve`

Start the FastAPI web server and job coordinator:

```bash
aurisdeck serve --host 0.0.0.0 --port 8080 --data-dir ./data
```

### `aurisdeck transcribe`

Transcribe one or more files locally (using an in-process coordinator) or remotely against a running server (`--server`):

```bash
# Local single-file transcription
aurisdeck transcribe meeting.mp4 --profile es-tecnico --format vtt,md

# Local single-file with diarization and custom hotwords
aurisdeck transcribe interview.wav --lang en --diarize --hotwords "AurisDeck,CTranslate2" --gpus 0

# Multitrack ZIP archive
aurisdeck transcribe recording.zip --mode multitrack --format vtt,srt --out-dir ./exports

# Batch mode against a remote AurisDeck server
aurisdeck transcribe ep1.wav ep2.wav --mode batch --server http://localhost:8080 --token "$AURISDECK_TOKEN"
```

Supported flags for `aurisdeck transcribe`:

- `--profile SLUG`: Profile slug to load.
- `--mode [single|multitrack|batch]`: Input mode (auto-inferred from inputs when omitted).
- `--lang CODE`: Language code (e.g., `es`, `en`, `auto`).
- `--diarize`: Enable speaker diarization.
- `--hotwords TEXT`: Comma-separated vocabulary hotwords.
- `--gpus SPEC`: Device selection: `auto`, `cpu`, or comma-separated GPU indices (e.g., `0,1`).
- `--format FMT`: Output format(s) (`vtt`, `srt`, `md`, `json`); repeatable or comma-separated.
- `--out-dir DIR`: Output directory (defaults to `.` with `--server`, or next to each input locally).
- `--server URL`: Remote AurisDeck server URL (also read from `AURISDECK_SERVER_URL`).
- `--token TOKEN`: Bearer API token for `--server` (also read from `AURISDECK_TOKEN`).
- `--data-dir DIR`: Local data directory when running without `--server`.

### `aurisdeck profiles`

```bash
aurisdeck profiles list --data-dir ./data
```

### `aurisdeck auth`

```bash
# Reset stored password hash for a user
aurisdeck auth reset-password --username default --data-dir ./data

# Generate a Bearer API token (displayed once)
aurisdeck auth create-token --username default --data-dir ./data
```

### `aurisdeck pull-model`

Pre-download a `faster-whisper` model into `data/models/`:

```bash
aurisdeck pull-model large-v3-turbo --data-dir ./data
```

---

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `AURISDECK_DATA_DIR` | `./data` (`/data` in Docker) | Root persistent data directory (SQLite DB, profiles, media, checkpoints, outputs, models, locales). |
| `AURISDECK_AUTH_MODE` | `none` | Authentication mode: `none` or `password`. |
| `AURISDECK_AUTH_PASSWORD` | unset | Shared password used when `AURISDECK_AUTH_MODE=password`. |
| `AURISDECK_BASE_PATH` | `""` | Optional URL subpath when served behind a reverse proxy (e.g., `/aurisdeck`). |
| `AURISDECK_MODEL_IDLE_UNLOAD_SECONDS` | `120` | Seconds of worker inactivity before unloading ASR/diarization models from VRAM/RAM (`0` disables auto-unload). |
| `AURISDECK_CPU_THREADS` | `0` | Global CPU thread count for CTranslate2 (`0` = automatic). |
| `HF_TOKEN` | unset | Hugging Face Hub token (required only for gated `pyannote` diarization models). |
| `HF_ENDPOINT` | unset | Custom Hugging Face Hub URL or mirror. |
| `HF_HOME` | `/data/models` in Docker | Hugging Face model cache directory. |
| `PUID` / `PGID` | `1000` / `1000` | Host UID/GID applied inside Docker to `/data` via the container entrypoint. |
| `AURISDECK_PORT` | `8080` | Host TCP port published by Docker Compose. |
| `INSTALL_DIARIZATION` | `false` | Docker build argument (`true` installs `.[diarization]`). |
| `AURISDECK_SERVER_URL` | unset | Default server URL for `aurisdeck transcribe --server`. |
| `AURISDECK_TOKEN` | unset | Default Bearer API token for `aurisdeck transcribe --server`. |

---

## License

This project is licensed under the [MIT License](LICENSE).
