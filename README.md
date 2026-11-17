# AurisDeck

Self-hosted, portable web application for speech-to-text transcription and optional speaker diarization, built for high accuracy across Spanish, English, and mixed technical audio.

## Planned Features

- **Browser-first & OS-independent UI:** Simple drag-and-drop workflow for everyday users plus an advanced view for JSON profile management.
- **Three input workflows:**
  - **Single file (`single`):** Audio or video (`.mp4`, `.mkv`, `.wav`, `.mp3`, `.flac`, etc.) with optional speaker diarization.
  - **Multitrack merge (`multitrack`):** Separate per-speaker audio tracks (or `.zip` archives) transcribed in parallel and merged chronologically with zero speaker-assignment error.
  - **Batch queue (`batch`):** Multiple independent files processed across available workers.
- **Context & difficult vocabulary:** Support for `initial_prompt`, CTranslate2 `hotwords`, external glossary files, and post-ASR normalization maps.
- **Dynamic hardware scaling:** Automatic CUDA device discovery (`0`, `1`, or `N` GPUs) with seamless fallback to CPU (`int8`).
- **Reusable JSON profiles:** Save, load, and override presets for languages, models, speaker formatting, and subtitle dimensions.
- **WebVTT (`.vtt`) export:** W3C-compliant `.vtt` cues with configurable `<v Speaker>` tags, visible `[Speaker]:` prefixes, or clean caption wrapping, plus `.srt`, `.md`, and `.json`.
- **Portable deployment:** Docker Compose (NVIDIA GPU & CPU-only variants) or isolated CPython 3.12 installation via `uv`.

## Project Status

Repository initialized. Core engine, FastAPI backend, and Docker packaging are under active planning.
