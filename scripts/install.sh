#!/usr/bin/env bash
set -euo pipefail

WITH_DIARIZATION=false

usage() {
    cat <<'EOF'
Usage: ./scripts/install.sh [OPTIONS]

Create an isolated Python 3.12 virtual environment with uv and install AurisDeck.

Options:
  --with-diarization    Install optional speaker diarization dependencies (pyannote.audio, torch, torchaudio)
  -h, --help            Show this help message and exit
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --with-diarization)
            WITH_DIARIZATION=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            echo >&2
            usage >&2
            exit 1
            ;;
    esac
done

if ! command -v ffmpeg >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Error: ffmpeg is required but was not found on PATH.
Install ffmpeg (which also provides ffprobe) using your package manager:
  - Debian / Ubuntu: sudo apt update && sudo apt install -y ffmpeg
  - Fedora:          sudo dnf install -y ffmpeg-free
  - Arch Linux:      sudo pacman -S ffmpeg
  - macOS (Homebrew): brew install ffmpeg
EOF
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Error: uv is required but was not found on PATH.
Install uv from https://docs.astral.sh/uv/:
  curl -LsSf https://astral.sh/uv/install.sh | sh
EOF
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [ "${WITH_DIARIZATION}" = true ]; then
    echo "Installing AurisDeck with optional diarization support (--extra diarization)..."
    uv sync --project "${PROJECT_DIR}" --python 3.12 --frozen --no-dev --extra diarization
else
    echo "Installing AurisDeck base package..."
    uv sync --project "${PROJECT_DIR}" --python 3.12 --frozen --no-dev
fi

cat <<EOF

AurisDeck installation complete.

Start the web server:
  source .venv/bin/activate && aurisdeck serve --host 0.0.0.0 --port 8080

Or run directly without activating the virtual environment:
  .venv/bin/aurisdeck serve --host 0.0.0.0 --port 8080

Then open http://localhost:8080 in your browser.
EOF

if [ "${WITH_DIARIZATION}" = true ]; then
    cat <<'EOF'

Optional Speaker Diarization (HF_TOKEN):
  Diarization uses pyannote/speaker-diarization-3.1 from Hugging Face.
  1. Accept the model license terms on Hugging Face for:
     - https://huggingface.co/pyannote/speaker-diarization-3.1
     - https://huggingface.co/pyannote/segmentation-3.0
  2. Export a read-only Hugging Face token before starting AurisDeck:
     export HF_TOKEN=hf_your_token_here
EOF
else
    cat <<'EOF'

Note:
  Optional speaker diarization dependencies were not installed.
  To enable diarization later, re-run:
    ./scripts/install.sh --with-diarization
  And set HF_TOKEN in your environment (see README.md for details).
EOF
fi
