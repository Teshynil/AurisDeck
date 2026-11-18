#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/install.sh [OPTIONS]

Create an isolated Python 3.12 virtual environment with uv and install AurisDeck
(including ASR and speaker diarization runtime dependencies).

Options:
  -h, --help            Show this help message and exit
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
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

echo "Installing AurisDeck..."
uv sync --project "${PROJECT_DIR}" --python 3.12 --frozen --no-dev

cat <<'EOF'

AurisDeck installation complete.

Start the web server:
  source .venv/bin/activate && aurisdeck serve --host 0.0.0.0 --port 8080

Or run directly without activating the virtual environment:
  .venv/bin/aurisdeck serve --host 0.0.0.0 --port 8080

Then open http://localhost:8080 in your browser to complete the initial setup wizard.
EOF
