"""Persistent /data layout, safe filename sanitization, and atomic file writes."""

from __future__ import annotations

import os
import re
import tempfile
import unicodedata
from pathlib import Path

_INVALID_FILENAME_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_DEFAULT_FILENAME = "archivo"
_MAX_FILENAME_CHARS = 200


def _validate_job_id(job_id: str) -> str:
    """Validate a job identifier used as a subdirectory under /data."""
    cleaned = job_id.strip()
    if (
        not cleaned
        or cleaned in {".", ".."}
        or "/" in cleaned
        or "\\" in cleaned
        or ".." in cleaned
        or _INVALID_FILENAME_CHARS_RE.search(cleaned)
    ):
        raise ValueError(f"Invalid job_id: {job_id!r}")
    return cleaned


class DataPaths:
    """Manages the canonical AurisDeck /data directory layout."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @property
    def db_path(self) -> Path:
        return self.root / "aurisdeck.db"

    @property
    def profiles(self) -> Path:
        return self.root / "profiles"

    @property
    def glossaries(self) -> Path:
        return self.root / "glossaries"

    @property
    def replacements(self) -> Path:
        return self.root / "replacements"

    @property
    def locales(self) -> Path:
        return self.root / "locales"

    @property
    def media(self) -> Path:
        return self.root / "media"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def ensure_dirs(self) -> None:
        """Idempotently create all top-level directories inside /data."""
        for directory in (
            self.root,
            self.profiles,
            self.glossaries,
            self.replacements,
            self.locales,
            self.media,
            self.checkpoints,
            self.outputs,
            self.tmp,
            self.models,
            self.logs,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def job_media_dir(self, job_id: str) -> Path:
        return self.media / _validate_job_id(job_id)

    def job_checkpoints_dir(self, job_id: str) -> Path:
        return self.checkpoints / _validate_job_id(job_id)

    def job_outputs_dir(self, job_id: str) -> Path:
        return self.outputs / _validate_job_id(job_id)

    def job_logs_dir(self, job_id: str) -> Path:
        return self.job_outputs_dir(job_id) / "logs"

    def worker_log_path(self, device_id: str) -> Path:
        safe_device = sanitize_filename((device_id or "worker").replace(":", "_"))
        return self.logs / f"worker-{safe_device}.log"

    def job_tmp_dir(self, job_id: str) -> Path:
        return self.tmp / _validate_job_id(job_id)


def _collapse_parent_dots(text: str) -> str:
    while ".." in text:
        text = text.replace("..", ".")
    return text


def sanitize_filename(name: str) -> str:
    """Return a cross-platform safe basename preserving intermediate dots."""
    normalized = unicodedata.normalize("NFC", name or "").replace("\\", "/")
    stripped_path = normalized.rstrip("/")
    basename = stripped_path.rsplit("/", 1)[-1] if stripped_path else ""
    basename = _INVALID_FILENAME_CHARS_RE.sub("_", basename).strip()

    if not basename or basename in {".", ".."}:
        return _DEFAULT_FILENAME

    parsed = Path(basename)
    stem = _collapse_parent_dots(parsed.stem).lstrip(".").strip(" .")
    suffix = _collapse_parent_dots(parsed.suffix).rstrip(" ")

    if suffix in {"", "."}:
        suffix = ""

    if not stem:
        stem = _DEFAULT_FILENAME

    max_stem_len = max(1, _MAX_FILENAME_CHARS - len(suffix))
    if len(stem) > max_stem_len:
        stem = stem[:max_stem_len].rstrip(" .") or _DEFAULT_FILENAME

    return f"{stem}{suffix}"


def is_ignored_archive_member(member_name: str) -> bool:
    """Return True for macOS metadata entries inside archives."""
    normalized = (member_name or "").replace("\\", "/").strip("/")
    if not normalized:
        return False
    parts = [part for part in normalized.split("/") if part]
    if "__MACOSX" in parts:
        return True
    basename = parts[-1]
    return basename == ".DS_Store" or basename.startswith("._")


def atomic_write_bytes(dest: Path, data: bytes, tmp_dir: Path) -> Path:
    """Write bytes to tmp_dir first and atomically move into dest via os.replace."""
    dest = Path(dest)
    tmp_dir = Path(tmp_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=tmp_dir,
            prefix=f".{dest.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, dest)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return dest


def atomic_write_text(
    dest: Path,
    content: str,
    tmp_dir: Path,
    encoding: str = "utf-8",
) -> Path:
    """Write UTF-8 text to tmp_dir first and atomically move into dest."""
    return atomic_write_bytes(dest, content.encode(encoding), tmp_dir)
