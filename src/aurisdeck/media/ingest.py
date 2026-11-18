"""Safe multitrack ZIP unpacking with ZIP Slip defense and media filtering."""

from __future__ import annotations

import re
import shutil
import stat
import zipfile
from collections.abc import Collection
from pathlib import Path, PurePosixPath

from aurisdeck.media.ffmpeg import MediaInvalidError
from aurisdeck.media.parsers import TrackHint, parse_track_hints, resolve_track_hint
from aurisdeck.storage.paths import is_ignored_archive_member, sanitize_filename

SUPPORTED_MEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".mp4", ".mov", ".webm", ".mkv"}
)

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class ZipSlipError(MediaInvalidError, ValueError):
    """Raised when a ZIP archive member attempts path traversal outside destination."""

    error_code: str = "ZIP_SLIP"

    def __init__(self, detail: str) -> None:
        super().__init__(detail=detail, error_code="ZIP_SLIP")


def _validate_archive_member_path(info: zipfile.ZipInfo, dest_root: Path) -> str:
    """Validate that a ZipInfo entry cannot escape dest_root via path traversal or symlinks."""
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ZipSlipError(f"Symbolic link entry disallowed in archive: {info.filename!r}")

    normalized = (info.filename or "").replace("\\", "/")
    if not normalized:
        raise ZipSlipError("Empty archive member path")

    if (
        normalized.startswith("/")
        or PurePosixPath(normalized).is_absolute()
        or _WINDOWS_DRIVE_RE.match(normalized)
    ):
        raise ZipSlipError(f"Absolute path disallowed in archive: {info.filename!r}")

    parts = [part for part in normalized.split("/") if part]
    if ".." in parts:
        raise ZipSlipError(f"Path traversal disallowed in archive: {info.filename!r}")

    candidate = (dest_root / normalized).resolve()
    if not candidate.is_relative_to(dest_root):
        raise ZipSlipError(f"Archive entry escapes destination directory: {info.filename!r}")

    return normalized


def _deduplicated_dest_path(dest_root: Path, raw_basename: str, used_names: set[str]) -> Path:
    """Compute a sanitized, collision-free destination path inside dest_root."""
    safe_name = sanitize_filename(raw_basename)
    candidate_name = safe_name
    if candidate_name.lower() in used_names or (dest_root / candidate_name).exists():
        parsed = Path(safe_name)
        stem = parsed.stem
        suffix = parsed.suffix
        counter = 2
        while True:
            candidate_name = sanitize_filename(f"{stem}_{counter}{suffix}")
            if (
                candidate_name.lower() not in used_names
                and not (dest_root / candidate_name).exists()
            ):
                break
            counter += 1

    used_names.add(candidate_name.lower())
    return dest_root / candidate_name


def _normalize_excluded_members(
    excluded_members: Collection[str] | None,
) -> tuple[set[str], set[str]]:
    """Return normalized archive paths and basenames to exclude."""
    excluded_paths: set[str] = set()
    excluded_names: set[str] = set()
    for raw in excluded_members or ():
        norm = str(raw or "").replace("\\", "/").strip().lstrip("/")
        if not norm:
            continue
        excluded_paths.add(norm.lower())
        if "/" not in norm:
            excluded_names.add(norm.lower())
    return excluded_paths, excluded_names


def unpack_multitrack_zip(
    zip_path: Path | str,
    dest_dir: Path | str,
    *,
    excluded_members: Collection[str] | None = None,
) -> list[tuple[Path, TrackHint]]:
    """Safely unpack supported audio/video tracks and parse `.aup` / `info.txt` hints."""
    archive_path = Path(zip_path)
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    dest_root = dest_path.resolve()
    excluded_paths, excluded_names = _normalize_excluded_members(excluded_members)

    info_entries: list[tuple[str, zipfile.ZipInfo]] = []
    aup_entries: list[tuple[str, zipfile.ZipInfo]] = []
    media_entries: list[tuple[str, zipfile.ZipInfo]] = []

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            members = zf.infolist()

            # Validate every member before extracting anything to disk.
            validated: list[tuple[str, zipfile.ZipInfo]] = []
            for member in members:
                normalized = _validate_archive_member_path(member, dest_root)
                validated.append((normalized, member))

            for normalized, member in validated:
                if member.is_dir() or normalized.endswith("/"):
                    continue
                if is_ignored_archive_member(normalized):
                    continue

                basename = normalized.rsplit("/", 1)[-1]
                lowered = basename.lower()
                if lowered == "info.txt":
                    info_entries.append((normalized, member))
                elif Path(lowered).suffix == ".aup":
                    aup_entries.append((normalized, member))
                elif Path(lowered).suffix in SUPPORTED_MEDIA_EXTENSIONS:
                    if normalized.lower() in excluded_paths or lowered in excluded_names:
                        continue
                    media_entries.append((normalized, member))

            info_entries.sort(key=lambda item: (item[0].count("/"), item[0]))
            aup_entries.sort(key=lambda item: (item[0].count("/"), item[0]))
            media_entries.sort(key=lambda item: item[0])

            info_text = "\n".join(
                zf.read(member).decode("utf-8", errors="replace") for _, member in info_entries
            )
            aup_texts = [
                zf.read(member).decode("utf-8", errors="replace") for _, member in aup_entries
            ]

            combined_hints: dict[str, TrackHint] = {}
            if info_text and not aup_texts:
                combined_hints.update(parse_track_hints(info_text=info_text))
            elif aup_texts:
                for idx, aup_text in enumerate(aup_texts):
                    combined_hints.update(
                        parse_track_hints(
                            info_text=info_text if idx == 0 else None,
                            aup_text=aup_text,
                        )
                    )

            used_names: set[str] = set()
            extracted: list[tuple[Path, TrackHint]] = []
            for normalized, member in media_entries:
                original_basename = normalized.rsplit("/", 1)[-1]
                target_path = _deduplicated_dest_path(dest_root, original_basename, used_names)
                with zf.open(member, "r") as src_stream, open(target_path, "wb") as dst_stream:
                    shutil.copyfileobj(src_stream, dst_stream)

                hint = resolve_track_hint(
                    target_path.name,
                    combined_hints,
                    original_filename=original_basename,
                )
                extracted.append((target_path, hint))
    except (zipfile.BadZipFile, OSError) as exc:
        raise MediaInvalidError(detail=f"Invalid ZIP archive {archive_path.name}: {exc}") from exc

    extracted.sort(key=lambda pair: str(pair[0]))
    return extracted
