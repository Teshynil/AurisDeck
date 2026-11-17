"""Subprocess wrappers for ffprobe media probing and ffmpeg 16 kHz WAV extraction."""

from __future__ import annotations

import math
import os
import subprocess
import uuid
from collections.abc import Sequence
from pathlib import Path


class MediaInvalidError(Exception):
    """Raised when an input media file is unreadable, lacks audio, or fails transcoding."""

    error_code: str = "MEDIA_INVALID"

    def __init__(
        self,
        detail: str,
        *,
        error_code: str = "MEDIA_INVALID",
    ) -> None:
        self.error_code = error_code
        self.detail = detail
        super().__init__(detail)


def _run_subprocess(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Execute a media CLI command without shell expansion."""
    try:
        return subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MediaInvalidError(detail=f"Failed to run {cmd[0]}: {exc}") from exc


def probe_duration_ms(path: Path | str) -> int:
    """Verify at least one audio stream exists and return duration in milliseconds."""
    media_path = Path(path)

    stream_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    stream_result = _run_subprocess(stream_cmd)
    if stream_result.returncode != 0:
        detail = stream_result.stderr.strip() or f"ffprobe exited with code {stream_result.returncode}"
        raise MediaInvalidError(detail=detail)

    stream_output = stream_result.stdout.strip()
    if not stream_output or stream_output.lower() == "video":
        raise MediaInvalidError(detail=f"No audio stream found in {media_path.name}")

    duration_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    duration_result = _run_subprocess(duration_cmd)
    if duration_result.returncode != 0:
        detail = (
            duration_result.stderr.strip()
            or f"ffprobe exited with code {duration_result.returncode}"
        )
        raise MediaInvalidError(detail=detail)

    raw_duration = duration_result.stdout.strip()
    try:
        duration_sec = float(raw_duration)
    except (TypeError, ValueError) as exc:
        raise MediaInvalidError(
            detail=f"Invalid media duration output {raw_duration!r} for {media_path.name}"
        ) from exc

    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise MediaInvalidError(
            detail=f"Non-positive media duration {duration_sec!r} for {media_path.name}"
        )

    return round(duration_sec * 1000)


def extract_wav_16k(
    src: Path | str,
    dst_wav: Path | str,
    normalize_audio: bool = False,
) -> None:
    """Extract mono 16 kHz PCM s16le WAV from src into dst_wav."""
    src_path = Path(src)
    dst_path = Path(dst_wav)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
    ]
    if normalize_audio:
        cmd.extend(["-af", "loudnorm"])
    cmd.append(str(dst_path))

    result = _run_subprocess(cmd)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"ffmpeg exited with code {result.returncode}"
        raise MediaInvalidError(detail=detail)


def mix_multitrack_audio(
    tracks: Sequence[tuple[Path | str, int]],
    dst_audio: Path | str,
    normalize_audio: bool = False,
) -> None:
    """Mix multiple audio tracks (with per-track offset_ms) into a single .m4a file."""
    if not tracks:
        raise MediaInvalidError(detail="At least one track is required to mix audio")

    dst_path = Path(dst_audio)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_name(f".{dst_path.stem}.tmp.{uuid.uuid4().hex[:8]}{dst_path.suffix}")

    cmd: list[str] = ["ffmpeg", "-y"]
    filter_parts: list[str] = []
    mix_inputs: list[str] = []

    for idx, (src, offset_ms) in enumerate(tracks):
        cmd.extend(["-i", str(Path(src))])
        delay = max(0, int(offset_ms or 0))
        if delay > 0:
            filter_parts.append(f"[{idx}:a]adelay={delay}:all=1[a{idx}]")
            mix_inputs.append(f"[a{idx}]")
        else:
            mix_inputs.append(f"[{idx}:a]")

    amix_chain = (
        f"{''.join(mix_inputs)}amix=inputs={len(tracks)}:duration=longest:normalize=0"
    )
    if normalize_audio:
        amix_chain += ",loudnorm"
    amix_chain += "[aout]"
    filter_parts.append(amix_chain)

    cmd.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[aout]",
            "-vn",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(tmp_path),
        ]
    )

    try:
        result = _run_subprocess(cmd)
        if result.returncode != 0:
            detail = result.stderr.strip() or f"ffmpeg exited with code {result.returncode}"
            raise MediaInvalidError(detail=detail)
        os.replace(tmp_path, dst_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
