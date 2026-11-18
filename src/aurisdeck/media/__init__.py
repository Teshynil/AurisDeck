"""Media probing, audio extraction, and safe ZIP ingestion."""

from aurisdeck.media.ffmpeg import MediaInvalidError, extract_wav_16k, probe_duration_ms
from aurisdeck.media.ingest import (
    SUPPORTED_MEDIA_EXTENSIONS,
    ZipSlipError,
    unpack_multitrack_zip,
)

__all__ = [
    "SUPPORTED_MEDIA_EXTENSIONS",
    "MediaInvalidError",
    "ZipSlipError",
    "extract_wav_16k",
    "probe_duration_ms",
    "unpack_multitrack_zip",
]
