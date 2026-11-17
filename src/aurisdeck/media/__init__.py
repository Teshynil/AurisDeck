"""Media probing, audio extraction, multitrack metadata parsing, and safe ZIP ingestion."""

from aurisdeck.media.ffmpeg import MediaInvalidError, extract_wav_16k, probe_duration_ms
from aurisdeck.media.ingest import (
    SUPPORTED_MEDIA_EXTENSIONS,
    ZipSlipError,
    unpack_multitrack_zip,
)
from aurisdeck.media.parsers import (
    AudacityAupParser,
    CraigInfoParser,
    TrackHint,
    parse_track_hints,
    resolve_track_hint,
)

__all__ = [
    "SUPPORTED_MEDIA_EXTENSIONS",
    "AudacityAupParser",
    "CraigInfoParser",
    "MediaInvalidError",
    "TrackHint",
    "ZipSlipError",
    "extract_wav_16k",
    "parse_track_hints",
    "probe_duration_ms",
    "resolve_track_hint",
    "unpack_multitrack_zip",
]
