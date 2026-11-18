"""Single-writer logging setup for AurisDeck main process, workers, and per-job task logs."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from aurisdeck.storage.paths import DataPaths

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(processName)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_AURISDECK_LOGGER_NAME = "aurisdeck"
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024
_DEFAULT_BACKUP_COUNT = 5

_NOISY_THIRD_PARTY_LOGGERS = (
    "faster_whisper",
    "ctranslate2",
    "huggingface_hub",
    "urllib3",
    "httpx",
    "httpcore",
    "multipart",
    "pyannote",
    "speechbrain",
    "pytorch_lightning",
    "lightning",
)


def parse_log_level(level: str | int | None) -> int:
    """Convert a string or int log level into a standard logging module level."""
    if isinstance(level, int):
        return level
    normalized = str(level or "INFO").strip().upper()
    mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return mapping.get(normalized, logging.INFO)


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)


def _close_owned_handlers(target_logger: logging.Logger) -> None:
    for handler in list(target_logger.handlers):
        if getattr(handler, "_aurisdeck_owned", False):
            target_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # noqa: BLE001
                pass


def _sync_third_party_loggers(numeric_level: int) -> None:
    third_party_level = logging.INFO if numeric_level <= logging.DEBUG else logging.WARNING
    for name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(third_party_level)


def _configure_rotating_logger(
    log_file: Path,
    *,
    level: str | int = "INFO",
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
    include_console: bool = True,
    attach_uvicorn_error: bool = False,
) -> logging.Logger:
    numeric_level = parse_log_level(level)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = _build_formatter()

    pkg_logger = logging.getLogger(_AURISDECK_LOGGER_NAME)
    _close_owned_handlers(pkg_logger)
    pkg_logger.setLevel(numeric_level)
    pkg_logger.propagate = False

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    file_handler._aurisdeck_owned = True  # type: ignore[attr-defined]
    pkg_logger.addHandler(file_handler)

    if include_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        console_handler._aurisdeck_owned = True  # type: ignore[attr-defined]
        pkg_logger.addHandler(console_handler)

    if attach_uvicorn_error:
        uvicorn_err = logging.getLogger("uvicorn.error")
        _close_owned_handlers(uvicorn_err)
        uvicorn_err.addHandler(file_handler)

    _sync_third_party_loggers(numeric_level)
    return pkg_logger


def configure_main_logging(
    paths: DataPaths,
    *,
    level: str | int = "INFO",
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
    include_console: bool = True,
) -> logging.Logger:
    """Configure the main process logger writing to data/logs/aurisdeck.log."""
    return _configure_rotating_logger(
        paths.logs / "aurisdeck.log",
        level=level,
        max_bytes=max_bytes,
        backup_count=backup_count,
        include_console=include_console,
        attach_uvicorn_error=True,
    )


def configure_worker_logging(
    log_path: str | Path | None,
    *,
    level: str | int = "INFO",
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
) -> logging.Logger:
    """Configure an isolated spawn worker's logger writing to its own worker log file."""
    if log_path is None:
        numeric_level = parse_log_level(level)
        pkg_logger = logging.getLogger(_AURISDECK_LOGGER_NAME)
        pkg_logger.setLevel(numeric_level)
        _sync_third_party_loggers(numeric_level)
        return pkg_logger

    return _configure_rotating_logger(
        Path(log_path),
        level=level,
        max_bytes=max_bytes,
        backup_count=backup_count,
        include_console=False,
        attach_uvicorn_error=False,
    )


def set_log_level(level: str | int) -> int:
    """Update the log level of the aurisdeck logger and all owned handlers at runtime."""
    numeric_level = parse_log_level(level)
    pkg_logger = logging.getLogger(_AURISDECK_LOGGER_NAME)
    pkg_logger.setLevel(numeric_level)
    for handler in pkg_logger.handlers:
        if getattr(handler, "_aurisdeck_owned", False):
            handler.setLevel(numeric_level)
    _sync_third_party_loggers(numeric_level)
    return numeric_level


@contextmanager
def task_file_handler(
    log_path: str | Path | None,
    *,
    level: str | int = "INFO",
) -> Iterator[Path | None]:
    """Temporarily attach a single-writer FileHandler for a job/task log file."""
    if not log_path:
        yield None
        return

    resolved = Path(log_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    numeric_level = parse_log_level(level)

    pkg_logger = logging.getLogger(_AURISDECK_LOGGER_NAME)
    prev_level = pkg_logger.level
    if prev_level == logging.NOTSET or numeric_level < prev_level:
        pkg_logger.setLevel(numeric_level)

    handler = logging.FileHandler(resolved, mode="a", encoding="utf-8")
    handler.setLevel(numeric_level)
    handler.setFormatter(_build_formatter())
    pkg_logger.addHandler(handler)
    try:
        yield resolved
    finally:
        pkg_logger.removeHandler(handler)
        try:
            handler.flush()
            handler.close()
        except Exception:  # noqa: BLE001
            pass
        pkg_logger.setLevel(prev_level)
