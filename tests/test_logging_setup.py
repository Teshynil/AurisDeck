"""Unit tests for AurisDeck logging configuration, rotation, and per-task handlers."""

from __future__ import annotations

import logging
from pathlib import Path

from aurisdeck.config import Settings
from aurisdeck.logging_setup import (
    configure_main_logging,
    configure_worker_logging,
    set_log_level,
    task_file_handler,
)
from aurisdeck.storage.paths import DataPaths


def test_settings_normalizes_log_level() -> None:
    settings = Settings(log_level="debug")
    assert settings.log_level == "DEBUG"


def test_data_paths_log_helpers(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    paths.ensure_dirs()
    assert paths.logs.is_dir()
    assert paths.job_logs_dir("job_123") == tmp_path / "outputs" / "job_123" / "logs"
    assert paths.worker_log_path("cuda:0") == tmp_path / "logs" / "worker-cuda_0.log"


def test_configure_main_logging_rotates_and_updates_level(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    paths.ensure_dirs()
    configure_main_logging(
        paths,
        level="WARNING",
        max_bytes=1200,
        backup_count=2,
        include_console=False,
    )

    test_logger = logging.getLogger("aurisdeck.test_main")
    test_logger.info("suppressed info message")
    test_logger.warning("visible warning message")

    log_file = paths.logs / "aurisdeck.log"
    content = log_file.read_text(encoding="utf-8")
    assert "suppressed info message" not in content
    assert "visible warning message" in content

    set_log_level("DEBUG")
    for idx in range(25):
        test_logger.debug("debug line %02d %s", idx, "x" * 80)

    assert (paths.logs / "aurisdeck.log.1").is_file()


def test_task_file_handler_isolates_job_logs(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    paths.ensure_dirs()
    worker_log = paths.worker_log_path("cpu")
    configure_worker_logging(worker_log, level="INFO")

    task_log = paths.job_logs_dir("job-abc") / "asr-t1.log"
    worker_logger = logging.getLogger("aurisdeck.test_worker")
    worker_logger.info("outside task")

    with task_file_handler(task_log, level="DEBUG"):
        worker_logger.debug("inside task debug")
        worker_logger.error("inside task error")

    worker_logger.info("after task")

    task_text = task_log.read_text(encoding="utf-8")
    assert "outside task" not in task_text
    assert "inside task debug" in task_text
    assert "inside task error" in task_text
    assert "after task" not in task_text

    worker_text = worker_log.read_text(encoding="utf-8")
    assert "outside task" in worker_text
    assert "after task" in worker_text
