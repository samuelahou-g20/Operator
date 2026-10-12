"""Uniform logging setup: console + rotating file per process/strategy.

``get_logger(name)`` returns a logger that writes to the console and to a
rotating file under ``logs/{name}/``. Idempotent: repeated calls return the same
configured logger without stacking handlers.

MUST NOT: be bypassed by modules calling ``logging.basicConfig`` or adding their
own handlers — every component obtains its logger from here.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from src.common import paths

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5


def get_logger(name: str) -> logging.Logger:
    """Return a console+file logger for ``name`` (idempotent)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT)

    # A TUI (the dashboard) owns the screen; a console handler on stderr corrupts it.
    # OPERATOR_LOG_FILE_ONLY routes such processes to the file handler only.
    if not os.environ.get("OPERATOR_LOG_FILE_ONLY"):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

    log_dir = paths.LOGS_DIR / name
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / f"{name}.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
