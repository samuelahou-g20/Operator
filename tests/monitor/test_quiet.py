"""Dashboard must not write to the terminal: tqdm + console logging corrupt the TUI."""

import logging

from logging.handlers import RotatingFileHandler

from src.common.logging import get_logger
from src.monitor.app import _quiet_console_logging, _silence_tqdm


def test_get_logger_file_only_env(monkeypatch):
    monkeypatch.setenv("OPERATOR_LOG_FILE_ONLY", "1")
    lg = get_logger("test_fileonly_unique")
    consoles = [h for h in lg.handlers
                if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)]
    assert consoles == []                                      # no console -> no TUI corruption
    assert any(isinstance(h, RotatingFileHandler) for h in lg.handlers)  # file log kept


def test_quiet_console_logging_strips_existing_console():
    lg = logging.getLogger("test_strip_unique")
    lg.handlers.clear()
    handler = logging.StreamHandler()
    lg.addHandler(handler)
    _quiet_console_logging()
    assert handler not in lg.handlers


def test_silence_tqdm_disables_bars():
    _silence_tqdm()
    import tqdm
    assert tqdm.tqdm(range(3)).disable is True
