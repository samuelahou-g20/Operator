"""Shared test fixtures."""

from __future__ import annotations

import pytest

import scripts.jwap_record as jrec
import scripts.jwap_report as jrep
import scripts.trade_bot as tb


@pytest.fixture(autouse=True)
def _jwap_sandbox(tmp_path, monkeypatch):
    """Every test writes JWAP records to a temp folder and never sleeps between retries. A record
    left in the real folder would be RESUMED by the live bot as a real parent."""
    monkeypatch.setattr(jrec, "RECORD_DIR", tmp_path / "jwap")
    monkeypatch.setattr(tb, "JWAP_RETRY_WAITS", (0, 0, 0))

    def offline(*a, **k):
        raise RuntimeError("tests are offline")
    monkeypatch.setattr(jrep, "binance_opens", offline)     # the report's only network call
    monkeypatch.setattr(jrep, "binance_tick", offline)
