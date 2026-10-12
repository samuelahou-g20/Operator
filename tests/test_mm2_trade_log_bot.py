"""Offline tests for the mm2 trade log bot's backfill/poll cycle (no network, no sheet)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from scripts import mm2_trade_log_bot as mod


class _StubWs:
    def __init__(self):
        self.appended: list[list] = []

    def append_rows(self, rows, **kw):
        self.appended.extend(rows)


class _StubConn:
    """Stands in for the sqlite dedupe state; real helpers are monkeypatched onto it."""


def _wire_state(monkeypatch, written: set[str]):
    monkeypatch.setattr(mod, "_unwritten_uids", lambda conn, uids: set(uids) - written)
    monkeypatch.setattr(mod, "_mark_written",
                        lambda conn, rows: written.update(rows["uid"].tolist()))
    monkeypatch.setattr(mod, "_set_cursor", lambda conn, ts: None)


def test_backfill_converges_past_inclusive_created_at_boundary(monkeypatch):
    """Regression: CoinRoutes' created_at_after is INCLUSIVE (confirmed live) — re-querying from
    a trade's own timestamp returns it again. Without stepping the cursor past it, a backfill
    that catches up to the last available trade stalls forever re-matching that one trade."""
    stuck_ts = pd.Timestamp("2026-08-11T11:01:28.651583+00:00")

    class _StubClient:
        calls = 0

        def trades(self, **filters):
            _StubClient.calls += 1
            if _StubClient.calls > 20:      # fast, deterministic failure instead of a real hang
                raise AssertionError("backfill did not converge — still looping")
            after = pd.Timestamp(filters["created_at_after"])
            before = pd.Timestamp(filters["created_at_before"])
            if after <= stuck_ts <= before:
                return pd.DataFrame([{"uid": "stuck-trade", "strategy": "mm-c23_oxt-gamma-j",
                                      "created_at": stuck_ts}])
            return pd.DataFrame()

    written: set[str] = set()
    _wire_state(monkeypatch, written)
    ws = _StubWs()

    since = stuck_ts.to_pydatetime() - timedelta(hours=1)
    until = stuck_ts.to_pydatetime() + timedelta(hours=1, minutes=30)
    result = mod.backfill(_StubClient(), ws, _StubConn(), since, until)

    assert result >= until               # loop actually terminates at/past the target
    assert len(ws.appended) == 1         # the one real trade logged exactly once, not repeatedly


def test_filter_mm2_keeps_only_known_strategies():
    trades = pd.DataFrame([
        {"uid": "a", "strategy": "mm-c23_oxt-gamma-j"},   # OXT — known
        {"uid": "b", "strategy": "some-other-desk-j"},    # not mm2
    ])
    out = mod.filter_mm2(trades)
    assert list(out["uid"]) == ["a"]


# --------------------------------------------------------------------------- #
# Live price table (second tab)
# --------------------------------------------------------------------------- #
def _price_conn():
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.executescript("CREATE TABLE price (token TEXT NOT NULL, ts TEXT NOT NULL, "
                       "mid REAL NOT NULL, PRIMARY KEY (token, ts));")
    return conn


def test_perp_is_not_priced():
    """PERP is the futures hedge strategy, not a token — it has no spot pair."""
    assert "PERP" in mod.MM2_STRATEGIES
    assert "PERP" not in mod.PRICE_TOKENS
    assert set(mod.PRICE_TOKENS) == set(mod.MM2_STRATEGIES) - {"PERP"}


def test_change_24h_is_blank_until_history_spans_the_window():
    conn, now = _price_conn(), datetime(2026, 9, 23, 12, tzinfo=timezone.utc)
    conn.execute("INSERT INTO price VALUES (?, ?, ?)",
                 ("ICP", (now - timedelta(hours=3)).isoformat(), 3.0))
    assert mod.change_24h(conn, "ICP", 3.3, now) is None      # 3h of history is not 24h


def test_change_24h_measures_against_the_sample_past_the_lookback():
    conn, now = _price_conn(), datetime(2026, 9, 23, 12, tzinfo=timezone.utc)
    for age, mid in ((timedelta(hours=30), 1.0), (timedelta(hours=25), 2.0),
                     (timedelta(hours=2), 9.0)):
        conn.execute("INSERT INTO price VALUES (?, ?, ?)",
                     ("ICP", (now - age).isoformat(), mid))
    # newest sample at least 24h old is the 25h one (2.0) — not the 30h one, not the 2h one
    assert mod.change_24h(conn, "ICP", 3.0, now) == pytest.approx(50.0)


def test_price_rows_layout_and_unpriced_token():
    conn, now = _price_conn(), datetime(2026, 9, 23, 12, tzinfo=timezone.utc)
    conn.execute("INSERT INTO price VALUES (?, ?, ?)",
                 ("ICP", (now - timedelta(hours=25)).isoformat(), 2.0))
    rows = mod.price_rows(conn, {"ICP": 3.0, "CC": None}, now)

    assert rows[0][0].startswith("MM2 live prices — 2026-09-23 12:00:00")
    assert rows[1] == ["Token", "Mid", "24h %"]
    assert len(rows) == 2 + len(mod.PRICE_TOKENS)              # title + header + one per token
    body = {r[0]: r[1:] for r in rows[2:]}
    assert body["ICP"] == ["3", "+50.00%"]
    assert body["CC"] == ["—", "—"]                            # no book -> blank, never a stale mid


def test_record_mids_skips_blanks_and_prunes_past_retention():
    conn, now = _price_conn(), datetime(2026, 9, 23, 12, tzinfo=timezone.utc)
    conn.execute("INSERT INTO price VALUES (?, ?, ?)",
                 ("ICP", (now - mod.PRICE_RETENTION - timedelta(hours=1)).isoformat(), 1.0))
    mod._record_mids(conn, {"ICP": 3.0, "CC": None}, now)

    kept = conn.execute("SELECT token, mid FROM price ORDER BY token").fetchall()
    assert kept == [("ICP", 3.0)]        # stale row pruned, unpriced CC never written
