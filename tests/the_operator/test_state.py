"""Durable gateway state: a restart keeps watching live orders and never resurrects a stale nonce."""

from __future__ import annotations

import json

from src.the_operator import policy, state
from src.the_operator.contract import Context

CTX = Context("r1", "mm3", "-1", "7", "u7", is_trader=True)


def _pending(nonce="n1", created=1000.0):
    return policy.Pending(nonce, CTX, "place_order", {"command": "/trade dnx buy"}, created, 250.0)


def test_round_trips_pending_and_fills(tmp_path):
    path = tmp_path / "state.json"
    fills = {"o1": {"oid": "o1", "base": "DNX", "side": "buy", "chat_id": "-1"}}
    state.dump({"n1": _pending()}, fills, path)
    pending, restored = state.load(path, now=1010.0)
    assert restored == fills
    p = pending["n1"]
    assert p.tool == "place_order" and p.proposal["command"] == "/trade dnx buy"
    assert p.notional_usd == 250.0
    assert p.ctx == CTX and p.ctx.is_trader       # the allowlist flag must survive the trip


def test_a_proposal_that_expired_while_we_were_down_is_dropped(tmp_path):
    path = tmp_path / "state.json"
    state.dump({"n1": _pending(created=1000.0)}, {"o1": {"oid": "o1"}}, path)
    pending, fills = state.load(path, now=1000.0 + policy.CONFIRM_TTL_S + 1)
    assert pending == {}                          # a trader must ask again for a fresh preview
    assert fills == {"o1": {"oid": "o1"}}         # but the live order is still watched


def test_missing_or_corrupt_state_is_empty_not_an_error(tmp_path):
    assert state.load(tmp_path / "nope.json") == ({}, {})
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert state.load(bad) == ({}, {})


def test_an_unreadable_row_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"pending": [{"nonce": "n1"}, None], "fills": {"o1": {}}}))
    pending, fills = state.load(path)
    assert pending == {} and fills == {"o1": {}}


def test_a_failed_write_is_survivable(tmp_path):
    missing = tmp_path / "nope" / "deep"
    missing.mkdir(parents=True)
    missing.chmod(0o500)                          # directory not writable
    try:
        state.dump({"n1": _pending()}, {}, missing / "state.json")   # must not raise
    finally:
        missing.chmod(0o700)


def test_dump_is_atomic_so_a_crash_cannot_leave_half_a_file(tmp_path):
    path = tmp_path / "state.json"
    state.dump({"n1": _pending()}, {}, path)
    state.dump({"n2": _pending("n2")}, {}, path)
    assert json.loads(path.read_text())["pending"][0]["nonce"] == "n2"
    assert not list(tmp_path.glob("*.tmp"))
