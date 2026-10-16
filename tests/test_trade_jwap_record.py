"""Offline tests for /jwap's durable record in trade_bot: record-before-act, safe retries, fill
tracking, the 5-minute cancel, stop, and resuming after a restart. No live API."""

from __future__ import annotations

import pandas as pd
import pytest

import scripts.jwap_record as jrec
import scripts.trade_bot as tb
from src.adapters.coinroutes.trade import CoinRoutesHTTPError

_T0 = pd.Timestamp("2026-09-28 10:00:30").timestamp()
_ENTRY = _T0 - 30 + 60                                       # 10:01:00


def _jason():
    idx = pd.date_range(end="2026-09-28 09:00", periods=3, freq="1h")
    return pd.DataFrame({"jsonk_dn": 100.0, "jsonk_up": 120.0, "days_dn": 2.0, "days_up": 2.0},
                        index=idx)


class _Client:
    """Prices for the gate (mid) and for prepare_order (touch)."""

    def __init__(self, mid=130.0):
        self.mid = mid

    def mid_on(self, pair, venues):
        return self.mid

    def touch_price(self, pair, side, *, probe_usd=25.0, markets=None):
        return "130"

    def strategy_exchanges(self, strategy):
        return ["binance", "binancefutures", "okex"]

    def book_venues(self, pair, side, *, probe_usd=25.0):
        return {"BINANCE", "BINANCEFUTURES", "OKEX"}


class _Trader:
    """CoinRoutes order API: ``place_errors`` are raised by successive place calls; ``land`` makes
    an erroring place still create the order (the dangerous case a blind retry would double)."""

    def __init__(self, place_errors=(), land=False, lookup_404s=0):
        self.placed: dict[str, dict] = {}
        self.place_calls, self.cancels = [], []
        self.place_errors, self.land, self.lookup_404s = list(place_errors), land, lookup_404s

    def place(self, **kw):
        self.place_calls.append(kw)
        cid = kw["client_order_id"]
        if self.place_errors:
            err = self.place_errors.pop(0)
            if self.land:
                self.placed[cid] = {"client_order_id": cid, "order_status": "open"}
            raise err
        self.placed[cid] = {"client_order_id": cid, "order_status": "open", "executed_qty": "0"}
        return self.placed[cid]

    def get_order(self, cid):
        if self.lookup_404s:
            self.lookup_404s -= 1
            raise CoinRoutesHTTPError("CoinRoutes get_order failed 404: Not Found", 404)
        if cid not in self.placed:
            raise CoinRoutesHTTPError("CoinRoutes get_order failed 404: Not Found", 404)
        return self.placed[cid]

    def cancel(self, cid):
        self.cancels.append(cid)
        self.placed[cid] = {**self.placed[cid], "order_status": "cancelled"}
        return {"ok": True}


@pytest.fixture(autouse=True)
def _stubs(monkeypatch):
    monkeypatch.setattr(tb, "jwap_jason", lambda base, sigma, now: _jason())
    monkeypatch.setattr(tb, "_send", lambda *a, **k: None)


def _start(client, text="/jwap sol sell size=1200 over=1h"):
    jwaps: dict = {}
    reply = tb.handle_jwap(text, client, None, "mm3", "c1", jwaps, now=_T0, user_id="42")
    assert reply.startswith("🟢"), reply
    (pid,) = jwaps
    return jwaps, pid


def _kinds(pid):
    return [e["kind"] for e in jrec.load(pid)]


def _poll(jwaps, client, trader, now):
    tb.poll_jwaps("api", {"mm3": client}, {"mm3": trader}, jwaps, now=now)


def test_start_records_the_command_who_and_the_arrival_gate():
    jwaps, pid = _start(_Client())
    created = jrec.load(pid)[0]
    assert created["kind"] == "created" and created["params"]["user_id"] == "42"
    assert created["command"] == "/jwap sol sell size=1200 over=1h"
    assert (created["arrival_mid"], created["arrival_level"], created["gate_open"]) == (130.0, 120.0, True)
    assert _kinds(pid) == ["created", "jason"]


def test_every_slice_is_recorded_even_when_the_gate_is_closed():
    client, trader = _Client(mid=110.0), _Trader()           # sell gate at 120: closed
    jwaps, pid = _start(client)
    _poll(jwaps, client, trader, _ENTRY + 1)
    (sl,) = [e for e in jrec.load(pid) if e["kind"] == "slice"]
    assert (sl["result"], sl["mid"], sl["level"], sl["days"]) == ("gate_closed", 110.0, 120.0, 2.0)
    assert trader.place_calls == []


def test_intent_is_written_before_the_order_and_uses_our_own_id():
    client, trader = _Client(), _Trader()
    jwaps, pid = _start(client)
    _poll(jwaps, client, trader, _ENTRY + 1)
    cid = jrec.child_id(pid, int(_ENTRY), 0)
    assert trader.place_calls[0]["client_order_id"] == cid
    k = _kinds(pid)
    assert k.index("slice") < k.index("intent") < k.index("placed")
    assert jwaps[pid]["oids"] == [cid]


@pytest.mark.parametrize("err", [CoinRoutesHTTPError("CoinRoutes place failed 404: x", 404),
                                 CoinRoutesHTTPError("CoinRoutes place failed 503: x", 503),
                                 tb.requests.exceptions.ReadTimeout("timed out")])
def test_a_transient_place_error_is_retried_when_the_order_did_not_land(err):
    client, trader = _Client(), _Trader(place_errors=[err])
    jwaps, pid = _start(client)
    _poll(jwaps, client, trader, _ENTRY + 1)
    assert len(trader.place_calls) == 2 and len(trader.placed) == 1
    assert "retry" in _kinds(pid) and "placed" in _kinds(pid)


def test_an_order_that_landed_despite_the_error_is_never_sent_twice():
    client = _Client()
    trader = _Trader(place_errors=[tb.requests.exceptions.ReadTimeout("timed out")], land=True)
    jwaps, pid = _start(client)
    _poll(jwaps, client, trader, _ENTRY + 1)
    assert len(trader.place_calls) == 1                       # found by id, not re-sent
    assert "placed" in _kinds(pid) and jwaps[pid]["fired"] == 1


def test_a_rejection_400_is_not_retried():
    client = _Client()
    trader = _Trader(place_errors=[CoinRoutesHTTPError("CoinRoutes place failed 400: bad", 400)])
    jwaps, pid = _start(client)
    _poll(jwaps, client, trader, _ENTRY + 1)
    assert len(trader.place_calls) == 1
    failed = [e for e in jrec.load(pid) if e["kind"] == "failed"]
    assert failed[0]["stage"] == "place" and "400" in failed[0]["error"]


def test_fills_are_recorded_and_a_child_still_working_after_5_minutes_is_cancelled():
    client, trader = _Client(), _Trader()
    jwaps, pid = _start(client, "/jwap sol sell size=1200 over=1h every=10m")
    _poll(jwaps, client, trader, _ENTRY + 1)
    cid = jwaps[pid]["oids"][0]
    trader.placed[cid] = {**trader.placed[cid], "executed_qty": "0.4", "avg_price": "130",
                          "net_consideration": "-52"}
    _poll(jwaps, client, trader, _ENTRY + 120)                # partial fill seen, not cancelled
    assert trader.cancels == []
    _poll(jwaps, client, trader, _ENTRY + 302)                # past 5 min: cancelled
    assert trader.cancels == [cid]
    _poll(jwaps, client, trader, _ENTRY + 363)                # final state recorded
    fills = [e["order"] for e in jrec.load(pid) if e["kind"] == "fill"]
    assert [(f["order_status"], f["executed_qty"]) for f in fills] == [
        ("open", "0"), ("open", "0.4"), ("cancelled", "0.4")]
    assert jwaps[pid]["children"][cid]["terminal"] is True


def test_done_waits_for_every_child_to_be_final():
    client, trader = _Client(), _Trader()
    jwaps, pid = _start(client, "/jwap sol sell size=1k over=5m")   # one slice
    _poll(jwaps, client, trader, _ENTRY + 1)
    assert pid in jwaps and "done" not in _kinds(pid)          # the child is still working
    cid = jwaps[pid]["oids"][0]
    trader.placed[cid] = {**trader.placed[cid], "order_status": "closed", "executed_qty": "7.69",
                          "avg_price": "130", "net_consideration": "-1000"}
    _poll(jwaps, client, trader, _ENTRY + 62)
    assert pid not in jwaps
    done = jrec.load(pid)[-1]
    assert done["kind"] == "done" and done["filled"] == 1000.0 and done["left_over"] == 0.0


def test_stop_cancels_working_children_and_finishes_once_they_settle():
    client, trader = _Client(), _Trader()
    jwaps, pid = _start(client)
    _poll(jwaps, client, trader, _ENTRY + 1)
    cid = jwaps[pid]["oids"][0]
    reply = tb.handle_jwap("/jwap stop sol", client, trader, "mm3", "c1", jwaps, now=_ENTRY + 30,
                           user_id="42")
    assert trader.cancels == [cid]
    stopped = [e for e in jrec.load(pid) if e["kind"] == "stopped"]
    assert stopped[0]["by"] == "42"
    # The stub cancels instantly, but the stop reads the order before the cancel lands.
    assert "stopping" in reply and pid in jwaps
    _poll(jwaps, client, trader, _ENTRY + 100)
    assert pid not in jwaps and jrec.load(pid)[-1]["kind"] == "done"
    assert len(trader.place_calls) == 1                        # no new slice after the stop


def test_a_restart_resumes_the_parent_and_records_the_slices_missed_while_down():
    client, trader = _Client(mid=110.0), _Trader()             # gate closed: no orders
    jwaps, pid = _start(client)
    _poll(jwaps, client, trader, _ENTRY + 1)                   # slice 0
    # The process dies here; 20 minutes later it starts again.
    resumed, notes = tb.resume_jwaps({"mm3": trader}, now=_ENTRY + 1200)
    p = resumed[pid]
    assert p["next"] == 4 and p["late"] == 3                   # slices 1-3 fell due while down
    assert [e["result"] for e in jrec.load(pid) if e["kind"] == "slice"][1:] == ["bot_down"] * 3
    assert "resumed" in notes[0][1] and "3 slice(s) missed" in notes[0][1]
    _poll(resumed, client, trader, _ENTRY + 1201)              # slice 4 runs, JASON refreshed
    assert resumed[pid]["next"] == 5


def test_a_restart_between_intent_and_answer_finds_the_order_by_its_id():
    client, trader = _Client(), _Trader()
    jwaps, pid = _start(client)
    cid = jrec.child_id(pid, int(_ENTRY), 0)
    jrec.append(pid, "slice", seq=0, due=_ENTRY, result="fired")
    jrec.append(pid, "intent", cid=cid, seq=0, place_kwargs={})
    trader.placed[cid] = {"client_order_id": cid, "order_status": "open"}   # it did land
    resumed, _ = tb.resume_jwaps({"mm3": trader}, now=_ENTRY + 10)
    assert resumed[pid]["oids"] == [cid] and resumed[pid]["fired"] == 1
    placed = [e for e in jrec.load(pid) if e["kind"] == "placed"]
    assert placed[0]["recovered"] is True
    assert trader.place_calls == []                            # never re-sent


def test_a_restart_between_intent_and_answer_records_an_order_that_never_landed():
    client, trader = _Client(), _Trader()
    jwaps, pid = _start(client)
    cid = jrec.child_id(pid, int(_ENTRY), 0)
    jrec.append(pid, "intent", cid=cid, seq=0, place_kwargs={})
    resumed, _ = tb.resume_jwaps({"mm3": trader}, now=_ENTRY + 10)
    failed = [e for e in jrec.load(pid) if e["kind"] == "failed"]
    assert failed[0]["stage"] == "restart" and resumed[pid]["errors"] == 1


def test_a_finished_parent_is_not_resumed():
    client, trader = _Client(), _Trader()
    jwaps, pid = _start(client)
    tb.handle_jwap("/jwap stop sol", client, trader, "mm3", "c1", jwaps, now=_ENTRY - 10)
    assert pid not in jwaps                                    # nothing working: done at once
    resumed, _ = tb.resume_jwaps({"mm3": trader}, now=_ENTRY + 10)
    assert resumed == {}


def test_ids_are_unique_across_a_restart():
    client = _Client()
    _jwaps, first = _start(client)
    _jwaps, second = _start(client)                            # a fresh process, same folder
    assert second == first + 1


def test_a_404_on_a_price_read_is_retried():
    client, trader = _Client(), _Trader()
    jwaps, pid = _start(client)
    calls = {"n": 0}

    def flaky(pair, venues):
        calls["n"] += 1
        if calls["n"] == 1:
            raise tb.AdapterError('CoinRoutes 404: "Not Found"', venue="coinroutes", code=404)
        return 130.0
    client.mid_on = flaky
    _poll(jwaps, client, trader, _ENTRY + 1)
    assert calls["n"] == 2 and len(trader.place_calls) == 1


# --------------------------------------------------------------------------- #
# /jwap report — the client page, only for this desk's parents
# --------------------------------------------------------------------------- #
def test_report_builds_an_html_page_for_this_desk():
    client, trader = _Client(), _Trader()
    jwaps, pid = _start(client)
    _poll(jwaps, client, trader, _ENTRY + 1)
    out = tb.jwap_report(f"/jwap report {pid}", "mm3", now=_ENTRY + 60)
    data, name, caption = out
    assert name.startswith("SOL_execution_") and name.endswith(".html")
    assert data.startswith(b"<!doctype html>") and b"jwap-" not in data
    assert caption.startswith("SOL execution report")


def test_report_refuses_another_desks_parent_and_bad_ids():
    client = _Client()
    _jwaps, pid = _start(client)
    assert tb.jwap_report(f"/jwap report #{pid}", "mm2") == f"No JWAP #{pid} on MM2."
    assert tb.jwap_report("/jwap report 999", "mm3") == "No JWAP #999 on MM3."
    assert tb.jwap_report("/jwap report", "mm3").startswith("Usage")
