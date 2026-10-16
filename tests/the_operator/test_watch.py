"""The MM watch posts a snapshot of what is not quoting, and why, every :00 and :30; the volume
watch posts sustained breaches once an hour. Tests move time with ``_Clock`` right before the round
that should speak."""

from __future__ import annotations

import json

import pytest

from src.the_operator import rapid, watch


def _adjust(**zero):
    """An order_adjustment block with every rule at 1, except ``rule=sides`` set to 0."""
    adj = {f"{r}_notional_multiplier_{s}": 1.0 for r, _ in watch._RULES for s in ("bid", "ask")}
    adj.update(notional_multiplier_bid=1.0, notional_multiplier_ask=1.0)
    for rule, sides in zero.items():
        for side in sides.split(","):
            adj[f"{rule}_notional_multiplier_{side}"] = 0.0
    return adj


def _entry(name, status, trading, reason="", **zero):
    return {"strategy_name": name, "latest_status": status, "latest_status_time_micros": 0,
            "last_error_status": "",
            "latest_status_details": json.dumps({
                "decision_state_validity_reason": reason,
                "debug_info": {"trading": trading, "order_adjustment": _adjust(**zero)}})}


OK = "LM2Strategy:REPOL:POL-EUR:BITVAVO"
STOP = "LM2Strategy:REPARTI:PARTI-USDT:BINANCE"
STOP2 = "LM2Strategy:REPARTI:PARTI-EUR:BITVAVO"
ERR = "LM2Strategy:RENEAR:NEAR-USD:BITSTAMP"
BAL = "LM2Strategy:RENYM:NYM-USD:KRAKEN"
OFF = "LM2Strategy:RELAPTOP:LAPTOP-EUR:KRAKEN"
BAND = {"price_boundaries": "bid,ask"}


@pytest.fixture
def box(monkeypatch):
    monkeypatch.setattr(rapid, "boxes", lambda program: {"tk1": "http://box"})
    return monkeypatch


def _serve(monkeypatch, entries):
    monkeypatch.setattr(rapid, "_get",
                        lambda url, path, timeout=None: {"get_strategy_status": entries})


class _Clock:
    def __init__(self) -> None:
        self.t = 1_000 * 3600.0 + 60                # 1 minute past an hour

    def __call__(self) -> float:
        return self.t

    def next_hour(self) -> None:
        self.t += 3600

    def next_half_hour(self) -> None:
        self.t += 1800


def _watcher(posted, say=None):
    say = say or (lambda chat, text: posted.append((chat, text)))
    return watch.Watcher({"mm3": ["-1", "-2"]}, say,
                         watch.Hourly(say, _Clock(), period_s=watch.MM_POST_S))


def _strategies(monkeypatch, entries):
    _serve(monkeypatch, entries)
    return rapid.scan("mm3", [])


def test_the_box_reports_why_a_strategy_stopped(box, monkeypatch):
    (s,) = _strategies(monkeypatch, [_entry(ERR, "STOP (ERROR)", False,
                                            reason="Delays:  Orderbook data is older than 5.0 seconds")])
    assert s.validity_reason.startswith("Delays:")
    assert s.adjust["price_boundaries_notional_multiplier_bid"] == 1.0


@pytest.mark.parametrize("entry,want", [
    (_entry(OK, "TRADING", True), None),
    (_entry(STOP, "STOP (DISABLED)", False, **BAND), "band"),
    (_entry(STOP, "STOP (DISABLED)", False, low_balance_policy="bid,ask", **BAND), "band"),
    (_entry(BAL, "STOP (DISABLED)", False, low_balance_policy="ask"), "balance"),
    (_entry(ERR, "STOP (ERROR)", False, reason="Delays: stale"), "error"),
    (_entry(ERR, "STOP (ERROR)", False, reason="Ref Price Out of Boundaries: Price 3.0"), "band"),
    (_entry(OK, "STOP (DISABLED)", False, descale_trigger="bid,ask"), "descale"),
    (_entry(OFF, "STOP (DISABLED)", False), "off"),
    (_entry(OK, None, None), "unknown"),
])
def test_cause_reads_the_rule_that_zeroed_the_size(box, monkeypatch, entry, want):
    (s,) = _strategies(monkeypatch, [entry])
    assert watch.cause(s) == want


def test_trading_with_both_sizes_at_zero_is_not_quoting(box, monkeypatch):
    """The docs' CASE 2: status TRADING, but no orders because the size multiplier is 0."""
    e = _entry(OK, "TRADING", True)
    details = json.loads(e["latest_status_details"])
    details["debug_info"]["order_adjustment"].update(notional_multiplier_bid=0.0,
                                                     notional_multiplier_ask=0.0)
    e["latest_status_details"] = json.dumps(details)
    (s,) = _strategies(monkeypatch, [e])
    assert watch.cause(s) == "zero"


def test_the_snapshot_groups_by_cause_with_the_fix(box, monkeypatch):
    found = _strategies(monkeypatch, [
        _entry(OK, "TRADING", True),
        _entry(STOP, "STOP (DISABLED)", False, low_balance_policy="bid,ask", **BAND),
        _entry(STOP2, "STOP (DISABLED)", False, **BAND),
        _entry(BAL, "STOP (DISABLED)", False, low_balance_policy="bid"),
        _entry(ERR, "STOP (ERROR)", False, reason="Delays:  Orderbook data is older than 5.0 seconds"),
        _entry(OFF, "STOP (DISABLED)", False),
    ])
    text = watch.render_snapshot("mm3", found, [])
    assert text.startswith("🔔 **MM watch · mm3** — 4 not quoting · ")     # off-by-hand not counted
    assert ("• **PARTI** — `BINANCE`, `BITVAVO` · also low balance · `/operator rebound PARTI`"
            in text)
    assert "• **NYM** — `KRAKEN` (bid only)" in text
    assert "• **NEAR** — `BITSTAMP`: Delays: Orderbook data is older than 5.0 seconds" in text
    assert "⏸ **Off by hand:** LAPTOP (1)" in text and "POL" not in text
    assert text.index("Price outside") < text.index("Low balance") < text.index("Error")


def test_only_hand_switched_off_strategies_post_nothing(box, monkeypatch):
    found = _strategies(monkeypatch, [_entry(OK, "TRADING", True), _entry(OFF, "STOP (DISABLED)", False)])
    assert watch.render_snapshot("mm3", found, []) == ""


def test_a_box_that_did_not_answer_is_said(box, monkeypatch):
    found = _strategies(monkeypatch, [_entry(STOP, "STOP (DISABLED)", False, **BAND)])
    text = watch.render_snapshot("mm3", found, ["box mm3/tk2 unreachable: timed out"])
    assert "1 box(es) did not answer" in text


def test_many_tokens_are_truncated_and_many_venues_folded(box, monkeypatch):
    many = [_entry(f"LM2Strategy:REX:X{i:02}-USDT:BINANCE", "STOP (DISABLED)", False, **BAND)
            for i in range(20)]
    venues = [_entry(f"LM2Strategy:REY:Y-USDT:V{i}", "STOP (DISABLED)", False, **BAND) for i in range(5)]
    text = watch.render_snapshot("mm3", _strategies(monkeypatch, many + venues), [])
    assert f"• …and {21 - watch.MAX_LINES} more tokens" in text
    found = _strategies(monkeypatch, venues)
    assert "• **Y** — `V0`, `V1`, `V2` +2 more" in watch.render_snapshot("mm3", found, [])


def test_the_snapshot_goes_out_every_half_hour_while_something_is_down(box, monkeypatch):
    """No hold, no "quoting again": each :00 and :30 restates what is down, to both chats."""
    _serve(monkeypatch, [_entry(OK, "TRADING", True), _entry(STOP, "STOP (DISABLED)", False, **BAND)])
    posted = []
    w = _watcher(posted)
    w.round_once()                                # a start spends its slot: nothing yet
    w.round_once()
    assert posted == []
    w.gate.clock.next_half_hour()
    w.round_once()
    assert [c for c, _ in posted] == ["-1", "-2"] and "PARTI" in posted[0][1]
    posted.clear()
    w.round_once()                                # same slot: no repeat
    assert posted == []
    w.gate.clock.next_half_hour()
    w.round_once()                                # still down: said again, as asked
    assert len(posted) == 2

    posted.clear()
    _serve(monkeypatch, [_entry(OK, "TRADING", True), _entry(STOP, "TRADING", True)])
    w.gate.clock.next_half_hour()
    w.round_once()                                # all quoting: silence, no recovery message
    assert posted == []


def test_a_snapshot_is_the_latest_scan_not_a_stale_one(box, monkeypatch):
    """A desk whose boxes stopped answering must not repost the last list it had."""
    _serve(monkeypatch, [_entry(STOP, "STOP (DISABLED)", False, **BAND)])
    posted = []
    w = _watcher(posted)
    w.round_once()
    w.gate.clock.next_half_hour()
    w.round_once()
    assert len(posted) == 2
    posted.clear()
    _serve(monkeypatch, [])                       # every box silent from here on
    w.gate.clock.next_half_hour()
    w.round_once()
    assert posted == []


def test_a_failing_scan_does_not_kill_the_round(box, monkeypatch):
    def boom(program, w):
        raise RuntimeError("registry exploded")
    monkeypatch.setattr(rapid, "scan", boom)
    posted = []
    _watcher(posted).round_once()                 # must not raise
    assert posted == []


def test_an_unpostable_chat_does_not_stop_the_other(box, monkeypatch):
    posted = []

    def say(chat, text):
        if chat == "-1":
            raise TimeoutError("telegram down")
        posted.append((chat, text))
    w = _watcher(posted, say)
    _serve(monkeypatch, [_entry(STOP, "STOP (DISABLED)", False, **BAND)])
    w.round_once()
    w.gate.clock.next_half_hour()
    w.round_once()
    assert [c for c, _ in posted] == ["-2"]


def test_a_box_that_stops_answering_neither_alarms_nor_clears():
    """For the volume watch: a coin missing from a reading is not news either way."""
    tracks = {"ELA": watch.Track("ok", streak=9), "DNX": watch.Track("low", streak=9,
                                                                    announced=True)}
    assert watch.settle(tracks, {"ELA": "ok"}) == ([], [])
    assert tracks["DNX"].streak == 9 and tracks["DNX"].announced   # untouched by the gap

# --------------------------------------------------------------------------------------
# The volume watch: the desk's 5%-50% rule, held for two rounds, minus today's exceptions
# --------------------------------------------------------------------------------------
from src.the_operator import volume            # noqa: E402


@pytest.fixture
def sheet(monkeypatch, tmp_path):
    monkeypatch.setattr(volume, "SNOOZE_PATH", tmp_path / "snooze.json")
    shares = {}
    monkeypatch.setattr(watch, "desk_shares", lambda desk: dict(shares))
    return shares


class _Venue:
    """Only what VolumeWatcher touches: it ranks by market size and shows the exchange."""

    exchange = "GATEIO"
    market = 100_000.0


def _vw(posted, cause="62% of GATEIO, 15bps vs 41bps touch", fix="/operator widen ELA"):
    say = lambda c, t: posted.append(t)            # noqa: E731
    return watch.VolumeWatcher({"mm3": ["-1"]}, say,
                               shares_fn=lambda d, t, w: [_Venue()],
                               brief_fn=lambda v: (cause, fix), hourly=watch.Hourly(say, _Clock()))


def test_volume_baseline_says_nothing(sheet):
    sheet.update({"ELA": 0.91, "XCH": 0.16})
    posted = []
    _vw(posted).round_once()
    assert posted == []


def test_a_breach_must_hold_before_it_is_announced(sheet):
    sheet.update({"ELA": 0.16})
    posted = []
    w = _vw(posted)
    w.round_once()                               # baseline, inside the band
    sheet["ELA"] = 0.91
    for _ in range(watch.VOLUME_ROUNDS - 1):
        w.round_once()
        assert posted == []
    w.hourly.clock.next_hour()
    w.round_once()
    assert "**ELA** 91%" in posted[0] and "Above 50%" in posted[0]
    assert "`/operator widen ELA`" in posted[0]        # the command, at the end

    posted.clear()
    w.hourly.clock.next_hour()
    w.round_once()                               # still breached: do not nag
    assert posted == []


def test_a_coin_that_dips_outside_and_comes_back_is_never_announced(sheet):
    sheet.update({"ELA": 0.16})
    posted = []
    w = _vw(posted)
    w.round_once()
    for _ in range(10):
        w.hourly.clock.next_hour()
        sheet["ELA"] = 0.91
        w.round_once()
        sheet["ELA"] = 0.16
        w.round_once()
    assert posted == []


def test_a_snoozed_coin_is_skipped_entirely(sheet):
    sheet.update({"ELA": 0.16})
    posted, diagnosed = [], []
    say = lambda c, t: posted.append(t)            # noqa: E731
    w = watch.VolumeWatcher({"mm3": ["-1"]}, say,
                            shares_fn=lambda d, t, wn: diagnosed.append(t) or [],
                            brief_fn=lambda v: ("x", "y"), hourly=watch.Hourly(say, _Clock()))
    w.round_once()
    volume.snooze("mm3", "ELA", "u7", "boss wants it high")
    sheet["ELA"] = 0.91
    for _ in range(watch.VOLUME_ROUNDS + 1):
        w.hourly.clock.next_hour()
        w.round_once()
    assert posted == [] and diagnosed == []      # not even diagnosed: an exception costs nothing


def test_both_sides_of_the_band_are_flagged(sheet):
    sheet.update({"ELA": 0.16, "DNX": 0.16})
    posted = []
    w = _vw(posted)
    w.round_once()
    sheet.update({"ELA": 0.91, "DNX": 0.0})
    for _ in range(watch.VOLUME_ROUNDS):
        w.round_once()
    w.hourly.clock.next_hour()
    w.round_once()
    assert "**ELA** 91%" in posted[0] and "**DNX** 0%" in posted[0]


def test_a_sheet_that_fails_does_not_kill_the_round(monkeypatch):
    def boom(desk):
        raise RuntimeError("gspread exploded")
    monkeypatch.setattr(watch, "desk_shares", boom)
    posted = []
    watch.VolumeWatcher({"mm3": ["-1"]}, lambda c, t: posted.append(t)).round_once()
    assert posted == []


def test_an_in_band_coin_is_in_neither_group_and_brings_no_command():
    assert watch.render_volume("mm3", [("ELA", 0.47, "fine", "/operator do nothing")]) == ""
    text = watch.render_volume("mm3", [("ELA", 0.47, "fine", "/operator nope"),
                                       ("DNX", 0.0, "not quoting", "/operator rebound DNX")])
    assert "Below 5%" in text and "Above" not in text
    assert "DNX" in text and "ELA" not in text and "/operator nope" not in text





def test_a_held_volume_breach_is_delivered_on_the_next_hour(sheet):
    sheet.update({"ELA": 0.16})
    posted = []
    w = _vw(posted)
    w.round_once()
    sheet["ELA"] = 0.91
    for _ in range(watch.VOLUME_ROUNDS + 2):
        w.round_once()
    assert posted == [] and len(w.held["mm3"]) == 1
    w.hourly.clock.next_hour()
    w.round_once()
    assert "**ELA** 91%" in posted[0]

