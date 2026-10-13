"""The MM watch: transitions only, baseline on the first round, and never a false recovery.

Both watches post through ``Hourly``: at most one message per desk per clock hour. Tests move the
hour with ``_Clock.next_hour`` right before the round that should speak."""

from __future__ import annotations

import json

import pytest

from src.the_operator import rapid, watch


def _entry(name, status, trading):
    return {"strategy_name": name, "latest_status": status, "latest_status_time_micros": 0,
            "last_error_status": "",
            "latest_status_details": json.dumps({"debug_info": {"trading": trading}})}


OK = "LM2Strategy:REPOL:POL-EUR:BITVAVO"
STOP = "LM2Strategy:REPARTI:PARTI-USDT:BINANCE"
ERR = "LM2Strategy:RENEAR:NEAR-USD:BITSTAMP"


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


def _watcher(posted, say=None):
    say = say or (lambda chat, text: posted.append((chat, text)))
    return watch.Watcher({"mm3": ["-1", "-2"]}, say, watch.Hourly(say, _Clock()))


def test_first_round_is_a_silent_baseline_not_a_wall_of_strategies(box, monkeypatch):
    """A restart says nothing in the chat, even with strategies already down."""
    _serve(monkeypatch, [_entry(OK, "TRADING", True), _entry(STOP, "STOP (DISABLED)", False)])
    posted = []
    w = _watcher(posted)
    w.round_once()
    w.hourly.clock.next_hour()
    w.round_once()
    assert posted == []


def test_a_silent_baseline_when_everything_is_quoting(box, monkeypatch):
    _serve(monkeypatch, [_entry(OK, "TRADING", True)])
    posted = []
    _watcher(posted).round_once()
    assert posted == []


def test_a_sustained_stop_is_announced_once_then_cleared_once(box, monkeypatch):
    _serve(monkeypatch, [_entry(OK, "TRADING", True), _entry(STOP, "TRADING", True)])
    posted = []
    w = _watcher(posted)
    w.round_once()                                # baseline, all fine
    w.round_once()                                # nothing changed
    assert posted == []

    _serve(monkeypatch, [_entry(OK, "TRADING", True), _entry(STOP, "STOP (DISABLED)", False)])
    for _ in range(watch.STOPPED_ROUNDS - 1):     # not yet held long enough
        w.round_once()
        assert posted == []
    w.hourly.clock.next_hour()
    w.round_once()
    text = posted[0][1]
    assert "Stopped quoting (1)" in text and "PARTI-USDT" in text
    assert "price left its band" in text and "/operator rebound PARTI" in text
    assert len(posted) == 2                       # posted to both chats

    posted.clear()
    w.hourly.clock.next_hour()
    w.round_once()                                # still stopped: do not nag
    assert posted == []

    _serve(monkeypatch, [_entry(OK, "TRADING", True), _entry(STOP, "TRADING", True)])
    for _ in range(watch.STOPPED_ROUNDS - 1):
        w.round_once()
        assert posted == []
    w.hourly.clock.next_hour()                    # the recovery lands in the next hourly message
    w.round_once()
    assert "Quoting again (1)" in posted[0][1]


def test_a_price_flapping_across_its_band_edge_is_never_announced(box, monkeypatch):
    """The whole reason this exists: a mid sitting on the edge crosses many times an hour."""
    posted = []
    w = _watcher(posted)
    _serve(monkeypatch, [_entry(STOP, "TRADING", True)])
    w.round_once()                                # baseline
    for _ in range(20):
        w.hourly.clock.next_hour()
        _serve(monkeypatch, [_entry(STOP, "STOP (DISABLED)", False)])
        w.round_once()
        _serve(monkeypatch, [_entry(STOP, "TRADING", True)])
        w.round_once()
    assert posted == []


def test_a_recovery_is_silent_when_the_stop_was_never_announced(box, monkeypatch):
    posted = []
    w = _watcher(posted)
    _serve(monkeypatch, [_entry(STOP, "TRADING", True)])
    w.round_once()
    _serve(monkeypatch, [_entry(STOP, "STOP (DISABLED)", False)])
    w.round_once()                                # one round bad: below the threshold
    _serve(monkeypatch, [_entry(STOP, "TRADING", True)])
    for _ in range(watch.STOPPED_ROUNDS + 1):
        w.hourly.clock.next_hour()
        w.round_once()
    assert posted == []


def test_an_error_is_announced_separately_from_a_deliberate_stop(box, monkeypatch):
    _serve(monkeypatch, [_entry(ERR, "TRADING", True), _entry(STOP, "TRADING", True)])
    posted = []
    w = _watcher(posted)
    w.round_once()
    _serve(monkeypatch, [_entry(ERR, "STOP (ERROR)", False),
                         _entry(STOP, "STOP (DISABLED)", False)])
    for _ in range(watch.STOPPED_ROUNDS):
        w.round_once()
    w.hourly.clock.next_hour()
    w.round_once()
    text = posted[0][1]
    assert "Stopped in error (1)" in text and "NEAR-USD" in text
    assert "Stopped quoting (1)" in text and "PARTI-USDT" in text


def test_a_box_that_stops_answering_neither_alarms_nor_clears(box, monkeypatch):
    """A strategy missing this round means its box did not reply — that is not news either way."""
    tracks = {OK: watch.Track("ok", streak=9), STOP: watch.Track("stopped", streak=9,
                                                                announced=True)}
    assert watch.settle(tracks, {OK: "ok"}) == ([], [])
    assert tracks[STOP].streak == 9 and tracks[STOP].announced   # untouched by the gap


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
    _serve(monkeypatch, [_entry(STOP, "TRADING", True)])
    w.round_once()
    _serve(monkeypatch, [_entry(STOP, "STOP (DISABLED)", False)])
    for _ in range(watch.STOPPED_ROUNDS):
        w.round_once()
    w.hourly.clock.next_hour()
    w.round_once()
    assert [c for c, _ in posted] == ["-2"]


def test_long_outages_are_truncated(box, monkeypatch):
    many = [_entry(f"LM2Strategy:REX:X{i}-USDT:BINANCE", "TRADING", True) for i in range(40)]
    _serve(monkeypatch, many)
    posted = []
    w = _watcher(posted)
    w.round_once()
    _serve(monkeypatch, [_entry(e["strategy_name"], "STOP (DISABLED)", False) for e in many])
    for _ in range(watch.STOPPED_ROUNDS):
        w.round_once()
    w.hourly.clock.next_hour()
    w.round_once()
    assert f"…and {40 - watch.MAX_LINES} more" in posted[0][1]


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


def test_what_is_already_broken_at_startup_is_never_replayed(box, monkeypatch):
    """A restart must not paste the whole standing outage into the chat a few rounds later."""
    _serve(monkeypatch, [_entry(STOP, "STOP (DISABLED)", False), _entry(OK, "TRADING", True)])
    posted = []
    w = _watcher(posted)
    w.round_once()                                  # baseline: silent
    for _ in range(watch.STOPPED_ROUNDS + 2):
        w.hourly.clock.next_hour()
        w.round_once()
    assert posted == []

    _serve(monkeypatch, [_entry(STOP, "TRADING", True), _entry(OK, "TRADING", True)])
    for _ in range(watch.STOPPED_ROUNDS):
        w.round_once()
    w.hourly.clock.next_hour()
    w.round_once()
    assert "Quoting again (1)" in posted[0][1]      # but its recovery still lands


def test_news_is_held_until_the_next_hour_and_a_start_waits_too(box, monkeypatch):
    """Scan often so an outage is found in minutes; tell the chat once an hour, at :00. The hour
    the watch starts in is spent, so news found right after a restart waits for :00 as well."""
    _serve(monkeypatch, [_entry(OK, "TRADING", True), _entry(STOP, "TRADING", True)])
    posted = []
    w = _watcher(posted)
    w.round_once()                                  # baseline
    _serve(monkeypatch, [_entry(OK, "TRADING", True), _entry(STOP, "STOP (DISABLED)", False)])
    for _ in range(watch.STOPPED_ROUNDS + 3):
        w.round_once()
    assert posted == []                             # held, not dropped
    assert w.held["mm3"][0] == [STOP]

    w.hourly.clock.next_hour()                      # :00
    w.round_once()
    assert "PARTI-USDT" in posted[0][1] and "mm3" not in w.held


def test_news_found_after_the_hour_was_spent_waits_for_the_next_one(box, monkeypatch):
    """An empty :00 still spends the hour: a stop found at :20 goes out at the next :00."""
    _serve(monkeypatch, [_entry(STOP, "TRADING", True)])
    posted = []
    w = _watcher(posted)
    w.round_once()
    w.hourly.clock.next_hour()
    w.round_once()                                  # :00, nothing to say
    _serve(monkeypatch, [_entry(STOP, "STOP (DISABLED)", False)])
    for _ in range(watch.STOPPED_ROUNDS):
        w.round_once()
    assert posted == []
    w.hourly.clock.next_hour()
    w.round_once()
    assert "PARTI-USDT" in posted[0][1]


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


def test_both_watches_share_one_message_per_hour(box, monkeypatch, sheet):
    """The desk hears once an hour in total, not once per watch."""
    posted = []
    say = lambda chat, text: posted.append((chat, text))   # noqa: E731
    hourly = watch.Hourly(say, _Clock())
    mm = watch.Watcher({"mm3": ["-1"]}, say, hourly)
    vw = watch.VolumeWatcher({"mm3": ["-1"]}, say, shares_fn=lambda d, t, w: [_Venue()],
                             brief_fn=lambda v: ("cause", "/operator widen ELA"), hourly=hourly)
    _serve(monkeypatch, [_entry(STOP, "TRADING", True)])
    sheet.update({"ELA": 0.16})
    mm.round_once()
    vw.round_once()
    _serve(monkeypatch, [_entry(STOP, "STOP (DISABLED)", False)])
    sheet["ELA"] = 0.91
    for _ in range(watch.STOPPED_ROUNDS):
        mm.round_once()
        vw.round_once()
    hourly.clock.next_hour()
    vw.round_once()                                 # whichever watch reaches :00 first posts both
    mm.round_once()
    assert len(posted) == 1
    assert "MM watch" in posted[0][1] and "Volume watch" in posted[0][1]
