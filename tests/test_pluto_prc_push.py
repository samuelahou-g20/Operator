"""Offline tests for the Pluto PRC sheet push (no Pluto, no Google)."""

from __future__ import annotations

import json

import pytest
import requests

from scripts import pluto_prc_push as pp


def _entry(code="C", biprod="ICPUSD", ntl=550_000.0, k=2.19, cpty="DB-MM2GAMMA_ICP", **greeks):
    v = [None] * 40
    v[1], v[2], v[3], v[4] = 147_852.78123, 393_931.641, 393_931.641, 937_557.3
    v[5] = v[6] = 10_886.77
    v[7], v[8], v[11] = 25_910.51, 1_181.66, -1_100.26
    for i, val in greeks.items():
        v[int(i[1:])] = val
    return {"posn": {"code": code, "biprod": biprod, "ntl_ccy": biprod[:3],
                     "ntl_amount": ntl, "K": k, "expiry_date": "2026-09-21T12:00:00Z",
                     "delivery_date": "2026-09-21T12:00:00Z", "expiry_cutoff": "12:00 UTC",
                     "counterparty": cpty, "settle_style": "Physical",
                     "was_rolled": False, "contract_size": 1.0,
                     "reval_vol": 0.6423071234, "official_vol": 0.6423071234,
                     "official_ref": "", "client": "", "instrument": "", "ave_px": None,
                     "official_pv": 0.0, "floating_profit_loss": 0.0,
                     "realized_profit_loss": 0.0, "total_profit_loss": 0.0},
            "greek": {"_values": v, "pv_ccy": "USD", "biprod": biprod}}


def _greeks(pv=335_013.0):
    b = ["ICPUSD", "POLUSD"]
    f = lambda a, c: dict(zip(b, [a, c]))  # noqa: E731
    return {"pv": pv, "deltas": f(26_802.91, -519_406.66), "delta_usds": f(63_790.92, -59_726.57),
            "gammas": f(14_069.38, 115_013.59), "gamma_usds": f(33_485.12, 13_225.41),
            "vegas": f(2_728.03, 1_803.38), "thetas": f(-1_438.26, -670.94),
            "spotrefs": f(2.38, 0.11499)}


# --------------------------------------------------------------------------- #
# summary block
# --------------------------------------------------------------------------- #
def test_summary_pv_is_the_per_biprod_position_sum_not_the_biprod_name():
    """Regression: Biprod and PV both have no source field, so PV once printed 'ICPUSD'."""
    pos = [_entry(biprod="ICPUSD"), _entry(biprod="ICPUSD"), _entry(biprod="POLUSD")]
    grid = pp.summary_block(pos, _greeks())
    icp = next(r for r in grid[1:] if r[0] == "ICPUSD")
    assert icp[2] == pytest.approx(2 * 147_852.78, abs=0.02)


def test_delta_mirrors_pluto_own_aggregate():
    """The Dashboard is reconciled against the Pluto summary tab, so delta must be Pluto's own
    per-biprod number -- every leg, CASH1 spot/loan balances included. Summing only the option
    and FX legs left POL at +2.37m against Pluto's -398,407, because POL's hedge is booked
    CASH1 rather than FX."""
    g = _greeks()
    g["deltas"]["ICPUSD"] = -398_407.0
    row = next(r for r in pp.summary_block([_entry(biprod="ICPUSD")], g)[1:] if r[0] == "ICPUSD")
    assert row[3] == -398_407.0


def test_pv_is_the_book_not_the_aggregate():
    """PV is summed per biprod -- Pluto's `pv` is portfolio-wide and cannot be split."""
    g = _greeks(pv=999_999_999.0)
    row = next(r for r in pp.summary_block([_entry(biprod="ICPUSD")], g)[1:] if r[0] == "ICPUSD")
    assert row[2] == pytest.approx(147_852.78, abs=0.01)


def test_summary_has_one_row_per_biprod_sorted():
    grid = pp.summary_block([_entry(biprod="POLUSD"), _entry(biprod="ICPUSD")], _greeks())
    assert [r[0] for r in grid[1:3]] == ["ICPUSD", "POLUSD"]


def test_summary_has_no_total_row():
    """These tabs get QUERY'd; a TOTAL row would be double-counted by any SUM over the range."""
    grid = pp.summary_block([_entry()], _greeks())
    assert not any(str(r[0]).upper() == "TOTAL" for r in grid)


def test_summary_carries_both_totals_in_the_metadata_instead():
    """Both totals are kept so the summed column can be reconciled against Pluto's own."""
    grid = pp.summary_block([_entry()], _greeks(pv=335_013.0))
    flat = [c for r in grid for c in r]
    assert "Summed PV" in flat and "Portfolio PV (Pluto)" in flat and 335_013.0 in flat


def test_summary_cash_legs_cancel_within_a_biprod():
    """The two ICP CASH1 legs are +/-476,000; a per-biprod PV must net them to zero."""
    pos = [_entry(code="CASH1", **{"_1": 476_000.0}), _entry(code="CASH1", **{"_1": -476_000.0})]
    assert pp.summary_block(pos, _greeks())[1][2] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# grid mechanics -- one contiguous rectangle per tab
# --------------------------------------------------------------------------- #
def test_grid_is_sized_to_the_data_not_padded():
    """Stale rows are removed by the write clearing the tab, so there is no pad to guess at."""
    g = dict(_greeks())
    many = {f"X{i}USD": 1.0 for i in range(10)}
    for k in ("deltas", "delta_usds", "gammas", "gamma_usds", "vegas", "thetas", "spotrefs"):
        g[k] = many
    assert len(pp.summary_block([], g)) == 11


def test_grid_is_at_least_as_tall_as_the_metadata_block():
    """A 2-biprod book still needs the metadata's full height or it would be truncated."""
    grid = pp.summary_block([_entry()], _greeks())
    assert len(grid) == 8
    assert grid[7][-2] == "Portfolio PV (Pluto)"


def test_grid_separates_metadata_from_the_data_by_a_blank_column():
    """A QUERY over the data range must not pick the metadata block up."""
    grid = pp.summary_block([_entry()], _greeks())
    assert grid[0][len(pp.SUM_COLS)] == ""
    assert grid[0][len(pp.SUM_COLS) + 1] == "metric"


def test_grid_metadata_names_the_risk_report_actually_used():
    """Omitting riskreport silently applies Pluto's DEFAULT and moves delta ~10%."""
    flat = [c for r in pp.summary_block([_entry()], _greeks()) for c in r]
    assert pp.RISK_REPORT in flat and pp.RISK_REPORT == "STICKY STRIKE FITTED"


def test_grid_is_rectangular():
    """One rectangle anchored at A1; a ragged row would shift columns."""
    grid = pp.summary_block([_entry()], _greeks())
    assert len({len(r) for r in grid}) == 1


def test_no_none_reaches_the_sheet():
    """A JSON null in a values payload is rejected; blanks must be empty strings."""
    grid = pp.summary_block([_entry()], _greeks())
    assert all(c is not None for r in grid for c in r)


# --------------------------------------------------------------------------- #
# call budget
# --------------------------------------------------------------------------- #
def test_fetch_makes_exactly_two_pluto_calls(monkeypatch):
    calls = []

    class R:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {}

    monkeypatch.setattr(requests, "get", lambda *a, **k: (calls.append(k["params"]), R())[1])
    monkeypatch.setenv("PLUTO_API_KEY", "k")
    monkeypatch.setenv("PLUTO_API_SECRET", "s")
    pp.fetch()
    assert [c["detailed"] for c in calls] == ["positions", "greeks"]
    assert all(c["riskreport"] == pp.RISK_REPORT for c in calls)


def _fake_sheets(monkeypatch, sent):
    class WS:
        def __init__(self, t): self.id = {pp.TAB_SUM: 22}[t]

    class SH:
        def worksheet(self, t): return WS(t)
        def batch_update(self, body): sent.append(body)

    monkeypatch.setattr(pp.gspread, "service_account",
                        lambda **kw: type("C", (), {"open_by_key": lambda s, k: SH()})())


def test_push_sends_one_call_for_the_summary_tab(monkeypatch):
    sent = []
    _fake_sheets(monkeypatch, sent)
    pp.push({pp.TAB_SUM: [["b"]]})
    assert len(sent) == 1
    assert [r["updateCells"]["range"]["sheetId"] for r in sent[0]["requests"]] == [22]


def test_push_range_is_unbounded_so_the_whole_tab_is_cleared(monkeypatch):
    """THE mechanism: an end index would leave anything past it behind. This test is the
    regression guard for a book that shrinks (they roll daily)."""
    sent = []
    _fake_sheets(monkeypatch, sent)
    pp.push({pp.TAB_SUM: [["b"]]})
    for r in sent[0]["requests"]:
        rng = r["updateCells"]["range"]
        assert rng["startRowIndex"] == 0 and rng["startColumnIndex"] == 0
        assert "endRowIndex" not in rng and "endColumnIndex" not in rng
        assert r["updateCells"]["fields"] == "userEnteredValue"


@pytest.mark.parametrize(("raw", "want"), [
    (2.5, {"numberValue": 2.5}), (7, {"numberValue": 7}),
    (True, {"boolValue": True}), (False, {"boolValue": False}),
    ("CASH1", {"stringValue": "CASH1"}), ("", {}), (None, {}),
])
def test_value_types_each_cell(raw, want):
    assert pp._value(raw) == want


def test_value_keeps_booleans_out_of_numberValue():
    """bool is an int subclass -- ordering the checks wrongly writes TRUE as 1."""
    assert pp._value(True) == {"boolValue": True}


def test_value_of_blank_is_empty_so_updateCells_clears_it():
    """An empty ExtendedValue is what wipes last cycle's longer book."""
    assert pp._value("") == {}


# --------------------------------------------------------------------------- #
# failure handling -- a bad cycle waits for the next tick; a sustained outage alerts
# --------------------------------------------------------------------------- #
@pytest.fixture
def failstate(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "FAIL_STATE", tmp_path / "fail.json")
    return pp.FAIL_STATE


@pytest.fixture
def alerts(monkeypatch):
    sent = []
    monkeypatch.setattr(pp, "alert_mm2", sent.append)
    return sent


def _run(monkeypatch, ok: bool):
    monkeypatch.setattr("sys.argv", ["pluto_prc_push"])
    if ok:
        monkeypatch.setattr(pp, "fetch", lambda: ([_entry()], _greeks()))
        monkeypatch.setattr(pp, "push", lambda blocks: None)
    else:
        def boom():
            raise RuntimeError("APIError: [503]: The service is currently unavailable.")
        monkeypatch.setattr(pp, "fetch", boom)
    return pp.main()


def test_a_failed_cycle_is_not_fatal_and_leaves_the_tabs_alone(failstate, alerts, monkeypatch):
    """The write is atomic, so a failure means last good data stands until the next tick."""
    assert _run(monkeypatch, ok=False) == 1
    assert alerts == []


def test_failures_accumulate_across_runs(failstate, alerts, monkeypatch):
    """Each tick is its own process, so the streak must survive on disk."""
    for _ in range(3):
        _run(monkeypatch, ok=False)
    assert json.loads(failstate.read_text())["streak"] == 3


def test_no_alert_at_or_below_the_threshold(failstate, alerts, monkeypatch):
    for _ in range(pp.ALERT_AFTER):
        _run(monkeypatch, ok=False)
    assert alerts == []


def test_alerts_once_the_streak_passes_the_threshold(failstate, alerts, monkeypatch):
    for _ in range(pp.ALERT_AFTER + 1):
        _run(monkeypatch, ok=False)
    assert len(alerts) == 1
    assert "STALE" in alerts[0] and "503" in alerts[0]


def test_alerts_only_once_per_outage_not_once_a_minute(failstate, alerts, monkeypatch):
    """At a 60s cadence, re-alerting every tick would be 60 messages an hour."""
    for _ in range(pp.ALERT_AFTER + 25):
        _run(monkeypatch, ok=False)
    assert len(alerts) == 1


def test_success_resets_the_streak(failstate, alerts, monkeypatch):
    for _ in range(3):
        _run(monkeypatch, ok=False)
    assert _run(monkeypatch, ok=True) == 0
    assert json.loads(failstate.read_text())["streak"] == 0


def test_a_later_outage_alerts_again_after_a_recovery(failstate, alerts, monkeypatch):
    """Recovery must clear `alerted`, or the second outage is silent."""
    for _ in range(pp.ALERT_AFTER + 1):
        _run(monkeypatch, ok=False)
    _run(monkeypatch, ok=True)
    for _ in range(pp.ALERT_AFTER + 1):
        _run(monkeypatch, ok=False)
    assert len(alerts) == 2


def test_one_success_inside_an_outage_restarts_the_count(failstate, alerts, monkeypatch):
    """'In a row' means in a row -- an intermittent sheet must not accumulate to an alert."""
    for _ in range(20):
        _run(monkeypatch, ok=False)
        _run(monkeypatch, ok=True)
    assert alerts == []


def test_a_corrupt_state_file_does_not_stop_the_push(failstate, alerts, monkeypatch):
    failstate.write_text("{not json")
    assert _run(monkeypatch, ok=True) == 0


def test_a_broken_alert_does_not_mask_the_outage(failstate, monkeypatch):
    """If Telegram is down the failure must still be recorded and reported as exit 1."""
    def bad(_):
        raise RuntimeError("telegram unreachable")
    monkeypatch.setattr(pp, "alert_mm2", bad)
    for _ in range(pp.ALERT_AFTER + 1):
        rc = _run(monkeypatch, ok=False)
    assert rc == 1
    assert json.loads(failstate.read_text())["streak"] == pp.ALERT_AFTER + 1


def test_alert_goes_to_the_mm2_chat(monkeypatch):
    calls = []
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_MM2_CHAT_ID", "-100111, -100222")
    monkeypatch.setattr(pp, "send", lambda api, chat, text: calls.append((api, chat, text)))
    pp.alert_mm2("boom")
    assert [c[1] for c in calls] == ["-100111", "-100222"]
    assert calls[0][0].endswith("/bottok")


def test_positions_are_still_fetched_for_per_biprod_pv():
    """Pluto's `pv` is one portfolio-wide scalar. Dropping the positions call to save a request
    would silently zero the PV column the Dashboard reads as "Live Options MTM"."""
    grid = pp.summary_block([_entry(biprod="ICPUSD")], _greeks())
    assert grid[1][2] == pytest.approx(147_852.78, abs=0.01)
    assert pp.summary_block([], _greeks())[1][2] == 0


# --------------------------------------------------------------------------- #
# option filtering -- the tab feeds columns named "Options ..."
# --------------------------------------------------------------------------- #
def test_pv_includes_the_intra_desk_loans():
    """Every leg counts. The loans make NEAR's PV read about -329,000 on a long call book --
    that is the intended number, not a contaminant."""
    loan = _entry(code="CASH1", cpty="DB-MM2_GAMMA_LOANS", **{"_1": -374_400.0})
    base = pp.summary_block([_entry()], _greeks())[1][2]
    assert pp.summary_block([_entry(), loan], _greeks())[1][2] == pytest.approx(base - 374_400.0)


@pytest.mark.parametrize("code", ["C", "P", "FX", "CASH1", "CASH2", "PERPF", "F1C"])
def test_pv_includes_every_leg_whatever_its_code(code):
    """Filtering by code was an earlier mistake: POL's hedge is a CASH1 like its loan is."""
    assert pp.summary_block([_entry(code=code, cpty="DB-MM2GAMMA_POL")], _greeks())[1][2] != 0


@pytest.mark.parametrize("cpty", ["BINANCEPM", "DERIBIT", "DB-MM2GAMMA_POL",
                                  "DB-MM2_GAMMA_LOANS", "DB-MMGAMMA_NEAR"])
def test_pv_includes_every_counterparty(cpty):
    grid = pp.summary_block([_entry(code="CASH1", cpty=cpty, **{"_1": 663_933.0})], _greeks())
    assert grid[1][2] == pytest.approx(663_933.0)


def test_summed_pv_reconciles_to_pluto_portfolio_scalar():
    """The per-biprod column must add up to Pluto's own total; both are in the metadata so a
    divergence is visible rather than silent."""
    pos = [_entry(biprod="ICPUSD", **{"_1": 100.0}), _entry(biprod="POLUSD", **{"_1": 250.0})]
    flat = [c for r in pp.summary_block(pos, _greeks(pv=350.0)) for c in r]
    assert flat[flat.index("Summed PV") + 1] == pytest.approx(350.0)
    assert flat[flat.index("Portfolio PV (Pluto)") + 1] == 350.0


def test_spot_ref_still_comes_from_the_aggregate():
    """It is market data, not a position sum -- a cash-only biprod must still show its spot."""
    grid = pp.summary_block([], _greeks())
    assert next(r for r in grid[1:] if r[0] == "ICPUSD")[1] == 2.38
