"""Offline tests for the ICP-MM nightly snapshot (no network, no sheet)."""

from __future__ import annotations

import json

import pytest

from scripts import icp_snapshot as ics


class FakeSheet:
    """Answers values_batch_get in the order the ranges were asked for, like the real API."""

    def __init__(self, by_range: dict):
        self.by_range = by_range
        self.params: dict | None = None

    def values_batch_get(self, ranges, params=None):
        self.params = params
        return {"valueRanges": [{"range": r, "values": self.by_range.get(r)} for r in ranges]}


def _sheet(**overrides):
    by_range = {cell: [[i + 1.0]] for i, cell in enumerate(ics.SCALARS.values())}
    by_range["Dashboard!B2"] = [["2026/08/25, 23:31:04"]]
    by_range["GammaSlide!I4:M13"] = (
        [["CALL", 46373.5, 10.0 - i, 0, 0] for i in range(8)]   # rows 4-11, padded and empty
        + [["CALL", 46373.5, 2.76, 300000, 828000],             # row 12 -> $F$5
           ["PUT", 46286, 2.19, -550000, -1204500]]             # row 13 -> $H$5
    )
    by_range["GammaSlide!V26:Z34"] = []
    by_range.update(overrides)
    return FakeSheet(by_range)


# --------------------------------------------------------------------------- #
# read_snapshot
# --------------------------------------------------------------------------- #
def test_read_snapshot_asks_for_raw_numbers_not_formatted_text():
    """Formatted values arrive as '2,949,935' / '60%' -- strings that silently poison arithmetic."""
    sheet = _sheet()
    ics.read_snapshot(sheet)
    assert sheet.params == {"valueRenderOption": "UNFORMATTED_VALUE"}


def test_read_snapshot_maps_every_declared_scalar():
    snap = ics.read_snapshot(_sheet())
    assert set(ics.SCALARS) <= set(snap)
    assert snap["sheet_update_ts"] == "2026/08/25, 23:31:04"


def test_read_snapshot_keeps_the_delta_that_includes_options_distinct_from_spot_only():
    """A P&L built on delta_spot_coins would drop the option delta; on delta_total_coins it
    must NOT also revalue the options. Conflating the two is the whole trap."""
    snap = ics.read_snapshot(_sheet())
    assert snap["delta_spot_coins"] != snap["delta_total_coins"]


def test_read_snapshot_drops_zero_size_option_lines():
    book = ics.read_snapshot(_sheet())["options_owned"]
    assert [ln["strike"] for ln in book] == [2.76, 2.19]


def test_read_snapshot_keeps_a_short_option_line():
    """Written options carry negative size; a truthiness filter must not treat -550k as empty."""
    book = ics.read_snapshot(_sheet())["options_owned"]
    assert book[1]["size_coin"] == -550000


def test_read_snapshot_decodes_expiries_to_iso_dates():
    book = ics.read_snapshot(_sheet())["options_owned"]
    assert [ln["expiry"] for ln in book] == ["2026-12-17", "2026-09-21"]


def test_read_snapshot_tolerates_an_empty_option_block():
    assert ics.read_snapshot(_sheet())["options_internal"] == []


def test_read_snapshot_tolerates_a_blank_cell():
    snap = ics.read_snapshot(_sheet(**{"GammaSlide!S29": None}))
    assert snap["adj_s29"] is None


# --------------------------------------------------------------------------- #
# write_snapshot -- the series must not gain duplicate days
# --------------------------------------------------------------------------- #
def test_write_snapshot_appends_a_new_day(tmp_path):
    p = tmp_path / "s.jsonl"
    ics.write_snapshot({"date": "2026-08-24", "spot": 2.4}, p)
    ics.write_snapshot({"date": "2026-08-25", "spot": 2.5}, p)
    assert [json.loads(x)["date"] for x in p.read_text().splitlines()] == ["2026-08-24", "2026-08-25"]


def test_write_snapshot_replaces_a_rerun_of_the_same_day(tmp_path):
    p = tmp_path / "s.jsonl"
    ics.write_snapshot({"date": "2026-08-25", "spot": 2.4}, p)
    ics.write_snapshot({"date": "2026-08-25", "spot": 2.5}, p)
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["spot"] == 2.5


def test_write_snapshot_keeps_the_series_in_date_order(tmp_path):
    """A day-over-day P&L reads consecutive rows; out-of-order arrivals would pair the wrong days."""
    p = tmp_path / "s.jsonl"
    for d in ("2026-08-25", "2026-08-23", "2026-08-24"):
        ics.write_snapshot({"date": d}, p)
    assert [json.loads(x)["date"] for x in p.read_text().splitlines()] == [
        "2026-08-23", "2026-08-24", "2026-08-25"]


def test_write_snapshot_creates_the_state_directory(tmp_path):
    p = tmp_path / "nested" / "s.jsonl"
    ics.write_snapshot({"date": "2026-08-25"}, p)
    assert p.exists()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("serial", "want"), [(46373.5, "2026-12-17"), (46286, "2026-09-21"),
                                              ("", ""), (None, None)])
def test_date_decodes_sheets_serials(serial, want):
    assert ics._date(serial) == want


# --------------------------------------------------------------------------- #
# vol_history -- manual marks go stale rather than wrong; age is the only warning
# --------------------------------------------------------------------------- #
def _series(*vols, start=24):
    return [{"date": f"2026-08-{start + i:02d}", "vol_1": v, "vol_2": 0.65}
            for i, v in enumerate(vols)]


def test_vol_history_reports_the_mark_in_force():
    assert ics.vol_history(_series(0.60, 0.60))["vol_1"]["vol"] == 0.60


def test_vol_history_finds_the_day_a_mark_changed():
    h = ics.vol_history(_series(0.60, 0.60, 0.65, 0.65))["vol_1"]
    assert (h["changed_on"], h["prior"], h["days"], h["bounded"]) == ("2026-08-26", 0.60, 1, False)


def test_vol_history_reports_the_MOST_RECENT_change_not_the_first():
    """Scanning forward would freeze the age at the first-ever re-mark and never warn again."""
    h = ics.vol_history(_series(0.60, 0.65, 0.65, 0.70))["vol_1"]
    assert h["changed_on"] == "2026-08-27" and h["prior"] == 0.65 and h["days"] == 0


def test_vol_history_bounds_the_age_when_no_change_is_in_the_series_yet():
    """A fresh series must say 'at least N days', never a confident 0 -- the mark could be months old."""
    h = ics.vol_history(_series(0.60, 0.60, 0.60))["vol_1"]
    assert h["bounded"] is True and h["days"] == 2 and h["changed_on"] is None


def test_vol_history_on_a_single_row_claims_nothing():
    h = ics.vol_history(_series(0.60))["vol_1"]
    assert h["bounded"] is True and h["days"] == 0


def test_vol_history_ignores_float_noise():
    """0.6 re-typed as 0.6000000001 is not a re-mark."""
    assert ics.vol_history(_series(0.60, 0.6 + 1e-15))["vol_1"]["changed_on"] is None


def test_vol_history_tracks_the_two_cells_independently():
    rows = _series(0.60, 0.60)
    rows[1]["vol_2"] = 0.70
    h = ics.vol_history(rows)
    assert h["vol_2"]["changed_on"] == "2026-08-25" and h["vol_1"]["changed_on"] is None


def test_vol_history_skips_a_cell_missing_from_every_row():
    assert "vol_2" not in ics.vol_history([{"date": "2026-08-25", "vol_1": 0.6}])


def test_describe_vol_flags_a_remark_today():
    h = ics.vol_history(_series(0.60, 0.65))["vol_1"]
    assert "re-marked TODAY from 0.6" in ics.describe_vol(h)


def test_describe_vol_hedges_an_unproven_age():
    assert "at least" in ics.describe_vol(ics.vol_history(_series(0.60, 0.60))["vol_1"])


def test_describe_vol_states_a_known_age_plainly():
    msg = ics.describe_vol(ics.vol_history(_series(0.60, 0.65, 0.65, 0.65))["vol_1"])
    assert "unchanged for 2d" in msg and "at least" not in msg   # changed 08-25, latest 08-27


# --------------------------------------------------------------------------- #
# vol_cell -- the sheet wires vol to option line POSITIONALLY
# --------------------------------------------------------------------------- #
def test_read_snapshot_tags_each_option_line_with_the_vol_that_prices_it():
    """Row 13 reads $H$5, rows 4-12 read $F$5. A line that moves row silently reprices."""
    book = ics.read_snapshot(_sheet())["options_owned"]
    assert [ln["vol_cell"] for ln in book] == ["vol_1", "vol_2"]


def test_read_snapshot_does_not_tag_the_internal_block():
    """Only the owned block has the two-bucket wiring; inventing one for internal would be a guess."""
    sheet = _sheet(**{"GammaSlide!V26:Z34": [["CALL", 46286, 2.19, 100, 219]]})
    assert "vol_cell" not in ics.read_snapshot(sheet)["options_internal"][0]
