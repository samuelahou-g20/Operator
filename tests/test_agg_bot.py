"""Offline tests for agg_bot's Pluto portfolio read (no network — a stubbed requests.get)."""

from __future__ import annotations

import pytest
import requests

from scripts import agg_bot as ab


class _Resp:
    status_code = 200

    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


_GREEKS = {"delta_usds": {"BTCUSD": 100.0, "UNIUSD": -50.0},
           "gamma_usds": {"BTCUSD": 10.0, "UNIUSD": 5.0},
           "vegas": {"BTCUSD": 1.0, "UNIUSD": 2.0},
           "thetas": {"BTCUSD": -1.0, "UNIUSD": -2.0},
           "spotrefs": {"BTCUSD": 65000.0, "UNIUSD": 7.5}, "pv": -1234.0}


@pytest.fixture
def spy(monkeypatch):
    seen = {}

    def _get(url, headers=None, verify=None, timeout=None, params=None):
        seen["url"], seen["params"] = url, params
        return _Resp(_GREEKS)

    monkeypatch.setattr(requests, "get", _get)
    return seen


def test_portfolio_read_sends_the_risk_report_param(spy):
    """Pluto resolves RiskReport as a saved object name for user G20 — the same selector as the UI.
    Omitting it silently applies Pluto's default model, which does NOT tie out: on the live book it
    moved BTC delta by -17.6% and flipped the sign of UNI delta."""
    ab.read_portfolio(ab._TREASURY)
    assert spy["params"]["RiskReport"] == "STICKY STRIKE FITTED"
    assert spy["params"]["portfolio"] == "TREASURYPROPTOTAL"
    assert spy["params"]["detailed"] == "greeks"


def test_risk_report_name_matches_the_ui_label_exactly():
    """Underscored / concatenated forms are rejected by Pluto with HTTP 400."""
    assert ab._TREASURY["risk_report"] == "STICKY STRIKE FITTED"
    assert "_" not in ab._TREASURY["risk_report"]


def test_portfolio_read_omits_the_param_when_not_configured(spy):
    ab.read_portfolio({**ab._TREASURY, "risk_report": None})
    assert "RiskReport" not in spy["params"]


def test_portfolio_read_shapes_rows_and_totals(spy):
    df, pv = ab.read_portfolio(ab._TREASURY)
    assert list(df.columns) == ab._PORT_COLS
    assert df.iloc[0]["Asset"] == "BTC"          # sorted by |delta $|, USD suffix stripped
    assert df.iloc[-1]["Asset"] == "TOTAL"
    assert pv == pytest.approx(-1234.0)


def test_portfolio_read_rejects_an_unexpected_payload(monkeypatch):
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: _Resp({"message": "Could not execute risk report"}))
    with pytest.raises(RuntimeError, match="unexpected greeks payload"):
        ab.read_portfolio(ab._TREASURY)


# --------------------------------------------------------------------------- #
# options book — same Pluto greeks path as treasury, its own portfolio
# --------------------------------------------------------------------------- #
def test_options_desk_reads_its_own_pluto_portfolio(spy):
    ab.read_portfolio(ab._OPTIONS)
    assert spy["params"]["portfolio"] == "TOTAL_OPTIONS"
    assert spy["params"]["detailed"] == "greeks"
    assert spy["params"]["RiskReport"] == "STICKY STRIKE FITTED"


def test_options_desk_is_pluto_sourced_not_a_sheet():
    assert ab._OPTIONS["source"] == "pluto"
    assert "sheet" not in ab._OPTIONS and "gid" not in ab._OPTIONS


def test_options_and_treasury_are_distinct_portfolios():
    assert ab._OPTIONS["portfolio"] != ab._TREASURY["portfolio"]
    assert ab._OPTIONS["label"] != ab._TREASURY["label"]


# --------------------------------------------------------------------------- #
# Sheet-backed tables (/agg MM2 theta column, /mm2gamma)
# --------------------------------------------------------------------------- #
class _StubWs:
    def __init__(self, values):
        self._v = values

    def get_all_values(self):
        return self._v


def _stub_sheet(monkeypatch, values):
    """Point read_dashboard's gspread client at ``values`` instead of Google."""
    class _Sh:
        def get_worksheet_by_id(self, gid):
            return _StubWs(values)

    class _Gc:
        def open_by_key(self, key):
            return _Sh()

    monkeypatch.setattr(ab.gspread, "service_account", lambda **kw: _Gc())


def test_mm2_shows_theta_with_the_sheets_hard_line_break():
    """The MM2 header is literally 'Time-Decay\\nD+1'; a space-joined name would never match."""
    names = [name for name, _disp in ab._MM2["columns"]]
    assert "Time-Decay\nD+1" in names
    assert ("Time-Decay\nD+1", "Theta D+1") in ab._MM2["columns"]
    assert "Theta D+1" in ab._MM2["sign"]          # theta is negative — coloured like the greeks


def test_mm2_theta_sits_with_the_risk_columns():
    disp = [d for _n, d in ab._MM2["columns"]]
    assert disp.index("Daily Net Δ $") < disp.index("Theta D+1") < disp.index("Daily PnL")


def test_read_dashboard_honours_a_per_table_header_row(monkeypatch):
    """The gamma book's headers are on row 5, not the aggregator's row 3."""
    values = [[""], [""], [""], [""],
              ["", "Coin", "", "", "Price"],
              ["", "FET", "", "", "0.21"],
              ["", "ICP", "", "", "3.09"],
              ["", "", "", "", ""],
              ["", "IGNORED", "", "", "9"]]
    _stub_sheet(monkeypatch, values)
    cfg = {"sheet": "s", "gid": 1, "header_row": 4,
           "columns": [("Coin", "Coin"), ("Price", "Price")]}
    df = ab.read_dashboard(cfg)

    assert list(df.columns) == ["Coin", "Price"]
    assert list(df["Coin"]) == ["FET", "ICP"]      # stops at the blank Coin, never reads past it


def test_read_dashboard_defaults_to_row_2_when_unconfigured(monkeypatch):
    values = [[""], [""], ["Coin", "Price"], ["FET", "0.21"], ["", ""]]
    _stub_sheet(monkeypatch, values)
    df = ab.read_dashboard({"sheet": "s", "gid": 1,
                            "columns": [("Coin", "Coin"), ("Price", "Price")]})
    assert list(df["Coin"]) == ["FET"]


def test_mm2gamma_is_a_different_workbook_from_the_mm2_aggregator():
    assert ab._MM2GAMMA["sheet"] != ab._MM2["sheet"]
    assert ab._MM2GAMMA["header_row"] == 4
    assert ab._MM2GAMMA["share"] is None           # this book has no market-share column
    assert [d for _n, d in ab._MM2GAMMA["columns"]] == [
        "Coin", "Price", "Start Bal", "Curr Bal", "Opt MTM", "Pluto Δ", "Net Δ coins", "Net Δ $"]


def test_mm2gamma_column_names_match_the_sheet_verbatim():
    names = [n for n, _d in ab._MM2GAMMA["columns"]]
    assert "Starting\nBalance" in names and "Net Delta\n(Coin)" in names


def test_mm4_reuses_the_mm3_column_spec_on_its_own_workbook():
    """MM4's Dashboard carries every column MM3's view renders, under the same header names and
    on the same header row — so the desk differs only in `sheet`/`gid`/`label`."""
    assert ab._MM4["sheet"] != ab._MM3["sheet"]
    assert ab._MM4["columns"] == ab._MM3["columns"]
    assert ab._MM4["share"] == "Mkt Share"
    assert "header_row" not in ab._MM4            # headers on the default row, like MM3
