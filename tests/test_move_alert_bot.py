"""Offline tests for move_alert_bot: delta reading, watchlist, move maths, message (no network)."""

from __future__ import annotations

from collections import deque

import pytest

from scripts import move_alert_bot as m
from src.adapters.errors import AdapterError


# --------------------------------------------------------------------------- #
# read_deltas -- by header name, never by column position
# --------------------------------------------------------------------------- #
def _sheet(rows, hdr=None):
    hdr = hdr or ["Coin", "Price", "Net Delta ($)"]
    return [[], [], hdr, *rows]


class FakeGC:
    def __init__(self, vals): self.vals = vals

    def open_by_key(self, _k):
        vals = self.vals

        class WS:
            def get_all_values(self): return vals

        return type("SH", (), {"get_worksheet_by_id": lambda s, g: WS()})()


@pytest.fixture
def sheet(monkeypatch):
    def install(vals):
        monkeypatch.setattr(m.gspread, "service_account", lambda **kw: FakeGC(vals))
    return install


def test_read_deltas_parses_coin_and_delta(sheet):
    sheet(_sheet([["ICP", "2.42", "146,860"], ["ETH", "2500", "-56,629"]]))
    assert m.read_deltas({"sheet": "k", "gid": 0}) == {"ICP": 146_860.0, "ETH": -56_629.0}


def test_read_deltas_finds_the_column_by_name_not_position(sheet):
    """A column inserted upstream must not silently shift this onto the wrong figure."""
    sheet(_sheet([["ICP", "x", "2.42", "146,860"]],
                 hdr=["Coin", "NEW COL", "Price", "Net Delta ($)"]))
    assert m.read_deltas({"sheet": "k", "gid": 0})["ICP"] == 146_860.0


def test_read_deltas_raises_when_the_column_is_gone(sheet):
    """Silently returning no deltas would make the bot go quiet, not fail."""
    sheet(_sheet([["ICP", "2.42"]], hdr=["Coin", "Price"]))
    with pytest.raises(RuntimeError, match="Net Delta"):
        m.read_deltas({"sheet": "k", "gid": 0})


def test_read_deltas_stops_at_the_totals_row(sheet):
    """ALL is the sum, not a token -- polling a price for it is meaningless."""
    sheet(_sheet([["ICP", "2.42", "100"], ["ALL", "", "999999"], ["JUNK", "1", "5"]]))
    d = m.read_deltas({"sheet": "k", "gid": 0})
    assert "ALL" not in d and "JUNK" not in d


def test_read_deltas_stops_at_a_blank_coin(sheet):
    sheet(_sheet([["ICP", "2.42", "100"], ["", "", ""], ["BELOW", "1", "5"]]))
    assert "BELOW" not in m.read_deltas({"sheet": "k", "gid": 0})


@pytest.mark.parametrize(("raw", "want"), [("146,860", 146_860.0), ("$1,234.5", 1234.5),
                                           ("(2,000)", -2000.0), ("-56,629", -56_629.0),
                                           ("", 0.0), ("n/a", 0.0), ("-", 0.0)])
def test_num_handles_the_sheet_spellings(raw, want):
    assert m._num(raw) == want


# --------------------------------------------------------------------------- #
# watchlist -- the floor is what keeps the CoinRoutes cost affordable
# --------------------------------------------------------------------------- #
def test_watchlist_keeps_only_material_deltas():
    d = {"BIG": 60_000.0, "SHORT": -80_000.0, "SMALL": 1_000.0}
    assert m.watchlist(d, set()) == ["BIG", "SHORT"]


def test_watchlist_is_sign_blind():
    """A short book is exactly as exposed to a move as a long one."""
    assert m.watchlist({"S": -60_000.0}, set()) == ["S"]


def test_watchlist_drops_tokens_known_to_have_no_book():
    """CAD/AUD/CHF/JPY and non-instrument rows like EXCHANGE 404; retrying each poll burns a
    call forever to learn the same thing."""
    assert m.watchlist({"EUR": 400_000.0, "CAD": 300_000.0}, {"CAD"}) == ["EUR"]


# --------------------------------------------------------------------------- #
# fetch_mids
# --------------------------------------------------------------------------- #
class FakeCR:
    def __init__(self, mids): self.mids, self.seen = mids, []

    def mid(self, symbol, markets=None):
        self.seen.append((symbol, markets))
        v = self.mids.get(symbol)
        if v is None:
            raise AdapterError(f"no book for {symbol}", venue="coinroutes")
        return v


def test_fetch_mids_returns_prices_and_flags_the_unpriceable():
    cr = FakeCR({"ICP-USDT": 2.42})
    mids, bad = m.fetch_mids(cr, ["ICP", "CAD"])
    assert mids == {"ICP": 2.42} and bad == {"CAD"}


def test_fetch_mids_restricts_an_ambiguous_ticker_to_its_own_venues():
    """AI-USDT is a different asset on binance/huobi; unrestricted the mid mixes two assets."""
    cr = FakeCR({"AI-USDT": 0.0209})
    m.fetch_mids(cr, ["AI"])
    assert ("AI-USDT", "OKEX;KUCOIN;BITGET") in cr.seen


def test_fetch_mids_leaves_unambiguous_tokens_unrestricted():
    cr = FakeCR({"ICP-USDT": 2.42})
    m.fetch_mids(cr, ["ICP"])
    assert ("ICP-USDT", None) in cr.seen


def test_fetch_mids_of_nothing_makes_no_calls():
    cr = FakeCR({})
    assert m.fetch_mids(cr, []) == ({}, set()) and cr.seen == []


# --------------------------------------------------------------------------- #
# window_move -- measured against the oldest sample still in the window
# --------------------------------------------------------------------------- #
def test_window_move_needs_two_samples():
    assert m.window_move(deque([(1000.0, 100.0)]), 1000.0) is None


def test_window_move_is_measured_from_the_oldest_sample_in_the_window():
    h = deque([(1000.0, 100.0), (1060.0, 102.0), (1120.0, 97.0)])
    pct, old, new = m.window_move(h, 1120.0)
    assert (old, new) == (100.0, 97.0) and pct == pytest.approx(-0.03)


def test_window_move_prunes_samples_older_than_the_window():
    """Otherwise the "5m move" silently becomes a since-start-of-day move."""
    stale = 1000.0
    h = deque([(stale, 50.0), (stale + 10_000, 100.0), (stale + 10_060, 101.0)])
    pct, old, _new = m.window_move(h, stale + 10_060)
    assert old == 100.0 and pct == pytest.approx(0.01)


def test_window_move_keeps_at_least_one_sample_when_all_are_stale():
    """Pruning to empty would restart the history and never produce a move at all."""
    h = deque([(0.0, 100.0), (1.0, 101.0)])
    assert m.window_move(h, 1e9) is None
    assert len(h) == 1


def test_window_move_survives_a_zero_price():
    assert m.window_move(deque([(1.0, 0.0), (2.0, 5.0)]), 2.0) is None


# --------------------------------------------------------------------------- #
# alert_text
# --------------------------------------------------------------------------- #
def test_alert_is_one_line():
    assert "\n" not in m.alert_text("MM3", "NYM", -0.032, 146_860.0, -4_699.5, 0.03012, 0.029156)


def test_alert_reads_as_expected():
    assert m.alert_text("MM3", "NYM", -0.032, 146_860.0, -4_699.5, 0.03012, 0.029156) == (
        "🔴 MM3 NYM -3.2%/5m · Δ +$146.9k · MtM -$4.7k · 0.03012 → 0.029156")


def test_arrow_follows_the_impact_not_the_price():
    """A fall is good news on a short book. The alert says "this is costing you", not "this fell"."""
    down_on_short = m.alert_text("MM3", "ETH", -0.021, -56_629.0, 1_189.2, 2542.0, 2489.2)
    down_on_long = m.alert_text("MM3", "NYM", -0.032, 146_860.0, -4_699.5, 0.0301, 0.0292)
    assert down_on_short.startswith("🟢") and down_on_long.startswith("🔴")


def test_alert_prices_big_and_small_tokens_readably():
    assert "2,542.00 → 2,489.20" in m.alert_text("MM3", "ETH", -0.021, -1.0, 1.0, 2542.0, 2489.2)
    assert "0.03012" in m.alert_text("MM3", "NYM", -0.03, 1.0, -1.0, 0.03012, 0.0292)


# --------------------------------------------------------------------------- #
# thresholds -- the defaults must actually be reachable on these books
# --------------------------------------------------------------------------- #
def test_the_default_threshold_is_reachable_on_the_real_book():
    """A $10,000 default needed a 7% move on MM3's largest position and would never have fired."""
    assert m.THRESHOLD / 146_860.0 < 0.02


def test_a_minimum_percent_move_is_required_as_well_as_a_dollar_impact():
    """The impact test alone scales the wrong way -- a bigger book trips on a smaller move, so a
    large position would alert on noise."""
    assert m.MIN_PCT > 0
    tiny_move_big_book = 5_000_000.0 * 0.001
    assert tiny_move_big_book >= m.THRESHOLD and 0.001 < m.MIN_PCT


# --------------------------------------------------------------------------- #
# should_alert -- one move, one message
# --------------------------------------------------------------------------- #
def test_first_sighting_alerts():
    assert m.should_alert(None, -4_469.0, 1000.0) is True


def test_a_move_that_merely_persists_stays_quiet():
    """"It is still bad" is not news. Re-arming on decay produced repeat alerts on every wobble."""
    assert m.should_alert((1000.0, -4_469.0), -3_099.0, 1194.0) is False


def test_a_sign_flip_inside_the_cooldown_stays_quiet():
    """THE regression: as the window rolls, the reference advances and the impact crosses zero on
    its own. Treating that as a new event gave four NYM alerts in eight minutes."""
    assert m.should_alert((1000.0, -4_469.0), 2_968.0, 1404.0) is False


def test_a_materially_worse_move_alerts_inside_the_cooldown():
    """Escalation is the one thing worth interrupting for while a move is already known."""
    assert m.should_alert((1000.0, -3_000.0), -4_600.0, 1100.0) is True


def test_escalation_is_measured_on_magnitude_not_direction():
    assert m.should_alert((1000.0, -3_000.0), 4_600.0, 1100.0) is True


def test_the_same_move_alerts_again_after_the_cooldown():
    after = 1000.0 + m.COOLDOWN.total_seconds()
    assert m.should_alert((1000.0, -4_469.0), -4_400.0, after) is True


def test_the_real_nym_episode_produces_one_alert_not_four():
    """Replay of the observed sequence: -4,469 / -3,099 / +2,968 / +3,310 within 8 minutes."""
    sent, last = 0, None
    for ts, impact in [(0.0, -4_469.0), (194.0, -3_099.0), (404.0, 2_968.0), (487.0, 3_310.0)]:
        if m.should_alert(last, impact, ts):
            sent += 1
            last = (ts, impact)
    assert sent == 1


def test_cooldown_is_longer_than_the_move_window():
    """A cooldown inside the window would let the same rolling move re-alert as it rolls."""
    assert m.COOLDOWN > m.WINDOW
