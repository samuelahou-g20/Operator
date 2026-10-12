"""Market share and the diagnosis ladder: the first supported cause, never a guess."""

from __future__ import annotations

import pytest

from src.the_operator import volume

DYN = {"base_min_spread_bps": 15.0, "base_max_spread_bps": 180.0, "levels": 4,
       "top_level_weight": 0.05, "total_value_usd": 5000.0,
       "reference_annualized_volatility_pct": 140.0,
       "volatility_min_spread_scaling_factor": 2.0,
       "volatility_max_spread_scaling_factor": 1.0,
       "volatility_size_scaling_factor": 0.5}


def _v(**over):
    base = dict(exchange="GATEIO", ccypair="ELA-USDT", status="TRADING", ours=5_000.0,
                market=100_000.0, mid=1.0, market_spread_bps=40.0,
                params={"ob_dynamic": dict(DYN)},
                calculator="LM2Strategy:REELA:ELA-USDT:GATEIO")
    base.update(over)
    return volume.VenueShare(**base)


def test_share_is_a_ratio_in_one_quote_currency():
    assert _v(ours=5_000.0, market=100_000.0).share == pytest.approx(0.05)


def test_a_thin_venue_has_no_readable_share():
    v = _v(market=volume.MIN_MARKET_QUOTE - 1)
    assert v.share is None
    cause, fix = volume.diagnose(v)
    assert "too thin to read a share" in cause and fix is None


def test_a_stopped_strategy_is_diagnosed_before_anything_else():
    """No spread or size conclusion is meaningful while a strategy is not quoting at all."""
    cause, fix = volume.diagnose(_v(status="STOP (DISABLED)", ours=0.0))
    assert cause.startswith("not quoting") and "rebound ELA" in fix


def test_being_the_market_suggests_widening_when_we_are_inside_the_touch():
    cause, fix = volume.diagnose(_v(ours=80_000.0, market_spread_bps=100.0))
    assert "we are 80% of this venue" in cause
    assert "widen" in fix and "15bps vs a 100bps market touch" in fix


def test_being_the_market_suggests_cutting_size_when_the_spread_is_not_the_cause():
    cause, fix = volume.diagnose(_v(ours=80_000.0, market_spread_bps=4.0))
    assert "effectively the market" in cause and "cut `ob_dynamic.total_value_usd`" in fix


def test_low_share_with_quotes_outside_the_touch_names_the_spread():
    cause, fix = volume.diagnose(_v(ours=1_000.0, market_spread_bps=10.0))
    assert "outside the touch" in cause
    assert "lower `ob_dynamic` min spread from 15bps" in fix


def test_low_share_with_a_competitive_spread_names_size():
    cause, fix = volume.diagnose(_v(ours=1_000.0, market_spread_bps=100.0))
    assert "the limit is size" in cause and "raise `ob_dynamic.total_value_usd`" in fix


def test_low_share_from_vol_throttling_names_the_reference_vol():
    """Realised vol well above reference scales size down; that is the cause, not the spread."""
    cause, fix = volume.diagnose(_v(ours=1_000.0, market_spread_bps=100.0),
                                 realized_vol_pct=560.0)      # 4x the 140% reference
    assert "throttled to" in cause and "reference_annualized_volatility_pct" in fix


def test_no_fills_at_all_is_said_plainly():
    cause, fix = volume.diagnose(_v(ours=0.0, market_spread_bps=10.0))
    assert "no fills in 24h" in cause and "behind the queue" in fix


def test_a_share_inside_the_band_suggests_nothing():
    cause, fix = volume.diagnose(_v(ours=20_000.0))
    assert cause == "share 20%" and fix is None


def test_unknown_market_volume_is_admitted_not_guessed():
    cause, fix = volume.diagnose(_v(market=None))
    assert "cannot be judged" in cause and fix is None


def test_mode_and_ladder_follow_whichever_group_the_strategy_runs():
    implicit = _v(params={"ob_dynamic": None,
                          "ob_implicit": {"min_spread_bps": 50.0, "total_value_usd": 1000.0}})
    assert implicit.mode == "ob_implicit"
    assert implicit.min_spread_bps == 50.0 and implicit.size_quote == 1000.0
    assert _v(params={}).mode is None and _v(params={}).min_spread_bps is None


# --------------------------------------------------------------------------------------
# Snooze — the desk's daily exceptions, which are told rather than configured
# --------------------------------------------------------------------------------------
from datetime import date, timedelta            # noqa: E402

from src.the_operator import execution, policy  # noqa: E402
from src.the_operator.contract import Context   # noqa: E402


@pytest.fixture
def snoozes(tmp_path, monkeypatch):
    monkeypatch.setattr(volume, "SNOOZE_PATH", tmp_path / "snooze.json")
    return tmp_path


def test_a_snooze_covers_today_only(snoozes):
    today = date(2026, 10, 5)
    volume.snooze("mm3", "ela", "u7", "boss wants it high", today=today)
    assert volume.is_snoozed("mm3", "ELA", today=today)
    assert not volume.is_snoozed("mm3", "ELA", today=today + timedelta(days=1))
    assert not volume.is_snoozed("mm2", "ELA", today=today)        # per desk
    assert volume.snoozed_today("mm3", today=today) == ["ELA"]


def test_a_missing_store_snoozes_nothing(snoozes):
    assert volume.is_snoozed("mm3", "ELA") is False
    assert volume.snoozed_today("mm3") == []


def test_execution_records_the_confirmed_snooze(snoozes):
    ctx = Context("r1", "mm3", "-1", "7", "u7", is_trader=True)
    p = policy.Pending("n1", ctx, "volume_snooze",
                       {"token": "ELA", "reason": "boss wants it high"}, 0.0)
    reply, tracked = execution.execute(p)
    assert tracked is None and "ELA volume alerts are off for mm3" in reply
    assert volume.is_snoozed("mm3", "ELA")


# --------------------------------------------------------------------------------------
# The desk-wide card: one line per coin, two command lines, nothing else
# --------------------------------------------------------------------------------------
def _card(monkeypatch, rows, strategies):
    from src.the_operator import rapid
    from src.the_operator.tools import engine

    class _S:
        def __init__(self, token, healthy):
            self.token, self.healthy = token, healthy
    monkeypatch.setattr(rapid, "scan", lambda program, w: [_S(t, h) for t, h in strategies])
    monkeypatch.setattr(volume, "snoozed_today", lambda desk: [])
    ctx = Context("r1", "mm3", "-1", "7", "u7")
    return engine._desk_card(ctx, "mm3", rows)


def test_the_card_is_one_line_per_coin_and_is_sent_as_is(monkeypatch):
    res = _card(monkeypatch, [("ELA", 1.01), ("DNX", 0.0), ("XCH", 0.16)],
                [("ELA", True), ("ELA", False), ("DNX", False)])
    assert res.verbatim and res.card
    assert "• **ELA** 101% — 1 of 2 venues stopped" in res.card
    assert "• **DNX** 0% — not quoting (1 venue)" in res.card
    assert "XCH" not in res.card                       # in-band coins are not listed
    assert "cumulative" not in res.card and "on purpose" not in res.card   # no caveat prose
    assert "Do not repeat it" in res.text              # the model is told to add nothing


def test_a_coin_above_the_band_is_never_sent_to_rebound(monkeypatch):
    """Restarting a stopped venue on a 101% coin pushes its share further up, not back."""
    res = _card(monkeypatch, [("ELA", 1.01), ("DNX", 0.0)],
                [("ELA", False), ("DNX", False)])
    assert "`/operator rebound <coin>` — DNX" in res.card
    assert "`/operator volume check <coin>` — ELA" in res.card


def test_a_sheet_row_that_is_not_a_token_here_is_not_listed_at_all(monkeypatch):
    """QASIM is a sheet row, not a coin this program quotes — nothing here can act on it."""
    res = _card(monkeypatch, [("QASIM", 0.0), ("DNX", 0.0)], [("DNX", False)])
    assert "QASIM" not in res.card and "**DNX** 0%" in res.card


def test_commands_collapse_to_one_line_each(monkeypatch):
    rows = [(f"C{i}", 0.0) for i in range(6)]
    res = _card(monkeypatch, rows, [(f"C{i}", False) for i in range(6)])
    assert res.card.count("/operator rebound") == 1
    assert "C0, C1, C2, C3, C4, C5" in res.card
