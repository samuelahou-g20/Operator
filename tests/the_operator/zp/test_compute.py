"""Hand-checked unit tests for the pure ZP formulas (no I/O, no network, synthetic fixtures)."""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from src.the_operator.zp import compute


# --------------------------------------------------------------------------------------
# summary_tables
# --------------------------------------------------------------------------------------
def test_summary_mark_to_last_trade_and_mm_other_split():
    trades = pd.DataFrame({
        "account": ["BINANCE MM", "BINANCE MM", "BINANCE GAMMA"],
        "acct_type": ["MM", "MM", "GAMMA"],
        "inst": ["BTC-USDT", "BTC-USDT", "BTC-USDT"],
        "side": ["B", "S", "B"],
        "qty": [1.0, 1.0, 2.0],
        "notional_usd": [100.0, 110.0, 190.0],
        "exch_ts": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-01"]),
    })
    last_px = 105.0
    mm, other, total = compute.summary_tables(trades, last_px)

    mm_row = mm[mm["account"] == "BINANCE MM"].iloc[0]
    # buy_avg=100, sell_avg=110 -> pnl = (105-100)*1 + (110-105)*1 = 10
    assert mm_row["pnl_$"] == pytest.approx(10.0)
    assert mm_row["buy_avg_price"] == pytest.approx(100.0)
    assert mm_row["sell_avg_price"] == pytest.approx(110.0)

    mm_total = mm[mm["account"] == "MM TOTAL"].iloc[0]
    assert mm_total["pnl_$"] == pytest.approx(10.0)

    other_row = other[other["account"] == "BINANCE GAMMA"].iloc[0]
    # buy_avg=95, pnl = (105-95)*2 = 20
    assert other_row["pnl_$"] == pytest.approx(20.0)
    other_total = other[other["account"] == "GAMMA TOTAL"].iloc[0]
    assert other_total["pnl_$"] == pytest.approx(20.0)

    assert total.iloc[0]["pnl_$"] == pytest.approx(30.0)
    # MM + Other must always reconstruct the total (no trade silently dropped)
    assert mm_total["pnl_$"] + other_total["pnl_$"] == pytest.approx(total.iloc[0]["pnl_$"])


def test_summary_tables_empty_input():
    mm, other, total = compute.summary_tables(pd.DataFrame(), last_px=100.0)
    assert mm.empty and other.empty and total.empty


# --------------------------------------------------------------------------------------
# period_ohlc
# --------------------------------------------------------------------------------------
def test_period_ohlc_pct_change():
    klines = pd.DataFrame({
        "open": [100.0, 105.0, 108.0],
        "high": [106.0, 109.0, 111.0],
        "low": [99.0, 104.0, 107.0],
        "close": [105.0, 108.0, 110.0],
    })
    out = compute.period_ohlc(klines)
    assert out["open"] == pytest.approx(100.0)
    assert out["high"] == pytest.approx(111.0)
    assert out["low"] == pytest.approx(99.0)
    assert out["close"] == pytest.approx(110.0)
    assert out["pct_change"] == pytest.approx(10.0)


def test_period_ohlc_none_on_empty():
    assert compute.period_ohlc(pd.DataFrame()) is None
    assert compute.period_ohlc(None) is None


# --------------------------------------------------------------------------------------
# realized_vol_ann_pct
# --------------------------------------------------------------------------------------
def test_realized_vol_ann_pct_hand_checked():
    # Fixed 1% log return every bar (60s) -> sigma of returns is exactly 0.01 (zero variance
    # around a constant, ddof=1 std would be 0 for a truly constant series, so alternate the sign
    # to get a known, nonzero, hand-computable stdev).
    rets = [0.01, -0.01] * 20   # 40 returns -> 41 closes
    closes = [100.0]
    for r in rets:
        closes.append(closes[-1] * np.exp(r))
    klines = pd.DataFrame({"close": closes})

    out = compute.realized_vol_ann_pct(klines, interval_seconds=60.0)
    sigma = pd.Series(rets).std()   # pandas default ddof=1, matches the implementation
    expected = sigma * (365 * 24 * 3600 / 60.0) ** 0.5 * 100
    assert out == pytest.approx(expected)


def test_realized_vol_ann_pct_none_when_too_few_returns():
    klines = pd.DataFrame({"close": [100.0, 101.0, 99.0]})   # 2 returns, needs >= 30
    assert compute.realized_vol_ann_pct(klines, interval_seconds=60.0) is None


def test_realized_vol_ann_pct_none_when_flat():
    klines = pd.DataFrame({"close": [100.0] * 40})   # zero variance
    assert compute.realized_vol_ann_pct(klines, interval_seconds=60.0) is None


def test_realized_vol_ann_pct_none_on_missing_or_empty_input():
    assert compute.realized_vol_ann_pct(None, 60.0) is None
    assert compute.realized_vol_ann_pct(pd.DataFrame(), 60.0) is None
    assert compute.realized_vol_ann_pct(pd.DataFrame({"open": [1.0] * 40}), 60.0) is None


# --------------------------------------------------------------------------------------
# markouts — sign, bar-ceil alignment, qty weighting
# --------------------------------------------------------------------------------------
def _ref_klines():
    idx = pd.date_range("2026-01-01 00:00:00", periods=5, freq="1min")
    return pd.DataFrame({"open": [100.0, 101.0, 103.0, 104.0, 105.0],
                         "high": [100.0] * 5, "low": [100.0] * 5, "close": [100.0] * 5}, index=idx)


def test_markouts_ceil_alignment_and_sign():
    # trade at 00:00:30, 60s window -> target ts = 00:01:30 -> ceil('1min') = 00:02:00 -> open=103
    mm_trades = pd.DataFrame({
        "exchange": ["BINANCE"], "inst": ["BTC-USDT"], "side": ["B"], "qty": [2.0],
        "price_usd": [100.0], "exch_ts": pd.to_datetime(["2026-01-01 00:00:30"]),
    })
    per_venue, per_trade = compute.markouts(mm_trades, _ref_klines(), windows=(60,))
    # buy: pnl_bps = 1e4*(ref/price - 1) = 1e4*(103/100 - 1) = 300
    assert per_trade["pnl_60s_bps"].iloc[0] == pytest.approx(300.0)
    assert per_trade["pnl_60s_$"].iloc[0] == pytest.approx((103.0 - 100.0) * 2.0)
    assert per_venue.loc[("BINANCE", "BTC-USDT"), "pnl_60s_bps"] == pytest.approx(300.0)
    assert per_venue.loc[("BINANCE", "BTC-USDT"), "volume_$"] == pytest.approx(200.0)


def test_markouts_sell_side_sign_flips():
    mm_trades = pd.DataFrame({
        "exchange": ["BINANCE"], "inst": ["BTC-USDT"], "side": ["S"], "qty": [1.0],
        "price_usd": [100.0], "exch_ts": pd.to_datetime(["2026-01-01 00:00:30"]),
    })
    _per_venue, per_trade = compute.markouts(mm_trades, _ref_klines(), windows=(60,))
    # sell: sign = -1 -> pnl_bps = -300 (price went up against a short)
    assert per_trade["pnl_60s_bps"].iloc[0] == pytest.approx(-300.0)


def test_markouts_qty_weighted_per_venue():
    mm_trades = pd.DataFrame({
        "exchange": ["BINANCE", "BINANCE"], "inst": ["BTC-USDT", "BTC-USDT"], "side": ["B", "B"],
        "qty": [1.0, 3.0], "price_usd": [100.0, 100.0],
        "exch_ts": pd.to_datetime(["2026-01-01 00:00:30", "2026-01-01 00:01:30"]),
    })
    # trade1 -> target 00:02:00 -> open 103 -> bps 300; trade2 -> target 00:02:30 -> ceil to
    # 00:03:00 -> open 104 -> bps 400. qty-weighted mean = (1*300 + 3*400)/4 = 375
    per_venue, _per_trade = compute.markouts(mm_trades, _ref_klines(), windows=(60,))
    assert per_venue.loc[("BINANCE", "BTC-USDT"), "pnl_60s_bps"] == pytest.approx(375.0)


def test_markouts_empty_inputs():
    per_venue, per_trade = compute.markouts(pd.DataFrame(), _ref_klines())
    assert per_venue.empty
    per_venue2, _ = compute.markouts(pd.DataFrame({"exchange": ["X"]}), None)
    assert per_venue2.empty


# --------------------------------------------------------------------------------------
# captures — market-key alias collapsing, TOTAL = sum/sum
# --------------------------------------------------------------------------------------
def test_captures_collapses_venue_alias_and_totals():
    # AI trades under two inst aliases on the same physical binance/USDT market
    trades = pd.DataFrame({
        "inst": ["AI-USDT", "AIGENSYN-USDT"], "exchange": ["BINANCE", "BINANCE"],
        "notional_usd": [1000.0, 500.0],
        "exch_ts": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:00:00"]),
    })
    market_volume_ts = pd.DataFrame({
        "exchange": ["BINANCE"], "pair": ["AIGENSYN-USDT"],
        "timestamp": pd.to_datetime(["2026-01-01 00:00:00"]), "notional_usd": [30000.0],
    })
    cap, _ts = compute.captures(trades, market_volume_ts)
    row = cap[cap["exchange"] == "BINANCE"].iloc[0]
    assert row["g20_volume_usd"] == pytest.approx(1500.0)      # both aliases summed
    assert row["market_volume_usd"] == pytest.approx(30000.0)
    assert row["capture"] == pytest.approx(1500.0 / 30000.0)

    total = cap[cap["exchange"] == "TOTAL"].iloc[0]
    assert total["g20_volume_usd"] == pytest.approx(cap[cap["exchange"] != "TOTAL"]["g20_volume_usd"].sum())
    assert total["capture"] == pytest.approx(total["g20_volume_usd"] / total["market_volume_usd"])


def test_captures_perp_and_spot_stay_separate():
    trades = pd.DataFrame({
        "inst": ["BTC-USDT", "P.BTC-USDT"], "exchange": ["BINANCE", "BINANCE"],
        "notional_usd": [100.0, 200.0],
        "exch_ts": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:00:00"]),
    })
    market_volume_ts = pd.DataFrame({
        "exchange": ["BINANCE", "BINANCE"], "pair": ["BTC-USDT", "P.BTC-USDT"],
        "timestamp": pd.to_datetime(["2026-01-01"] * 2), "notional_usd": [1000.0, 2000.0],
    })
    cap, _ts = compute.captures(trades, market_volume_ts)
    spot = cap[(cap["exchange"] == "BINANCE") & (cap["pair"] == "BTC-USDT")].iloc[0]
    perp = cap[(cap["exchange"] == "BINANCE") & (cap["pair"] == "P.BTC-USDT")].iloc[0]
    assert spot["g20_volume_usd"] == pytest.approx(100.0)
    assert perp["g20_volume_usd"] == pytest.approx(200.0)


# --------------------------------------------------------------------------------------
# ob_widths — TOB / BOB formulas
# --------------------------------------------------------------------------------------
def test_ob_widths_tob_bob_formula():
    ts = pd.Timestamp("2026-01-01 00:00:00")
    raw = pd.DataFrame([
        {"exchange": "BINANCE", "ccypair": "BTC-USDT", "book_id": 1,
         "timestamp_micros": int(ts.value / 1000), "price": 99.0, "size": 1, "type": "BID"},
        {"exchange": "BINANCE", "ccypair": "BTC-USDT", "book_id": 1,
         "timestamp_micros": int(ts.value / 1000), "price": 98.0, "size": 1, "type": "BID"},
        {"exchange": "BINANCE", "ccypair": "BTC-USDT", "book_id": 1,
         "timestamp_micros": int(ts.value / 1000), "price": 101.0, "size": 1, "type": "ASK"},
        {"exchange": "BINANCE", "ccypair": "BTC-USDT", "book_id": 1,
         "timestamp_micros": int(ts.value / 1000), "price": 102.0, "size": 1, "type": "ASK"},
    ])
    summary, series = compute.ob_widths(raw)
    assert summary is not None
    row = summary.iloc[0]
    # TOB = 1e4*(min_ask-max_bid)/min_ask = 1e4*(101-99)/101
    assert row["tob_median"] == pytest.approx(10000 * (101 - 99) / 101)
    # BOB = 1e4*(max_ask-min_bid)/max_ask = 1e4*(102-98)/102
    assert row["bob_median"] == pytest.approx(10000 * (102 - 98) / 102)
    assert series is not None and not series.empty


def test_ob_widths_none_on_empty():
    assert compute.ob_widths(None) == (None, None)
    assert compute.ob_widths(pd.DataFrame()) == (None, None)


# --------------------------------------------------------------------------------------
# ob_compliance — composition fractions sum to 1, 400bps rule
# --------------------------------------------------------------------------------------
def _snapshot(book_id, exch="BINANCE", pair="BTC-USDT", ts=0):
    return {"exchange": exch, "ccypair": pair, "book_id": book_id, "timestamp_micros": ts}


def test_ob_compliance_fractions_sum_to_one_and_threshold():
    rows = []
    # book 1: two-sided, narrow (compliant, bob = 1e4*(101-99)/mid ~ small)
    rows += [{**_snapshot(1), "price": 100.0, "type": "BID"}, {**_snapshot(1), "price": 101.0, "type": "ASK"}]
    # book 2: two-sided, wide (bid 50, ask 150 -> mid=100, bob=1e4*(150-50)/100=10000 >= 400 -> not compliant)
    rows += [{**_snapshot(2), "price": 50.0, "type": "BID"}, {**_snapshot(2), "price": 150.0, "type": "ASK"}]
    # book 3: one-sided (bid only)
    rows += [{**_snapshot(3), "price": 100.0, "type": "BID"}]
    # book 4: empty book (no entries at all -> NaN type row, kept so `total` counts it)
    rows += [{**_snapshot(4), "price": None, "type": None}]
    raw = pd.DataFrame(rows)
    comp = compute.ob_compliance(raw)
    assert comp is not None
    r = comp.iloc[0]
    assert r["snaps"] == 4
    total_frac = r["compliance"] + r["wide"] + r["one_sided"] + r["empty"]
    assert total_frac == pytest.approx(1.0)
    assert r["compliance"] == pytest.approx(1 / 4)   # only book 1
    assert r["wide"] == pytest.approx(1 / 4)          # book 2: two-sided but >= 400bps
    assert r["one_sided"] == pytest.approx(1 / 4)     # book 3
    assert r["empty"] == pytest.approx(1 / 4)         # book 4
    assert r["uptime"] == pytest.approx(2 / 4)        # books 1 & 2 are two-sided


def test_ob_compliance_boundary_is_strict_less_than():
    # bid=98, ask=102 -> mid=100, bob = 1e4*(102-98)/100 = 400 exactly -> NOT compliant (< 400 rule)
    rows = [{**_snapshot(1), "price": 98.0, "type": "BID"}, {**_snapshot(1), "price": 102.0, "type": "ASK"}]
    raw = pd.DataFrame(rows)
    comp = compute.ob_compliance(raw)
    assert comp.iloc[0]["compliance"] == pytest.approx(0.0)
    assert comp.iloc[0]["wide"] == pytest.approx(1.0)


def test_ob_compliance_nodata_below_uptime_floor():
    # 100 empty-book snapshots, 1 two-sided -> uptime 1/101 < 0.05 -> nodata
    rows = [{**_snapshot(i), "price": None, "type": None} for i in range(1, 101)]
    rows += [{**_snapshot(101), "price": 100.0, "type": "BID"}, {**_snapshot(101), "price": 101.0, "type": "ASK"}]
    raw = pd.DataFrame(rows)
    comp = compute.ob_compliance(raw)
    assert bool(comp.iloc[0]["nodata"]) is True


# --------------------------------------------------------------------------------------
# delta_pnl — realised/unrealised on a buy-sell-flip sequence
# --------------------------------------------------------------------------------------
def test_delta_pnl_buy_sell_flip_realised_unrealised():
    # buy 10 @100 -> sell 15 @110 (closes 10, flips short 5 @110) -> buy 5 @90 (closes short, flat)
    trades = pd.DataFrame({
        "side": ["B", "S", "B"], "qty": [10.0, 15.0, 5.0], "price_usd": [100.0, 110.0, 90.0],
        "exch_ts": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:01:00", "2026-01-01 00:02:00"]),
    })
    out = compute.delta_pnl(trades, md_klines=None)
    assert out is not None
    # step 1: position 10 @ avg 100, unrealised = 0 (price==avg)
    assert out.iloc[0]["cum_realised_pnl"] == pytest.approx(0.0)
    # step 2: closes 10 of the long selling at 110 vs. a 100 cost basis -> +100 realised;
    # flips short 5 @ avg 110 (closing_qty = (avg-price)*closing_qty = (100-110)*(-10) = +100)
    assert out.iloc[1]["cum_realised_pnl"] == pytest.approx(100.0)
    assert out.iloc[1]["cum_delta"] == pytest.approx(-5.0 * 110.0)   # position -5 @ price 110
    # step 3: closes the short 5 buying at 90 vs. a 110 cost basis -> +100 more realised -> flat
    assert out.iloc[2]["cum_realised_pnl"] == pytest.approx(200.0)
    assert out.iloc[2]["cum_delta"] == pytest.approx(0.0)            # flat book
    assert out.iloc[2]["cum_unrealised_pnl"] == pytest.approx(0.0)   # flat book


def test_delta_pnl_none_on_empty():
    assert compute.delta_pnl(pd.DataFrame(), None) is None
    assert compute.delta_pnl(None, None) is None


# --------------------------------------------------------------------------------------
# flow_capture_bps
# --------------------------------------------------------------------------------------
def test_flow_capture_bps():
    total = pd.DataFrame([{"account": "MM TOTAL", "buy_avg_price": 100.0, "sell_avg_price": 102.0}])
    bps = compute.flow_capture_bps(total)
    mid = (100.0 + 102.0) / 2
    assert bps == pytest.approx((102.0 - 100.0) / mid * 1e4)


def test_flow_capture_bps_none_when_no_volume():
    total = pd.DataFrame([{"account": "MM TOTAL", "buy_avg_price": 0.0, "sell_avg_price": 0.0}])
    assert compute.flow_capture_bps(total) is None
