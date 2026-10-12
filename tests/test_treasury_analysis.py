"""Offline tests for treasury_analysis: argument parsing, the cube request, the P&L split,
markouts, episodes, reconciliation and the option trade detection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from scripts import treasury_analysis as ta

T0 = datetime(2026, 9, 23, 8, tzinfo=timezone.utc)
T1 = datetime(2026, 9, 30, 8, tzinfo=timezone.utc)
H = timedelta(hours=1)


def _perp(ccy, cp, ntl, delta_usd=0.0):
    return {"code": "PERPF", "ccy": ccy, "cp": cp, "ntl": ntl, "K": 0.0, "expiry": "2026-10-01",
            "pv": 0.0, "delta_usd": delta_usd}


def _opt(code, ccy, K, expiry, ntl, pv=0.0, delta_usd=0.0, cp="DERIBIT"):
    return {"code": code, "ccy": ccy, "cp": cp, "ntl": ntl, "K": K, "expiry": expiry,
            "pv": pv, "delta_usd": delta_usd}


def _cube(ts, rows):
    return {"ts": ts, "pv": sum(r["pv"] for r in rows), "positions": rows}


def _fills(rows):
    """rows of (ts, q, px[, maker, fee_usd, order_id, cp])."""
    out = []
    for i, r in enumerate(rows):
        ts, q, px = r[:3]
        maker, fee, oid, cp = (list(r[3:]) + [True, 0.0, f"o{i}", "BINANCE"][len(r) - 3:])
        out.append({"ts": pd.Timestamp(ts), "cp": cp, "q": q, "usd": q * px, "px": px,
                    "maker": maker, "fee_usd": fee, "order_id": oid})
    return pd.DataFrame(out, columns=ta._FILL_COLS)


def _candles(start, closes, step=H):
    idx = pd.DatetimeIndex([pd.Timestamp(start) + i * step for i in range(len(closes))])
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx)


# --------------------------------------------------------------------------- #
# arguments, sampling, request
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("words", "want"), [
    (["btc", "7d"], ("BTC", timedelta(days=7), "7d")),
    (["7d", "btc"], ("BTC", timedelta(days=7), "7d")),
    (["hype", "24h"], ("HYPE", timedelta(hours=24), "24h")),
    (["eth", "1m"], ("ETH", timedelta(days=30), "1m")),
])
def test_parse_args(words, want):
    assert ta.parse_args(words) == want


@pytest.mark.parametrize("words", [[], ["7d"], ["btc"], ["btc", "7d", "1m"], ["btc", "eth", "7d"],
                                   ["btc", "4m"]])
def test_parse_args_refuses(words):
    with pytest.raises(ValueError):
        ta.parse_args(words)


@pytest.mark.parametrize(("span", "n"), [(timedelta(hours=24), 25), (timedelta(days=7), 29),
                                         (timedelta(days=30), 31)])
def test_sample_times_cover_the_window(span, n):
    times = ta.sample_times(T1, span)
    assert len(times) == n and times[0] == T1 - span and times[-1] == T1


def test_rpc_body_is_compact():
    """Pluto answers "Json Error" to the same payload with default JSON spacing."""
    body = ta.rpc_body(datetime(2026, 9, 23, 8, 5, tzinfo=timezone.utc))
    assert ", " not in body and ": " not in body
    assert '"start_date":"2026-09-23"' in body and '"start_time":"08:05"' in body
    assert '"riskreport":"STICKY STRIKE FITTED"' in body


def test_reduce_cube_refuses_misaligned_greeks():
    cube = {"timestamp": 0, "positions": [{"biprod": "BTCUSD"}],
            "greeks": [{"biprod": "ETHUSD", "_values": [None, 1.0, 0, 0, 0]}]}
    with pytest.raises(RuntimeError, match="line up"):
        ta.reduce_cube(cube)


def test_fetch_cubes_dedupes_on_pluto_time():
    same = _cube(T0, [])
    cubes = ta.fetch_cubes([T0, T0 + timedelta(minutes=30), T1],
                           fetch=lambda t: same if t < T1 else _cube(T1, []))
    assert [c["ts"] for c in cubes] == [T0, T1]


# --------------------------------------------------------------------------- #
# prices
# --------------------------------------------------------------------------- #
def test_price_at_never_looks_ahead():
    """A bar's close is only known once it closes: at 08:30 the 08:00 bar is still open."""
    px = ta.price_series(_candles(T0, [100, 110, 120]), now=T1)
    got = ta.price_at(px, [T0 + timedelta(minutes=30), T0 + H, T0 + 2.5 * H, T0 - H])
    assert got[0] != got[0] and list(got[1:3]) == [100, 110] and got[3] != got[3]   # NaN, ..., NaN


def test_last_open_bar_is_priced_at_now():
    now = T0 + timedelta(minutes=90)
    px = ta.price_series(_candles(T0, [100, 105]), now=now)
    assert ta.price_at(px, [now])[0] == 105


# --------------------------------------------------------------------------- #
# trade statistics
# --------------------------------------------------------------------------- #
def test_pnl_split_matches_the_mark_to_market_identity():
    # Start long 2 at 100, buy 1 at 105, end at 110:
    # holding 2*10 = 20, trading 1*(110-105) = 5; and V1 - V0 - cash = 3*110 - 2*100 - 105 = 25.
    c = _candles(T0 - H, [100, 100, 105, 110, 110])
    px = ta.price_series(c, now=T1)
    fills = _fills([(T0 + H, 1.0, 105.0, True, -0.5)])
    st = ta.trade_stats(fills, px, c, T0, T0 + 3 * H, T1, pos0=2.0)
    assert (st["holding"], st["trading"], st["fees"]) == (20.0, 5.0, -0.5)
    assert st["holding"] + st["trading"] == pytest.approx(3 * 110 - 2 * 100 - 105)


def test_markouts_are_positive_when_the_price_moves_our_way():
    c = _candles(T0 - H, [100] + [102] * 33)       # the T0 bar closes at 102
    px = ta.price_series(c, now=T1)
    fills = _fills([(T0, 1.0, 100.0), (T0, -1.0, 100.0)])      # buy and sell at 100, then +2%
    mk = ta.trade_stats(fills, px, c, T0, T0 + 3 * H, T1, pos0=0.0)["mk1h"]
    assert mk["buy"] == pytest.approx(200.0) and mk["sell"] == pytest.approx(-200.0)
    assert mk["usd"] == pytest.approx(0.0) and mk["n"] == 2


def test_markouts_skip_fills_too_recent_to_judge():
    c = _candles(T0 - H, [100] * 5)
    px = ta.price_series(c, now=T0 + 2 * H)
    fills = _fills([(T0, 1.0, 100.0), (T0 + 1.5 * H, 1.0, 100.0)])
    st = ta.trade_stats(fills, px, c, T0, T0 + 2 * H, T0 + 2 * H, pos0=0.0)
    assert st["mk1h"]["n"] == 1 and st["mk24h"]["n"] == 0


def test_maker_share_is_by_notional():
    c = _candles(T0 - H, [100] * 4)
    px = ta.price_series(c, now=T1)
    fills = _fills([(T0, 3.0, 100.0, True), (T0, -1.0, 100.0, False)])
    assert ta.trade_stats(fills, px, c, T0, T0 + H, T1, 0.0)["maker_share"] == pytest.approx(0.75)


def test_episodes_split_on_side_change_and_pause():
    fills = _fills([(T0, 1.0, 1.0), (T0 + timedelta(minutes=5), 1.0, 1.0),
                    (T0 + timedelta(minutes=10), -1.0, 1.0),
                    (T0 + timedelta(minutes=50), -1.0, 1.0)])
    assert list(ta.episodes(fills)) == [1, 1, 2, 3]


def test_worst_episodes_lists_losers_only_worst_first():
    c = _candles(T0 - H, [100] * 2 + [90] * 60)
    px = ta.price_series(c, now=T1)
    fills = _fills([(T0, 1.0, 100.0), (T0 + timedelta(minutes=1), 1.0, 100.0),    # buys, then -10
                    (T0 + 2 * H, -5.0, 100.0)])                                   # a sell, +50
    worst = ta.worst_episodes(fills, px, T1)
    assert len(worst) == 1 and worst.iloc[0]["n"] == 2 and worst.iloc[0]["mk"] == pytest.approx(-20)


def test_reconcile_flags_position_moves_the_fills_do_not_explain():
    fills = _fills([(T0, 2.0, 1.0, True, 0.0, "a", "BINANCE")])
    bad = ta.reconcile(fills, {"BINANCE": 1.0, "DERIBIT": -6.0}, {"BINANCE": 3.0, "DERIBIT": -1.0})
    assert bad == [{"cp": "DERIBIT", "dpos": 5.0, "traded": 0.0}]


# --------------------------------------------------------------------------- #
# Pluto side
# --------------------------------------------------------------------------- #
def test_delta_path_splits_perps_options_coins():
    rows = [_perp("BTC", "BINANCE", -1.0, -80_000), _opt("F1C", "BTC", 90_000, "2026-12-25", 5, 0, 30_000),
            {**_perp("BTC", "DB-X", 1.0, 80_000), "code": "CASH1"}, _perp("ETH", "BINANCE", 1.0, 2_000)]
    d = ta.delta_path([_cube(T0, rows)], "BTC").iloc[0]
    assert (d["perps"], d["options"], d["coins"], d["net"]) == (-80_000, 30_000, 80_000, 30_000)


def test_worst_exposure_picks_the_costliest_interval():
    dpath = pd.DataFrame({"ts": [T0, T0 + H, T0 + 2 * H], "perps": 0.0, "options": 0.0,
                          "coins": 0.0, "net": [1_000.0, -1_000.0, 0.0]})
    px = ta.price_series(_candles(T0 - H, [100, 100, 110, 99]), now=T1)
    w = ta.worst_exposure(dpath, px)
    # +10% while net -1,000 costs -100; the first step (+0%) costs nothing.
    assert w["t_lo"] == T0 + H and w["pnl"] == pytest.approx(-100.0)


def test_option_rows_statuses_and_timing():
    t = [T0, T0 + H, T0 + 2 * H]
    c0 = _cube(t[0], [_opt("C", "HYPE", 90.01, "2026-12-25", 5_000),
                      _opt("F2C", "HYPE", 82, "2026-09-23", 2_550),
                      _opt("F1C", "HYPE", 100, "2026-10-09", 20),
                      _opt("F1C", "HYPE", 120, "2026-12-25", 500)])
    c1 = _cube(t[1], [_opt("C", "HYPE", 90.01, "2026-12-25", 5_000),
                      _opt("F1C", "HYPE", 120, "2026-12-25", 250),
                      _opt("P", "HYPE", 70, "2026-12-25", 10),
                      _opt("C", "HYPE", 99, "2026-12-25", 0)])            # zero-size: not a trade
    c2 = _cube(t[2], [_opt("C", "HYPE", 90.01, "2026-12-25", 5_000),
                      _opt("F1C", "HYPE", 120, "2026-12-25", 250),
                      _opt("F1P", "HYPE", 80, "2026-10-16", 10),
                      _opt("C", "BTC", 1, "2026-12-25", 1)])               # other symbol
    rows = ta.option_rows([c0, c1, c2], "HYPE").set_index("K")
    assert rows["status"].to_dict() == {70: "opened+closed", 80: "opened", 82: "expired",
                                        90.01: "held", 100: "closed", 120: "resized"}
    assert rows.loc[80, "t_lo"] == t[1] and rows.loc[80, "t_hi"] == t[2]
    assert rows.loc[70, "peak"] == 10 and pd.isna(rows.loc[90.01, "t_hi"])


# --------------------------------------------------------------------------- #
# handle
# --------------------------------------------------------------------------- #
class _RefSvc:
    def venues_by_tag(self, key, value):
        assert (key, value) == ("client", "N009-TREASURY_PROP")
        return {1: "binance"}


def test_handle_end_to_end(monkeypatch):
    now = T0 + 30 * H
    c0 = _cube(T0, [_perp("BTC", "BINANCE", 2.0, 200.0), _opt("F1C", "BTC", 120, "2026-12-25", 1, 5.0)])
    c1 = _cube(now, [_perp("BTC", "BINANCE", 3.0, 330.0), _opt("F1C", "BTC", 120, "2026-12-25", 1, 9.0),
                     _opt("F1P", "BTC", 90, "2026-12-25", 2, 1.0)])
    fills = _fills([(T0 + H, 1.0, 105.0, True, -1.0)])
    monkeypatch.setattr(ta, "fetch_perp_fills", lambda *a: fills)
    candles = _candles(T0 - 2 * H, [100, 100, 105] + [110] * 30)
    text, pngs = ta.handle(["btc", "30h"], refsvc=_RefSvc(), trades=object(), now=now,
                           fetch=lambda t: c0 if t < now else c1,
                           candles_fn=lambda sym, since, venues: (candles, "binance perp · 1h"))
    assert "Perp position 2.00 → 3.00 BTC" in text
    assert "P&L (perps, before funding): +$24" in text      # 2*10 + 5 - 1
    assert "opened 2.00 × 90 put" in text and "⚠" not in text
    assert len(pngs) == 2                                   # chart + options table


def test_handle_flags_unreconciled_venue(monkeypatch):
    c0 = _cube(T0, [_perp("BTC", "DERIBIT", -6.0)])
    c1 = _cube(T1, [_perp("BTC", "DERIBIT", -1.0)])
    monkeypatch.setattr(ta, "fetch_perp_fills", lambda *a: _fills([]))
    text, _ = ta.handle(["btc", "7d"], refsvc=_RefSvc(), trades=object(), now=T1,
                        fetch=lambda t: c0 if t < T1 else c1,
                        candles_fn=lambda *a: (pd.DataFrame(), None))
    assert "⚠ DERIBIT: position moved 5.00 but fills show 0" in text
    assert "No candle source" in text


def test_handle_bad_args_is_a_message_not_a_crash():
    text, pngs = ta.handle(["7d"])
    assert text.startswith("❌") and pngs == []
