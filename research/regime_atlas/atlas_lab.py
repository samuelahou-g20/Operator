"""Regime-atlas research lab — shared data loaders, stats, and the CANONICAL FROZEN sleeve
builders for the crypto regime-atlas track (Phases 1-3).

Everything here reads the lake ONLY via ``src.data.query`` and judges performance in EXCESS of
the risk-free rate (FRED ``tb3ms``). One daily 365-day grid is used throughout so the sleeves, the
carry book, and rf all line up.

rf treatment (uniform): a sleeve's EXCESS daily return = its $1-capital total return minus rf.
  - long spot / beta          excess = w * (spot_ret - rf)          (capital in spot forgoes rf)
  - cash / flat leg           excess = 0                            (earns exactly rf)
  - dollar-neutral L/S perp   excess = sum_i w_i * perp_total_i     (collateral earns rf -> cancels)
  - CARRY (stored v8.0 book)  excess = net - rf                     (repo's established honesty convention)
where perp_total = perp_price_return - funding_paid_by_long (a short earns the funding).

MUST NOT: write/mutate the lake, re-run the funding-carry engine (CARRY is read from stored
results), or tune the sleeve definitions — sleeves 1-6 are preregistered and frozen in
``_build_sleeve_audit.py``; the implementations below must match that header exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = next((q for q in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
             if (q / "pyproject.toml").exists()), Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import config as cfg  # noqa: E402
from src.common import universe as U  # noqa: E402
from src.data import query  # noqa: E402

START, END = "2021-01-01", "2026-05-31"
SOURCE, TF, ANN = "binance", "1h", 365
CARRY_RESULTS = ROOT / "results" / "funding_carry" / "v8.0" / "20260703T145150Z" / "returns.parquet"


def date_grid(start: str = START, end: str = END) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")


# ---------------------------------------------------------------- risk-free (FRED tb3ms)

def rf_annual(didx: pd.DatetimeIndex) -> pd.Series:
    """Annualized risk-free (decimal) forward-filled onto the daily grid."""
    tb = query.get("rates", "fred", "tb3ms", "rates")["value"]
    return tb.reindex(tb.index.union(didx)).ffill().reindex(didx)


def rf_daily(didx: pd.DatetimeIndex) -> pd.Series:
    """Daily risk-free (decimal) = annual / 365."""
    return rf_annual(didx) / ANN


def macro(symbol: str, didx: pd.DatetimeIndex) -> pd.Series:
    """A FRED context series (sp500, dgs10) forward-filled onto the grid. Note the rates schema
    stores levels/yields scaled by 1/100 (dgs10 0.0448 = 4.48%; sp500 value*100 = the index level)."""
    s = query.get("rates", "fred", symbol, "rates")["value"]
    return s.reindex(s.index.union(didx)).ffill().reindex(didx)


# ---------------------------------------------------------------- lake loaders (daily panels)

def registry_symbols(asset_type: str) -> list[str]:
    """Distinct instrument symbols of ``asset_type`` from the binance universe registry."""
    uni = U.load_universe(SOURCE)
    return sorted({v["symbol"] for v in uni.values() if v.get("asset_type") == asset_type})


def registry_status() -> pd.DataFrame:
    """Per perp-symbol registry facts: listed date + status (trading|delisted)."""
    uni = U.load_universe(SOURCE)
    rows = [{"symbol": v["symbol"], "listed": v.get("listed"), "status": v.get("status")}
            for k, v in uni.items() if v.get("asset_type") == "perps"]
    df = pd.DataFrame(rows).drop_duplicates("symbol").set_index("symbol")
    df["listed"] = pd.to_datetime(df["listed"], errors="coerce")
    return df


def spot_daily(symbols: list[str], didx: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(daily close, daily dollar volume) for spot names, resampled from 1h bars.
    Dollar volume approximated as close * base-volume summed over the day."""
    close, dvol = {}, {}
    start, end = str(didx[0].date()), str(didx[-1].date())
    for s in symbols:
        df = query.get("spot", SOURCE, s, "bars", timeframe=TF, start=start, end=end,
                       columns=["close", "volume"])
        if df.empty:
            continue
        d = df.resample("1D").agg({"close": "last", "volume": "sum"})
        close[s] = d["close"]
        dvol[s] = d["close"] * d["volume"]
    return pd.DataFrame(close).reindex(didx), pd.DataFrame(dvol).reindex(didx)


def perp_daily(symbols: list[str], didx: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(daily perp close, daily funding sum) for perp names. funding sum = sum of funding rates
    settled that UTC day (funding earned by the SHORT when positive)."""
    close, fund = {}, {}
    start, end = str(didx[0].date()), str(didx[-1].date())
    for s in symbols:
        p = query.get("perps", SOURCE, s, "bars", timeframe=TF, start=start, end=end, columns=["close"])
        if p.empty:
            continue
        close[s] = p["close"].resample("1D").last()
        f = query.get("perps", SOURCE, s, "funding_rate", start=start, end=end, columns=["rate"])
        fund[s] = f["rate"].resample("1D").sum() if not f.empty else pd.Series(dtype=float)
    PC = pd.DataFrame(close).reindex(didx)
    FS = pd.DataFrame(fund).reindex(didx)
    # funding NaN but perp trades that day -> 0 transfer; funding where no perp stays NaN (untradeable)
    FS = FS.where(PC.isna(), FS.fillna(0.0))
    return PC, FS


# ---------------------------------------------------------------- liquid universe & breadth

def liquid_membership(dollarvol: pd.DataFrame, adv_window: int = 60, top_n: int = 30,
                      min_history: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point-in-time top-``top_n`` membership by trailing ``adv_window``-day mean dollar volume,
    refreshed on the first calendar day of each month and held constant through the month.
    Returns (bool membership day x name, the trailing-ADV panel)."""
    adv = dollarvol.rolling(adv_window, min_periods=min_history).mean()
    didx = dollarvol.index
    memb = pd.DataFrame(False, index=didx, columns=dollarvol.columns)
    for ms in pd.date_range(didx[0], didx[-1], freq="MS"):
        hist = adv.loc[:ms]
        if hist.empty:
            continue
        top = hist.iloc[-1].dropna().sort_values(ascending=False).head(top_n).index
        memb.loc[ms: ms + pd.offsets.MonthEnd(0), top] = True
    return memb, adv


def liquid_pool(adv: pd.DataFrame, floor_usd: float = 5e6) -> pd.DataFrame:
    """Broader point-in-time 'liquid universe' for breadth: names whose trailing ADV exceeds a
    fixed dollar floor. Grows over time with the market — a descriptive breadth base, not a
    tradable set."""
    return adv > floor_usd


def breadth_above_ma(close: pd.DataFrame, pool: pd.DataFrame, ma: int = 200) -> pd.Series:
    """Share of the point-in-time liquid ``pool`` trading above its own ``ma``-day moving average."""
    above = close > close.rolling(ma, min_periods=ma // 2).mean()
    inpool = pool & close.notna()
    num = (above & inpool).sum(axis=1)
    den = inpool.sum(axis=1)
    return (num / den.where(den > 0, np.nan))


# ---------------------------------------------------------------- stats

def max_drawdown(excess: pd.Series) -> float:
    """Max drawdown of the additive (sum-of-daily-excess) equity curve."""
    eq = excess.fillna(0.0).cumsum()
    return float((eq - eq.cummax()).min())


def ann_stats(excess: pd.Series, ann: int = ANN) -> dict:
    """Annualized excess-return stats for a daily excess series."""
    e = excess.dropna()
    if len(e) < 2 or e.std() == 0:
        return {"ann_return": float(e.sum() / len(e) * ann) if len(e) else 0.0,
                "ann_vol": 0.0, "sharpe": 0.0, "max_dd": max_drawdown(e), "calmar": 0.0, "n": len(e)}
    ar = e.mean() * ann
    av = e.std() * np.sqrt(ann)
    dd = max_drawdown(e)
    return {"ann_return": float(ar), "ann_vol": float(av), "sharpe": float(ar / av),
            "max_dd": dd, "calmar": float(ar / abs(dd)) if dd < 0 else np.inf, "n": len(e)}


def per_year(excess: pd.Series, ann: int = ANN) -> pd.DataFrame:
    """Year x {excess_return, sharpe} table (sum of daily excess per year, plus annualized Sharpe)."""
    e = excess.dropna()
    rows = {}
    for yr, g in e.groupby(e.index.year):
        s = ann_stats(g, ann)
        rows[yr] = {"excess_ret": float(g.sum()), "sharpe": s["sharpe"], "ann_vol": s["ann_vol"]}
    return pd.DataFrame(rows).T


def vol_scale_per_year(excess: pd.Series, target: float = 0.10, ann: int = ANN) -> tuple[pd.Series, pd.Series]:
    """Scale each calendar year's excess returns to ``target`` annualized vol (ex-post, for
    heatmap comparability only — NOT tradable). Returns (scaled series, per-year scale factor)."""
    e = excess.copy()
    out = pd.Series(index=e.index, dtype=float)
    scales = {}
    for yr, g in e.groupby(e.index.year):
        v = g.std() * np.sqrt(ann)
        k = target / v if v and v > 0 else 0.0
        scales[yr] = k
        out.loc[g.index] = g * k
    return out, pd.Series(scales)


# ---------------------------------------------------------------- sleeve primitives

def perp_total_ret(perp_close: pd.DataFrame, funding_sum: pd.DataFrame) -> pd.DataFrame:
    """Per-unit-LONG perp total return = price return - funding paid by the long."""
    return perp_close.pct_change() - funding_sum


def inv_vol(daily_ret: pd.DataFrame, window: int = 30, min_periods: int = 15) -> pd.DataFrame:
    """1 / rolling std of daily returns (inverse-vol weight input)."""
    v = daily_ret.rolling(window, min_periods=min_periods).std()
    return (1.0 / v.replace(0.0, np.nan))


def _fridays(didx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return didx[didx.weekday == 4]


def ls_weights(signal: pd.DataFrame, membership: pd.DataFrame, ivol: pd.DataFrame,
               valid: pd.DataFrame, mode: str, long_high: bool = True,
               rebal: str = "fri") -> pd.DataFrame:
    """Build a signed, dollar-neutral, inverse-vol weekly weight matrix.

    mode='sign'    : long names with signal>0, short signal<0 (time-series trend).
    mode='tercile' : long top tercile, short bottom tercile of the signal among members
                     (long_high=True); long_high=False flips (long losers — reversal).
    Weights set on each Friday from the as-of signal, held through the week (ffill). Long leg
    sums to +1, short leg to -1 (before any vol scaling)."""
    cols = signal.columns
    W = pd.DataFrame(np.nan, index=signal.index, columns=cols)
    for d in _fridays(signal.index):
        sig = signal.loc[d]
        iv = ivol.loc[d]
        ok = membership.loc[d] & valid.loc[d] & sig.notna() & iv.notna()
        w = pd.Series(0.0, index=cols)
        s = sig[ok]
        if len(s) >= 6:
            if mode == "sign":
                longs, shorts = s[s > 0].index, s[s < 0].index
            else:
                hi, lo = s.quantile(2 / 3), s.quantile(1 / 3)
                top, bot = s[s >= hi].index, s[s <= lo].index
                longs, shorts = (top, bot) if long_high else (bot, top)
            if len(longs):
                w[longs] = iv[longs] / iv[longs].sum()
            if len(shorts):
                w[shorts] = -iv[shorts] / iv[shorts].sum()
        W.loc[d] = w
    return W.ffill().fillna(0.0)


def book_ls(W: pd.DataFrame, perp_total: pd.DataFrame, c_oneway: float) -> dict:
    """Book a signed perp weight matrix. excess = w(shifted) . perp_total - one-way cost * turnover.
    (Collateral rf cancels for a dollar-neutral book, so this is already excess-of-rf.)"""
    pos = W.shift(1).fillna(0.0)
    gross = (pos * perp_total).sum(axis=1)
    turnover = (W - W.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = c_oneway * turnover
    return {"excess": gross - cost, "gross": gross, "turnover": turnover, "cost": cost, "W": W}


# ---------------------------------------------------------------- CANONICAL FROZEN SLEEVES (1-6)

def sleeve_ts_trend_lf(spot_close: pd.DataFrame, rf_d: pd.Series, c_spot: float,
                       names=("btcusdt", "ethusdt"), ma: int = 100) -> dict:
    """1. TS-TREND-LF — BTC+ETH long/flat, long the name when spot>``ma``-day MA else cash, equal
    split. Long-only -> holds SPOT; the flat half earns rf (excess 0). Weekly (Friday) decision."""
    names = [n for n in names if n in spot_close.columns]
    didx = spot_close.index
    fri = didx.isin(_fridays(didx))
    long_flag = pd.DataFrame(False, index=didx, columns=names)
    for n in names:
        ma_s = spot_close[n].rolling(ma, min_periods=ma // 2).mean()
        cond01 = (spot_close[n] > ma_s).astype(float)      # 1.0 long / 0.0 flat
        wk = cond01.where(fri)                              # keep only Friday decisions
        long_flag[n] = wk.ffill().fillna(0.0) > 0.5
    ret = spot_close[names].pct_change()
    excess_by = {n: 0.5 * (ret[n] - rf_d).where(long_flag[n], 0.0) for n in names}
    excess_gross = sum(excess_by.values())
    turnover = 0.5 * long_flag.astype(float).diff().abs().sum(axis=1)  # spot notional flipped
    cost = c_spot * turnover
    return {"excess": excess_gross - cost, "gross": excess_gross, "turnover": turnover,
            "cost": cost, "long_flag": long_flag}


def sleeve_ts_trend_ls(perp_close, funding_sum, membership, c_perp: float, lookback: int = 90) -> dict:
    """2. TS-TREND-LS — universe time-series trend: long names with trailing ``lookback``-day
    perp return > 0, short those < 0 (via perps, incl. funding). Inverse-30d-vol within legs,
    dollar-neutral, weekly Friday rebalance."""
    ptot = perp_total_ret(perp_close, funding_sum)
    sig = perp_close.pct_change(lookback)
    W = ls_weights(sig, membership, inv_vol(perp_close.pct_change()), perp_close.notna(), mode="sign")
    return book_ls(W, ptot, c_perp)


def sleeve_xs_mom(perp_close, funding_sum, membership, c_perp: float,
                  window: int = 30, skip: int = 3) -> dict:
    """3. XS-MOM — cross-sectional momentum: rank trailing ``window``-day return skipping the last
    ``skip`` days; long top tercile, short bottom tercile (perps, incl. funding). Inverse-vol,
    weekly."""
    ptot = perp_total_ret(perp_close, funding_sum)
    sig = perp_close.shift(skip).pct_change(window)
    W = ls_weights(sig, membership, inv_vol(perp_close.pct_change()), perp_close.notna(),
                   mode="tercile", long_high=True)
    return book_ls(W, ptot, c_perp)


def sleeve_rev(perp_close, funding_sum, membership, c_perp: float, window: int = 7) -> dict:
    """4. REV — short-horizon reversal: rank trailing ``window``-day return; long losers tercile,
    short winners tercile (perps, incl. funding). Inverse-vol, weekly."""
    ptot = perp_total_ret(perp_close, funding_sum)
    sig = perp_close.pct_change(window)
    W = ls_weights(sig, membership, inv_vol(perp_close.pct_change()), perp_close.notna(),
                   mode="tercile", long_high=False)
    return book_ls(W, ptot, c_perp)


def sleeve_beta_btc(spot_close: pd.DataFrame, rf_d: pd.Series) -> dict:
    """5a. BETA-BTC — hold BTC spot (long-only). excess = spot_ret - rf."""
    ret = spot_close["btcusdt"].pct_change()
    return {"excess": (ret - rf_d), "gross": (ret - rf_d),
            "turnover": pd.Series(0.0, index=ret.index), "cost": pd.Series(0.0, index=ret.index)}


def sleeve_beta_ew(spot_close: pd.DataFrame, membership: pd.DataFrame, rf_d: pd.Series,
                   c_spot: float) -> dict:
    """5b. BETA-EW — hold the equal-weight liquid index (top-30 spot, monthly membership,
    rebalanced weekly to equal). excess = mean member spot_ret - rf, net of spot rebal cost."""
    ret = spot_close.pct_change()
    didx = spot_close.index
    W = pd.DataFrame(np.nan, index=didx, columns=spot_close.columns)
    for d in _fridays(didx):
        m = membership.loc[d] & spot_close.loc[d].notna()
        w = pd.Series(0.0, index=spot_close.columns)      # full row incl. zeros for non-members
        if m.sum():
            w[m[m].index] = 1.0 / m.sum()
        W.loc[d] = w
    W = W.ffill().fillna(0.0)                              # hold the full weekly vector between Fridays
    gross = (W.shift(1) * ret).sum(axis=1) - rf_d
    turnover = (W - W.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = c_spot * turnover
    return {"excess": gross - cost, "gross": gross, "turnover": turnover, "cost": cost, "W": W}


def carry_excess(didx: pd.DatetimeIndex) -> pd.Series:
    """6. CARRY — stored v8.0 book. excess = stored net - rf (repo honesty convention). 'net'
    already credits idle cash at rf, so excess is 0 on idle days and the funding spread minus rf
    when deployed."""
    r = pd.read_parquet(CARRY_RESULTS)
    net = r["net"]
    net.index = pd.to_datetime(net.index)
    net = net.reindex(didx)
    return net - rf_daily(didx)


# ---------------------------------------------------------------- cost constants (from v3.1.yaml)

def costs() -> dict:
    """Cost table + derived one-way costs (bps as decimals). carry_pair = spot+perp taker + 2 slip;
    perp single-leg = perp taker + slip; spot single-leg = spot taker + slip."""
    c = cfg.load_config("strategies/funding_carry/v3.1.yaml")["costs"]
    return {**c,
            "carry_pair_oneway": c["spot_taker"] + c["perp_taker"] + 2 * c["slippage_per_leg"],
            "perp_oneway": c["perp_taker"] + c["slippage_per_leg"],
            "spot_oneway": c["spot_taker"] + c["slippage_per_leg"]}
