"""Shared funding-carry backtest engine (ARCHITECTURE §11).

Runs the machinery common to every version: loads funding + 1h spot/perp bars via ``query``,
builds per-name daily carry (funding income + the delta-neutral leg ``r_spot − r_perp``), then
for each rebalance day calls the version's pure ``decide`` to get target weights, books P&L on
*actual* turnover (doc §1), and writes the return series, attribution and metrics to
``results/{name}/{version}/{run_id}/``.

Multi-venue (v7): ``config["sources"]`` + a per-source ``universe`` dict trade each same-venue
delta-neutral pair as its own asset, columns keyed ``venue:pair`` (no cross-venue legs). Signals
and decide are pure per-column functions, so they need no changes; margin is reported per venue
(each venue is its own unified-margin account) and the summary gains a cross-venue capital-drift
diagnostic. Single-``source`` configs keep bare-symbol keys — v0–v6 outputs are unchanged.

Returns are booked **per dollar of equity (margin)**, not notional: the optional ``margin``
config block (v5) sets the deploy ratio — ``portfolio`` (spot collateralizes the perp, ratio 1.0;
what v0–v4 implicitly assumed) or ``isolated`` (ratio k/(k+1) for perp leverage k) — and adds a
margin-safety report (daily uniMMR vs Binance's 1.05 liquidation threshold). No block → ratio 1.0.

A version supplies two pure callables (in ``v{n}/signals.py``):
  ``signal(F, config) -> DataFrame[day × asset]``   causal as-of signal (e.g. APR estimate)
  ``decide(asof_row, feats, prev_weights, config) -> Series`` target weights over held names

``feats`` is a per-day ``{feature_name: Series[asset]}`` dict built from the optional
``config["features"]`` block (causal, eligibility-masked) so a decision can veto on more than the
signal (e.g. funding volatility); it is ``{}`` when no features are declared (v0/v1 ignore it).

MUST NOT: hold any version-specific signal/selection logic, or open lake files outside ``query``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.common import logging as common_logging
from src.common import paths
from src.data import query

_log = common_logging.get_logger("funding_carry.engine")
_YEAR = 365.0


def round_trip_cost(costs: dict) -> tuple[float, float]:
    """(one_way, round_trip) cost fractions: both legs + slippage, ×2 for enter+exit."""
    one_way = costs["spot_taker"] + costs["perp_taker"] + 2 * costs["slippage_per_leg"]
    return one_way, 2 * one_way


def _daily_close(asset_type, source, symbol, tf, start, end, didx) -> pd.Series:
    bars = query.get(asset_type, source, symbol, "bars", timeframe=tf, start=start, end=end)
    return bars["close"].resample("1D").last().reindex(didx)


def _daily_funding(source, symbol, start, end, didx) -> pd.Series:
    f = query.get("perps", source, symbol, "funding_rate", start=start, end=end)
    return f["rate"].resample("1D").sum().reindex(didx).fillna(0.0)


def _assets(config: dict) -> list[tuple[str, str, str]]:
    """``(source, symbol, column_key)`` per tradeable asset.

    Single-source configs keep bare-symbol column keys, so v0–v6 outputs are byte-identical;
    multi-venue configs (v7: ``sources`` list + per-source ``universe`` dict) key every column
    ``venue:pair`` in config-order."""
    if "sources" in config:
        return [(src, sym, f"{src}:{sym}")
                for src in config["sources"] for sym in config["universe"][src]]
    return [(config["source"], sym, sym) for sym in config["universe"]]


# ---------------------------------------------------------------- margin accounting (v5)
# Returns are booked per dollar of EQUITY (margin), not per dollar of notional. The optional
# ``config["margin"]`` block declares the margin setup; the engine derives the deploy ratio
# (pair notional per $1 equity) and scales the book accordingly. Absent block → deploy 1.0,
# which preserves v0–v4 byte-for-byte and equals the portfolio-margin setup they assumed.


def _deploy_ratio(margin_cfg: dict | None) -> float:
    """Pair notional deployable per $1 of equity.

    ``isolated``: spot (N) + perp margin (N/k) = equity → N/E = k/(k+1) for perp leverage k.
    ``portfolio``: spot itself collateralizes the perp (Binance PM) → N/E = 1.
    ``reserve`` holds back a cash fraction in either mode.
    """
    if not margin_cfg:
        return 1.0
    deploy = 1.0 - margin_cfg.get("reserve", 0.0)
    if margin_cfg.get("mode") == "isolated":
        k = margin_cfg["perp_leverage"]
        deploy *= k / (k + 1.0)
    return deploy


def _margin_report(margin_cfg: dict, pos: pd.DataFrame, keep: pd.Series) -> tuple[dict, pd.Series | None]:
    """Margin-safety summary (+ daily uniMMR series in portfolio mode).

    Portfolio mode: uniMMR_t = adjusted equity / maintenance margin, with per-name collateral
    ratios (haircuts) and MMRs from the config maps; Binance liquidates at uniMMR ≤ 1.05.
    Isolated mode: each perp must survive alone — the per-name buffer is ~1/k of notional.

    Multi-venue (``venue:pair`` columns): each venue is its own unified-margin account, so
    uniMMR is computed per venue over that venue's positions only — conservatively with NO
    idle-cash credit (where spare cash sits is an execution decision the backtest can't see).
    Per-venue maps come from ``margin["venues"][venue]``, falling back to the top-level maps.
    """
    deploy = _deploy_ratio(margin_cfg)
    if margin_cfg.get("mode") == "isolated":
        k = margin_cfg["perp_leverage"]
        return {"mode": "isolated", "deploy_ratio": round(deploy, 4), "perp_leverage": k,
                "per_name_buffer": round(1.0 / k, 4),
                "note": "each perp liquidates alone if it rallies ~buffer intraday"}, None

    def _maps(cfg: dict, cols, bare) -> tuple[pd.Series, pd.Series]:
        cr_d, cr_o = cfg.get("collateral_default", 0.5), cfg.get("collateral_overrides", {})
        mm_d, mm_o = cfg.get("mmr_default", 0.02), cfg.get("mmr_overrides", {})
        return (pd.Series({c: cr_o.get(b, cr_d) for c, b in zip(cols, bare)}),
                pd.Series({c: mm_o.get(b, mm_d) for c, b in zip(cols, bare)}))

    venues = sorted({c.split(":", 1)[0] for c in pos.columns if ":" in c})
    if not venues:
        cr, mm = _maps(margin_cfg, pos.columns, pos.columns)
        adj = (pos * cr).sum(axis=1) + (1.0 - pos.sum(axis=1)).clip(lower=0.0)   # haircut spot + cash
        mmreq = (pos * mm).sum(axis=1)
        uni = (adj / mmreq.replace(0.0, np.nan))[keep]
        lived = uni.dropna()
        return {"mode": "portfolio", "deploy_ratio": round(deploy, 4), "liq_threshold": 1.05,
                "unimmr_min": round(float(lived.min()), 1) if len(lived) else None,
                "unimmr_p01": round(float(lived.quantile(0.01)), 1) if len(lived) else None,
                "unimmr_median": round(float(lived.median()), 1) if len(lived) else None}, uni

    per, uni_worst = {}, None
    for v in venues:
        vcfg = (margin_cfg.get("venues") or {}).get(v, margin_cfg)
        cols = [c for c in pos.columns if c.startswith(v + ":")]
        cr, mm = _maps(vcfg, cols, [c.split(":", 1)[1] for c in cols])
        sub = pos[cols]
        uni = ((sub * cr).sum(axis=1) / (sub * mm).sum(axis=1).replace(0.0, np.nan))[keep]
        lived = uni.dropna()
        per[v] = {"unimmr_min": round(float(lived.min()), 1) if len(lived) else None,
                  "unimmr_median": round(float(lived.median()), 1) if len(lived) else None}
        uni_worst = uni if uni_worst is None else pd.concat([uni_worst, uni], axis=1).min(axis=1)
    return {"mode": "portfolio", "deploy_ratio": round(deploy, 4), "liq_threshold": 1.05,
            "venues": per, "note": "per-venue uniMMR, no idle-cash credit"}, uni_worst


# ---------------------------------------------------------------- feature panel (selection vetoes)
# Builders derive per-name feature frames from a context of raw panels (funding F, basis level B,
# basis return L, eligibility E) for a version's decide() to veto or size on. Add a builder here
# (e.g. dollar_volume) when a point-version needs it; the decide contract is unchanged. Pre-listing
# days are 0 in F, so a just-listed name reads as low-vol here — the ``age`` veto is the guard.


def _feat_funding_vol(ctx: dict, spec: dict) -> pd.DataFrame:
    """Rolling std of daily funding income, on the annualized APR axis (× year) — funding vol."""
    w = spec["window"]
    return ctx["F"].rolling(f"{w}D", min_periods=2).std() * _YEAR


def _feat_neg_freq(ctx: dict, spec: dict) -> pd.DataFrame:
    """Fraction of the trailing window with negative funding (flip-proneness), in [0, 1]."""
    w = spec["window"]
    return (ctx["F"] < 0).rolling(f"{w}D", min_periods=1).mean()


def _feat_basis(ctx: dict, spec: dict) -> pd.DataFrame:
    """Rolling mean of |perp−spot|/spot — how dislocated the two legs sit (blowout gauge)."""
    w = spec["window"]
    return ctx["basis"].abs().rolling(f"{w}D", min_periods=1).mean()


def _feat_basis_vol(ctx: dict, spec: dict) -> pd.DataFrame:
    """Rolling std of the daily basis return (the delta-neutral leg), annualized — basis risk."""
    w = spec["window"]
    return ctx["leg"].rolling(f"{w}D", min_periods=2).std() * (_YEAR ** 0.5)


def _feat_age(ctx: dict, spec: dict) -> pd.DataFrame:
    """Days since a name's first eligible day (listing proxy; 0 on that day) — the maturity gauge.
    Names already trading at sample start float as "old" (they age from the first sample day)."""
    seen = ctx["E"].cummax()
    return seen.cumsum().where(seen) - 1.0


def _feat_cvar(ctx: dict, spec: dict) -> pd.DataFrame:
    """Rolling CVaR of the daily pair return (funding + basis leg): mean of the worst ``q``-fraction
    of days in the window — a negative number, the typical bad-day loss. Tail gauge for sizing."""
    w, q = spec["window"], spec.get("q", 0.05)
    k = max(1, int(round(q * w)))

    def tail_mean(x):
        x = x[~np.isnan(x)]
        return np.sort(x)[:k].mean() if len(x) >= w // 2 else np.nan

    return (ctx["F"] + ctx["leg"]).rolling(w, min_periods=w // 2).apply(tail_mean, raw=True)


_FEATURE_BUILDERS = {"funding_vol": _feat_funding_vol, "neg_freq": _feat_neg_freq,
                     "basis": _feat_basis, "basis_vol": _feat_basis_vol,
                     "age": _feat_age, "cvar": _feat_cvar}


def _build_features(config: dict, ctx: dict, didx, syms, E: pd.DataFrame) -> dict:
    """Per-name feature frames declared under ``config['features']`` — causal, eligibility-masked."""
    panel = {}
    for spec in config.get("features", []):
        panel[spec["name"]] = (
            _FEATURE_BUILDERS[spec["name"]](ctx, spec).reindex(index=didx, columns=syms).where(E)
        )
    return panel


def _cash_yield(cash_cfg: dict, pos: pd.DataFrame, didx) -> tuple[pd.Series, pd.Series]:
    """(daily cash P&L credit, daily rf rate) on idle cash (v8).

    The rf path comes from the lake via ``rf_series`` ({source, symbol} → the canonical
    ``rates`` series, decimal annual, e.g. fred/tb3ms) or, as an offline fallback, from an
    inline monthly ``rf_apr`` map ({"YYYY-MM": annual rate}). Either way it is forward-filled
    to days — a flat APR would misprice the 2022 0→5% ramp, which is the whole effect.
    Credit = idle × rf × ``capture`` (fraction of the bill rate actually earned on-venue);
    the returned rf series is un-captured, for excess-Sharpe measurement."""
    if "rf_series" in cash_cfg:
        s = cash_cfg["rf_series"]
        obs = query.get("rates", s["source"], s["symbol"], "rates")["value"]
        rf = obs.reindex(didx, method="ffill").fillna(0.0) / _YEAR
    else:
        rf_m = pd.Series({pd.Timestamp(f"{k}-01"): float(v)
                          for k, v in cash_cfg["rf_apr"].items()}).sort_index()
        key = didx.to_period("M").to_timestamp()
        rf = pd.Series(rf_m.reindex(key, method="ffill").to_numpy(), index=didx).fillna(0.0) / _YEAR
    idle = (1.0 - pos.sum(axis=1)).clip(lower=0.0)
    return idle * rf * cash_cfg.get("capture", 1.0), rf


def _metrics(net: pd.Series, gross: pd.Series, turnover: pd.Series, deployed: pd.Series,
             rf: pd.Series | None = None) -> dict:
    net = net.dropna()
    years = len(net) / _YEAR
    ann_ret = net.mean() * _YEAR
    ann_vol = net.std() * np.sqrt(_YEAR)
    cum = (1 + net).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())
    var = float(np.percentile(net, 5)) if len(net) else float("nan")
    cvar = float(net[net <= var].mean()) if (net <= var).any() else var
    by_year = {}
    for y, g in net.groupby(net.index.year):
        v = g.std() * np.sqrt(_YEAR)
        rf_y = float(rf.reindex(g.index).fillna(0.0).mean() * _YEAR) if rf is not None else 0.0
        by_year[int(y)] = {"ann_return": round(g.mean() * _YEAR, 4),
                           "sharpe": round((g.mean() * _YEAR - rf_y) / v, 2) if v > 0 else None,
                           "days": int(len(g))}
    # Sharpe = (E[r] − E[rf]) / σ(r) — a ratio that ignores rf is not a Sharpe ratio.
    # Configs with a rate path (v8+) get the true figure; older configs (no rf source,
    # calibrated in the ~0% era) degrade to rf=0 and their stored outputs are unchanged.
    ann_rf = float(rf.reindex(net.index).fillna(0.0).mean() * _YEAR) if rf is not None else 0.0
    out_extra = {"ann_rf": round(ann_rf, 4)} if rf is not None else {}
    return {
        "ann_return": round(float(ann_ret), 4), "ann_vol": round(float(ann_vol), 4),
        "sharpe": round(float((ann_ret - ann_rf) / ann_vol), 3) if ann_vol > 0 else float("nan"),
        **out_extra,
        "max_drawdown": round(max_dd, 4), "cvar95": round(cvar, 5),
        "total_return": round(float(cum.iloc[-1] - 1), 4) if len(net) else 0.0,
        "gross_ann_return": round(float(gross.dropna().mean() * _YEAR), 4),
        "pct_days_deployed": round(float((deployed > 0).mean()), 3),
        "avg_names_held": round(float(deployed[deployed > 0].mean()), 2) if (deployed > 0).any() else 0.0,
        "ann_turnover": round(float(turnover.sum() / years), 1), "n_days": int(len(net)),
        "by_year": by_year,
    }


def run(config: dict, signal_fn, decide_fn, overlay_fn=None) -> dict:
    """Run a funding-carry backtest with the version's ``signal_fn``/``decide_fn``; save + return.

    ``overlay_fn`` (optional, v4) is a **portfolio-level** hook ``overlay_fn(shadow_net, config) ->
    exposure``: given the *shadow* book's daily net-return series it returns a causal exposure in
    [0,1] that gates every weight (``W *= exposure``). Used for the drawdown overlay — de-risk when
    the shadow's own equity is in a sustained drawdown. ``None`` (default) = no overlay (v0–v3)."""
    bt, c = config["backtest"], config["costs"]
    assets = _assets(config)
    syms = [key for _, _, key in assets]
    start, end, tf = bt["start"], bt["end"], bt["timeframe"]
    didx = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")

    fund_d, leg_d, basis_d, elig_d = {}, {}, {}, {}
    for src, sym, key in assets:
        spot = _daily_close("spot", src, sym, tf, start, end, didx)
        perp = _daily_close("perps", src, sym, tf, start, end, didx)
        leg_d[key] = spot.pct_change() - perp.pct_change()        # exact long-spot/short-perp move
        basis_d[key] = (perp - spot) / spot                       # basis level (blowout gauge)
        fund_d[key] = _daily_funding(src, sym, start, end, didx)
        elig_d[key] = spot.notna() & perp.notna()                 # tradeable only when both legs exist
    F, L = pd.DataFrame(fund_d), pd.DataFrame(leg_d)
    E = pd.DataFrame(elig_d)                                       # point-in-time eligibility
    R = F + L
    # As-of signal, masked to eligibility: a symbol with no data (pre-listing / post-delisting)
    # gets NaN, so decide() never enters it and force-exits it the day it stops trading.
    A = signal_fn(F, config).reindex(index=didx, columns=syms).where(E)
    features = _build_features(config, {"F": F, "leg": L, "basis": pd.DataFrame(basis_d), "E": E},
                               didx, syms, E)
    one_way, c_rt = round_trip_cost(c)

    # Per-day target weights via the version's decide (may be stateful → thread prev row).
    W = pd.DataFrame(0.0, index=didx, columns=syms)
    prev = pd.Series(0.0, index=syms)
    for d in didx:
        feats_d = {fn: features[fn].loc[d] for fn in features}
        w = decide_fn(A.loc[d], feats_d, prev, config)
        row = pd.Series(0.0, index=syms)
        if w is not None and len(w):
            row.loc[w.index] = w.values
        W.loc[d] = row
        prev = row

    if overlay_fn is not None:
        # Portfolio overlay (v4): gate the shadow's weights on its OWN realized-equity drawdown.
        # Book the shadow once (un-gated) to get its net-return path, then W *= exposure (causal).
        pos0 = W.shift(1).fillna(0.0)
        gross0 = (pos0 * R.fillna(0.0)).sum(axis=1)
        turn0 = (pos0 - pos0.shift(1).fillna(0.0)).abs().sum(axis=1)
        exposure = overlay_fn(gross0 - one_way * turn0, config).reindex(didx).fillna(1.0)
        W = W.mul(exposure, axis=0)

    # Margin accounting (v5): book everything per dollar of EQUITY. Scaling the weights by the
    # deploy ratio makes returns/turnover/attribution equity-based; deploy=1.0 (default / PM) is
    # the setup v0–v4 implicitly assumed, so their outputs are unchanged.
    margin_cfg = config.get("margin")
    deploy = _deploy_ratio(margin_cfg)
    if deploy != 1.0:
        W = W * deploy

    pos = W.shift(1).fillna(0.0)                                  # held during day d = decision at d-1
    deployed = (pos > 0).sum(axis=1)
    # fillna(0) so a NaN return on a NOT-held day (pre-listing / post-delisting) contributes 0
    # rather than poisoning the portfolio sum; held days always have data (eligibility mask).
    gross = (pos * R.fillna(0.0)).sum(axis=1)
    turnover = (pos - pos.shift(1).fillna(0.0)).abs().sum(axis=1)
    funding_pnl_n, basis_pnl_n = pos * F.fillna(0.0), pos * L.fillna(0.0)
    cost_n = one_way * (pos - pos.shift(1).fillna(0.0)).abs()

    warm = didx[0] + pd.Timedelta(days=bt["warmup_days"])
    keep = pd.Series(didx >= warm, index=didx)

    # Idle-cash yield (v8): credit overlay-out cash at the configured rf path. The overlay's
    # shadow book stays uncredited (it is fully deployed by construction, idle ≈ 0).
    cash_cfg = config.get("cash")
    cash, rf = _cash_yield(cash_cfg, pos, didx) if cash_cfg else (None, None)

    results = {}
    for mult in bt.get("cost_stress_mult", [1.0]):
        net = gross - one_way * mult * turnover
        if cash is not None:
            net = net + cash
        results[f"{mult}x"] = _metrics(net[keep], gross[keep], turnover[keep], deployed[keep],
                                       rf=rf[keep] if rf is not None else None)

    margin_report, unimmr = (None, None)
    if margin_cfg:
        margin_report, unimmr = _margin_report(margin_cfg, pos, keep)

    # Multi-venue diagnostic: the per-venue capital split drifts as selection moves between
    # venues — in reality that is cash transferred between exchanges. Reported, not modeled.
    venue_report = None
    if "sources" in config:
        shares = pd.DataFrame({s: pos[[col for col in pos.columns if col.startswith(s + ":")]]
                               .sum(axis=1) for s in config["sources"]})
        drift = (shares.diff().abs().sum(axis=1) / 2.0)[keep]
        venue_report = {
            "capital_share_mean": {s: round(float(shares[s][keep].mean()), 3) for s in shares},
            "daily_transfer_mean": round(float(drift.mean()), 4),
            "daily_transfer_max": round(float(drift.max()), 4),
        }

    version = config.get("version", "v0")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = paths.RESULTS_DIR / config["name"] / version / run_id
    out.mkdir(parents=True, exist_ok=True)
    km = keep.to_numpy()
    net_1x = gross - one_way * turnover
    if cash is not None:
        net_1x = net_1x + cash
    ret_out = {"gross": gross, "net": net_1x, "turnover": turnover, "deployed": deployed}
    if rf is not None:
        ret_out["rf"] = rf                     # persisted so reports/consumers need no lake read
    if unimmr is not None:
        ret_out["unimmr"] = unimmr.reindex(didx)
    pd.DataFrame(ret_out)[km].to_parquet(out / "returns.parquet")
    attribution = {"funding_pnl": funding_pnl_n.sum(axis=1), "basis_pnl": basis_pnl_n.sum(axis=1),
                   "cost": cost_n.sum(axis=1)}
    if cash is not None:
        attribution["cash"] = cash
    pd.DataFrame(attribution)[km].to_parquet(out / "attribution_daily.parquet")
    rows = []
    for sym in syms:
        held = (pos[sym] > 0) & keep
        net_n = funding_pnl_n[sym] + basis_pnl_n[sym] - cost_n[sym]
        rows.append({"symbol": sym, "pct_days_held": round(float(held.mean()), 3),
                     "mean_apr_held": round(float(A[sym][held].mean()), 4) if held.any() else None,
                     "funding_pnl": round(float(funding_pnl_n[sym][keep].sum()), 5),
                     "basis_pnl": round(float(basis_pnl_n[sym][keep].sum()), 5),
                     "cost": round(float(cost_n[sym][keep].sum()), 5),
                     "net_pnl": round(float(net_n[keep].sum()), 5),
                     "hit_rate": round(float((net_n[held] > 0).mean()), 3) if held.any() else None})
    pd.DataFrame(rows).set_index("symbol").to_parquet(out / "attribution_asset.parquet")
    W[km].to_parquet(out / "weights.parquet")
    A[km].to_parquet(out / "apr.parquet")
    summary = {"version": version, "config": config, "c_rt": round(c_rt, 5),
               "one_way": round(one_way, 5), "period": [str(didx[0].date()), str(didx[-1].date())],
               "results": results, "margin": margin_report, "venues": venue_report}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    _log.info("%s %s: net@1x ann=%.2f%% sharpe=%.2f turnover=%.1fx (%s)",
              config["name"], version, results["1.0x"]["ann_return"] * 100,
              results["1.0x"]["sharpe"], results["1.0x"]["ann_turnover"], out)
    summary["run_dir"] = str(out)
    return summary
