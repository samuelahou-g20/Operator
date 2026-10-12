"""Funding-carry research lab — the shared, reproducible backtest + selection harness.

Every funding-carry research notebook imports this module so results are reproducible from one
place. It mirrors the production engine exactly (parity-checked in `validate_engine`) but loads the
panels once and evaluates knobs/vetos/weightings in memory, so sweeps are fast:

  load_panels / universe        daily funding F, basis return L, basis level, eligibility E
  signal_apr / hysteresis_weights   v1 signal + vectorized per-name Layer-D hysteresis
  feature_panels / veto_mask / held_matrix / weights_from_held / book   v2 selection primitives
  evaluate / evaluate_select / evaluate_v2   book a config on a (masked) window → engine._metrics
  sweep / select                 grid sweep + gated selection (max net Sharpe s.t. gate)

Used by notebooks/holdout_validation.ipynb (v1 knob validation) and notebooks/v2_selection.ipynb
(v2 selection + basis veto). research/ may import src; not vice-versa. Reads the lake via query.

Run:  uv run python research/funding_carry/lab.py            # engine-parity check + a demo sweep
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import config as cfg  # noqa: E402
from src.data import query  # noqa: E402
from src.strategies.funding_carry import engine  # noqa: E402

MAJORS = ["btcusdt", "ethusdt", "bnbusdt", "xrpusdt", "solusdt"]
SOURCE = "binance"
TF = "1h"

# Knob grid. θ_keep < θ_in enforced downstream.
HALFLIVES = [3, 5, 7, 10, 14]
THETA_INS = [0.04, 0.06, 0.08, 0.10, 0.12]
THETA_KEEPS = [0.0, 0.02, 0.04]
INCUMBENT = {"ewma_halflife": 7, "theta_in_apr": 0.08, "theta_keep_apr": 0.02}


def universe() -> tuple[list[str], list[str]]:
    """(majors, non-majors) — names with all three legs present in the lake (funding+perp1h+spot1h)."""
    covf = query.coverage("perps", SOURCE, "funding_rate")
    covp = query.coverage("perps", SOURCE, "bars", timeframe=TF)
    covs = query.coverage("spot", SOURCE, "bars", timeframe=TF)
    both = sorted(set(covf.index) & set(covp.index) & set(covs.index))
    nonmajors = [s for s in both if s not in MAJORS]
    return [s for s in MAJORS if s in both], nonmajors


# ---------------------------------------------------------------- panels (load once, sweep in memory)


def load_panels(symbols: list[str], start="2021-01-01", end="2026-05-31") -> dict:
    """Daily funding-income F, delta-neutral leg L, eligibility E over one date grid (engine parity)."""
    didx = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    fund, leg, basis, elig = {}, {}, {}, {}
    for s in symbols:
        spot = engine._daily_close("spot", SOURCE, s, TF, start, end, didx)
        perp = engine._daily_close("perps", SOURCE, s, TF, start, end, didx)
        leg[s] = spot.pct_change() - perp.pct_change()
        basis[s] = (perp - spot) / spot
        fund[s] = engine._daily_funding(SOURCE, s, start, end, didx)
        elig[s] = spot.notna() & perp.notna()
    F, L, E = pd.DataFrame(fund), pd.DataFrame(leg), pd.DataFrame(elig)
    return {"F": F, "L": L, "basis": pd.DataFrame(basis), "E": E, "R": F + L,
            "didx": didx, "symbols": symbols}


# ---------------------------------------------------------------- v1 signal + vectorized hysteresis


def signal_apr(F: pd.DataFrame, hl: float) -> pd.DataFrame:
    """v1 signal: EWMA-mean daily funding × 365 = as-of APR (identical to v1.signals.signal)."""
    return F.ewm(halflife=hl).mean() * engine._YEAR


def hysteresis_weights(A: pd.DataFrame, theta_in: float, theta_keep: float) -> pd.DataFrame:
    """v1 decision, vectorized per name (max_n=null case): enter ≥θ_in, keep ≥θ_keep, else exit;
    equal-weight the held set. NaN (ineligible/pre-listing/delisted) → force-exit."""
    a = A.to_numpy()
    ge_in, ge_keep = a >= theta_in, a >= theta_keep      # NaN → False on both (auto force-exit)
    T, N = a.shape
    H = np.empty((T, N), dtype=bool)
    held = np.zeros(N, dtype=bool)
    for t in range(T):
        held = (held & ge_keep[t]) | (~held & ge_in[t])
        H[t] = held
    counts = H.sum(axis=1, keepdims=True)
    W = np.where(H, 1.0 / np.where(counts == 0, 1, counts), 0.0)
    return pd.DataFrame(W, index=A.index, columns=A.columns)


# ---------------------------------------------------------------- factor diagnostics (v3.2 research)


def funding_factor(sig: pd.DataFrame, window: int = 90):
    """Common funding factor G (cross-sectional mean of the carry signal) and each name's rolling
    loading β_i = cov(sig_i, G)/var(G). Used to test Layer-G factor-aware sizing."""
    G = sig.mean(axis=1)
    mS, mG = sig.rolling(window, min_periods=40).mean(), G.rolling(window, min_periods=40).mean()
    cov = (sig.mul(G, axis=0)).rolling(window, min_periods=40).mean().sub(mS.mul(mG, axis=0), axis=0)
    beta = cov.div(G.rolling(window, min_periods=40).var(), axis=0)
    return G, beta.where(sig.notna())


def drawdown(returns: pd.DataFrame | pd.Series):
    """Running drawdown (≤0) of an equity built from daily returns; masks where returns are NaN."""
    eq = (1 + returns.fillna(0)).cumprod()
    dd = eq / eq.cummax() - 1
    return dd.where(returns.notna())


def overlay_state(dd, qexit: float = 0.15, qreenter: float = 0.50, lookback: int = 252,
                  expanding: bool = True):
    """Point-in-time drawdown on/off state (1=in, 0=out) from a drawdown series/frame ``dd`` (≤0).
    Exit when dd falls below the ``qexit`` percentile of its own PAST distribution; re-enter when it
    heals back above the ``qreenter`` percentile — percentiles estimated point-in-time (expanding, or
    rolling ``lookback``). Works on a Series (portfolio overlay) or a DataFrame (per-asset). The
    shadow's dd is passed untouched, so recovery is always observable (never on the flat traded book)."""
    df = dd.to_frame() if isinstance(dd, pd.Series) else dd
    roll = df.expanding(min_periods=lookback) if expanding else df.rolling(lookback, min_periods=min(lookback, 120))
    qe, qr = roll.quantile(qexit).to_numpy(), roll.quantile(qreenter).to_numpy()
    a = df.to_numpy()
    st = np.ones_like(a)
    s = np.ones(a.shape[1])
    for t in range(a.shape[0]):
        ex = (s == 1) & (a[t] <= qe[t]) & ~np.isnan(qe[t])
        re = (s == 0) & (a[t] >= qr[t]) & ~np.isnan(qr[t])
        s = np.where(ex, 0.0, np.where(re, 1.0, s))
        st[t] = s
    out = pd.DataFrame(st, index=df.index, columns=df.columns)
    return out.iloc[:, 0] if isinstance(dd, pd.Series) else out


def shrink(sig: pd.DataFrame, omega, prior: pd.Series | None = None) -> pd.DataFrame:
    """Cross-sectional James-Stein shrinkage of a per-name signal toward a prior (Layer-H research).
    ``omega`` (scalar or day×name, in [0,1]; 1 = no shrink) is the signal weight; ``prior`` defaults
    to the per-day cross-sectional mean. μ_shr = ω·sig + (1−ω)·prior."""
    pri = sig.mean(axis=1) if prior is None else prior
    if np.isscalar(omega):
        return (sig * omega).add(pri * (1 - omega), axis=0)
    return sig.mul(omega) + (1 - omega).mul(pri, axis=0)


def neff(returns: pd.DataFrame, min_periods: int = 60) -> float:
    """Effective number of independent bets in a set of columns: (Σλ)²/Σλ² of their correlation
    matrix (λ = eigenvalues). N_eff ≈ N when uncorrelated, ≪ N when a common factor dominates."""
    C = returns.corr(min_periods=min_periods).dropna(how="all").dropna(how="all", axis=1)
    if C.shape[0] < 2:
        return float(C.shape[0])
    C = C.loc[C.index, C.index]
    ev = np.linalg.eigvalsh(np.nan_to_num(C.to_numpy()))
    ev = ev[ev > 1e-9]
    return float((ev.sum() ** 2) / (ev ** 2).sum())


# ---------------------------------------------------------------- v2 selection (generic veto + weighting)


def feature_panels(panels: dict, specs: list) -> dict:
    """Build the declared feature frames using the ENGINE's own builders (exact production parity)."""
    ctx = {"F": panels["F"], "leg": panels["L"], "basis": panels["basis"]}
    didx, syms, E = panels["didx"], panels["symbols"], panels["E"]
    return {sp["name"]: engine._FEATURE_BUILDERS[sp["name"]](ctx, sp)
            .reindex(index=didx, columns=syms).where(E) for sp in specs}


def veto_mask(feats: dict, rules: list, ref: pd.DataFrame) -> pd.DataFrame:
    """Boolean day×name frame: True = clears every veto rule (NaN feature → passes, unknown)."""
    ok = pd.DataFrame(True, index=ref.index, columns=ref.columns)
    for r in rules:
        f = feats[r["feature"]]
        ok &= ~(f > r["cap"]) if r["direction"] == "max" else ~(f < r["cap"])
    return ok


def held_matrix(score, veto_ok, theta_in, theta_keep, k_in, k_keep) -> pd.DataFrame:
    """Vectorized selection: veto → θ-gate → top-K by score with rank hysteresis (None K = no cap)."""
    a, ok = score.to_numpy(), veto_ok.to_numpy()
    ki = np.inf if k_in is None else k_in
    kk = np.inf if k_keep is None else k_keep
    T, N = a.shape
    H = np.zeros((T, N), dtype=bool)
    held = np.zeros(N, dtype=bool)
    for t in range(T):
        cand = ~np.isnan(a[t]) & ok[t]
        rank = np.full(N, np.inf)
        idx = np.where(cand)[0]
        if len(idx):
            rank[idx[np.argsort(-a[t][idx], kind="stable")]] = np.arange(1, len(idx) + 1)
        enter = (~held) & cand & (a[t] >= theta_in) & (rank <= ki)
        keep = held & cand & (a[t] >= theta_keep) & (rank <= kk)
        held = enter | keep
        H[t] = held
    return pd.DataFrame(H, index=score.index, columns=score.columns)


def weights_from_held(H: pd.DataFrame, size: pd.DataFrame | None = None) -> pd.DataFrame:
    """Equal weight the held set, or weight ∝ ``size`` (e.g. 1/basis_vol) among the held, renormalized."""
    if size is None:
        s = H.astype(float)
    else:
        s = size.where(H, 0.0).clip(lower=0.0).fillna(0.0)   # held name w/ undefined size → 0 weight
    tot = s.sum(axis=1)
    return s.div(tot.where(tot > 0, 1.0), axis=0)


def deadband_weights(W: pd.DataFrame, band: float) -> pd.DataFrame:
    """No-trade band: keep a continuing name's weight until its target drifts > ``band`` (relative);
    entries take target, exits go to 0, then renormalize to stay fully invested. Threads prev weights
    exactly as the engine's decide does (matches v3.1 to 1e-6)."""
    Wt = W.to_numpy()
    T, N = Wt.shape
    out = np.empty((T, N))
    prev = np.zeros(N)
    for t in range(T):
        tgt = Wt[t]
        held_now, cont = tgt > 0, (tgt > 0) & (prev > 0)
        w = np.where(held_now, tgt, 0.0)                       # entries/continuing → target; exits → 0
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(tgt - prev) / np.where(tgt > 0, tgt, 1.0)
        w = np.where(cont & (rel < band), prev, w)             # within band → hold prev (no trade)
        s = w.sum()
        if s > 0:
            w = w / s
        out[t] = w
        prev = w
    return pd.DataFrame(out, index=W.index, columns=W.columns)


def book(W: pd.DataFrame, panels: dict, costs: dict, mask=None, warmup_days=30,
         cost_mults=(1.0, 2.0)) -> dict:
    """Book P&L for a weight matrix (shared by every evaluator). Returns metrics + net series."""
    R, didx = panels["R"], panels["didx"]
    pos = W.shift(1).fillna(0.0)
    deployed = (pos > 0).sum(axis=1)
    gross = (pos * R.fillna(0.0)).sum(axis=1)
    turnover = (pos - pos.shift(1).fillna(0.0)).abs().sum(axis=1)
    one_way, _ = engine.round_trip_cost(costs)
    keep = pd.Series(didx >= didx[0] + pd.Timedelta(days=warmup_days), index=didx)
    if mask is not None:
        keep = keep & mask.reindex(didx).fillna(False)
    out = {"W": W, "gross": gross, "turnover": turnover, "deployed": deployed,
           "net": (gross - one_way * turnover)[keep]}
    for m in cost_mults:
        net = gross - one_way * m * turnover
        out[f"{m}x"] = engine._metrics(net[keep], gross[keep], turnover[keep], deployed[keep])
    return out


def evaluate_select(panels, hl, theta_in, theta_keep, k_in, k_keep, costs,
                    feature_specs=(), veto_rules=(), score_kind="funding", size_kind="equal",
                    band=None, mask=None, warmup_days=30, cost_mults=(1.0, 2.0)) -> dict:
    """General v2 evaluator: pick a score, apply the veto, select top-K (rank hysteresis), weight, book.

    score_kind: 'funding' (EWMA-APR) or 'carry_over_risk' (EWMA-APR / basis_vol — risk-adjusted).
    size_kind:  'equal' or 'inv_basis_vol' (weight ∝ 1/basis_vol among the held).
    band:       None, or a relative no-trade band on weight changes (v3.1 deadband).
    """
    F, E, didx, syms = panels["F"], panels["E"], panels["didx"], panels["symbols"]
    apr = signal_apr(F, hl).reindex(index=didx, columns=syms).where(E)
    specs = list(feature_specs)
    if score_kind == "carry_over_risk" or size_kind == "inv_basis_vol":
        if not any(s["name"] == "basis_vol" for s in specs):
            specs.append({"name": "basis_vol", "window": 30})
    feats = feature_panels(panels, specs)
    bvol = feats.get("basis_vol")
    score = apr if score_kind == "funding" else apr / bvol.replace(0.0, np.nan)
    veto_ok = veto_mask(feats, list(veto_rules), apr)
    if size_kind == "inv_basis_vol":                       # a name we can't size isn't held (engine parity)
        veto_ok = veto_ok & bvol.notna() & (bvol > 0)
    H = held_matrix(score, veto_ok, theta_in, theta_keep, k_in, k_keep)
    size = (1.0 / bvol.replace(0.0, np.nan)) if size_kind == "inv_basis_vol" else None
    W = weights_from_held(H, size)
    if band is not None:
        W = deadband_weights(W, band)
    out = book(W, panels, costs, mask, warmup_days, cost_mults)
    out["A"] = apr
    return out


def evaluate_v2(panels, hl, theta_in, theta_keep, k_in, k_keep, fv_cap, nf_cap, costs,
                fv_win=30, nf_win=30, mask=None, warmup_days=30, cost_mults=(1.0, 2.0)) -> dict:
    """v2.0 shim (funding-instability veto) over evaluate_select — kept for existing call sites."""
    return evaluate_select(
        panels, hl, theta_in, theta_keep, k_in, k_keep, costs,
        feature_specs=[{"name": "funding_vol", "window": fv_win}, {"name": "neg_freq", "window": nf_win}],
        veto_rules=[{"feature": "funding_vol", "direction": "max", "cap": fv_cap},
                    {"feature": "neg_freq", "direction": "max", "cap": nf_cap}],
        mask=mask, warmup_days=warmup_days, cost_mults=cost_mults)


# ---------------------------------------------------------------- evaluate one knob set on a window


def evaluate(panels: dict, hl, theta_in, theta_keep, costs: dict,
             mask: pd.Series | None = None, warmup_days: int = 14,
             cost_mults=(1.0, 2.0)) -> dict:
    """Book the v1 strategy for one knob set; return {mult: metrics, 'net': net series, 'W': weights}.
    `mask` restricts the metric window (held state still evolves over the full grid, so no re-warm)."""
    F, R, E, didx = panels["F"], panels["R"], panels["E"], panels["didx"]
    A = signal_apr(F, hl).reindex(index=didx, columns=panels["symbols"]).where(E)
    W = hysteresis_weights(A, theta_in, theta_keep)
    pos = W.shift(1).fillna(0.0)
    deployed = (pos > 0).sum(axis=1)
    gross = (pos * R.fillna(0.0)).sum(axis=1)
    turnover = (pos - pos.shift(1).fillna(0.0)).abs().sum(axis=1)
    one_way, _ = engine.round_trip_cost(costs)

    warm = didx[0] + pd.Timedelta(days=warmup_days)
    keep = pd.Series(didx >= warm, index=didx)
    if mask is not None:
        keep = keep & mask.reindex(didx).fillna(False)

    out = {"W": W, "A": A, "gross": gross, "turnover": turnover, "deployed": deployed}
    net1 = gross - one_way * turnover
    out["net"] = net1[keep]
    for m in cost_mults:
        net = gross - one_way * m * turnover
        out[f"{m}x"] = engine._metrics(net[keep], gross[keep], turnover[keep], deployed[keep])
    return out


def _window_mask(didx, start=None, end=None) -> pd.Series:
    m = pd.Series(True, index=didx)
    if start is not None:
        m &= didx >= pd.Timestamp(start)
    if end is not None:
        m &= didx <= pd.Timestamp(end)
    return m


# ---------------------------------------------------------------- grid sweep + gated selection


def sweep(panels: dict, costs: dict, mask=None, warmup_days=14) -> pd.DataFrame:
    """Evaluate every knob combo (θ_keep<θ_in) on the (masked) window; one row of metrics each."""
    rows = []
    for hl in HALFLIVES:
        for ti in THETA_INS:
            for tk in THETA_KEEPS:
                if tk >= ti:
                    continue
                r = evaluate(panels, hl, ti, tk, costs, mask=mask, warmup_days=warmup_days)
                m1, m2 = r["1.0x"], r["2.0x"]
                rows.append({
                    "hl": hl, "theta_in": ti, "theta_keep": tk,
                    "net_apr": m1["ann_return"], "net_apr_2x": m2["ann_return"],
                    "sharpe": m1["sharpe"], "sharpe_2x": m2["sharpe"],
                    "gross_apr": m1["gross_ann_return"], "max_dd": m1["max_drawdown"],
                    "turnover": m1["ann_turnover"], "pct_deployed": m1["pct_days_deployed"],
                    "avg_names": m1["avg_names_held"],
                })
    return pd.DataFrame(rows)


def select(sweep_df: pd.DataFrame) -> dict:
    """Gated selection: among knobs with net_apr>0 AND net_apr_2x>0, take max net Sharpe."""
    gated = sweep_df[(sweep_df["net_apr"] > 0) & (sweep_df["net_apr_2x"] > 0)]
    pool = gated if len(gated) else sweep_df
    best = pool.sort_values("sharpe", ascending=False).iloc[0]
    return {"ewma_halflife": int(best["hl"]),
            "theta_in_apr": float(best["theta_in"]),
            "theta_keep_apr": float(best["theta_keep"]),
            "gated": bool(len(gated)), "row": best}


# ---------------------------------------------------------------- engine parity check


def validate_engine() -> dict:
    """Confirm the vectorized path reproduces engine.run on the majors v1 base config."""
    conf = cfg.load_config("strategies/funding_carry/v1.yaml")
    ref = engine.run(conf, __import__(
        "src.strategies.funding_carry.v1.signals", fromlist=["signal", "decide"]).signal,
        __import__("src.strategies.funding_carry.v1.signals", fromlist=["signal", "decide"]).decide,
    )["results"]
    panels = load_panels(conf["universe"], conf["backtest"]["start"], conf["backtest"]["end"])
    mine = evaluate(panels, INCUMBENT["ewma_halflife"], INCUMBENT["theta_in_apr"],
                    INCUMBENT["theta_keep_apr"], conf["costs"],
                    warmup_days=conf["backtest"]["warmup_days"])
    diff = {k: (ref["1.0x"][k], mine["1.0x"][k]) for k in ("ann_return", "sharpe", "ann_turnover")}
    ok = all(abs(a - b) < 1e-6 for a, b in diff.values())
    return {"ok": ok, "compare": diff}


if __name__ == "__main__":
    v = validate_engine()
    print("engine parity:", "OK" if v["ok"] else "MISMATCH", v["compare"])
    maj, non = universe()
    print(f"universe: {len(maj)} majors, {len(non)} non-majors")
    p = load_panels(maj)
    costs = cfg.load_config("strategies/funding_carry/v1.yaml")["costs"]
    sw = sweep(p, costs, mask=_window_mask(p["didx"], end="2023-12-31"))
    b = select(sw)
    print("tuned-on-majors-early best:", {k: b[k] for k in list(b)[:3]}, "gated:", b["gated"])
