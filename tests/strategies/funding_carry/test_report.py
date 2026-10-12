import json

import numpy as np
import pandas as pd

from src.strategies.funding_carry.report import build


def _make_run(run_dir):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2021-01-08", periods=420, freq="D", name="ts")
    gross = pd.Series(rng.normal(0.0004, 0.003, len(idx)), index=idx)
    turnover = pd.Series(np.abs(rng.normal(0.15, 0.1, len(idx))), index=idx)
    deployed = pd.Series(rng.integers(0, 4, len(idx)), index=idx)
    one_way = 0.0019
    pd.DataFrame({"gross": gross, "net": gross - one_way * turnover,
                  "turnover": turnover, "deployed": deployed}).to_parquet(run_dir / "returns.parquet")
    pd.DataFrame({"funding_pnl": gross * 1.1, "basis_pnl": gross * 0.0,
                  "cost": one_way * turnover}).to_parquet(run_dir / "attribution_daily.parquet")
    pd.DataFrame({
        "symbol": ["btcusdt", "ethusdt"], "pct_days_held": [0.6, 0.5], "mean_apr_held": [0.16, 0.18],
        "funding_pnl": [0.14, 0.15], "basis_pnl": [0.0, 0.0], "cost": [0.16, 0.13],
        "net_pnl": [-0.02, 0.02], "hit_rate": [0.72, 0.71],
    }).set_index("symbol").to_parquet(run_dir / "attribution_asset.parquet")
    meta = {
        "version": "v0",
        "config": {"name": "funding_carry", "version": "v0", "source": "binance",
                   "universe": ["btcusdt", "ethusdt"],
                   "signal": {"apr_window_days": 7, "gate_breakeven_days": 30, "max_n": None},
                   "costs": {"spot_taker": 0.001, "perp_taker": 0.0005, "slippage_per_leg": 0.0002}},
        "c_rt": 0.0038, "one_way": one_way,
        "period": ["2021-01-08", "2022-03-04"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(meta))


def test_report_self_contained_with_gross_column(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _make_run(run)
    out = build(run, out_path=tmp_path / "r.html")
    html = out.read_text(encoding="utf-8")
    # self-contained vector figures (inline SVG, no external assets, no raster)
    assert "src=\"http" not in html and "<link" not in html
    assert "data:image/png" not in html
    assert html.count("<svg") >= 6
    # gross KPI column present (the fix requested), plus net columns and carry sections
    assert "Gross" in html and "Net @1×" in html and "Net @2×" in html
    assert "P&amp;L attribution" in html and "Per-asset breakdown" in html
    assert "Gross Ann Return" in html      # headline card
    assert "Sharpe @1×" in html            # one Sharpe card (rf degrades to 0 on legacy runs)


def test_report_cash_and_rf_aware(tmp_path):
    # a v8-style run: returns carry the persisted rf series, attribution a cash column
    run = tmp_path / "run"
    run.mkdir()
    _make_run(run)
    ret = pd.read_parquet(run / "returns.parquet")
    ret["rf"] = 0.0001
    ret.to_parquet(run / "returns.parquet")
    attr = pd.read_parquet(run / "attribution_daily.parquet")
    attr["cash"] = 0.0001
    attr.to_parquet(run / "attribution_daily.parquet")

    html = build(run, out_path=tmp_path / "r.html").read_text(encoding="utf-8")
    assert "Cash yield" in html            # waterfall gained the cash bar
    assert "excess @1×" in html and ">rf<" in html   # by-year gained rf + excess columns
    # the rendered Sharpe card is rf-adjusted: (E[r] − E[rf]) / σ(r) on the cash-inclusive net
    net1 = ret["gross"] - 0.0019 * ret["turnover"] + attr["cash"]
    want = (net1.mean() - 0.0001) * 365 / (net1.std() * (365 ** 0.5))
    assert f"{want:.2f}" in html
    assert "vs 0" not in html and "sharpe_raw" not in html
