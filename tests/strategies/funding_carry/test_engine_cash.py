import numpy as np
import pandas as pd

from src.strategies.funding_carry import engine


def _didx(start, n):
    return pd.date_range(start, periods=n, freq="D")


def test_rf_map_forward_fills_months():
    idx = _didx("2022-01-30", 4)                        # spans a month boundary
    pos = pd.DataFrame(0.0, index=idx, columns=["a"])   # fully idle
    cash, rf = engine._cash_yield({"rf_apr": {"2022-01": 0.365, "2022-02": 0.730}}, pos, idx)
    assert np.allclose(rf.values, [0.001, 0.001, 0.002, 0.002])   # apr/365, ffill at the boundary
    assert np.allclose(cash.values, rf.values)                    # idle = 1, capture default 1.0


def test_cash_credit_is_idle_times_rf_times_capture():
    idx = _didx("2022-06-01", 3)
    pos = pd.DataFrame({"a": [1.0, 0.6, 0.0]}, index=idx)         # deployed -> partial -> flat
    cash, _ = engine._cash_yield({"rf_apr": {"2022-06": 0.365}, "capture": 0.5}, pos, idx)
    assert np.allclose(cash.values, [0.0, 0.4 * 0.001 * 0.5, 1.0 * 0.001 * 0.5])


def test_rf_before_first_map_entry_is_zero():
    idx = _didx("2021-12-30", 4)
    pos = pd.DataFrame(0.0, index=idx, columns=["a"])
    cash, rf = engine._cash_yield({"rf_apr": {"2022-01": 0.365}}, pos, idx)
    assert np.allclose(rf.values, [0.0, 0.0, 0.001, 0.001])


def test_rf_series_mode_reads_lake(monkeypatch):
    # rf_series pulls the canonical decimal-annual rates series via query and ffills to days
    obs = pd.Series([0.0365, 0.0730],
                    index=pd.to_datetime(["2022-06-01", "2022-07-01"]), name="value")
    monkeypatch.setattr(engine.query, "get",
                        lambda *a, **k: obs.to_frame())
    idx = pd.to_datetime(["2022-06-15", "2022-07-02"])
    pos = pd.DataFrame(0.0, index=idx, columns=["a"])
    cash, rf = engine._cash_yield({"rf_series": {"source": "fred", "symbol": "tb3ms"}}, pos, idx)
    assert np.allclose(rf.values, [0.0365 / 365, 0.0730 / 365])
    assert np.allclose(cash.values, rf.values)                    # fully idle, capture 1.0


def test_metrics_sharpe_accounts_for_rf():
    idx = _didx("2024-01-01", 200)
    rng = np.random.default_rng(7)
    net = pd.Series(0.001 + rng.normal(0, 0.0005, len(idx)), index=idx)
    zero = pd.Series(0.0, index=idx)
    plain = engine._metrics(net, net, zero, zero)
    assert "ann_rf" not in plain                                  # v0–v7 shape preserved
    rf = pd.Series(0.0005, index=idx)
    with_rf = engine._metrics(net, net, zero, zero, rf=rf)
    assert "sharpe_raw" not in with_rf                            # one Sharpe, the real one
    assert with_rf["sharpe"] < plain["sharpe"]                    # rf haircut is real
    assert np.isclose(with_rf["ann_rf"], 0.0005 * 365, atol=1e-4)
    # Sharpe = (E[r] − E[rf]) / σ(r), the owner's definition
    direct = (net.mean() - rf.mean()) * 365 / (net.std() * np.sqrt(365))
    assert np.isclose(with_rf["sharpe"], round(direct, 3))
    # per-year sharpe carries the same haircut
    assert with_rf["by_year"][2024]["sharpe"] < plain["by_year"][2024]["sharpe"]
