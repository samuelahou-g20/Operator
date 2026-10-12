import numpy as np
import pandas as pd

from src.strategies.funding_carry import engine


def test_deploy_ratio_default_is_one():
    assert engine._deploy_ratio(None) == 1.0
    assert engine._deploy_ratio({}) == 1.0


def test_deploy_ratio_portfolio():
    assert engine._deploy_ratio({"mode": "portfolio"}) == 1.0
    assert np.isclose(engine._deploy_ratio({"mode": "portfolio", "reserve": 0.1}), 0.9)


def test_deploy_ratio_isolated_k_over_k_plus_one():
    # the owner's example: 100k at 4x perp leverage -> 80k/80k book -> deploy 0.8
    assert np.isclose(engine._deploy_ratio({"mode": "isolated", "perp_leverage": 4}), 0.8)
    assert np.isclose(engine._deploy_ratio({"mode": "isolated", "perp_leverage": 1}), 0.5)
    assert np.isclose(engine._deploy_ratio({"mode": "isolated", "perp_leverage": 10}), 10 / 11)


def _pos():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame({"btcusdt": 0.5, "altusdt": 0.5}, index=idx)


def test_margin_report_portfolio_unimmr():
    cfg = {"mode": "portfolio", "collateral_default": 0.5,
           "collateral_overrides": {"btcusdt": 0.95},
           "mmr_default": 0.02, "mmr_overrides": {"btcusdt": 0.004}}
    pos = _pos()
    keep = pd.Series(True, index=pos.index)
    report, uni = engine._margin_report(cfg, pos, keep)
    # adjEq = 0.5*0.95 + 0.5*0.5 = 0.725 ; MM = 0.5*0.004 + 0.5*0.02 = 0.012 ; uniMMR = 60.4
    assert report["mode"] == "portfolio" and report["deploy_ratio"] == 1.0
    assert np.isclose(uni.iloc[0], 0.725 / 0.012, rtol=1e-6)
    assert report["unimmr_min"] == round(0.725 / 0.012, 1)
    assert report["liq_threshold"] == 1.05


def test_margin_report_counts_cash_when_underdeployed():
    cfg = {"mode": "portfolio", "collateral_default": 1.0, "mmr_default": 0.01}
    pos = _pos() * 0.5                                       # only half deployed -> half cash
    keep = pd.Series(True, index=pos.index)
    _, uni = engine._margin_report(cfg, pos, keep)
    # adjEq = 0.5 (spot at cr=1) + 0.5 (cash) = 1.0 ; MM = 0.5*0.01
    assert np.isclose(uni.iloc[0], 1.0 / 0.005, rtol=1e-6)


def test_margin_report_isolated():
    report, uni = engine._margin_report({"mode": "isolated", "perp_leverage": 4},
                                        _pos(), pd.Series(True, index=_pos().index))
    assert uni is None
    assert report["deploy_ratio"] == 0.8 and report["per_name_buffer"] == 0.25
