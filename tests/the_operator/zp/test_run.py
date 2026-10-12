"""Offline tests for ``run_zp``'s dynamic-mode source preference: gamma-dash first (richer — live
vol history + the real ladder), falling back to the strategy-boxes source only when gamma-dash
comes back empty. No network calls — every fetcher is monkeypatched."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.the_operator.zp import run, sources

_START, _END = datetime(2026, 1, 1), datetime(2026, 1, 2)


def _stub_common_fetchers(monkeypatch):
    monkeypatch.setattr(sources, "fetch_trades", lambda token, start, end, warnings: pd.DataFrame())
    monkeypatch.setattr(sources, "fetch_klines",
                        lambda exchange, pair, timeframe, start, end: None)


def test_run_zp_prefers_gamma_dash_over_strategy_params(monkeypatch):
    _stub_common_fetchers(monkeypatch)
    gamma_payload = {"rows": [("RENEAR", "BINANCE", "USDT", {"reference_annualized_volatility_pct": 40},
                              "mm2", [])], "series": [], "source": "gamma-dash"}
    monkeypatch.setattr(sources, "fetch_gamma_dash",
                        lambda names, hours, start, end, warnings: gamma_payload)

    def _must_not_be_called(*a, **kw):
        raise AssertionError("fetch_strategy_params should not be called when gamma-dash succeeds")

    monkeypatch.setattr(sources, "fetch_strategy_params", _must_not_be_called)

    result = run.run_zp("NEAR", _START, _END, "mm3", want_ob=False)
    assert result.gamma_dash is gamma_payload


def test_run_zp_falls_back_to_strategy_params_when_gamma_dash_empty(monkeypatch):
    _stub_common_fetchers(monkeypatch)
    monkeypatch.setattr(sources, "fetch_gamma_dash",
                        lambda names, hours, start, end, warnings: None)
    box_payload = {"rows": [("BUY", "BINANCE", "USDT", {"reference_annualized_volatility_pct": 40},
                            "mm3", [])], "series": [], "source": "strategy_boxes"}
    monkeypatch.setattr(sources, "fetch_strategy_params",
                        lambda token, program, warnings: box_payload)

    result = run.run_zp("NEAR", _START, _END, "mm3", want_ob=False)
    assert result.gamma_dash is box_payload


def test_run_zp_dmc_none_when_both_sources_empty(monkeypatch):
    _stub_common_fetchers(monkeypatch)
    monkeypatch.setattr(sources, "fetch_gamma_dash",
                        lambda names, hours, start, end, warnings: None)
    monkeypatch.setattr(sources, "fetch_strategy_params",
                        lambda token, program, warnings: None)

    result = run.run_zp("NEAR", _START, _END, "mm3", want_ob=False)
    assert result.gamma_dash is None
