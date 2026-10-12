import numpy as np
import pandas as pd

from src.strategies.funding_carry import engine
from src.strategies.funding_carry.v2 import backtest as v2bt


def test_assets_single_source_keeps_bare_keys():
    cfg = {"source": "binance", "universe": ["btcusdt", "ethusdt"]}
    assert engine._assets(cfg) == [("binance", "btcusdt", "btcusdt"),
                                   ("binance", "ethusdt", "ethusdt")]


def test_assets_multi_source_composite_keys_in_config_order():
    cfg = {"sources": ["binance", "bybit"],
           "universe": {"binance": ["btcusdt"], "bybit": ["aerousdt", "b3usdt"]}}
    assert engine._assets(cfg) == [("binance", "btcusdt", "binance:btcusdt"),
                                   ("bybit", "aerousdt", "bybit:aerousdt"),
                                   ("bybit", "b3usdt", "bybit:b3usdt")]


def test_canon_base_strips_multiplier_conventions():
    # binance prefix and bybit suffix forms map to the same base coin
    assert v2bt._canon_base("1000shibusdt") == "shib"
    assert v2bt._canon_base("shib1000usdt") == "shib"
    assert v2bt._canon_base("1000000cheemsusdt") == "cheems"
    assert v2bt._canon_base("1mbabydogeusdt") == "babydoge"


def test_canon_base_keeps_genuine_digit_names():
    for sym, base in [("1inchusdt", "1inch"), ("c98usdt", "c98"), ("a8usdt", "a8"),
                      ("usd1usdt", "usd1"), ("hpos10iusdt", "hpos10i"), ("2zusdt", "2z")]:
        assert v2bt._canon_base(sym) == base


def _registry(*symbols):
    out = {}
    for s in symbols:
        out[f"{s}_perp"] = {"symbol": s, "asset_type": "perps"}
        out[f"{s}_spot"] = {"symbol": s, "asset_type": "spot"}
    return out


def _mock_universe(monkeypatch, by_source):
    monkeypatch.setattr(v2bt.universe, "load_universe", lambda src: by_source[src])


def test_resolve_breadth_drops_overlap_keeps_unique(monkeypatch):
    cov = {("perps", "binance", "funding_rate", None): ["btcusdt", "1000shibusdt"],
           ("perps", "binance", "bars", "1h"): ["btcusdt", "1000shibusdt"],
           ("spot", "binance", "bars", "1h"): ["btcusdt", "1000shibusdt"],
           ("perps", "bybit", "funding_rate", None): ["btcusdt", "shib1000usdt", "aerousdt"],
           ("perps", "bybit", "bars", "1h"): ["btcusdt", "shib1000usdt", "aerousdt"],
           ("spot", "bybit", "bars", "1h"): ["btcusdt", "shib1000usdt", "aerousdt"]}

    def fake_coverage(asset_type, source, data_type, timeframe=None, **kw):
        return pd.DataFrame(index=cov[(asset_type, source, data_type, timeframe)])

    monkeypatch.setattr(v2bt.query, "coverage", fake_coverage)
    _mock_universe(monkeypatch, {"binance": _registry("btcusdt", "1000shibusdt"),
                                 "bybit": _registry("btcusdt", "shib1000usdt", "aerousdt")})
    cfg = {"sources": ["binance", "bybit"],
           "universe": {"binance": "all", "bybit": "breadth"},
           "backtest": {"timeframe": "1h"}}
    resolved = v2bt._resolve_universe(cfg)
    assert resolved["binance"] == ["1000shibusdt", "btcusdt"]
    assert resolved["bybit"] == ["aerousdt"]          # btc + shib overlap -> stay on binance


def test_resolve_all_mode_keeps_both_instances(monkeypatch):
    monkeypatch.setattr(v2bt.query, "coverage",
                        lambda *a, **k: pd.DataFrame(index=["btcusdt"]))
    _mock_universe(monkeypatch, {"binance": _registry("btcusdt"),
                                 "bybit": _registry("btcusdt")})
    cfg = {"sources": ["binance", "bybit"],
           "universe": {"binance": "all", "bybit": "all"},
           "backtest": {"timeframe": "1h"}}
    resolved = v2bt._resolve_universe(cfg)
    assert resolved == {"binance": ["btcusdt"], "bybit": ["btcusdt"]}


def test_resolve_single_source_unchanged(monkeypatch):
    monkeypatch.setattr(v2bt.query, "coverage",
                        lambda *a, **k: pd.DataFrame(index=["btcusdt", "ethusdt"]))
    _mock_universe(monkeypatch, {"binance": _registry("btcusdt", "ethusdt")})
    cfg = {"source": "binance", "universe": "all", "backtest": {"timeframe": "1h"}}
    assert v2bt._resolve_universe(cfg) == ["btcusdt", "ethusdt"]
    cfg = {"source": "binance", "universe": ["solusdt"], "backtest": {"timeframe": "1h"}}
    assert v2bt._resolve_universe(cfg) == ["solusdt"]


def test_registry_is_authoritative_over_lake(monkeypatch, capsys):
    # lake has an orphan symbol not in the registry -> excluded with a loud warning;
    # registry has an instrument without lake data -> skipped with a note
    monkeypatch.setattr(v2bt.query, "coverage",
                        lambda *a, **k: pd.DataFrame(index=["btcusdt", "orphanusdt"]))
    _mock_universe(monkeypatch, {"binance": _registry("btcusdt", "nodatausdt")})
    cfg = {"source": "binance", "universe": "all", "backtest": {"timeframe": "1h"}}
    assert v2bt._resolve_universe(cfg) == ["btcusdt"]
    out = capsys.readouterr().out
    assert "orphanusdt" in out and "WARNING" in out
    assert "nodatausdt" in out


def _pos_multi():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame({"binance:btcusdt": 0.5, "bybit:aerousdt": 0.25}, index=idx)


def test_margin_report_per_venue():
    cfg = {"mode": "portfolio", "collateral_default": 0.5, "mmr_default": 0.02,
           "collateral_overrides": {"btcusdt": 0.95}, "mmr_overrides": {"btcusdt": 0.004},
           "venues": {"bybit": {"collateral_default": 0.4, "mmr_default": 0.05}}}
    pos = _pos_multi()
    keep = pd.Series(True, index=pos.index)
    report, uni = engine._margin_report(cfg, pos, keep)
    # binance falls back to top-level maps (bare-name overrides apply through the venue prefix)
    assert np.isclose(report["venues"]["binance"]["unimmr_min"], round(0.95 / 0.004, 1))
    # bybit uses its own venue block
    assert np.isclose(report["venues"]["bybit"]["unimmr_min"], round(0.4 / 0.05, 1))
    # the returned series is the worst venue per day
    assert np.isclose(uni.iloc[0], 0.4 / 0.05)


def test_margin_report_single_venue_path_unchanged():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    pos = pd.DataFrame({"btcusdt": 0.5, "altusdt": 0.5}, index=idx)
    cfg = {"mode": "portfolio", "collateral_default": 0.5,
           "collateral_overrides": {"btcusdt": 0.95},
           "mmr_default": 0.02, "mmr_overrides": {"btcusdt": 0.004}}
    report, uni = engine._margin_report(cfg, pos, pd.Series(True, index=idx))
    assert "venues" not in report                     # legacy shape preserved
    assert np.isclose(uni.iloc[0], 0.725 / 0.012, rtol=1e-6)
