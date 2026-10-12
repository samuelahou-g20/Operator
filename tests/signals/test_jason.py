"""Vol-blend parameters and the JSON level at k in src.signals.jason."""
import numpy as np
import pandas as pd

from src.signals.jason import _OUTPUT_COLUMNS, _vol_blend_params, compute_jason_vectorized

CRYPTO_DAY_S = 24 * 3600.0


def test_4h_keeps_historical_spec():
    assert _vol_blend_params("4h", 14400, "crypto", CRYPTO_DAY_S) == (9, 6.0)


def test_1h_pins_wall_clock_hours():
    assert _vol_blend_params("1h", 3600, "crypto", CRYPTO_DAY_S) == (36, 12.0)


def test_1h_pin_overrides_asset_class_tau():
    assert _vol_blend_params("1h", 3600, "equity", 16 * 3600.0) == (36, 12.0)


def test_unpinned_freq_falls_back_to_one_trading_day():
    assert _vol_blend_params("15min", 900, "crypto", CRYPTO_DAY_S) == (144, 96.0)


def _ohlc(n=24 * 10, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.r_[100.0, close[:-1]]
    spread = np.abs(rng.normal(0, 0.005, n))
    return pd.DataFrame(
        {"open": open_, "close": close,
         "high": np.maximum(open_, close) * (1 + spread),
         "low": np.minimum(open_, close) * (1 - spread)},
        index=pd.date_range("2026-01-01", periods=n, freq="1h"),
    )


def test_json_k_at_2_equals_json2():
    out = compute_jason_vectorized(_ohlc(), "T", "1h", json_k=2.0)
    np.testing.assert_allclose(out["jsonk_dn"], out["json2_dn"], rtol=1e-9)
    np.testing.assert_allclose(out["jsonk_up"], out["json2_up"], rtol=1e-9)


def test_json_k_is_opt_in():
    ohlc = _ohlc()
    plain = compute_jason_vectorized(ohlc.copy(), "T", "1h")
    with_k = compute_jason_vectorized(ohlc.copy(), "T", "1h", json_k=1.5)
    assert list(plain.columns) == _OUTPUT_COLUMNS
    assert with_k[_OUTPUT_COLUMNS].equals(plain)


def test_smaller_k_sits_closer_to_the_pivots():
    out = compute_jason_vectorized(_ohlc(), "T", "1h", json_k=1.5)
    assert (out["jsonk_dn"] > out["json2_dn"]).all() and (out["jsonk_dn"] < out["high_pivot"]).all()
    assert (out["jsonk_up"] < out["json2_up"]).all() and (out["jsonk_up"] > out["low_pivot"]).all()
