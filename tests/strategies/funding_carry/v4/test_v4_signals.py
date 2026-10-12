import numpy as np
import pandas as pd

from src.strategies.funding_carry.v4 import signals


def _cfg(qexit=0.15, qreenter=0.50, lookback=252):
    return {"overlay": {"qexit": qexit, "qreenter": qreenter, "lookback": lookback}}


def _series():
    # 300d mild uptrend with noise (calibration) -> 60d drawdown -> 140d recovery to new highs
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=500, freq="D")
    r = np.concatenate([0.002 + rng.normal(0, 0.002, 300),
                        np.full(60, -0.004),
                        np.full(140, 0.004)])
    return pd.Series(r, index=idx)


def test_exposure_is_binary():
    e = signals.overlay(_series(), _cfg())
    assert set(np.unique(e.values)) <= {0.0, 1.0}


def test_inactive_before_lookback():
    e = signals.overlay(_series(), _cfg(lookback=252))
    assert e.iloc[:252].eq(1.0).all()          # no history to calibrate -> stay invested


def test_derisks_in_drawdown_and_reenters_on_recovery():
    e = signals.overlay(_series(), _cfg())
    assert e.iloc[300:360].eq(0.0).any()        # exits during the sustained drawdown
    assert e.iloc[-1] == 1.0                     # back in after recovery to new highs


def test_never_acts_if_lookback_exceeds_sample():
    e = signals.overlay(_series(), _cfg(lookback=600))
    assert e.eq(1.0).all()                       # never enough history -> always invested (no-op)
