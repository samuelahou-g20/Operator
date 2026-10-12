import numpy as np
import pandas as pd

from src.strategies.funding_carry import engine


def _didx(n):
    return pd.date_range("2024-01-01", periods=n, freq="D")


def test_age_counts_days_since_first_eligible():
    idx = _didx(5)
    E = pd.DataFrame({"old": [True] * 5,
                      "new": [False, False, True, True, True]}, index=idx)
    age = engine._feat_age({"E": E}, {})
    assert list(age["old"]) == [0, 1, 2, 3, 4]                    # ages from sample start
    assert age["new"].isna().tolist() == [True, True, False, False, False]
    assert list(age["new"].dropna()) == [0, 1, 2]                 # 0 on the first eligible day


def test_age_survives_eligibility_gaps():
    # a data gap after listing must not reset the clock (calendar age, like the research)
    idx = _didx(5)
    E = pd.DataFrame({"gappy": [False, True, False, True, True]}, index=idx)
    age = engine._feat_age({"E": E}, {})
    assert list(age["gappy"].dropna()) == [0, 1, 2, 3]


def test_cvar_is_worst_k_mean():
    idx = _didx(5)
    F = pd.DataFrame({"a": [0.01, -0.02, 0.03, -0.05, 0.04]}, index=idx)
    leg = pd.DataFrame(0.0, index=idx, columns=["a"])
    # window 4, default q=0.05 -> k=1 (the single worst day); min_periods = 2
    cv = engine._feat_cvar({"F": F, "leg": leg}, {"window": 4})
    assert np.isnan(cv["a"].iloc[0])                              # fewer than window//2 obs
    assert np.allclose(cv["a"].iloc[1:], [-0.02, -0.02, -0.05, -0.05])


def test_cvar_q_sets_tail_size():
    idx = _didx(4)
    F = pd.DataFrame({"a": [0.01, -0.02, 0.03, -0.05]}, index=idx)
    leg = pd.DataFrame(0.0, index=idx, columns=["a"])
    # q=0.5 on window 4 -> k=2: mean of the two worst days
    cv = engine._feat_cvar({"F": F, "leg": leg}, {"window": 4, "q": 0.5})
    assert np.isclose(cv["a"].iloc[-1], (-0.05 + -0.02) / 2)


def test_cvar_uses_pair_return_not_funding_alone():
    idx = _didx(4)
    F = pd.DataFrame({"a": [0.01, 0.01, 0.01, 0.01]}, index=idx)
    leg = pd.DataFrame({"a": [0.0, -0.04, 0.0, 0.0]}, index=idx)  # basis blowout day
    cv = engine._feat_cvar({"F": F, "leg": leg}, {"window": 4})
    assert np.isclose(cv["a"].iloc[-1], 0.01 - 0.04)              # tail comes from F + leg
