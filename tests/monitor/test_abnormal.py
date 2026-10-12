"""Abnormal-move scoring: long-horizon sigma projection and two-scale significance."""

from src.monitor.app import _abn_score, _sigma_from_vol


def test_sigma_from_vol_scales_with_horizon():
    s_1h = _sigma_from_vol(100.0, 3600)
    s_15m = _sigma_from_vol(100.0, 900)
    assert s_1h is not None and s_15m is not None
    assert abs(s_15m / s_1h - 0.5) < 1e-9         # 15m = 1/4 h -> sqrt(1/4) = 0.5
    assert _sigma_from_vol(None, 900) is None
    assert _sigma_from_vol(0.0, 900) is None


def test_abn_score_picks_more_extreme_scale():
    # move +1%; short sigma 0.5% -> z=2 (S); long sigma 0.2% -> z=5 (L) -> L governs
    z, tag = _abn_score(0.01, 0.005, 0.002)
    assert tag == "L" and abs(z - 5.0) < 1e-9


def test_abn_score_sign_and_short_only():
    z, tag = _abn_score(-0.01, 0.005, None)       # only the short scale is available
    assert tag == "S" and abs(z + 2.0) < 1e-9


def test_abn_score_none_when_no_move_or_no_sigma():
    assert _abn_score(None, 0.01, 0.01) == (None, None)
    assert _abn_score(0.01, None, None) == (None, None)
    assert _abn_score(0.01, 0.0, 0.0) == (None, None)
