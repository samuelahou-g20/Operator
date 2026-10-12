"""Presentation helpers: value formatting and color coding."""

from src.monitor.app import _fmt_age, _fmt_price, _pivot_cell, _trend_cell, _vol_cell, _z_cell


def test_fmt_price_scales():
    assert _fmt_price(64210.5) == "64,210.5"
    assert _fmt_price(3.14159) == "3.142"
    assert _fmt_price(0.00012345) == "0.000123"


def test_fmt_age_units():
    assert _fmt_age(5) == "5s ago"
    assert _fmt_age(125) == "2m ago"
    assert _fmt_age(7200) == "2h ago"


def _rgb(cell):
    import re
    return tuple(int(x) for x in re.findall(r"\d+", str(cell.style)))


def test_trend_cell_score_and_gradient():
    up, strong_up, down = _trend_cell(74.0), _trend_cell(95.0), _trend_cell(-84.0)
    assert up.plain == "▲ 74" and str(up.style).startswith("rgb(")
    assert down.plain == "▼ 84"
    ru, gu, bu = _rgb(up)
    rd, gd, bd = _rgb(down)
    assert gu > ru and gu > bu           # up -> green dominant
    assert rd > gd and rd > bd           # down -> red dominant
    assert _rgb(strong_up)[1] > gu       # stronger up -> more saturated green (gradient)
    assert _trend_cell(None).plain == "—"


def test_z_cell_thresholds():
    assert "magenta" in str(_z_cell(-0.5).style)   # beyond the pivot
    assert "green" in str(_z_cell(0.4).style)
    assert "yellow" in str(_z_cell(1.5).style)
    assert "red" in str(_z_cell(2.5).style)
    assert _z_cell(None).plain == "—"


def test_pivot_cell():
    assert _pivot_cell(64210.5).plain == "64,210.5"
    assert _pivot_cell(None).plain == "—"


def test_vol_cell_thresholds():
    assert "green" in str(_vol_cell(20.0).style)
    assert "yellow" in str(_vol_cell(60.0).style)
    assert "red" in str(_vol_cell(95.0).style)
    assert _vol_cell(None).plain == "—"


def test_abn_cell_flags_over_threshold():
    from src.monitor.app import _abn_cell
    quiet = _abn_cell(1.0, 4.0)
    flagged = _abn_cell(-5.0, 4.0)
    assert quiet.plain == "▲ 1.0σ" and "dim" in str(quiet.style)
    assert flagged.plain == "▼ 5.0σ" and "reverse" in str(flagged.style) and "red" in str(flagged.style)
    assert _abn_cell(None, 4.0).plain == "—"


def test_abn_cell_tag_suffix():
    from src.monitor.app import _abn_cell
    assert _abn_cell(3.0, 4.0, "L").plain == "▲ 3.0σ L"
    assert _abn_cell(-5.0, 4.0, "S").plain == "▼ 5.0σ S"
