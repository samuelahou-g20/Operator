"""Sort key: numeric-mirror lookup, symbol fallback, and None-sorts-last in both directions."""

from src.monitor.app import _sort_value

_RK = {("G", "BTCUSDT"): "binance|btcusdt", ("G", "ETHUSDT"): "binance|ethusdt"}
_NUMS = {
    "binance|btcusdt": {"trend_st": 74.0, "z_up": None},
    "binance|ethusdt": {"trend_st": -12.0, "z_up": 1.5},
}


def test_symbol_column_sorts_by_symbol_string():
    assert _sort_value("symbol", "BTCUSDT", "G", _RK, _NUMS, float("inf")) == "BTCUSDT"


def test_numeric_column_uses_mirror():
    assert _sort_value("trend_st", "BTCUSDT", "G", _RK, _NUMS, float("inf")) == 74.0
    assert _sort_value("trend_st", "ETHUSDT", "G", _RK, _NUMS, float("inf")) == -12.0


def test_none_sorts_last_both_directions():
    # ascending: missing -> +inf (last); descending: missing -> -inf (still last after reverse)
    assert _sort_value("z_up", "BTCUSDT", "G", _RK, _NUMS, float("inf")) == float("inf")
    assert _sort_value("z_up", "BTCUSDT", "G", _RK, _NUMS, float("-inf")) == float("-inf")


def test_unknown_symbol_is_none_valued():
    assert _sort_value("trend_st", "NOPE", "G", _RK, _NUMS, float("inf")) == float("inf")
