"""Offline tests for twap_bot's parsing, venue guard and per-candle reduction (no network)."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from scripts import twap_bot as tw


# --------------------------------------------------------------------------- #
# parse_ts — the window is ALWAYS UTC
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw", ["2026/08/20-00:00:00UTC", "2026/08/20-00:00:00",
                                 "2026-08-20T00:00:00Z", "2026/08/20-00:00", "2026/08/20"])
def test_parse_ts_accepts_the_usual_spellings(raw):
    assert tw.parse_ts(raw) == datetime(2026, 8, 20, tzinfo=timezone.utc)


def test_parse_ts_is_timezone_aware_utc():
    """A naive/local window would silently shift the average."""
    assert tw.parse_ts("2026/08/20-12:30:00UTC").tzinfo is timezone.utc


@pytest.mark.parametrize("raw", ["yesterday", "20/08/2026", "", "2026/13/99"])
def test_parse_ts_rejects_junk(raw):
    with pytest.raises(ValueError, match="can't read time"):
        tw.parse_ts(raw)


# --------------------------------------------------------------------------- #
# parse_timeframe — a trader's spelling -> ccxt's
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("raw", "want"), [("1hr", "1h"), ("1hour", "1h"), ("4hr", "4h"),
                                           ("15min", "15m"), ("daily", "1d"), ("1h", "1h"),
                                           ("1m", "1m"), ("1W", "1w")])
def test_parse_timeframe_normalises(raw, want):
    assert tw.parse_timeframe(raw) == want


# --------------------------------------------------------------------------- #
# parse_command
# --------------------------------------------------------------------------- #
_FULL = ("/twap btw start=2026/08/20-00:00:00UTC end=2026/08/24-00:00:00UTC "
         "candle=1hr exchange=bitget type=default")


def test_parse_command_reads_every_field():
    c = tw.parse_command(_FULL)
    assert c["token"] == "BTW" and c["tf"] == "1h"
    assert c["exchange"] == "bitget" and c["type"] == "default"
    assert c["start"] == datetime(2026, 8, 20, tzinfo=timezone.utc)
    assert c["end"] == datetime(2026, 8, 24, tzinfo=timezone.utc)


def test_parse_command_defaults_candle_exchange_and_type():
    c = tw.parse_command("/twap eth start=2026/08/23 end=2026/08/24")
    assert (c["tf"], c["exchange"], c["type"]) == (tw.DEFAULT_TF, tw.DEFAULT_EXCHANGE, "default")


def test_parse_command_accepts_aliases_for_the_keys():
    c = tw.parse_command("/twap eth start=2026/08/23 end=2026/08/24 tf=4hr venue=okx")
    assert c["tf"] == "4h" and c["exchange"] == "okx"


def test_parse_command_maps_coinroutes_venue_names_to_ccxt():
    """So a venue copied out of an order record (`gateio`, `okex`) just works."""
    assert tw.parse_command("/twap eth start=2026/08/23 end=2026/08/24 exchange=gateio")["exchange"] == "gate"
    assert tw.parse_command("/twap eth start=2026/08/23 end=2026/08/24 exchange=okex")["exchange"] == "okx"


def test_parse_command_requires_both_bounds():
    for cmd in ("/twap btw start=2026/08/20", "/twap btw end=2026/08/24", "/twap btw"):
        with pytest.raises(ValueError):
            tw.parse_command(cmd)


def test_parse_command_rejects_an_inverted_window():
    with pytest.raises(ValueError, match="end must be after start"):
        tw.parse_command("/twap btw start=2026/08/24 end=2026/08/20")


def test_parse_command_rejects_an_unknown_reduction():
    with pytest.raises(ValueError, match="type must be one of"):
        tw.parse_command("/twap btw start=2026/08/23 end=2026/08/24 type=median")


def test_parse_command_rejects_an_unknown_argument():
    with pytest.raises(ValueError, match="unknown argument"):
        tw.parse_command("/twap btw start=2026/08/23 end=2026/08/24 foo=1")


def test_parse_command_ignores_a_botname_suffix():
    assert tw.parse_command("/twap@G20TwapBot btw start=2026/08/23 end=2026/08/24")["token"] == "BTW"


# --------------------------------------------------------------------------- #
# check_venue — an ambiguous ticker on the wrong venue is a DIFFERENT asset
# --------------------------------------------------------------------------- #
def test_check_venue_blocks_an_ambiguous_ticker_on_the_wrong_exchange():
    """AI is Gensyn on okx/kucoin/bitget/mexc; binance lists an unrelated AI. A TWAP taken there
    would be the wrong coin's average — plausible-looking, which is the dangerous kind of wrong."""
    with pytest.raises(ValueError, match="DIFFERENT asset"):
        tw.check_venue("AI", "binance")
    with pytest.raises(ValueError, match="DIFFERENT asset"):
        tw.check_venue("ai", "htx")


def test_check_venue_allows_an_ambiguous_ticker_on_a_correct_exchange():
    for v in ("okx", "kucoin", "bitget", "mexc"):
        tw.check_venue("AI", v)


def test_check_venue_leaves_unambiguous_tokens_alone():
    for v in ("binance", "bitget", "kucoin"):
        tw.check_venue("BTW", v)


# --------------------------------------------------------------------------- #
# twap — equal weight per candle is what makes it TIME-weighted
# --------------------------------------------------------------------------- #
def _candles(rows):
    idx = pd.date_range("2026-08-20", periods=len(rows), freq="1h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_twap_default_is_the_ohlc4_mean_across_candles():
    df = _candles([[1.0, 3.0, 1.0, 3.0], [2.0, 2.0, 2.0, 2.0]])
    assert tw.twap(df, "default") == pytest.approx(2.0)      # (2.0 + 2.0) / 2


def test_twap_alternative_reductions():
    df = _candles([[1.0, 4.0, 2.0, 3.0]])
    assert tw.twap(df, "close") == pytest.approx(3.0)
    assert tw.twap(df, "hl2") == pytest.approx(3.0)          # (4+2)/2
    assert tw.twap(df, "hlc3") == pytest.approx(3.0)         # (4+2+3)/3
    assert tw.twap(df, "ohlc4") == pytest.approx(2.5)        # (1+4+2+3)/4


def test_twap_weights_every_candle_equally_regardless_of_volume():
    """A quiet hour must count as much as a busy one — otherwise it is a VWAP, a different thing."""
    df = _candles([[10.0, 10.0, 10.0, 10.0], [20.0, 20.0, 20.0, 20.0]])
    df["volume"] = [1.0, 1_000_000.0]
    assert tw.twap(df, "default") == pytest.approx(15.0)     # not pulled toward 20


def test_twap_of_a_single_candle_is_that_candle():
    assert tw.twap(_candles([[1.0, 2.0, 3.0, 4.0]]), "default") == pytest.approx(2.5)


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def test_fmt_keeps_precision_on_a_sub_penny_token():
    assert tw._fmt(0.020372417) == "0.020372417"
    assert tw._fmt(0.44086678) == "0.44086678"
    assert tw._fmt(2427.0836) == "2,427.0836"


# --------------------------------------------------------------------------- #
# CSV attachment — the data behind the number
# --------------------------------------------------------------------------- #
def _ohlc_frame():
    idx = pd.date_range("2026-09-01", periods=3, freq="1h", tz="UTC")
    return pd.DataFrame({"open": [10.0, 11.0, 12.0], "high": [12.0, 13.0, 14.0],
                         "low": [8.0, 9.0, 10.0], "close": [11.0, 12.0, 13.0],
                         "volume": [1.0, 2.0, 3.0]}, index=idx)


def test_twap_csv_carries_the_reduced_price_column():
    """The reduced column is the point: its mean IS the reported TWAP, so the reply is checkable
    from the file instead of having to re-derive the reduction rule."""
    import io

    df = _ohlc_frame()
    out = pd.read_csv(io.StringIO(tw.twap_csv(df, "default").decode()))
    assert list(out.columns) == ["ts_utc", "open", "high", "low", "close", "volume",
                                 "price_default"]
    assert out["price_default"].mean() == pytest.approx(tw.twap(df, "default"))
    assert len(out) == len(df)


def test_twap_csv_column_is_named_after_the_reduction_used():
    import io

    for kind in ("close", "hl2", "hlc3"):
        out = pd.read_csv(io.StringIO(tw.twap_csv(_ohlc_frame(), kind).decode()))
        assert f"price_{kind}" in out.columns
        assert out[f"price_{kind}"].mean() == pytest.approx(tw.twap(_ohlc_frame(), kind))


def test_twap_csv_keeps_timestamps_as_readable_utc():
    import io

    out = pd.read_csv(io.StringIO(tw.twap_csv(_ohlc_frame(), "default").decode()))
    assert out["ts_utc"].iloc[0] == "2026-09-01 00:00:00"


def test_csv_name_states_symbol_venue_timeframe_and_window():
    cmd = {"exchange": "binance", "tf": "1h",
           "start": datetime(2026, 9, 1, tzinfo=timezone.utc),
           "end": datetime(2026, 9, 2, tzinfo=timezone.utc)}
    name = tw.csv_name(cmd, "BTC/USDT:USDT")
    assert name == "twap_BTCUSDTUSDT_binance_1h_20260901T0000_20260902T0000.csv"
    assert "/" not in name and ":" not in name      # must be a legal filename


def test_handle_twap_returns_no_csv_on_a_failure():
    """Every error path must still return the 3-tuple, with nothing to attach."""
    reply, csv, name = tw.handle_twap("/twap btc")      # missing start=/end=
    assert reply.startswith("❌") and csv is None and name is None
