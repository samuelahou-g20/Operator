import pandas as pd
import pytest

from src.common import paths
from src.data import lake_io, query
from src.data.etl.perps import normalize
from src.data.fetching.chunking import RateLimiter
from src.data.fetching.hyperliquid import backfill

_H_MS = 3_600_000
_D_MS = 86_400_000
_JAN1 = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC in ms

_NO_THROTTLE = RateLimiter(0)

_META = {"universe": [
    {"name": "BTC", "szDecimals": 5, "maxLeverage": 40},
    {"name": "kPEPE", "szDecimals": 0, "maxLeverage": 10},   # multiplier coin, exact case matters
    {"name": "MATIC", "szDecimals": 1, "isDelisted": True},  # isDelisted present only when true
]}

_SPOT_META = {
    "tokens": [
        {"name": "USDC", "index": 0, "isCanonical": True, "fullName": None},
        {"name": "PURR", "index": 1, "isCanonical": True, "fullName": None},
        {"name": "UBTC", "index": 8, "isCanonical": False, "fullName": "Unit Bitcoin"},
        {"name": "UETH", "index": 9, "isCanonical": False, "fullName": "Unit Ethereum"},
    ],
    "universe": [
        {"tokens": [1, 0], "name": "PURR/USDC", "index": 0, "isCanonical": True},
        {"tokens": [8, 0], "name": "@142", "index": 142, "isCanonical": False},
        {"tokens": [9, 8], "name": "@999", "index": 999, "isCanonical": False},  # non-USDC quote
    ],
}


def _candle_rows(n, start=_JAN1, step=_H_MS, vols=None, interval="1h"):
    """Ascending candle dicts as /info candleSnapshot returns them (strings for o/h/l/c/v)."""
    vols = vols or ["10.0"] * n
    return [{"t": start + i * step, "T": start + (i + 1) * step - 1, "s": "BTC",
             "i": interval, "o": "1.0", "c": "1.5", "h": "2.0", "l": "0.5",
             "v": vols[i], "n": 7} for i in range(n)]


def _candle_server(rows, calls):
    """Serve `rows` the way candleSnapshot does: one shot, window-filtered, oldest-first."""
    def fake(rest_url, body):
        calls.append((rest_url, body))
        req = body["req"]
        sel = [r for r in rows if req["startTime"] <= r["t"] <= req["endTime"]]
        return sorted(sel, key=lambda r: r["t"])
    return fake


def _funding_rows(n, start=_JAN1, step=_H_MS, rates=None, premiums=None):
    rates = rates or ["0.0000125"] * n
    premiums = premiums or ["0.0003"] * n
    return [{"coin": "BTC", "fundingRate": rates[i], "premium": premiums[i],
             "time": start + i * step} for i in range(n)]


def _funding_server(rows, calls, page=500):
    """Serve `rows` the way fundingHistory does: OLDEST-first, `page` rows per call,
    startTime-floored and endTime-capped."""
    def fake(rest_url, body):
        calls.append((rest_url, body))
        sel = [r for r in rows
               if r["time"] >= body["startTime"] and r["time"] <= body.get("endTime", 2**62)]
        return sorted(sel, key=lambda r: r["time"])[:page]
    return fake


class _LogRecorder:
    def __init__(self):
        self.warnings = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args)

    def info(self, msg, *args):
        pass


# ---------------------------------------------------------------- request construction


def test_candle_request_construction(monkeypatch):
    calls = []
    monkeypatch.setattr(backfill, "_post_json", _candle_server([], calls))
    end = _JAN1 + 24 * _H_MS - 1
    df = backfill._fetch_candles("https://x", "BTC", backfill.INTERVALS["1h"],
                                 _JAN1, end, _NO_THROTTLE)
    assert df.empty
    rest_url, body = calls[0]
    assert rest_url == "https://x"
    assert body == {"type": "candleSnapshot",
                    "req": {"coin": "BTC", "interval": "1h", "startTime": _JAN1, "endTime": end}}


def test_funding_request_construction(monkeypatch):
    calls = []
    monkeypatch.setattr(backfill, "_post_json", _funding_server([], calls))
    end = _JAN1 + 24 * _H_MS - 1
    backfill._fetch_funding("https://x", "BTC", _JAN1, end, _NO_THROTTLE)
    _, body = calls[0]
    assert body == {"type": "fundingHistory", "coin": "BTC", "startTime": _JAN1, "endTime": end}


def test_interval_codes():
    assert backfill.INTERVALS == {"1m": "1m", "1h": "1h", "1d": "1d"}


# ---------------------------------------------------------------- oldest-first pagination


def test_funding_pagination_oldest_first_terminates_and_covers(monkeypatch):
    rows = _funding_rows(5)
    calls = []
    monkeypatch.setattr(backfill, "_post_json", _funding_server(rows, calls, page=2))
    df = backfill._fetch_funding("https://x", "BTC", _JAN1, _JAN1 + 4 * _H_MS, _NO_THROTTLE)
    assert len(calls) == 3                                   # 5 rows at 2/page
    assert list(df["funding_time"]) == [r["time"] for r in rows]  # ascending, complete, no dupes
    # each page moves the floor forward to just past the newest row already received
    assert calls[1][1]["startTime"] == rows[1]["time"] + 1
    assert calls[2][1]["startTime"] == rows[3]["time"] + 1


def test_funding_pagination_no_progress_guard(monkeypatch):
    # a server stuck on the same page (ignoring startTime) must not loop forever
    rows = _funding_rows(2)
    calls = []

    def stuck(rest_url, body):
        calls.append(body)
        return rows

    monkeypatch.setattr(backfill, "_post_json", stuck)
    df = backfill._fetch_funding("https://x", "BTC", _JAN1, _JAN1 + 24 * _H_MS, _NO_THROTTLE)
    assert len(calls) == 2                                   # bounded, no infinite loop
    assert sorted(set(df["funding_time"])) == [r["time"] for r in rows]


def test_funding_clips_to_window(monkeypatch):
    rows = _funding_rows(6)
    monkeypatch.setattr(backfill, "_post_json", _funding_server(rows, []))
    start, end = _JAN1 + _H_MS, _JAN1 + 3 * _H_MS
    df = backfill._fetch_funding("https://x", "BTC", start, end, _NO_THROTTLE)
    assert df["funding_time"].min() >= start and df["funding_time"].max() <= end
    assert len(df) == 3


# ---------------------------------------------------------------- candle window semantics


def test_candles_truncation_clamped_and_warned_not_raised(monkeypatch):
    # 1h history only reaches back ~5000 candles: a start beyond that is clamped by
    # the server; the connector warns and keeps what it got (expected, not an error)
    available_start = _JAN1 + 100 * _H_MS
    rows = _candle_rows(5, start=available_start)
    log = _LogRecorder()
    monkeypatch.setattr(backfill, "_post_json", _candle_server(rows, []))
    monkeypatch.setattr(backfill, "_log", log)
    df = backfill._fetch_candles("https://x", "BTC", "1h",
                                 _JAN1, available_start + 5 * _H_MS, _NO_THROTTLE)
    assert len(df) == 5
    assert df["start_time"].min() == available_start
    assert any("retention-truncated" in w for w in log.warnings)


def test_daily_candles_drop_leading_zero_volume(monkeypatch):
    # 1d pre-listing rows are price-backfilled with v="0.0": drop LEADING zeros only
    rows = _candle_rows(5, step=_D_MS, vols=["0.0", "0.0", "5.0", "0.0", "7.0"], interval="1d")
    monkeypatch.setattr(backfill, "_post_json", _candle_server(rows, []))
    df = backfill._fetch_candles("https://x", "BTC", "1d",
                                 _JAN1, _JAN1 + 5 * _D_MS, _NO_THROTTLE)
    assert list(df["start_time"]) == [r["t"] for r in rows[2:]]   # listing detected at row 2
    assert df["volume"].tolist() == ["5.0", "0.0", "7.0"]         # middle zero kept

    monkeypatch.setattr(backfill, "_post_json",
                        _candle_server(_candle_rows(3, step=_D_MS,
                                                    vols=["0.0"] * 3, interval="1d"), []))
    all_zero = backfill._fetch_candles("https://x", "BTC", "1d",
                                       _JAN1, _JAN1 + 3 * _D_MS, _NO_THROTTLE)
    assert all_zero.empty                                         # never listed in window


def test_hourly_candles_keep_zero_volume_rows(monkeypatch):
    # the leading-zero drop is listing detection for 1d only; a thin market's real
    # zero-volume 1h bars must survive
    rows = _candle_rows(3, vols=["0.0", "0.0", "5.0"])
    monkeypatch.setattr(backfill, "_post_json", _candle_server(rows, []))
    df = backfill._fetch_candles("https://x", "BTC", "1h",
                                 _JAN1, _JAN1 + 3 * _H_MS, _NO_THROTTLE)
    assert len(df) == 3


# ---------------------------------------------------------------- delisted funding truncation


def test_truncate_delisted_funding_after_last_nonzero():
    rows = _funding_rows(6, rates=["0.001", "-0.002", "0.003", "0.0", "0.0", "0.0"],
                         premiums=["0.1", "0.1", "0.0", "0.0", "0.0", "0.0"])
    df = pd.DataFrame({
        "funding_time": pd.Series([r["time"] for r in rows], dtype="int64"),
        "funding_rate": [r["fundingRate"] for r in rows],
        "premium": [r["premium"] for r in rows],
    })
    out = backfill._truncate_delisted_funding(df)
    assert len(out) == 3
    assert float(out["funding_rate"].iloc[-1]) != 0.0

    # a nonzero PREMIUM alone also counts as live (rate can settle at exactly 0)
    rows = _funding_rows(4, rates=["0.001", "0.0", "0.0", "0.0"],
                         premiums=["0.0", "0.0", "0.0002", "0.0"])
    df = pd.DataFrame({
        "funding_time": pd.Series([r["time"] for r in rows], dtype="int64"),
        "funding_rate": [r["fundingRate"] for r in rows],
        "premium": [r["premium"] for r in rows],
    })
    assert len(backfill._truncate_delisted_funding(df)) == 3

    assert backfill._truncate_delisted_funding(
        df.assign(funding_rate="0.0", premium="0.0")).empty


def test_backfill_small_truncates_delisted_funding(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    rows = _funding_rows(4, rates=["0.001", "-0.002", "0.0", "0.0"],
                         premiums=["0.1", "0.1", "0.0", "0.0"])
    monkeypatch.setattr(backfill, "_post_json", _funding_server(rows, []))
    coins = {"matic": {"coin": "MATIC", "delisted": True}}
    cfg = {"asset_type": "perps", "rest_url": "https://x"}
    out = backfill._backfill_small(cfg, ("funding_rate", None, "matic"),
                                   _JAN1, _JAN1 + 4 * _H_MS, coins, _NO_THROTTLE, False)
    assert out["rows"] == 2
    df = pd.read_parquet(paths.data_path("raw", "perps", "hyperliquid", "matic", "funding_rate")
                         / "matic.parquet")
    assert list(df["funding_time"]) == [rows[0]["time"], rows[1]["time"]]


# ---------------------------------------------------------------- run-time key resolution


def test_perp_coin_map(monkeypatch):
    monkeypatch.setattr(backfill, "_post_json", lambda *a: _META)
    coins = backfill._perp_coins("https://x", _NO_THROTTLE)
    assert coins == {
        "btc": {"coin": "BTC", "delisted": False},
        "kpepe": {"coin": "kPEPE", "delisted": False},   # exact API case preserved
        "matic": {"coin": "MATIC", "delisted": True},
    }


def test_spot_pair_resolution(monkeypatch):
    monkeypatch.setattr(backfill, "_post_json", lambda *a: _SPOT_META)
    pairs = backfill._spot_pairs("https://x", _NO_THROTTLE)
    # canonical pairs go by name, non-canonical MUST use "@{index}"; non-USDC quotes dropped
    assert pairs == {"purr": "PURR/USDC", "ubtc": "@142"}


def test_coin_keys_dispatch_per_asset_type(monkeypatch):
    def fake(rest_url, body):
        return _SPOT_META if body["type"] == "spotMeta" else _META

    monkeypatch.setattr(backfill, "_post_json", fake)
    spot = backfill._coin_keys({"asset_type": "spot", "rest_url": "https://x"}, _NO_THROTTLE)
    assert spot["ubtc"] == {"coin": "@142", "delisted": False}
    perps = backfill._coin_keys({"asset_type": "perps", "rest_url": "https://x"}, _NO_THROTTLE)
    assert perps["btc"] == {"coin": "BTC", "delisted": False}


# ---------------------------------------------------------------- market dispatch


def test_run_backfill_dispatches_per_market(monkeypatch):
    seen = []
    monkeypatch.setattr(backfill, "_backfill_market",
                        lambda cfg, *a: seen.append(cfg["asset_type"]))
    conf = {"max_workers": 1,
            "markets": {"perps": {"asset_type": "perps"}, "spot": {"asset_type": "spot"}}}
    backfill.run_backfill(conf)
    assert seen == ["perps", "spot"]
    seen.clear()
    backfill.run_backfill(conf, market="spot")
    assert seen == ["spot"]


def test_run_backfill_unknown_market():
    with pytest.raises(ValueError):
        backfill.run_backfill({"markets": {"perps": {}}}, market="nope")


# ---------------------------------------------------------------- idempotency + skips


_CFG = {"asset_type": "perps", "rest_url": "https://x"}
_COINS = {"btc": {"coin": "BTC", "delisted": False}}


def test_backfill_small_empty_is_skip_not_error(monkeypatch, tmp_path):
    # pre-listing window returns [] -> warn + skip, no file
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(backfill, "_post_json", lambda *a: [])
    out = backfill._backfill_small(_CFG, ("bars", "1h", "btc"),
                                   _JAN1, _JAN1 + _H_MS, _COINS, _NO_THROTTLE, False)
    assert out["missing"] and out["rows"] == 0
    target = paths.data_path("raw", "perps", "hyperliquid", "btc", "bars", "1h") / "btc.parquet"
    assert not target.exists()


def test_backfill_small_unknown_symbol_skips_without_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")

    def _boom(*a):
        raise AssertionError("must not fetch a symbol missing from meta")

    monkeypatch.setattr(backfill, "_post_json", _boom)
    out = backfill._backfill_small(_CFG, ("bars", "1h", "gone"),
                                   _JAN1, _JAN1 + _H_MS, _COINS, _NO_THROTTLE, False)
    assert out["missing"]


def test_backfill_small_skips_existing_unless_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    out_dir = paths.data_path("raw", "perps", "hyperliquid", "btc", "funding_rate")
    lake_io.write_table(pd.DataFrame({"funding_time": [1], "funding_rate": ["0.9"],
                                      "premium": ["0.1"]}),
                        out_dir / "btc.parquet")

    def _boom(*a):
        raise AssertionError("must not fetch when the target exists and overwrite=False")

    monkeypatch.setattr(backfill, "_post_json", _boom)
    out = backfill._backfill_small(_CFG, ("funding_rate", None, "btc"),
                                   _JAN1, _JAN1 + _H_MS, _COINS, _NO_THROTTLE, False)
    assert out["skipped"]

    monkeypatch.setattr(backfill, "_post_json", _funding_server(_funding_rows(2), []))
    out = backfill._backfill_small(_CFG, ("funding_rate", None, "btc"),
                                   _JAN1, _JAN1 + _H_MS, _COINS, _NO_THROTTLE, True)
    assert out["rows"] == 2                       # overwrite refetched cleanly
    df = pd.read_parquet(out_dir / "btc.parquet")
    assert list(df["funding_time"]) == [_JAN1, _JAN1 + _H_MS]   # old content replaced


def test_backfill_small_rerun_no_duplicates(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(backfill, "_post_json", _candle_server(_candle_rows(3), []))
    unit = ("bars", "1h", "btc")
    for overwrite in (False, True, False):
        backfill._backfill_small(_CFG, unit, _JAN1, _JAN1 + 2 * _H_MS,
                                 _COINS, _NO_THROTTLE, overwrite)
    df = pd.read_parquet(paths.data_path("raw", "perps", "hyperliquid", "btc", "bars", "1h")
                         / "btc.parquet")
    assert list(df["start_time"]) == [_JAN1, _JAN1 + _H_MS, _JAN1 + 2 * _H_MS]


# ---------------------------------------------------------------- end-to-end roundtrip


def _conf():
    return {
        "max_workers": 1, "throttle_per_sec": 0, "rest_url": "https://x",
        "markets": {
            "perps": {"asset_type": "perps",
                      "data_types": {"bars": {"timeframes": ["1h", "1d"],
                                              "storage": {"1h": "small", "1d": "small"}},
                                     "funding_rate": {"storage": "small"}},
                      "backfill": {"start": "2024-01-01", "symbols": ["btc"]}},
            "spot": {"asset_type": "spot",
                     "data_types": {"bars": {"timeframes": ["1h"],
                                             "storage": {"1h": "small"}}},
                     "backfill": {"start": "2024-01-01", "symbols": ["ubtc"]}},
        },
    }


def test_end_to_end_raw_normalize_query(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "PROCESSED_DIR", tmp_path / "processed")
    candle_calls = []
    hourly = _candle_server(_candle_rows(24), candle_calls)
    daily = _candle_server(_candle_rows(2, step=_D_MS, interval="1d"), candle_calls)
    funding = _funding_server(_funding_rows(3, rates=["0.0000125", "-0.0002", "0.0003"]), [])

    def fake(rest_url, body):
        if body["type"] == "meta":
            return _META
        if body["type"] == "spotMeta":
            return _SPOT_META
        if body["type"] == "candleSnapshot":
            server = daily if body["req"]["interval"] == "1d" else hourly
            return server(rest_url, body)
        assert body["type"] == "fundingHistory"
        return funding(rest_url, body)

    monkeypatch.setattr(backfill, "_post_json", fake)
    backfill.run_backfill(_conf(), start="2024-01-01", end="2024-01-02")

    # spot candles must be requested by the "@{index}" pair key, never the name
    spot_coins = {c[1]["req"]["coin"] for c in candle_calls if c[1]["req"]["interval"] == "1h"}
    assert spot_coins == {"BTC", "@142"}

    for asset_type, symbol in (("perps", "btc"), ("spot", "ubtc")):
        n = normalize.normalize("hyperliquid", symbol, "bars", timeframe="1h",
                                asset_type=asset_type)
        assert n == 24
        got = query.get(asset_type, "hyperliquid", symbol, "bars", timeframe="1h",
                        stage="processed")
        assert len(got) == 24
        assert got.index[0] == pd.Timestamp("2024-01-01 00:00:00")
        assert got.index[-1] == pd.Timestamp("2024-01-01 23:00:00")
        assert list(got.columns) == ["open", "high", "low", "close", "volume"]
        assert got["volume"].iloc[0] == 10.0                    # string -> float64

    assert normalize.normalize("hyperliquid", "btc", "bars", timeframe="1d",
                               asset_type="perps") == 2
    day = query.get("perps", "hyperliquid", "btc", "bars", timeframe="1d", stage="processed")
    assert list(day.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]

    assert normalize.normalize("hyperliquid", "btc", "funding_rate", asset_type="perps") == 3
    fr = query.get("perps", "hyperliquid", "btc", "funding_rate", stage="processed")
    assert list(fr["rate"]) == [0.0000125, -0.0002, 0.0003]     # sign preserved
    assert fr["next_funding_ts"].iloc[0] == fr.index[1]         # ts.shift(-1)
    assert fr["next_funding_ts"].iloc[1] == fr.index[2]
    assert pd.isna(fr["next_funding_ts"].iloc[-1])
    assert (fr.index[1] - fr.index[0]) == pd.Timedelta(hours=1)  # HOURLY cadence, no 8h assumption

    # rerun without overwrite: idempotent, nothing duplicated
    backfill.run_backfill(_conf(), start="2024-01-01", end="2024-01-02")
    raw = pd.read_parquet(paths.data_path("raw", "perps", "hyperliquid", "btc", "bars", "1h")
                          / "btc.parquet")
    assert len(raw) == 24
