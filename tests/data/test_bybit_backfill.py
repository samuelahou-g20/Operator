import pandas as pd
import pytest

from src.common import paths
from src.data import lake_io, query
from src.data.etl.perps import normalize
from src.data.fetching.bybit import backfill
from src.data.fetching.chunking import RateLimiter

_H_MS = 3_600_000
_JAN1 = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC in ms

_NO_THROTTLE = RateLimiter(0)


def _kline_rows(n, start=_JAN1, step=_H_MS):
    """Full ascending kline set as the vendor would hold it (ints for asserts)."""
    return [[start + i * step, "1.0", "2.0", "0.5", "1.5", "10.0", "15.0"] for i in range(n)]


def _kline_server(rows, calls, page=1000):
    """Serve `rows` the way v5 kline does: newest-first, `page` rows per call."""
    def fake(rest_url, path, params):
        calls.append((path, dict(params)))
        sel = [r for r in rows if params["start"] <= r[0] <= params["end"]]
        sel = sorted(sel, key=lambda r: -r[0])[: min(page, params["limit"])]
        return {"list": [[str(c) for c in r] for r in sel]}
    return fake


def _funding_rows(n, start=_JAN1, step=8 * _H_MS, rates=None):
    rates = rates or ["0.0001"] * n
    return [{"symbol": "BTCUSDT", "fundingRate": rates[i],
             "fundingRateTimestamp": str(start + i * step)} for i in range(n)]


def _funding_server(rows, calls, page=200):
    """Serve `rows` the way v5 funding/history does: newest-first, endTime-capped."""
    def fake(rest_url, path, params):
        calls.append((path, dict(params)))
        sel = [r for r in rows if int(r["fundingRateTimestamp"]) <= params["endTime"]]
        sel = sorted(sel, key=lambda r: -int(r["fundingRateTimestamp"]))
        return {"list": sel[: min(page, params["limit"])]}
    return fake


# ---------------------------------------------------------------- request construction


def test_kline_request_construction(monkeypatch):
    calls = []
    monkeypatch.setattr(backfill, "_get_json", _kline_server([], calls))
    end = _JAN1 + 24 * _H_MS - 1
    df = backfill._fetch_klines("https://x", "linear", "btcusdt", backfill.INTERVALS["1h"],
                                _JAN1, end, _NO_THROTTLE)
    assert df.empty
    path, params = calls[0]
    assert path == "/v5/market/kline"
    assert params == {"category": "linear", "symbol": "BTCUSDT", "interval": "60",
                      "start": _JAN1, "end": end, "limit": 1000}


def test_funding_request_construction_no_start_time(monkeypatch):
    # startTime alone errors on this endpoint: paging is endTime-only, floor client-side.
    calls = []
    monkeypatch.setattr(backfill, "_get_json", _funding_server([], calls))
    end = _JAN1 + 8 * _H_MS
    backfill._fetch_funding("https://x", "linear", "btcusdt", _JAN1, end, _NO_THROTTLE)
    path, params = calls[0]
    assert path == "/v5/market/funding/history"
    assert params == {"category": "linear", "symbol": "BTCUSDT", "endTime": end, "limit": 200}
    assert "startTime" not in params and "start" not in params


def test_interval_codes():
    assert backfill.INTERVALS == {"1m": "1", "5m": "5", "1h": "60", "1d": "D"}


def test_result_unwrap_raises_on_nonzero_retcode():
    assert backfill._result({"retCode": 0, "result": {"list": [1]}}, "/p") == {"list": [1]}
    with pytest.raises(RuntimeError, match="retCode=10006"):
        backfill._result({"retCode": 10006, "retMsg": "rate limit"}, "/p")


# ---------------------------------------------------------------- walk-back pagination


def test_kline_walkback_terminates_and_covers(monkeypatch):
    rows = _kline_rows(5)
    calls = []
    monkeypatch.setattr(backfill, "_get_json", _kline_server(rows, calls, page=2))
    df = backfill._fetch_klines("https://x", "linear", "btcusdt", "60",
                                _JAN1, _JAN1 + 4 * _H_MS, _NO_THROTTLE)
    assert len(calls) == 3                                  # 5 rows at 2/page
    assert sorted(df["start_time"]) == [r[0] for r in rows]  # nothing lost, no dupes
    # each page walks end back to just before the oldest row already received
    assert calls[1][1]["end"] == rows[3][0] - 1
    assert calls[2][1]["end"] == rows[1][0] - 1


def test_funding_walkback_terminates_and_covers(monkeypatch):
    rows = _funding_rows(5)
    calls = []
    monkeypatch.setattr(backfill, "_get_json", _funding_server(rows, calls, page=2))
    df = backfill._fetch_funding("https://x", "linear", "btcusdt",
                                 _JAN1, _JAN1 + 4 * 8 * _H_MS, _NO_THROTTLE)
    assert len(calls) == 3
    assert list(sorted(df["funding_time"])) == [int(r["fundingRateTimestamp"]) for r in rows]
    assert calls[1][1]["endTime"] == int(rows[3]["fundingRateTimestamp"]) - 1


def test_fetch_clips_to_window(monkeypatch):
    # server rows outside [start, end] (funding walk has no server-side floor) are clipped
    rows = _funding_rows(6)
    monkeypatch.setattr(backfill, "_get_json", _funding_server(rows, []))
    start = _JAN1 + 8 * _H_MS
    end = _JAN1 + 3 * 8 * _H_MS
    df = backfill._fetch_funding("https://x", "linear", "btcusdt", start, end, _NO_THROTTLE)
    assert df["funding_time"].min() >= start and df["funding_time"].max() <= end
    assert len(df) == 3


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


_CFG = {"asset_type": "perps", "rest_url": "https://x", "category": "linear"}


def test_backfill_small_empty_is_skip_not_error(monkeypatch, tmp_path):
    # delisted/pre-listing window returns retCode 0 with an empty list -> warn + skip
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(backfill, "_get_json", lambda *a: {"list": []})
    out = backfill._backfill_small(_CFG, ("bars", "1h", "abcusdt"),
                                   _JAN1, _JAN1 + _H_MS, _NO_THROTTLE, False)
    assert out["missing"] and out["rows"] == 0
    target = paths.data_path("raw", "perps", "bybit", "abcusdt", "bars", "1h") / "abcusdt.parquet"
    assert not target.exists()


def test_backfill_small_skips_existing_unless_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    out_dir = paths.data_path("raw", "perps", "bybit", "btcusdt", "funding_rate")
    lake_io.write_table(pd.DataFrame({"funding_time": [1], "funding_rate": ["0.9"]}),
                        out_dir / "btcusdt.parquet")

    def _boom(*a):
        raise AssertionError("must not fetch when the target exists and overwrite=False")

    monkeypatch.setattr(backfill, "_get_json", _boom)
    out = backfill._backfill_small(_CFG, ("funding_rate", None, "btcusdt"),
                                   _JAN1, _JAN1 + 8 * _H_MS, _NO_THROTTLE, False)
    assert out["skipped"]

    monkeypatch.setattr(backfill, "_get_json", _funding_server(_funding_rows(2), []))
    out = backfill._backfill_small(_CFG, ("funding_rate", None, "btcusdt"),
                                   _JAN1, _JAN1 + 8 * _H_MS, _NO_THROTTLE, True)
    assert out["rows"] == 2                       # overwrite refetched cleanly
    df = pd.read_parquet(out_dir / "btcusdt.parquet")
    assert list(df["funding_time"]) == [_JAN1, _JAN1 + 8 * _H_MS]   # old content replaced


def test_backfill_small_rerun_no_duplicates(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(backfill, "_get_json", _kline_server(_kline_rows(3), [], page=2))
    unit = ("bars", "1h", "btcusdt")
    for overwrite in (False, True, False):
        backfill._backfill_small(_CFG, unit, _JAN1, _JAN1 + 2 * _H_MS, _NO_THROTTLE, overwrite)
    df = pd.read_parquet(paths.data_path("raw", "perps", "bybit", "btcusdt", "bars", "1h")
                         / "btcusdt.parquet")
    assert list(df["start_time"]) == [_JAN1, _JAN1 + _H_MS, _JAN1 + 2 * _H_MS]  # ascending, deduped


def test_backfill_small_clamps_window_to_delisted_life(monkeypatch, tmp_path):
    # probed live: a Closed perp's klines return empty unless the window sits inside its
    # life — bars requests must clamp start/end to the (launch, delivery) bounds
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    calls = []
    launch, delivery = _JAN1 + 2 * _H_MS, _JAN1 + 5 * _H_MS
    monkeypatch.setattr(backfill, "_get_json", _kline_server(_kline_rows(4, start=launch), calls))
    out = backfill._backfill_small(_CFG, ("bars", "1h", "deadusdt"),
                                   _JAN1, _JAN1 + 100 * _H_MS, _NO_THROTTLE, False,
                                   life={"deadusdt": (launch, delivery)})
    assert out["rows"] == 4
    assert calls[0][1]["start"] == launch and calls[0][1]["end"] == delivery


def test_backfill_small_life_outside_window_skips_without_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")

    def _boom(*a):
        raise AssertionError("must not fetch when the clamped window is empty")

    monkeypatch.setattr(backfill, "_get_json", _boom)
    out = backfill._backfill_small(_CFG, ("bars", "1h", "deadusdt"),
                                   _JAN1, _JAN1 + _H_MS, _NO_THROTTLE, False,
                                   life={"deadusdt": (_JAN1 + 10 * _H_MS, _JAN1 + 20 * _H_MS)})
    assert out["missing"] and out["rows"] == 0


# ---------------------------------------------------------------- end-to-end roundtrip


def _conf():
    return {
        "max_workers": 1, "throttle_per_sec": 0, "rest_url": "https://x",
        "markets": {
            "perps": {"asset_type": "perps", "category": "linear",
                      "data_types": {"bars": {"timeframes": ["1h"], "storage": {"1h": "small"}},
                                     "funding_rate": {"storage": "small"}}},
            "spot": {"asset_type": "spot", "category": "spot",
                     "data_types": {"bars": {"timeframes": ["1h"], "storage": {"1h": "small"}}}},
        },
        "backfill": {"start": "2024-01-01", "symbols": ["btcusdt"]},
    }


def test_end_to_end_raw_normalize_query(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "PROCESSED_DIR", tmp_path / "processed")
    klines = _kline_server(_kline_rows(24), [])
    funding = _funding_server(_funding_rows(3, rates=["0.0001", "-0.0002", "0.0003"]), [])

    def fake(rest_url, path, params):
        return klines(rest_url, path, params) if path == "/v5/market/kline" \
            else funding(rest_url, path, params)

    monkeypatch.setattr(backfill, "_get_json", fake)
    backfill.run_backfill(_conf(), start="2024-01-01", end="2024-01-01")

    for asset_type in ("perps", "spot"):
        n = normalize.normalize("bybit", "btcusdt", "bars", timeframe="1h", asset_type=asset_type)
        assert n == 24
        got = query.get(asset_type, "bybit", "btcusdt", "bars", timeframe="1h", stage="processed")
        assert len(got) == 24
        assert got.index[0] == pd.Timestamp("2024-01-01 00:00:00")
        assert got.index[-1] == pd.Timestamp("2024-01-01 23:00:00")
        assert list(got.columns) == ["open", "high", "low", "close", "volume"]
        assert got["volume"].iloc[0] == 10.0

    assert normalize.normalize("bybit", "btcusdt", "funding_rate", asset_type="perps") == 3
    fr = query.get("perps", "bybit", "btcusdt", "funding_rate", stage="processed")
    assert list(fr["rate"]) == [0.0001, -0.0002, 0.0003]            # sign preserved
    assert fr["next_funding_ts"].iloc[0] == fr.index[1]             # ts.shift(-1)
    assert fr["next_funding_ts"].iloc[1] == fr.index[2]
    assert pd.isna(fr["next_funding_ts"].iloc[-1])

    # rerun without overwrite: idempotent, nothing duplicated
    backfill.run_backfill(_conf(), start="2024-01-01", end="2024-01-01")
    raw = pd.read_parquet(paths.data_path("raw", "perps", "bybit", "btcusdt", "bars", "1h")
                          / "btcusdt.parquet")
    assert len(raw) == 24
