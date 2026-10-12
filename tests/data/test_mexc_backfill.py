import pandas as pd
import pytest

from src.common import paths
from src.data import lake_io, query
from src.data.etl.perps import normalize
from src.data.fetching.chunking import RateLimiter
from src.data.fetching.mexc import backfill

_H_MS = 3_600_000
_JAN1 = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC in ms

_NO_THROTTLE = RateLimiter(0)


# ---------------------------------------------------------------- spot server (JSON array rows)


def _spot_rows(n, start=_JAN1, step=_H_MS):
    """Full ascending spot-kline set as MEXC /api/v3/klines returns it."""
    return [[start + i * step, "1.0", "2.0", "0.5", "1.5", "10.0", start + (i + 1) * step, "15.0"]
            for i in range(n)]


def _spot_server(rows, calls, page=1000):
    """Serve `rows` the way /api/v3/klines does: ascending rows within [startTime, endTime]."""
    def fake(rest_url, path, params):
        calls.append((path, dict(params)))
        sel = [r for r in rows if params["startTime"] <= r[0] <= params["endTime"]]
        sel = sorted(sel, key=lambda r: r[0])[: min(page, params["limit"])]
        return [[str(c) for c in r] for r in sel]
    return fake


# ---------------------------------------------------------------- perp server (columnar, seconds)


def _perp_rows(n, start=_JAN1, step=_H_MS):
    """openTime-in-ms rows for building the columnar seconds payload the contract host returns."""
    return [[start + i * step, "1.0", "2.0", "0.5", "1.5", "10.0"] for i in range(n)]


def _perp_server(rows, calls):
    """Serve `rows` the way /api/v1/contract/kline/{sym} does: columnar arrays in SECONDS."""
    def fake(rest_url, path, params):
        calls.append((path, dict(params)))
        sel = [r for r in rows if params["start"] <= r[0] // 1000 <= params["end"]]
        sel = sorted(sel, key=lambda r: r[0])
        return {"success": True, "code": 0, "data": {
            "time": [r[0] // 1000 for r in sel],
            "open": [r[1] for r in sel], "high": [r[2] for r in sel],
            "low": [r[3] for r in sel], "close": [r[4] for r in sel],
            "vol": [r[5] for r in sel],
        }}
    return fake


# ---------------------------------------------------------------- request construction


def test_spot_request_construction(monkeypatch):
    calls = []
    monkeypatch.setattr(backfill, "_get_json", _spot_server([], calls))
    end = _JAN1 + 24 * _H_MS - 1
    df = backfill._fetch_klines_spot("https://x", "btcusdt", backfill.SPOT_INTERVALS["1h"],
                                     _JAN1, end, _NO_THROTTLE)
    assert df.empty
    path, params = calls[0]
    assert path == "/api/v3/klines"
    assert params["symbol"] == "BTCUSDT" and params["interval"] == "60m" and params["limit"] == 1000
    assert params["startTime"] == _JAN1 and params["endTime"] == end   # bounded window, not lone endTime


def test_perp_request_construction_underscore_symbol_and_seconds(monkeypatch):
    calls = []
    monkeypatch.setattr(backfill, "_get_json", _perp_server([], calls))
    end = _JAN1 + 24 * _H_MS - 1
    backfill._fetch_klines_perp("https://x", "btcusdt", backfill.PERP_INTERVALS["1h"],
                                _JAN1, end, _NO_THROTTLE)
    path, params = calls[0]
    assert path == "/api/v1/contract/kline/BTC_USDT"          # lake btcusdt -> MEXC BTC_USDT
    assert params["interval"] == "Min60"
    assert params["end"] == end // 1000                       # seconds, not ms
    assert params["start"] == _JAN1 // 1000                   # window floored to start


def test_interval_codes():
    assert backfill.SPOT_INTERVALS["1h"] == "60m"
    assert backfill.PERP_INTERVALS["1h"] == "Min60"


def test_perp_symbol_mapping():
    assert backfill._to_perp_symbol("btcusdt") == "BTC_USDT"
    assert backfill._to_perp_symbol("shib1000usdt") == "SHIB1000_USDT"


# ---------------------------------------------------------------- walk-back pagination


def test_spot_pagination_terminates_and_covers(monkeypatch):
    rows = _spot_rows(5)
    calls = []
    monkeypatch.setattr(backfill, "_get_json", _spot_server(rows, calls, page=2))
    df = backfill._fetch_klines_spot("https://x", "btcusdt", "60m",
                                     _JAN1, _JAN1 + 4 * _H_MS, _NO_THROTTLE)
    assert len(calls) == 3                                    # 5 rows at 2/page
    assert sorted(df["start_time"]) == [r[0] for r in rows]   # nothing lost, no dupes
    starts = [c[1]["startTime"] for c in calls]
    assert starts == sorted(starts) and len(set(starts)) == 3   # forward walk, strictly advancing


def test_perp_walkback_terminates_and_covers(monkeypatch):
    rows = _perp_rows(5)
    calls = []
    monkeypatch.setattr(backfill, "_get_json", _perp_server(rows, calls))
    df = backfill._fetch_klines_perp("https://x", "btcusdt", "Min60",
                                     _JAN1, _JAN1 + 4 * _H_MS, _NO_THROTTLE)
    # one bounded window covers the whole range here; times normalized back to ms
    assert sorted(df["start_time"]) == [r[0] for r in rows]


def test_spot_fetch_clips_to_window(monkeypatch):
    rows = _spot_rows(6)
    monkeypatch.setattr(backfill, "_get_json", _spot_server(rows, []))
    start = _JAN1 + 1 * _H_MS
    end = _JAN1 + 3 * _H_MS
    df = backfill._fetch_klines_spot("https://x", "btcusdt", "60m", start, end, _NO_THROTTLE)
    assert df["start_time"].min() >= start and df["start_time"].max() <= end
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


_SPOT_CFG = {"asset_type": "spot", "rest_url": "https://x"}


def test_backfill_small_empty_is_skip_not_error(monkeypatch, tmp_path):
    # unlisted / pre-listing window returns an empty array -> warn + skip, no file written
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(backfill, "_get_json", lambda *a: [])
    out = backfill._backfill_small(_SPOT_CFG, ("bars", "1h", "abcusdt"),
                                   _JAN1, _JAN1 + _H_MS, _NO_THROTTLE, False)
    assert out["missing"] and out["rows"] == 0
    target = paths.data_path("raw", "spot", "mexc", "abcusdt", "bars", "1h") / "abcusdt.parquet"
    assert not target.exists()


def test_backfill_small_skips_existing_unless_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    out_dir = paths.data_path("raw", "spot", "mexc", "btcusdt", "bars", "1h")
    lake_io.write_table(pd.DataFrame({"start_time": [1], "open": ["1"], "high": ["1"],
                                      "low": ["1"], "close": ["1"], "volume": ["1"]}),
                        out_dir / "btcusdt.parquet")

    def _boom(*a):
        raise AssertionError("must not fetch when the target exists and overwrite=False")

    monkeypatch.setattr(backfill, "_get_json", _boom)
    out = backfill._backfill_small(_SPOT_CFG, ("bars", "1h", "btcusdt"),
                                   _JAN1, _JAN1 + _H_MS, _NO_THROTTLE, False)
    assert out["skipped"]

    monkeypatch.setattr(backfill, "_get_json", _spot_server(_spot_rows(2), []))
    out = backfill._backfill_small(_SPOT_CFG, ("bars", "1h", "btcusdt"),
                                   _JAN1, _JAN1 + _H_MS, _NO_THROTTLE, True)
    assert out["rows"] == 2                       # overwrite refetched cleanly
    df = pd.read_parquet(out_dir / "btcusdt.parquet")
    assert list(df["start_time"]) == [_JAN1, _JAN1 + _H_MS]   # old content replaced


def test_backfill_small_rerun_no_duplicates(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(backfill, "_get_json", _spot_server(_spot_rows(3), [], page=2))
    unit = ("bars", "1h", "btcusdt")
    for overwrite in (False, True, False):
        backfill._backfill_small(_SPOT_CFG, unit, _JAN1, _JAN1 + 2 * _H_MS, _NO_THROTTLE, overwrite)
    df = pd.read_parquet(paths.data_path("raw", "spot", "mexc", "btcusdt", "bars", "1h")
                         / "btcusdt.parquet")
    assert list(df["start_time"]) == [_JAN1, _JAN1 + _H_MS, _JAN1 + 2 * _H_MS]  # ascending, deduped


# ---------------------------------------------------------------- end-to-end roundtrip


def _conf():
    return {
        "max_workers": 1, "throttle_per_sec": 0,
        "markets": {
            "perps": {"asset_type": "perps", "rest_url": "https://contract.x",
                      "data_types": {"bars": {"timeframes": ["1h"], "storage": {"1h": "small"}}}},
            "spot": {"asset_type": "spot", "rest_url": "https://api.x",
                     "data_types": {"bars": {"timeframes": ["1h"], "storage": {"1h": "small"}}}},
        },
        "backfill": {"start": "2024-01-01", "symbols": ["btcusdt"]},
    }


def test_end_to_end_raw_normalize_query(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "PROCESSED_DIR", tmp_path / "processed")
    spot = _spot_server(_spot_rows(24), [])
    perp = _perp_server(_perp_rows(24), [])

    def fake(rest_url, path, params):
        return spot(rest_url, path, params) if path == "/api/v3/klines" \
            else perp(rest_url, path, params)

    monkeypatch.setattr(backfill, "_get_json", fake)
    backfill.run_backfill(_conf(), start="2024-01-01", end="2024-01-01")

    for asset_type in ("perps", "spot"):
        n = normalize.normalize("mexc", "btcusdt", "bars", timeframe="1h", asset_type=asset_type)
        assert n == 24
        got = query.get(asset_type, "mexc", "btcusdt", "bars", timeframe="1h", stage="processed")
        assert len(got) == 24
        assert got.index[0] == pd.Timestamp("2024-01-01 00:00:00")
        assert got.index[-1] == pd.Timestamp("2024-01-01 23:00:00")
        assert list(got.columns) == ["open", "high", "low", "close", "volume"]
        assert got["volume"].iloc[0] == 10.0

    # rerun without overwrite: idempotent, nothing duplicated
    backfill.run_backfill(_conf(), start="2024-01-01", end="2024-01-01")
    raw = pd.read_parquet(paths.data_path("raw", "spot", "mexc", "btcusdt", "bars", "1h")
                          / "btcusdt.parquet")
    assert len(raw) == 24
