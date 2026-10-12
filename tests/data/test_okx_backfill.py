import pandas as pd
import pytest

from src.common import paths
from src.data import lake_io, query
from src.data.etl.perps import normalize
from src.data.fetching.chunking import RateLimiter
from src.data.fetching.okx import backfill

_H_MS = 3_600_000
_JAN1 = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC in ms

_NO_THROTTLE = RateLimiter(0)


def _candle_rows(n, start=_JAN1, step=_H_MS, confirm="1"):
    """Ascending [ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm] rows (ints for asserts)."""
    return [[start + i * step, "1.0", "2.0", "0.5", "1.5", "5.0", "0.05", "210.0", confirm]
            for i in range(n)]


def _candle_server(rows, calls, page=300):
    """Serve `rows` the way v5 candles do: newest-first, `after` = strictly older."""
    def fake(rest_url, path, params):
        calls.append((path, dict(params)))
        sel = [r for r in rows if r[0] < params["after"]]
        sel = sorted(sel, key=lambda r: -r[0])[: min(page, params["limit"])]
        return [[str(c) for c in r] for r in sel]
    return fake


def _funding_rows(n, start=_JAN1, step=8 * _H_MS, rates=None):
    rates = rates or ["0.0001"] * n
    return [{"instId": "BTC-USDT-SWAP", "fundingRate": rates[i],
             "fundingTime": str(start + i * step)} for i in range(n)]


def _funding_server(rows, calls, page=400, inclusive_after=False):
    """Serve funding history: newest-first, `after` pages backwards (strictly older
    by default; `inclusive_after` simulates a boundary-inclusive vendor response)."""
    def fake(rest_url, path, params):
        calls.append((path, dict(params)))
        if inclusive_after:
            sel = [r for r in rows if int(r["fundingTime"]) <= params["after"]]
        else:
            sel = [r for r in rows if int(r["fundingTime"]) < params["after"]]
        sel = sorted(sel, key=lambda r: -int(r["fundingTime"]))
        return sel[: min(page, params["limit"])]
    return fake


# ---------------------------------------------------------------- instId + envelope


def test_inst_id_derivation():
    assert backfill._inst_id("btcusdt", "SWAP") == "BTC-USDT-SWAP"
    assert backfill._inst_id("btcusdt", "SPOT") == "BTC-USDT"
    assert backfill._inst_id("dogeusdt", "SWAP") == "DOGE-USDT-SWAP"
    with pytest.raises(ValueError):
        backfill._inst_id("btcbusd", "SWAP")


def test_data_unwrap():
    assert backfill._data({"code": "0", "data": [1, 2]}, "/p") == [1, 2]
    assert backfill._data({"code": "51001", "msg": "Instrument ID does not exist"}, "/p") == []
    with pytest.raises(RuntimeError, match="code=50011"):
        backfill._data({"code": "50011", "msg": "rate limit"}, "/p")


def test_ct_vals_map(monkeypatch):
    monkeypatch.setattr(backfill, "_get_json", lambda *a: [
        {"instId": "BTC-USDT-SWAP", "ctVal": "0.01", "ctMult": "1"},
        {"instId": "DOGE-USDT-SWAP", "ctVal": "1000", "ctMult": "1"},
    ])
    ct = backfill._ct_vals("https://x", _NO_THROTTLE)
    assert ct == {"BTC-USDT-SWAP": 0.01, "DOGE-USDT-SWAP": 1000.0}


# ---------------------------------------------------------------- request construction


def test_candle_request_construction(monkeypatch):
    calls = []
    monkeypatch.setattr(backfill, "_get_json", _candle_server([], calls))
    end = _JAN1 + 24 * _H_MS - 1
    df = backfill._fetch_candles("https://x", "/api/v5/market/history-candles", "BTC-USDT-SWAP",
                                 backfill.BARS["1h"], _JAN1, end, 0.01, _NO_THROTTLE)
    assert df.empty
    path, params = calls[0]
    assert path == "/api/v5/market/history-candles"
    assert params == {"instId": "BTC-USDT-SWAP", "bar": "1H", "after": end + 1, "limit": 300}


def test_funding_request_construction(monkeypatch):
    calls = []
    monkeypatch.setattr(backfill, "_get_json", _funding_server([], calls))
    end = _JAN1 + 8 * _H_MS
    backfill._fetch_funding("https://x", "BTC-USDT-SWAP", _JAN1, end, _NO_THROTTLE)
    path, params = calls[0]
    assert path == "/api/v5/public/funding-rate-history"
    assert params == {"instId": "BTC-USDT-SWAP", "after": end + 1, "limit": 400}


def test_bar_codes():
    assert backfill.BARS["1h"] == "1H"
    assert backfill.BARS["1m"] == "1m"


# ---------------------------------------------------------------- walk-back pagination


def test_candle_walkback_terminates_and_covers(monkeypatch):
    rows = _candle_rows(5)
    calls = []
    monkeypatch.setattr(backfill, "_get_json", _candle_server(rows, calls, page=2))
    df = backfill._fetch_candles("https://x", "/api/v5/market/history-candles", "BTC-USDT-SWAP",
                                 "1H", _JAN1, _JAN1 + 4 * _H_MS, 0.01, _NO_THROTTLE)
    assert len(calls) == 3                                   # 5 rows at 2/page
    assert sorted(df["start_time"]) == [r[0] for r in rows]  # nothing lost, no dupes
    assert calls[1][1]["after"] == rows[3][0]                # walks `after` back to the oldest seen


def test_candles_drop_in_progress_bar(monkeypatch):
    rows = _candle_rows(3) + _candle_rows(1, start=_JAN1 + 3 * _H_MS, confirm="0")
    monkeypatch.setattr(backfill, "_get_json", _candle_server(rows, []))
    df = backfill._fetch_candles("https://x", "/api/v5/market/candles", "BTC-USDT-SWAP",
                                 "1H", _JAN1, _JAN1 + 4 * _H_MS, 0.01, _NO_THROTTLE)
    assert len(df) == 3                                      # confirm=="0" dropped at fetch time
    assert _JAN1 + 3 * _H_MS not in set(df["start_time"])
    assert (df["ct_val"] == 0.01).all()                      # raw is self-contained


def test_funding_inclusive_after_still_terminates(monkeypatch):
    # if the vendor pages inclusively on `after`, the no-progress guard must stop the walk
    rows = _funding_rows(3)
    calls = []
    monkeypatch.setattr(backfill, "_get_json",
                        _funding_server(rows, calls, page=2, inclusive_after=True))
    df = backfill._fetch_funding("https://x", "BTC-USDT-SWAP",
                                 _JAN1 - 24 * _H_MS, _JAN1 + 2 * 8 * _H_MS, _NO_THROTTLE)
    assert len(calls) <= 4                                   # bounded, no infinite loop
    assert sorted(set(df["funding_time"])) == [int(r["fundingTime"]) for r in rows]


def test_funding_retention_cap_empty_is_skip(monkeypatch, tmp_path):
    # a window older than ~3 months returns code "0" with empty data -> warn + skip
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(backfill, "_get_json", lambda *a: [])
    cfg = {"asset_type": "perps", "rest_url": "https://x", "inst_type": "SWAP"}
    out = backfill._backfill_small(cfg, ("funding_rate", None, "btcusdt"),
                                   _JAN1, _JAN1 + 8 * _H_MS, {}, _NO_THROTTLE, False)
    assert out["missing"] and out["rows"] == 0
    target = paths.data_path("raw", "perps", "okx", "btcusdt", "funding_rate") / "btcusdt.parquet"
    assert not target.exists()


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


def test_backfill_small_skips_existing_unless_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    cfg = {"asset_type": "spot", "rest_url": "https://x", "inst_type": "SPOT"}
    out_dir = paths.data_path("raw", "spot", "okx", "btcusdt", "bars", "1h")
    lake_io.write_table(pd.DataFrame({"start_time": [1]}), out_dir / "btcusdt.parquet")

    def _boom(*a):
        raise AssertionError("must not fetch when the target exists and overwrite=False")

    monkeypatch.setattr(backfill, "_get_json", _boom)
    out = backfill._backfill_small(cfg, ("bars", "1h", "btcusdt"),
                                   _JAN1, _JAN1 + _H_MS, {}, _NO_THROTTLE, False)
    assert out["skipped"]

    monkeypatch.setattr(backfill, "_get_json", _candle_server(_candle_rows(2), []))
    out = backfill._backfill_small(cfg, ("bars", "1h", "btcusdt"),
                                   _JAN1, _JAN1 + _H_MS, {}, _NO_THROTTLE, True)
    assert out["rows"] == 2
    df = pd.read_parquet(out_dir / "btcusdt.parquet")
    assert list(df["start_time"]) == [_JAN1, _JAN1 + _H_MS]   # ascending, replaced cleanly


def test_backfill_small_missing_ct_val_skips(monkeypatch, tmp_path):
    # a swap absent from instruments (delisted, not enumerable) is skipped pre-fetch
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")

    def _boom(*a):
        raise AssertionError("must not hit candles without a ct_val")

    monkeypatch.setattr(backfill, "_get_json", _boom)
    cfg = {"asset_type": "perps", "rest_url": "https://x", "inst_type": "SWAP"}
    out = backfill._backfill_small(cfg, ("bars", "1h", "goneusdt"),
                                   _JAN1, _JAN1 + _H_MS, {}, _NO_THROTTLE, False)
    assert out["missing"]


# ---------------------------------------------------------------- end-to-end roundtrip


def _conf():
    return {
        "max_workers": 1, "throttle_per_sec": 0, "rest_url": "https://x",
        "markets": {
            "perps": {"asset_type": "perps", "inst_type": "SWAP",
                      "data_types": {"bars": {"timeframes": ["1h"], "storage": {"1h": "small"}},
                                     "funding_rate": {"storage": "small"}}},
            "spot": {"asset_type": "spot", "inst_type": "SPOT",
                     "data_types": {"bars": {"timeframes": ["1h"], "storage": {"1h": "small"}}}},
        },
        "backfill": {"start": "2024-01-01", "symbols": ["btcusdt"]},
    }


def test_end_to_end_raw_normalize_query(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "PROCESSED_DIR", tmp_path / "processed")
    candles = _candle_server(_candle_rows(24), [])
    funding = _funding_server(_funding_rows(3, rates=["0.0001", "-0.0002", "0.0003"]), [])

    def fake(rest_url, path, params):
        if path == "/api/v5/public/instruments":
            return [{"instId": "BTC-USDT-SWAP", "ctVal": "0.01", "ctMult": "1"}]
        if path == "/api/v5/market/history-candles":
            return candles(rest_url, path, params)
        assert path == "/api/v5/public/funding-rate-history"
        return funding(rest_url, path, params)

    monkeypatch.setattr(backfill, "_get_json", fake)
    backfill.run_backfill(_conf(), start="2024-01-01", end="2024-01-01")

    # swap bars: volume must be vol x ctVal x ctMult = 5.0 x 0.01
    assert normalize.normalize("okx", "btcusdt", "bars", timeframe="1h", asset_type="perps") == 24
    perp = query.get("perps", "okx", "btcusdt", "bars", timeframe="1h", stage="processed")
    assert len(perp) == 24
    assert perp.index[0] == pd.Timestamp("2024-01-01 00:00:00")
    assert perp["volume"].iloc[0] == pytest.approx(0.05)

    # spot bars: ct_val is 1.0, volume already base units
    assert normalize.normalize("okx", "btcusdt", "bars", timeframe="1h", asset_type="spot") == 24
    spot = query.get("spot", "okx", "btcusdt", "bars", timeframe="1h", stage="processed")
    assert spot["volume"].iloc[0] == pytest.approx(5.0)

    assert normalize.normalize("okx", "btcusdt", "funding_rate", asset_type="perps") == 3
    fr = query.get("perps", "okx", "btcusdt", "funding_rate", stage="processed")
    assert list(fr["rate"]) == [0.0001, -0.0002, 0.0003]            # sign preserved
    assert fr["next_funding_ts"].iloc[0] == fr.index[1]             # ts.shift(-1)
    assert pd.isna(fr["next_funding_ts"].iloc[-1])

    # rerun without overwrite: idempotent, nothing duplicated
    backfill.run_backfill(_conf(), start="2024-01-01", end="2024-01-01")
    raw = pd.read_parquet(paths.data_path("raw", "perps", "okx", "btcusdt", "bars", "1h")
                          / "btcusdt.parquet")
    assert len(raw) == 24
