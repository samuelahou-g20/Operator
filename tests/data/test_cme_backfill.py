import pandas as pd
import pytest

from src.common import paths
from src.data.fetching.cme import backfill


def _ohlcv_with_spread():
    ts = pd.to_datetime(["2024-03-01", "2024-03-01", "2024-03-01"], utc=True)
    return pd.DataFrame({
        "ts_event": ts,
        "open": [1.0, 2.0, 3.0], "high": [1.0, 2.0, 3.0], "low": [1.0, 2.0, 3.0],
        "close": [1.0, 2.0, 3.0], "volume": [1.0, 1.0, 1.0],
        "symbol": ["ESH4", "ESM4", "ESH4-ESM4"],
    })


def test_outrights_drops_spreads():
    out = backfill._outrights(_ohlcv_with_spread(), "symbol")
    assert set(out["symbol"]) == {"ESH4", "ESM4"}


def test_write_partitions_writes_outrights_only(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    rows = backfill._write_partitions(_ohlcv_with_spread(), "es", "bars", "1m")
    base = paths.data_path("raw", "futures", "cme", "es", "bars", "1m")
    contracts = sorted(p.name for p in base.glob("contract=*"))
    assert contracts == ["contract=ESH4", "contract=ESM4"]  # spread not written to raw
    assert rows == 2


# ----------------------------------------------------------------- fetch-span chunking

def test_per_tf_scalar_and_map():
    assert backfill._per_tf("ohlcv-1m", "1m") == "ohlcv-1m"            # scalar applies to all
    assert backfill._per_tf({"1m": "ohlcv-1m", "1d": "ohlcv-1d"}, "1d") == "ohlcv-1d"


def test_fetch_chunk_lookup_with_default():
    dt = {"bars": {"fetch_chunk": {"1m": "year", "1d": "all"}},
          "trades": {"fetch_chunk": "day"}, "contracts": {}}
    assert backfill._fetch_chunk(dt, "bars", "1m") == "year"
    assert backfill._fetch_chunk(dt, "bars", "1d") == "all"
    assert backfill._fetch_chunk(dt, "trades") == "day"
    assert backfill._fetch_chunk(dt, "contracts") == "day"             # default = back-compat


def test_spans_all_is_one_request():
    sp = backfill._spans(pd.Timestamp("2018-06-01"), pd.Timestamp("2020-03-01"), "all")
    assert sp == [("2018-06-01", "2020-03-02")]                         # end is exclusive (+1d)


def test_spans_year_partial_first_period():
    sp = backfill._spans(pd.Timestamp("2018-06-01"), pd.Timestamp("2020-03-01"), "year")
    assert sp == [("2018-06-01", "2019-01-01"),
                  ("2019-01-01", "2020-01-01"),
                  ("2020-01-01", "2020-03-02")]


def test_spans_year_aligned_start_no_stub():
    sp = backfill._spans(pd.Timestamp("2019-01-01"), pd.Timestamp("2020-12-31"), "year")
    assert sp == [("2019-01-01", "2020-01-01"), ("2020-01-01", "2021-01-01")]


def test_spans_month_and_day():
    months = backfill._spans(pd.Timestamp("2021-01-15"), pd.Timestamp("2021-03-02"), "month")
    assert months == [("2021-01-15", "2021-02-01"),
                      ("2021-02-01", "2021-03-01"),
                      ("2021-03-01", "2021-03-03")]
    days = backfill._spans(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-03"), "day")
    assert days == [("2021-01-01", "2021-01-02"),
                    ("2021-01-02", "2021-01-03"),
                    ("2021-01-03", "2021-01-04")]


@pytest.mark.parametrize("chunk", ["all", "year", "month", "day"])
def test_spans_cover_range_contiguously(chunk):
    sp = backfill._spans(pd.Timestamp("2018-06-01"), pd.Timestamp("2020-03-01"), chunk)
    assert sp[0][0] == "2018-06-01"
    assert sp[-1][1] == "2020-03-02"                                   # union == [start, end]
    for (_, a_end), (b_start, _) in zip(sp, sp[1:]):
        assert a_end == b_start                                        # no gap, no overlap


def test_spans_rejects_unknown_chunk():
    with pytest.raises(ValueError):
        backfill._spans(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-02-01"), "week")


class _FakeStore:
    def __init__(self, df):
        self._df = df

    def to_df(self):
        return self._df


class _FakeTimeseries:
    """Records get_range calls and returns one ts_event-indexed row per request."""

    def __init__(self, calls):
        self.calls = calls

    def get_range(self, dataset, symbols, schema, stype_in, start, end):
        self.calls.append((schema, start, end, tuple(symbols), stype_in))
        idx = pd.to_datetime([start], utc=True)
        idx.name = "ts_event"
        if schema == "definition":
            df = pd.DataFrame(
                {"raw_symbol": [symbols[0]], "expiration": pd.to_datetime([end], utc=True),
                 "instrument_class": ["F"]},
                index=idx,
            )
        else:
            df = pd.DataFrame(
                {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
                 "volume": [1.0], "symbol": ["ESH8"]},
                index=idx,
            )
        return _FakeStore(df)


class _FakeClient:
    def __init__(self, calls):
        self.timeseries = _FakeTimeseries(calls)


def _bars_only_config(chunk):
    return {
        "dataset": "GLBX.MDP3", "api_key_env": "X", "max_workers": 2, "throttle_per_sec": 0,
        "data_types": {"bars": {"timeframes": ["1m"], "schema": {"1m": "ohlcv-1m"},
                                "storage": {"1m": "big"}, "fetch_chunk": {"1m": chunk}}},
        "symbols": {"es": {"parent": "ES.FUT"}},
        "backfill": {"start": "2018-01-01", "symbols": ["es"]},
    }


def test_run_backfill_year_chunk_issues_one_request_per_span(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    calls: list = []
    monkeypatch.setattr(backfill, "_client", lambda config: _FakeClient(calls))
    backfill.run_backfill(_bars_only_config("year"), symbols=["es"],
                          start="2018-06-01", end="2020-03-01")
    bars_calls = [c for c in calls if c[0] == "ohlcv-1m"]
    expected = backfill._spans(pd.Timestamp("2018-06-01"), pd.Timestamp("2020-03-01"), "year")
    assert len(bars_calls) == len(expected) == 3            # 3 yearly spans, not ~640 days
    base = paths.data_path("raw", "futures", "cme", "es", "bars", "1m")
    assert list(base.glob("contract=*/date=*/part.parquet"))


def test_run_backfill_all_chunk_issues_single_request(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    calls: list = []
    monkeypatch.setattr(backfill, "_client", lambda config: _FakeClient(calls))
    backfill.run_backfill(_bars_only_config("all"), symbols=["es"],
                          start="2018-06-01", end="2020-03-01")
    assert len([c for c in calls if c[0] == "ohlcv-1m"]) == 1


def _seed_bar_partitions(symbol, contracts_dates):
    for contract, date in contracts_dates:
        part = (paths.data_path("raw", "futures", "cme", symbol, "bars", "1m")
                / f"contract={contract}" / f"date={date}")
        part.mkdir(parents=True)
        (part / "part.parquet").write_bytes(b"x")  # presence is all _bar_contract_ranges needs


_DEFS_CONFIG = {
    "dataset": "GLBX.MDP3", "symbols": {"es": {"parent": "ES.FUT"}},
    "data_types": {"bars": {"timeframes": ["1m"]}, "contracts": {"schema": "definition"}},
}


def test_backfill_definitions_targets_outrights_per_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    _seed_bar_partitions("es", [("ESH4", "2024-02-01"), ("ESM4", "2024-05-01")])
    calls: list = []
    backfill._backfill_definitions(_FakeClient(calls), _DEFS_CONFIG, "es", "2024-01-01", "2024-07-01")
    def_calls = [c for c in calls if c[0] == "definition"]
    assert len(def_calls) == 2                            # one request per contract, not the parent
    assert all(c[4] == "raw_symbol" for c in def_calls)   # never stype_in='parent' (no spread bloat)
    assert {c[3][0] for c in def_calls} == {"ESH4", "ESM4"}
    assert (paths.data_path("raw", "futures", "cme", "es", "contracts") / "es.parquet").exists()


def test_backfill_definitions_falls_back_to_parent_without_bars(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")  # no bar partitions seeded
    calls: list = []
    backfill._backfill_definitions(_FakeClient(calls), _DEFS_CONFIG, "es", "2024-01-01", "2024-07-01")
    def_calls = [c for c in calls if c[0] == "definition"]
    assert len(def_calls) == 1 and def_calls[0][4] == "parent"  # single parent request fallback


@pytest.mark.parametrize("chunk", ["all", "year", "month", "day"])
def test_spans_reversed_range_is_noop(chunk):
    # end < start must produce no spans for every mode (no malformed start>end request).
    assert backfill._spans(pd.Timestamp("2021-06-01"), pd.Timestamp("2020-01-01"), chunk) == []


def test_per_tf_missing_timeframe_raises_clear_error():
    with pytest.raises(ValueError, match="missing timeframe"):
        backfill._per_tf({"1m": "ohlcv-1m"}, "1h")


def test_fetch_daily_uses_span_api_and_per_timeframe_schema(monkeypatch, tmp_path):
    # Guards the cross-module dependency: fetch_daily must use the new _backfill_span API
    # and resolve each timeframe's schema (a per-tf map), not pass the whole schema dict.
    from src.data.fetching.cme import fetcher

    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    calls: list = []
    monkeypatch.setattr(backfill, "_client", lambda config: _FakeClient(calls))
    config = {
        "dataset": "GLBX.MDP3", "api_key_env": "X", "max_workers": 2, "throttle_per_sec": 0,
        "data_types": {
            "bars": {"timeframes": ["1d", "1h", "1m"],
                     "schema": {"1d": "ohlcv-1d", "1h": "ohlcv-1h", "1m": "ohlcv-1m"},
                     "storage": {"1d": "big", "1h": "big", "1m": "big"},
                     "fetch_chunk": {"1d": "all", "1h": "all", "1m": "year"}},
            "trades": {"schema": "trades", "storage": "big", "fetch_chunk": "day"},
        },
        "symbols": {"es": {"parent": "ES.FUT"}},
        "backfill": {"start": "2024-01-01", "symbols": ["es"]},
    }
    fetcher.fetch_daily(config, "2024-03-01")
    schemas_used = {c[0] for c in calls}                       # each is a string (not a dict)
    assert {"ohlcv-1d", "ohlcv-1h", "ohlcv-1m", "trades"} <= schemas_used
    for tf in ("1d", "1h", "1m"):
        base = paths.data_path("raw", "futures", "cme", "es", "bars", tf)
        assert list(base.glob("contract=*/date=*/part.parquet"))
