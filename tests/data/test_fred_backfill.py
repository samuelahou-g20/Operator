import numpy as np
import pandas as pd
import pytest

from src.common import paths
from src.data import lake_io, query
from src.data.etl.perps import normalize
from src.data.fetching.fred import backfill

_CFG = {"rest_url": "https://x", "api_key_env": "FRED_API_KEY",
        "backfill": {"symbols": ["tb3ms"]}}

_OBS = {"observations": [
    {"date": "2022-01-01", "value": "0.15"},
    {"date": "2022-02-01", "value": "."},          # FRED missing marker — must be dropped
    {"date": "2022-03-01", "value": "0.44"},
]}


def test_request_construction_and_missing_dropped(monkeypatch):
    calls = []

    def fake(rest_url, params):
        calls.append((rest_url, dict(params)))
        return _OBS

    monkeypatch.setattr(backfill, "_get_json", fake)
    df = backfill._fetch_series("https://x", "TB3MS", "k123")
    assert calls[0] == ("https://x", {"series_id": "TB3MS", "api_key": "k123",
                                      "file_type": "json"})
    assert list(df["date"]) == ["2022-01-01", "2022-03-01"]       # "." dropped
    assert list(df["value"]) == [0.15, 0.44]


def test_run_backfill_writes_and_skips_unless_overwrite(monkeypatch, tmp_path):
    monkeypatch.setenv("FRED_API_KEY", "k123")
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(backfill, "_get_json", lambda *a: _OBS)
    backfill.run_backfill(_CFG)
    target = paths.data_path("raw", "rates", "fred", "tb3ms", "rates") / "tb3ms.parquet"
    assert target.exists()
    df = pd.read_parquet(target)
    assert list(df["value"]) == [0.15, 0.44]                       # native percent units in raw

    def _boom(*a):
        raise AssertionError("must not fetch when the target exists and overwrite=False")

    monkeypatch.setattr(backfill, "_get_json", _boom)
    backfill.run_backfill(_CFG)                                    # skip path
    monkeypatch.setattr(backfill, "_get_json", lambda *a: _OBS)
    backfill.run_backfill(_CFG, overwrite=True)                    # refresh path
    assert pd.read_parquet(target).shape == df.shape


def test_missing_api_key_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    with pytest.raises(Exception):
        backfill.run_backfill(_CFG)


def test_normalize_percent_to_decimal_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "PROCESSED_DIR", tmp_path / "processed")
    out = paths.data_path("raw", "rates", "fred", "tb3ms", "rates")
    lake_io.write_table(pd.DataFrame({"date": ["2022-01-01", "2022-03-01"],
                                      "value": [0.15, 0.44]}), out / "tb3ms.parquet")
    rows = normalize.normalize("fred", "tb3ms", "rates", asset_type="rates")
    assert rows == 2
    got = query.get("rates", "fred", "tb3ms", "rates", stage="processed")
    assert np.allclose(got["value"].values, [0.0015, 0.0044])      # percent -> decimal
    assert got.index[0] == pd.Timestamp("2022-01-01")
