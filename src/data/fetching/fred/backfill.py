"""FRED macro-series backfill via the official API (ARCHITECTURE §6.1).

`run_backfill` pulls each configured series (e.g. TB3MS — 3-month T-bill, the
risk-free rate) in ONE request per series: the observations endpoint returns a monthly
series' full history in a single page. Writes raw one-file-per-symbol in the series' native
units (TB3MS: percent per annum) — unit conversion is normalize's job. Missing observations
(value ".") are dropped at the fetch boundary. Requires a free API key in the env var named
by ``api_key_env`` (https://fred.stlouisfed.org/docs/api/api_key.html); note the fredgraph
CSV endpoint is NOT reachable from this network — only api.stlouisfed.org is.

MUST NOT: write outside raw/, convert units, run quality checks, or normalize to canonical.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

import pandas as pd

from src.common import config as cfg
from src.common import logging as common_logging
from src.common import paths
from src.data import lake_io

_log = common_logging.get_logger("backfill")

SOURCE = "fred"
ASSET_TYPE = "rates"

# Vendor column names chosen once here (FRED observation fields -> these names at fetch time).
VENDOR_COLUMNS: dict[str, list[str]] = {"rates": ["date", "value"]}


def _get_json(rest_url: str, params: dict) -> dict:
    url = f"{rest_url}/fred/series/observations?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "operator-fetcher"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 — fixed https host
        return json.load(resp)


def _fetch_series(rest_url: str, series_id: str, api_key: str) -> pd.DataFrame:
    payload = _get_json(rest_url, {
        "series_id": series_id, "api_key": api_key, "file_type": "json",
    })
    obs = payload.get("observations") or []
    rows = [(o["date"], float(o["value"])) for o in obs if o.get("value") not in (".", None)]
    return pd.DataFrame(rows, columns=VENDOR_COLUMNS["rates"])


def run_backfill(config: dict, symbols: list[str] | None = None, start: Any = None,
                 end: Any = None, overwrite: bool = False) -> None:
    """Fetch every configured series into raw/ (one file per symbol, idempotent).

    ``start``/``end`` clip the stored observations (dates are the vendor's, monthly for
    TB3MS). A series file is skipped when present unless ``overwrite`` — FRED revises
    history, so the daily fetcher refreshes with overwrite=True (sanctioned: the file is
    a full-series snapshot, not an append log).
    """
    api_key = cfg.require_env(config["api_key_env"])
    rest_url = config["rest_url"]
    for symbol in symbols or config["backfill"]["symbols"]:
        out_dir = paths.data_path("raw", ASSET_TYPE, SOURCE, symbol, "rates")
        target = out_dir / f"{symbol}.parquet"
        if target.exists() and not overwrite:
            _log.info("fred %s exists, skipping (use --overwrite to refresh)", symbol)
            continue
        df = _fetch_series(rest_url, symbol.upper(), api_key)
        if start is not None:
            df = df[pd.to_datetime(df["date"]) >= pd.Timestamp(start)]
        if end is not None:
            df = df[pd.to_datetime(df["date"]) <= pd.Timestamp(end)]
        if df.empty:
            _log.warning("no fred data for %s", symbol)
            continue
        rows = lake_io.write_table(df.drop_duplicates("date").sort_values("date"), target)
        _log.info("fred %s: %d observations (%s..%s)", symbol, rows,
                  df["date"].iloc[0], df["date"].iloc[-1])
