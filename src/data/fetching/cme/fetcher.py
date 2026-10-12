"""CME futures daily incremental fetch via Databento (ARCHITECTURE §6.1).

`fetch_daily(config, date)` backfills one day of every configured bar timeframe + trades
and refreshes the definitions, then runs the §12 quality checks per contract on that day's
bars (raw Databento ohlcv already has open/high/low/close/volume; only ts_event -> ts is
needed). The daily span is one day, so each (data_type, timeframe) is a single-day
`_backfill_span`; per-timeframe schema is resolved with `backfill._per_tf`.

NOTE: requires DATABENTO_API_KEY; not live-verified at build time.

MUST NOT: write outside raw/, normalize to processed (ETL does that), or skip the
quality checks (that distinguishes the daily fetch from a backfill).
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.common import config as cfg
from src.common import logging as common_logging
from src.data import quality
from src.data.etl.resample import pandas_freq
from src.data.fetching.cme import backfill

_log = common_logging.get_logger("daily_fetch")

_CONT_SCHEMA = {"1m": "ohlcv-1m", "1h": "ohlcv-1h", "1d": "ohlcv-1d"}


def fetch_recent_continuous(
    config: dict, symbol: str, timeframe: str, start: Any, end: Any
) -> pd.DataFrame:
    """Databento `.{roll}.0` continuous bars as a ts-indexed float OHLCV frame (no lake write).

    Same roll resolution as `scripts.fetch_cme_continuous` (per-symbol `roll:`, else
    `continuous_roll`, else c) so history, live, and this catch-up all share one series.
    Databento historical is current to ~the last complete hour, so this is what the dashboard
    uses to bridge a stale CME lake up to now on open.

    MUST NOT: write the lake (returns the frame for the caller to seed/persist).
    """
    import databento as db  # local import: only this path needs the SDK

    client = db.Historical(cfg.require_env(config["api_key_env"]))
    info = config["symbols"].get(symbol.lower(), {})
    root = info.get("parent", f"{symbol.upper()}.FUT").split(".")[0]
    roll = info.get("roll", config.get("continuous_roll", "c"))
    cont = f"{root}.{roll}.0"

    def _get(end_ts: Any) -> Any:
        return client.timeseries.get_range(
            dataset=config["dataset"], symbols=[cont], stype_in="continuous",
            schema=_CONT_SCHEMA[timeframe],
            start=pd.Timestamp(start).isoformat(), end=pd.Timestamp(end_ts).isoformat(),
        )

    try:
        data = _get(end)
    except Exception as exc:  # noqa: BLE001
        # Historical lags live by ~1h; if `end` overshoots the available range, retry at the
        # available end the API reports (live fills the final gap). Re-raise anything else.
        avail = re.search(r"available up to '([^']+)'", str(exc))
        if avail is None:
            raise
        data = _get(pd.Timestamp(avail.group(1)))
    df = data.to_df()
    if df.empty:
        return df
    out = pd.DataFrame({"ts": pd.DatetimeIndex(df.index).tz_convert("UTC").tz_localize(None)})
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = df[col].astype("float64").to_numpy()
    return out.dropna(subset=["close"]).drop_duplicates("ts").sort_values("ts").set_index("ts")


def fetch_daily(config: dict, date: Any) -> None:
    client = backfill._client(config)
    symbols = config["backfill"]["symbols"]
    day = pd.Timestamp(date).normalize().strftime("%Y-%m-%d")
    next_day = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    dt_cfg = config["data_types"]
    bars_cfg = dt_cfg.get("bars")
    timeframes = list(bars_cfg["timeframes"]) if bars_cfg else []

    for symbol in symbols:
        for timeframe in timeframes:
            schema = backfill._per_tf(bars_cfg["schema"], timeframe)
            backfill._backfill_span(client, config, (symbol, "bars", schema, timeframe, day, next_day))
        if "trades" in dt_cfg:
            backfill._backfill_span(
                client, config, (symbol, "trades", dt_cfg["trades"]["schema"], None, day, next_day)
            )
        if "contracts" in dt_cfg:
            backfill._backfill_definitions(client, config, symbol, day, next_day)

        # Quality on the day's bars, per timeframe, per contract (raw ohlcv is canonical-ish).
        parent = config["symbols"][symbol]["parent"]
        for timeframe in timeframes:
            schema = backfill._per_tf(bars_cfg["schema"], timeframe)
            bars = backfill._fetch_df(client, config, parent, schema, day, next_day)
            if bars.empty:
                continue
            bars = bars.assign(ts=pd.to_datetime(bars["ts_event"], utc=True).dt.tz_localize(None))
            for contract, group in bars.groupby("symbol"):
                quality.run_checks(group, f"cme:{symbol}:{contract}:bars:{timeframe}:{day}",
                                   freq=pandas_freq(timeframe))
        _log.info("cme daily fetch done: %s %s", symbol, day)
