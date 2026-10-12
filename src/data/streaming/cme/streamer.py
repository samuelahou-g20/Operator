"""CME live streamer: Databento Live -> normalize -> Redis (ARCHITECTURE §6.2).

Subscribes to a source's Databento Live feed, resolves each record's instrument_id to
its contract via the SymbolMappingMsg stream, normalizes to the canonical schema
(+ `ts` int64 epoch-ns), and publishes one JSON message per event to
`live:cme:{contract}:{data_type}` (per-contract channels — the tradeable instrument;
the aggregator derives bars per contract). Outright contracts only. Handles reconnects.

NOTE: Databento Live needs a live entitlement (separate from historical); normalizers
+ dispatch are unit-tested offline, live-running is verified when entitlement exists.

MUST NOT: persist to the lake, build bars (aggregator.py), or contain strategy logic.
"""

from __future__ import annotations

import json
import time
from typing import Any

from src.common import config as cfg
from src.common import logging as common_logging
from src.data.streaming import get_redis, raw_channel

_log = common_logging.get_logger("streamer")

_SIDE = {"A": "sell", "B": "buy", "N": "none"}
# canonical data_type -> Databento Live schema
_SCHEMA_BY_DATA_TYPE = {"trades": "trades", "l1": "mbp-1"}


def _norm_trade(rec: Any) -> dict:
    return {"ts": int(rec.ts_event), "price": float(rec.pretty_price),
            "size": float(rec.size), "side": _SIDE.get(rec.side, "none")}


def _norm_l1(rec: Any) -> dict:
    top = rec.levels[0]
    return {"ts": int(rec.ts_event),
            "bid_px": float(rec.pretty_bid_px_00), "bid_sz": float(top.bid_sz),
            "ask_px": float(rec.pretty_ask_px_00), "ask_sz": float(top.ask_sz)}


# DBN record class name -> (canonical data_type, normalizer). Keyed by class name so the
# dispatch is unit-testable with lightweight fakes (DBN records are Rust-backed).
_NORMALIZERS = {"TradeMsg": ("trades", _norm_trade), "MBP1Msg": ("l1", _norm_l1)}


class CMEStreamer:
    SOURCE = "cme"

    def __init__(
        self,
        config: dict,
        symbols: list[str],
        data_types: list[str] | None = None,
        *,
        redis: Any | None = None,
        reconnect_delay: float = 5.0,
        continuous: bool = False,
    ) -> None:
        streaming = config.get("streaming", {})
        self._dataset = config["dataset"]
        self._key_env = config["api_key_env"]
        self._continuous = continuous
        roots = {s: config["symbols"][s]["parent"].split(".")[0] for s in symbols}  # es -> ES
        if continuous:
            # .{roll}.0 front-month continuous (matches the historical fetch's roll), published
            # under the parent lake symbol so the dashboard's `es` channel gets front-month live
            # trades. roll is per-symbol (`roll:` in cme.yaml, else `continuous_roll`, else c).
            default_roll = config.get("continuous_roll", "c")
            self._sub_symbols = [
                f"{root}.{config['symbols'][s].get('roll', default_roll)}.0"
                for s, root in roots.items()
            ]
            self._stype = "continuous"
            self._root_to_lake = {r.upper(): s for s, r in roots.items()}       # ES -> es
        else:
            self._sub_symbols = [config["symbols"][s]["parent"] for s in symbols]
            self._stype = "parent"
            self._root_to_lake = {}
        self._data_types = data_types or streaming.get("data_types", ["trades"])
        self._redis = redis if redis is not None else get_redis()
        self._reconnect_delay = reconnect_delay
        self._id_to_symbol: dict[int, str] = {}
        self._stop = False

    def handle(self, record: Any) -> tuple[str, dict] | None:
        """Process one DBN record: learn symbol mappings, or publish a normalized event."""
        kind = type(record).__name__
        if kind == "SymbolMappingMsg":
            if self._continuous:  # continuous input symbol -> parent lake symbol (GC.v.0 -> gc)
                lake = self._root_to_lake.get(record.stype_in_symbol.split(".")[0].upper())
                if lake is not None:
                    self._id_to_symbol[record.instrument_id] = lake
            else:
                self._id_to_symbol[record.instrument_id] = record.stype_out_symbol
            return None
        entry = _NORMALIZERS.get(kind)
        if entry is None:
            return None  # system/error/other records
        symbol = self._id_to_symbol.get(record.instrument_id)
        if symbol is None or (not self._continuous and "-" in symbol):
            return None  # unmapped yet, or a calendar spread (outrights only, per-contract mode)
        data_type, normalize = entry
        payload = normalize(record)
        channel = raw_channel(self.SOURCE, symbol.lower(), data_type)
        self._redis.publish(channel, json.dumps(payload))
        return channel, payload

    def run(self) -> None:
        import databento as db

        schemas = [_SCHEMA_BY_DATA_TYPE[dt] for dt in self._data_types]
        while not self._stop:
            client = None
            try:
                client = db.Live(key=cfg.require_env(self._key_env))
                for schema in schemas:
                    client.subscribe(dataset=self._dataset, schema=schema,
                                     stype_in=self._stype, symbols=self._sub_symbols)
                _log.info("databento live: %s %s (%s) schemas=%s",
                          self._dataset, self._sub_symbols, self._stype, schemas)
                for record in client:
                    if self._stop:
                        break
                    self.handle(record)
            except Exception as exc:  # noqa: BLE001 — reconnect on any live failure
                _log.warning("databento live error: %s; reconnecting in %ss",
                             exc, self._reconnect_delay)
                time.sleep(self._reconnect_delay)
            finally:
                if client is not None:
                    try:
                        client.stop()
                    except Exception:  # noqa: BLE001
                        pass

    def stop(self) -> None:
        self._stop = True
