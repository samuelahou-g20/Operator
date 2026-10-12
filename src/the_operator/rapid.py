"""Client for the rapid-engine quoting boxes: which LM2 strategies are trading, and their params.

The boxes are the deployments that actually quote — one process per box, many
``LM2Strategy:{flow}:{ccypair}:{exchange}`` calculators inside each. They serve three routes we
use, all unauthenticated:

    GET  /rest/status/full                  live state, incl. every strategy's status + debug_info
    GET  /rest/calculators/params           the full parameter set per calculator
    GET  /rest/calculators/overrides        what has been overridden away from the deployed config

The box registry (``config/zp/strategy_boxes.yaml``, generated from AWS SSM by ``scripts.zp_boxes``)
is keyed by PROGRAM (mm1..mm5, perp), and a program's boxes are the only ones a desk's request may
touch — a coin is quoted by one program, which is not always the desk that asked, so a caller that
wants another program's boxes must say so explicitly.

``sources.fetch_strategy_params`` already reads ``/rest/calculators/params`` for the ZP; this module
is the live-operations view (status, overrides) and deliberately does not duplicate that read.

``push_param`` is the one write: ``POST /rest/calculators/overrides/modify`` replaces a whole
attribute group on a live strategy. ONLY ``execution`` may call it — a parameter push is a money
action and reaches the box after a human confirm, never from a tool handler.

MUST NOT: hold desk policy, call Telegram, or let anything but ``execution`` reach ``push_param``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

from src.common.config import load_config
from src.common.logging import get_logger

_log = get_logger("operator")

BOXES_CONFIG = "zp/strategy_boxes.yaml"
TIMEOUT_S = 10          # a slow box must not hold a chat reply; it becomes a warning instead
MAX_WORKERS = 10
STALE_S = 300           # status is a heartbeat every ~30s; five minutes silent is itself a finding
TRADING = "TRADING"


@dataclass(frozen=True)
class Strategy:
    """One LM2 calculator's live state, flattened out of a box's ``/rest/status/full``."""

    program: str
    box: str
    name: str            # LM2Strategy:REPOL:POL-EUR:BITVAVO
    flow: str
    ccypair: str
    exchange: str
    status: str          # TRADING | STOP (ERROR) | STOP (DISABLED) | ...
    trading: bool | None
    resetting: bool | None
    reported: datetime | None    # the status HEARTBEAT (~30s), NOT when this state was entered:
    last_error: str              # the box restamps it continuously, so it dates the read, not the stop

    @property
    def token(self) -> str:
        return token_of(self.ccypair)

    @property
    def healthy(self) -> bool:
        return self.status == TRADING and self.trading is not False

    @property
    def errored(self) -> bool:
        """Stopped by a failure, as opposed to DISABLED, which is somebody's deliberate off."""
        return "ERROR" in self.status.upper()

    def stale(self, now: datetime | None = None) -> bool:
        if self.reported is None:
            return True
        return ((now or datetime.now(timezone.utc)) - self.reported).total_seconds() > STALE_S


def token_of(ccypair: str) -> str:
    """``NEAR-USDT`` / perp ``P.NEAR-USDT`` -> ``NEAR``."""
    p = str(ccypair).upper()
    return (p[2:] if p.startswith("P.") else p).split("-", 1)[0]


def calculator_token(calculator: str) -> str:
    """The base token out of a ``LM2Strategy:{flow}:{ccypair}:{exchange}`` name ("" if malformed)."""
    parts = str(calculator).split(":")
    return token_of(parts[2]) if len(parts) > 2 else ""


def boxes(program: str) -> dict[str, str]:
    """``{box_name: url}`` for one program, empty when the program is unknown."""
    return dict(load_config(BOXES_CONFIG).get(program) or {})


def _get(url: str, path: str, timeout: float = TIMEOUT_S):
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=timeout) as r:
        return json.load(r)


def _micros(v) -> datetime | None:
    try:
        return datetime.fromtimestamp(int(v) / 1e6, tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _strategies_from_status(payload: dict, program: str, box: str) -> list[Strategy]:
    out = []
    for entry in (payload or {}).get("get_strategy_status") or []:
        name = str(entry.get("strategy_name") or "")
        parts = name.split(":")
        if parts[0] != "LM2Strategy" or len(parts) < 4:
            continue
        try:
            details = json.loads(entry.get("latest_status_details") or "{}")
        except (TypeError, ValueError):
            details = {}
        debug = details.get("debug_info") or {}
        out.append(Strategy(
            program=program, box=box, name=name, flow=parts[1], ccypair=parts[2], exchange=parts[3],
            status=str(entry.get("latest_status") or "?"),
            trading=debug.get("trading"), resetting=debug.get("resetting"),
            reported=_micros(entry.get("latest_status_time_micros")),
            last_error=str(entry.get("last_error_status") or ""),
        ))
    return out


def scan(program: str, warnings: list[str]) -> list[Strategy]:
    """Every LM2 strategy across one program's boxes. A box that errors is a warning, not a
    failure — a partial answer about a quoting book beats no answer."""
    reg = boxes(program)
    if not reg:
        warnings.append(f"no quoting boxes registered for program {program!r}.")
        return []
    found: list[Strategy] = []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(reg))) as pool:
        futures = {pool.submit(_get, url, "/rest/status/full"): name for name, url in reg.items()}
        for future, name in futures.items():
            try:
                found += _strategies_from_status(future.result(), program, name)
            except Exception as e:  # noqa: BLE001 - one unreachable box must not sink the scan
                warnings.append(f"box {program}/{name} unreachable: {e}")
    # Worst first: errors, then deliberate stops, then the book that is working.
    return sorted(found, key=lambda s: (s.healthy, not s.errored, s.exchange, s.name))


def overrides(program: str, warnings: list[str]) -> dict[str, dict]:
    """``{calculator: {param: value}}`` currently overridden across a program's boxes.

    An override is a hand-set value sitting on top of the deployed config, so it is the first thing
    to check when a strategy behaves unlike its configuration says it should."""
    reg = boxes(program)
    out: dict[str, dict] = {}
    if not reg:
        warnings.append(f"no quoting boxes registered for program {program!r}.")
        return out
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(reg))) as pool:
        futures = {pool.submit(_get, url, "/rest/calculators/overrides"): name
                   for name, url in reg.items()}
        for future, name in futures.items():
            try:
                for calc, params in (future.result() or {}).items():
                    if params:
                        out[calc] = params
            except Exception as e:  # noqa: BLE001
                warnings.append(f"box {program}/{name} overrides unreachable: {e}")
    return out


@dataclass(frozen=True)
class Boundary:
    """One strategy's price band and where the price actually is relative to it.

    ``price_boundaries`` with ``stop_both_sides`` is what stops a strategy: when the price leaves
    the band the strategy goes STOP (DISABLED), which is why re-bounding is the desk's daily chore.
    """

    box: str
    url: str
    calculator: str
    exchange: str
    ccypair: str
    status: str
    mid: float | None
    band: dict              # the WHOLE price_boundaries group, as the box serves it

    @property
    def price_min(self) -> float | None:
        return self.band.get("price_min")

    @property
    def price_max(self) -> float | None:
        return self.band.get("price_max")

    @property
    def enabled(self) -> bool:
        return bool(self.band.get("enabled"))

    @property
    def outside(self) -> bool | None:
        """True when the price has left the band — unknown (None) without a mid or a band."""
        if self.mid is None or self.price_min is None or self.price_max is None:
            return None
        return not (self.price_min <= self.mid <= self.price_max)

    @property
    def half_width_pct(self) -> float | None:
        """The band's half-width as a percentage of its centre (the desk runs ~2.5%)."""
        lo, hi = self.price_min, self.price_max
        if lo is None or hi is None or lo <= 0 or hi <= lo:
            return None
        centre = (lo + hi) / 2
        return (hi - centre) / centre * 100


def _box_boundaries(url: str, box: str, token: str) -> list[Boundary]:
    params = _get(url, "/rest/calculators/params")
    mids: dict[str, float] = {}
    status: dict[str, str] = {}
    for entry in (_get(url, "/rest/status/full") or {}).get("get_strategy_status") or []:
        name = str(entry.get("strategy_name") or "")
        status[name] = str(entry.get("latest_status") or "?")
        try:
            debug = (json.loads(entry.get("latest_status_details") or "{}").get("debug_info") or {})
        except (TypeError, ValueError):
            continue
        if isinstance(debug.get("implied_mid"), (int, float)):
            mids[name] = float(debug["implied_mid"])
    out = []
    for name, node in (params or {}).items():
        parts = name.split(":")
        if parts[0] != "LM2Strategy" or len(parts) < 4 or calculator_token(name) != token:
            continue
        band = (node or {}).get("price_boundaries")
        if not isinstance(band, dict):
            continue
        out.append(Boundary(box=box, url=url, calculator=name, exchange=parts[3], ccypair=parts[2],
                            status=status.get(name, "?"), mid=mids.get(name), band=dict(band)))
    return out


def boundaries(program: str, token: str, warnings: list[str]) -> list[Boundary]:
    """Every ``token`` strategy's price band across one program's boxes, with its current mid."""
    reg = boxes(program)
    if not reg:
        warnings.append(f"no quoting boxes registered for program {program!r}.")
        return []
    found: list[Boundary] = []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(reg))) as pool:
        futures = {pool.submit(_box_boundaries, url, name, token.upper()): name
                   for name, url in reg.items()}
        for future, name in futures.items():
            try:
                found += future.result()
            except Exception as e:  # noqa: BLE001 - one unreachable box is a warning, not a failure
                warnings.append(f"box {program}/{name} unreachable: {e}")
    return sorted(found, key=lambda b: (b.exchange, b.ccypair))


def live_band(url: str, calculator: str) -> tuple[dict, float | None] | None:
    """One strategy's CURRENT ``price_boundaries`` group and its current mid, read fresh.

    The push replaces the whole group, so the confirmed min/max are merged into exactly what the
    box is running right now — never into a copy taken at preview time."""
    params = _get(url, "/rest/calculators/params")
    band = ((params or {}).get(calculator) or {}).get("price_boundaries")
    if not isinstance(band, dict):
        return None
    mid = None
    for entry in (_get(url, "/rest/status/full") or {}).get("get_strategy_status") or []:
        if str(entry.get("strategy_name")) != calculator:
            continue
        try:
            debug = json.loads(entry.get("latest_status_details") or "{}").get("debug_info") or {}
        except (TypeError, ValueError):
            break
        if isinstance(debug.get("implied_mid"), (int, float)):
            mid = float(debug["implied_mid"])
        break
    return dict(band), mid


def strategy_params(program: str, token: str, warnings: list[str]) -> list[dict]:
    """Every ``token`` strategy's full parameter set across one program's boxes.

    ``{box, url, calculator, exchange, ccypair, params}`` per strategy — the caller picks the group
    and field it cares about, so nothing here assumes which lever is being pulled."""
    reg = boxes(program)
    if not reg:
        warnings.append(f"no quoting boxes registered for program {program!r}.")
        return []

    def one(name: str, url: str) -> list[dict]:
        out = []
        for calc, node in (_get(url, "/rest/calculators/params") or {}).items():
            parts = calc.split(":")
            if parts[0] != "LM2Strategy" or len(parts) < 4 or calculator_token(calc) != token:
                continue
            out.append({"box": name, "url": url, "calculator": calc, "exchange": parts[3],
                        "ccypair": parts[2], "params": node or {}})
        return out

    found: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(reg))) as pool:
        futures = {pool.submit(one, name, url): name for name, url in reg.items()}
        for future, name in futures.items():
            try:
                found += future.result()
            except Exception as e:  # noqa: BLE001 - one unreachable box is a warning
                warnings.append(f"box {program}/{name} unreachable: {e}")
    return sorted(found, key=lambda r: (r["exchange"], r["ccypair"]))


def live_group(url: str, calculator: str, attribute: str) -> dict | None:
    """One strategy's CURRENT parameter group, read fresh (the push replaces the whole group)."""
    group = ((_get(url, "/rest/calculators/params") or {}).get(calculator) or {}).get(attribute)
    return dict(group) if isinstance(group, dict) else None


def token_view(program: str, token: str, warnings: list[str]) -> list[dict]:
    """Everything one token's strategies expose, from BOTH box endpoints in one pass.

    Per strategy: ``params`` (its full config), ``status``/``mid``/``market_bid``/``market_ask``
    (what it currently sees) and ``our_quote_24h`` (its own executed notional over 24h, in the pair's QUOTE currency, from the
    box's ``recent_volume_summary``). Built together because answering "why is our share low" needs
    the configuration and the live book side by side, and two separate scans of 20 boxes is slow
    and can straddle a change."""
    reg = boxes(program)
    if not reg:
        warnings.append(f"no quoting boxes registered for program {program!r}.")
        return []
    token = token.upper()

    def one(name: str, url: str) -> list[dict]:
        params = _get(url, "/rest/calculators/params")
        status_full = _get(url, "/rest/status/full") or {}
        live: dict[str, dict] = {}
        for entry in status_full.get("get_strategy_status") or []:
            try:
                debug = json.loads(entry.get("latest_status_details") or "{}").get("debug_info") or {}
            except (TypeError, ValueError):
                debug = {}
            book = debug.get("ob_bidask") or [None, None]
            live[str(entry.get("strategy_name"))] = {
                "status": str(entry.get("latest_status") or "?"),
                "mid": debug.get("implied_mid"),
                "market_bid": book[0] if len(book) > 1 else None,
                "market_ask": book[1] if len(book) > 1 else None,
            }
        # 24h executed notional per (exchange, ccypair), both sides summed.
        vol: dict[tuple, float] = {}
        for row in (status_full.get("recent_volume_summary") or {}).get("volumes_24h") or []:
            key = (str(row.get("exchange")), str(row.get("ccypair")))
            vol[key] = vol.get(key, 0.0) + abs(float(row.get("executed_cost") or 0.0))

        out = []
        for calc, node in (params or {}).items():
            parts = calc.split(":")
            if parts[0] != "LM2Strategy" or len(parts) < 4 or calculator_token(calc) != token:
                continue
            row = {"box": name, "url": url, "calculator": calc, "exchange": parts[3],
                   "ccypair": parts[2], "params": node or {},
                   "our_quote_24h": vol.get((parts[3], parts[2]), 0.0)}
            row.update(live.get(calc) or {"status": "?", "mid": None, "market_bid": None,
                                          "market_ask": None})
            out.append(row)
        return out

    found: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(reg))) as pool:
        futures = {pool.submit(one, name, url): name for name, url in reg.items()}
        for future, name in futures.items():
            try:
                found += future.result()
            except Exception as e:  # noqa: BLE001 - one unreachable box is a warning
                warnings.append(f"box {program}/{name} unreachable: {e}")
    return sorted(found, key=lambda r: (r["exchange"], r["ccypair"]))


def push_param(url: str, calculator: str, attribute: str, value: dict, timeout: float = 30) -> dict:
    """``POST /rest/calculators/overrides/modify`` — replace one attribute group on a live strategy.

    The box takes the WHOLE group, so ``value`` must be the complete dict: a partial one drops the
    fields it omits. Callers read the current group and merge into it. Only ``execution`` may call
    this, and only for a proposal a human has confirmed."""
    body = json.dumps({"action": "update", "calculator": calculator, "param": attribute,
                       "value": value}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/rest/calculators/overrides/modify", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)
