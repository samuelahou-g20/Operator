"""Market share per venue, and why it is where it is.

The desk's rule of thumb is 5%-50% of a coin's traded volume; outside that band somebody usually
has to act. The aggregator sheet carries the headline per coin, but it is a DAILY CUMULATIVE
number at desk level — it cannot say which venue is responsible, and a coin can read 60% while
every strategy is currently stopped. This computes the same ratio per venue, from the boxes'
own 24h executed notional over the market's 24h klines, and then walks a diagnosis ladder.

Share is a ratio of two amounts in the SAME quote currency, so no FX conversion is involved and
none is implied: a EUR pair's share is EUR over EUR.

The ladder stops at the first cause the data supports, because the second cause is unknowable
while the first holds — a strategy that is not quoting has no spread problem.

MUST NOT: change anything (the fixes it names are carried out by ``mm_rebound`` / ``mm_set_param``
after a human confirms), invent a market spread or a volume it could not read, or treat a share
outside the band as wrong — the desk sets some coins low on purpose.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from src.common.logging import get_logger
from src.common.paths import LOGS_DIR
from src.the_operator import rapid
from src.the_operator.zp import dynamic, sources

_log = get_logger("operator")

LOW, HIGH = 0.05, 0.50          # the desk's rule of thumb
MIN_MARKET_QUOTE = 5_000.0      # below this a venue's share is noise, not a signal
HOURS = 24


@dataclass
class VenueShare:
    """One strategy's share of its venue, with the configuration that produced it."""

    exchange: str
    ccypair: str
    status: str
    ours: float
    market: float | None
    mid: float | None
    market_spread_bps: float | None
    params: dict
    calculator: str

    @property
    def share(self) -> float | None:
        if not self.market or self.market < MIN_MARKET_QUOTE:
            return None
        return self.ours / self.market

    @property
    def quoting(self) -> bool:
        return self.status == rapid.TRADING

    @property
    def mode(self) -> str | None:
        """Which ladder group this strategy actually runs (they are mutually exclusive)."""
        for name in ("ob_dynamic", "ob_implicit", "ob_explicit", "ob_asymmetric"):
            if isinstance(self.params.get(name), dict):
                return name
        return None

    @property
    def ladder(self) -> dict:
        return self.params.get(self.mode) or {} if self.mode else {}

    @property
    def min_spread_bps(self) -> float | None:
        g = self.ladder
        v = g.get("base_min_spread_bps", g.get("min_spread_bps"))
        return float(v) if isinstance(v, (int, float)) else None

    @property
    def size_quote(self) -> float | None:
        v = self.ladder.get("total_value_usd")
        return float(v) if isinstance(v, (int, float)) else None


def _market_quote_volume(exchange: str, pair: str, hours: int) -> float | None:
    """The venue's traded notional over ``hours``, in the pair's quote currency."""
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    bars = sources.fetch_klines(exchange, pair, "1h", end - timedelta(hours=hours), end)
    if bars is None or bars.empty:
        return None
    mid = (bars["open"] + bars["high"] + bars["low"] + bars["close"]) / 4
    return float((mid * bars["volume"]).sum())


def shares(program: str, token: str, warnings: list[str], hours: int = HOURS) -> list[VenueShare]:
    """Every venue quoting ``token``, with our share of it over the last ``hours``."""
    rows = rapid.token_view(program, token, warnings)
    if not rows:
        return []

    def market(row: dict) -> float | None:
        try:
            return _market_quote_volume(row["exchange"], row["ccypair"], hours)
        except Exception as e:  # noqa: BLE001 - a venue without klines keeps its share unknown
            warnings.append(f"market volume unavailable for {row['exchange']} {row['ccypair']}: "
                            f"{str(e)[:90]}")
            return None

    with ThreadPoolExecutor(max_workers=min(8, len(rows))) as pool:
        markets = list(pool.map(market, rows))

    out = []
    for row, mkt in zip(rows, markets):
        bid, ask = row.get("market_bid"), row.get("market_ask")
        spread = ((ask - bid) / ((ask + bid) / 2) * 1e4) if bid and ask else None
        out.append(VenueShare(exchange=row["exchange"], ccypair=row["ccypair"],
                              status=row["status"], ours=float(row["our_quote_24h"] or 0.0),
                              market=mkt, mid=row.get("mid"), market_spread_bps=spread,
                              params=row["params"], calculator=row["calculator"]))
    return out


def diagnose(v: VenueShare, realized_vol_pct: float | None = None) -> tuple[str, str | None]:
    """``(cause, suggested fix)`` for one venue — the first rung the data supports.

    The fix names a parameter and a direction, never a predicted share: how much volume a setting
    buys is not measured anywhere here."""
    share = v.share
    if not v.quoting:
        return (f"not quoting ({v.status})",
                f"`/operator rebound {rapid.token_of(v.ccypair)}` if the price left its band")
    if v.market is None:
        return "market volume unavailable, so share cannot be judged", None
    if v.market < MIN_MARKET_QUOTE:
        return f"venue traded only {v.market:,.0f} in {HOURS}h — too thin to read a share", None
    if share is not None and share > HIGH:
        tight = (v.min_spread_bps is not None and v.market_spread_bps is not None
                 and v.min_spread_bps < v.market_spread_bps / 2)
        return (f"we are {share:.0%} of this venue — effectively the market here",
                (f"widen `{v.mode}.{'base_min_spread_bps' if 'base_min' in str(v.ladder) else 'min_spread_bps'}` "
                 f"(ours {v.min_spread_bps:g}bps vs a {v.market_spread_bps:.0f}bps market touch)"
                 if tight else
                 f"cut `{v.mode}.total_value_usd` from {v.size_quote:,.0f}" if v.size_quote else None))
    if share is not None and share < LOW:
        if v.ours == 0:
            return (f"quoting but no fills in {HOURS}h",
                    (f"our `{v.mode}` min spread is {v.min_spread_bps:g}bps against a "
                     f"{v.market_spread_bps:.0f}bps market touch — we are behind the queue"
                     if v.min_spread_bps and v.market_spread_bps
                     and v.min_spread_bps > v.market_spread_bps / 2 else
                     "check the ladder is inside the market's touch"))
        if (v.min_spread_bps is not None and v.market_spread_bps is not None
                and v.min_spread_bps > v.market_spread_bps / 2):
            return (f"share {share:.1%}: our quotes sit outside the touch",
                    f"lower `{v.mode}` min spread from {v.min_spread_bps:g}bps "
                    f"(market touch is {v.market_spread_bps:.0f}bps wide)")
        applied = dynamic.applied(v.ladder, realized_vol_pct) if v.mode == "ob_dynamic" else None
        if applied and applied.get("size_mult", 1.0) < 0.9:
            return (f"share {share:.1%}: size is throttled to "
                    f"{applied['size_mult']:.0%} by vol scaling",
                    f"raise `ob_dynamic.reference_annualized_volatility_pct` "
                    f"(now {applied['ref_vol']:.0f}%) or `total_value_usd`")
        if v.size_quote is not None:
            return (f"share {share:.1%}: spread is competitive, so the limit is size",
                    f"raise `{v.mode}.total_value_usd` from {v.size_quote:,.0f}")
        return f"share {share:.1%}, below the {LOW:.0%} floor", None
    return (f"share {share:.0%}" if share is not None else "share unknown"), None


# --------------------------------------------------------------------------------------
# Snooze — the desk's daily exceptions, told to the Operator rather than configured
# --------------------------------------------------------------------------------------
# Which coins are deliberately low or high changes day to day, so there is no list to keep in
# config. Instead a trader dismisses an alert ("ELA is meant to be high today") and the coin goes
# quiet until the end of the UTC day. The exception list maintains itself, from the people who
# decide it.
SNOOZE_PATH = LOGS_DIR / "operator" / "volume_snooze.json"


def _snoozes() -> dict:
    try:
        return json.loads(SNOOZE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def snooze(desk: str, token: str, author: str, reason: str, today: date | None = None) -> date:
    """Silence ``token`` on ``desk`` for the rest of the UTC day. Returns the day it covers."""
    day = today or datetime.now(timezone.utc).date()
    data = _snoozes()
    data[f"{desk}:{token.upper()}"] = {"until": day.isoformat(), "by": author, "reason": reason}
    try:
        SNOOZE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNOOZE_PATH.write_text(json.dumps(data), encoding="utf-8")
    except OSError as e:
        _log.warning("could not save volume snooze: %s", e)
    return day


def is_snoozed(desk: str, token: str, today: date | None = None) -> bool:
    row = _snoozes().get(f"{desk}:{token.upper()}")
    if not isinstance(row, dict):
        return False
    return str(row.get("until")) == (today or datetime.now(timezone.utc).date()).isoformat()


def snoozed_today(desk: str, today: date | None = None) -> list[str]:
    day = (today or datetime.now(timezone.utc).date()).isoformat()
    return sorted(k.split(":", 1)[1] for k, v in _snoozes().items()
                  if k.startswith(f"{desk}:") and isinstance(v, dict) and v.get("until") == day)


def brief(v: VenueShare) -> tuple[str, str | None]:
    """``(a few words, the command to fix it)`` — the alert form of ``diagnose``.

    The alert goes to people who already know the rule and the band; what they need per coin is
    the reason and what to type. Long form stays in ``diagnose`` for the pull tool."""
    token = rapid.token_of(v.ccypair)
    spread, touch = v.min_spread_bps, v.market_spread_bps
    if not v.quoting:
        return f"not quoting on {v.exchange}", f"/operator rebound {token}"
    if v.market is None or v.market < MIN_MARKET_QUOTE:
        return f"{v.exchange} too thin to read", None
    share = v.share
    if share is not None and share > HIGH:
        if spread and touch and spread < touch / 2:
            return (f"{share:.0%} of {v.exchange}, our {spread:g}bps vs {touch:.0f}bps touch",
                    f"/operator widen {token} min spread on {v.exchange}")
        return (f"{share:.0%} of {v.exchange}" + (f", size {v.size_quote:,.0f}" if v.size_quote else ""),
                f"/operator cut {token} size on {v.exchange}")
    if share is not None and share < LOW:
        if v.ours == 0 and spread and touch and spread > touch / 2:
            return (f"no fills on {v.exchange}, {spread:g}bps vs {touch:.0f}bps touch",
                    f"/operator tighten {token} min spread on {v.exchange}")
        if spread and touch and spread > touch / 2:
            return (f"outside the touch on {v.exchange}, {spread:g}bps vs {touch:.0f}bps",
                    f"/operator tighten {token} min spread on {v.exchange}")
        if v.size_quote:
            return (f"spread competitive on {v.exchange}, size only {v.size_quote:,.0f}",
                    f"/operator raise {token} size on {v.exchange}")
    return f"{v.exchange} {share:.0%}" if share is not None else f"{v.exchange} unknown", None
