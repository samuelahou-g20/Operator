"""Read-only market/desk-dashboard tools: thin wrappers over the existing Telegram bots and
``positioning.py``.

Each handler reuses a bot's own fetch + render functions verbatim (no reimplemented fetch/render
logic here) and repackages the result as a ``ToolResult``. MUST NOT: call Telegram, open the data
lake, place/amend/cancel anything, or hold policy/desk-routing decisions beyond picking the
per-desk config a bot already defines.
"""

from __future__ import annotations

import pandas as pd

from scripts import agg_bot, jason_bot, ls_bot, twap_bot, vol_bot
from src.common.config import load_config
from src.common.deadline import run_with_deadline
from src.the_operator import cache, positioning
from src.the_operator.contract import Attachment, Context, Risk, ToolResult, ToolSpec

_TEXT_LIMIT = 6000


def _df_text(df: pd.DataFrame) -> str:
    """A dataframe as plain text, capped so one huge table can't blow the model's context."""
    text = df.to_string(index=False)
    if len(text) > _TEXT_LIMIT:
        text = text[:_TEXT_LIMIT] + "\n…truncated"
    return text


# --------------------------------------------------------------------------- #
# aggregator
# --------------------------------------------------------------------------- #
_AGG_CFG_BY_DESK = {
    "mm3": agg_bot._MM3, "mm2": agg_bot._MM2, "treasury": agg_bot._TREASURY,
    "options": agg_bot._OPTIONS, "mm5": agg_bot._MM5,
}


def _aggregator(ctx: Context, args: dict) -> ToolResult:
    cfg = _AGG_CFG_BY_DESK[ctx.desk]   # ctx.desk is gateway-built, not user input — let a miss raise
    if cfg.get("source") == "pluto":
        df, pv = agg_bot.read_portfolio(cfg)
        png = agg_bot.render_portfolio_image(cfg, df)
        caption = f"{cfg['label']} · PV ${pv:,.0f} · {len(df) - 1} assets"
    else:
        df = agg_bot.read_dashboard(cfg)
        png = agg_bot.render_table_image(cfg, df)
        caption = f"{cfg['label']} · {len(df)} rows"
    text = f"{_df_text(df)}\n\n{caption}"
    return ToolResult(text=text, attachments=[
        Attachment(filename="aggregator.png", data=png, kind="photo", caption=caption)])


AGGREGATOR = ToolSpec(
    name="aggregator",
    description=(
        "The desk's aggregator dashboard: per-coin price, 24h %, daily P&L, net/daily delta, "
        "trading P&L and market share (mm3/mm2, from the live Google Sheet), or per-asset greeks "
        "(spot, delta $, gamma $, vega, theta) plus portfolio PV (treasury/options/mm5, from "
        "Pluto). No arguments — the desk is taken from the requesting chat. Returns the table as "
        "text plus a rendered PNG. Use for 'what's my book/P&L/delta look like right now'."
    ),
    input_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    risk=Risk.READ,
    handler=_aggregator,
    desks=frozenset({"mm3", "mm2", "treasury", "options", "mm5"}),
)


# --------------------------------------------------------------------------- #
# jason_levels
# --------------------------------------------------------------------------- #
def _jason_levels(ctx: Context, args: dict) -> ToolResult:
    universe = args["universe"]
    if universe not in jason_bot.UNIVERSES:
        return ToolResult(text=f"❌ unknown universe {universe!r}. "
                                f"Options: {', '.join(jason_bot.UNIVERSES)}")
    df = jason_bot.build_table(jason_bot.fetch_jason(universe))
    png = jason_bot.render_table_image(df, universe)
    caption = jason_bot._caption(universe, df)
    text = f"{_df_text(df)}\n\n{caption}"
    return ToolResult(text=text, attachments=[
        Attachment(filename="jason_levels.png", data=png, kind="photo", caption=caption)])


JASON_LEVELS = ToolSpec(
    name="jason_levels",
    description=(
        "JASON support/resistance levels: per-pair pivot vol, z-scores off the high/low pivot, "
        "and the 2-std price targets, from the Pluto jason_levels API. universe picks the ccypair "
        "basket: 'default' (the standard desk watchlist), 'majors' (BTC/ETH/SOL/XRP/BNB/HYPE/"
        "PAXG), 'mm3' (mm3 desk names) or 'rwa' (~30 most-liquid Binance tradfi/equity perps). "
        "Returns the table as text plus a rendered PNG. Use for 'where are the levels on X' or a "
        "levels check across a basket."
    ),
    input_schema={
        "type": "object",
        "properties": {"universe": {"type": "string", "enum": ["default", "majors", "mm3", "rwa"]}},
        "required": ["universe"],
        "additionalProperties": False,
    },
    risk=Risk.READ,
    handler=_jason_levels,
)


# --------------------------------------------------------------------------- #
# twap
# --------------------------------------------------------------------------- #
def _twap(ctx: Context, args: dict) -> ToolResult:
    cmd = (f"/twap {args['token']} start={args['start']} end={args['end']} "
           f"candle={args['candle']} exchange={args['exchange']} type={args['method']}")
    reply, csv_bytes, name = twap_bot.handle_twap(cmd)   # never raises; ❌-prefixed text on failure
    attachments = [Attachment(filename=name or "twap.csv", data=csv_bytes, kind="document")] \
        if csv_bytes is not None else []
    return ToolResult(text=reply, attachments=attachments)


TWAP = ToolSpec(
    name="twap",
    description=(
        "Time-weighted average price of a token over an explicit UTC window, computed from live "
        "exchange candles (equal weight per candle — this is a TWAP, not volume-weighted). "
        "token: a ticker, e.g. 'btc' or 'btw'. start/end: UTC timestamps, e.g. "
        "'2026/08/20-00:00:00UTC' (also accepts '2026-08-20T00:00:00Z' or a bare date "
        "'2026/08/20'); always interpreted as UTC regardless of a trailing UTC/Z. candle: a "
        "ccxt timeframe such as '1m','5m','15m','1h','4h','1d' or a trader spelling like '1hr', "
        "'1day','daily'. exchange: a ccxt exchange id, e.g. 'binance','bybit','okx','bitget'. "
        "method: 'default'/'ohlc4' = (O+H+L+C)/4, 'close' = C, 'hl2' = (H+L)/2, "
        "'hlc3' = (H+L+C)/3. Returns the TWAP with context (venue, candle count, range) as text, "
        "and the underlying candle CSV as a document."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "token": {"type": "string"},
            "start": {"type": "string"},
            "end": {"type": "string"},
            "candle": {"type": "string"},
            "exchange": {"type": "string"},
            "method": {"type": "string", "enum": ["default", "ohlc4", "close", "hl2", "hlc3"]},
        },
        "required": ["token", "start", "end", "candle", "exchange", "method"],
        "additionalProperties": False,
    },
    risk=Risk.READ,
    handler=_twap,
)


# --------------------------------------------------------------------------- #
# vol_surface
# --------------------------------------------------------------------------- #
def _surface_text(surface: dict, jvols: dict) -> str:
    labels = [lab for lab, _ in vol_bot._VOL_TENORS]
    order = sorted(surface, key=lambda u: -surface[u].get("1M", 0.0))
    lines = ["Deribit ATM implied vol"]
    for u in order:
        tenors = ", ".join(f"{lab}={surface[u][lab] * 100:.0f}%" for lab in labels if lab in surface[u])
        lines.append(f"{u}: {tenors}")
    return "\n".join(lines)


def _vol_surface(ctx: Context, args: dict) -> ToolResult:
    surface = run_with_deadline(vol_bot._fetch_iv_surface, deadline=vol_bot._FETCH_DEADLINE)
    if not surface:
        return ToolResult(text="No Deribit implied-vol data available right now.")
    jvols = run_with_deadline(vol_bot._fetch_jason_vols, list(surface), deadline=vol_bot._FETCH_DEADLINE)
    if not vol_bot._render_lock.acquire(timeout=vol_bot._RENDER_TIMEOUT):
        raise RuntimeError("vol_bot renderer is stuck — flag to ops")
    try:
        png = vol_bot.render_surface_png(surface, jvols)
    finally:
        vol_bot._render_lock.release()
    return ToolResult(text=_surface_text(surface, jvols), attachments=[
        Attachment(filename="vol_surface.png", data=png, kind="photo",
                   caption="Deribit ATM implied vol")])


VOL_SURFACE = ToolSpec(
    name="vol_surface",
    description=(
        "The Deribit ATM implied-vol term structure (1W/2W/1M) for BTC/ETH/SOL/XRP/AVAX/HYPE, "
        "plus each name's JASON realized vol (vol up/down off the pivots, whole-period). No "
        "arguments. Returns a compact underlying -> tenor:iv text table plus a rendered PNG. Use "
        "for 'what's implied vol doing' or an RV/IV comparison."
    ),
    input_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    risk=Risk.READ,
    handler=_vol_surface,
)


# --------------------------------------------------------------------------- #
# ls_ranking
# --------------------------------------------------------------------------- #
def _ls_ranking(ctx: Context, args: dict) -> ToolResult:
    universe = args["universe"]
    if universe not in ls_bot.UNIVERSES:
        return ToolResult(text=f"❌ unknown universe {universe!r}. "
                                f"Options: {', '.join(ls_bot.UNIVERSES)}")
    cfg = load_config("data/binance.yaml")
    df, skipped, short_hist = ls_bot.build_ranking(cfg, ls_bot.UNIVERSES[universe])
    png = ls_bot.render_table_image(df, universe)
    caption = ls_bot._caption(skipped, short_hist)
    text = f"{_df_text(df)}\n\n{caption}"
    return ToolResult(text=text, attachments=[
        Attachment(filename="ls_ranking.png", data=png, kind="photo", caption=caption)])


LS_RANKING = ToolSpec(
    name="ls_ranking",
    description=(
        "Cross-universe long/short trend ranking: each ticker's short (7d half-life) and long "
        "(28d half-life) JASON trend score on 4h bars, JASON support/resistance z-scores, "
        "2-week realized vol, a continuous 1-5 regime, and resting-state-delta sizing — sorted "
        "most-long to most-short. universe must be one of the configured names (e.g. 'default', "
        "'clarity'); an unknown name returns an error instead of a table. Returns the ranking as "
        "text plus a rendered PNG. Use for 'where's the trend across the book/universe'."
    ),
    input_schema={
        "type": "object",
        "properties": {"universe": {"type": "string"}},
        "required": ["universe"],
        "additionalProperties": False,
    },
    risk=Risk.READ,
    handler=_ls_ranking,
)


# --------------------------------------------------------------------------- #
# positioning
# --------------------------------------------------------------------------- #
POSITIONING_TTL_S = 3600   # daily ratios and 8h funding; an hour is fresh enough and spares Binance


def positioning_cached(ttl_s: float = POSITIONING_TTL_S) -> tuple[str, float]:
    return cache.cached("positioning", ttl_s, positioning.snapshot)


def _positioning(ctx: Context, args: dict) -> ToolResult:
    text, age = positioning_cached(0 if args.get("refresh") else POSITIONING_TTL_S)
    return ToolResult(text=f"{text}\n({cache.age_note(age)})")


POSITIONING = ToolSpec(
    name="positioning",
    description=(
        "Where the market is positioned, free and fast (~5s): Crypto Fear & Greed (today, yesterday, "
        "7d avg, 30d range) and, for BTC/ETH/SOL/XRP/DOGE on Binance perps, the retail account "
        "long/short ratio, the top-trader position long/short ratio and the last-24h funding sum, "
        "each with its percentile vs the last 30 days, plus fixed-threshold flags (retail vs top "
        "traders diverging, crowded long/short, extreme greed/fear). Use to check whether the crowd "
        "is actually positioned the way X sentiment says. Cached 1h; refresh: true forces a new pull."
    ),
    input_schema={"type": "object", "properties": {"refresh": {"type": "boolean"}},
                  "required": ["refresh"], "additionalProperties": False},
    risk=Risk.READ,
    handler=_positioning,
)


TOOLS: list[ToolSpec] = [AGGREGATOR, JASON_LEVELS, TWAP, VOL_SURFACE, LS_RANKING, POSITIONING]
