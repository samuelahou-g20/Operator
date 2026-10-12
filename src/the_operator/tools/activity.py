"""Operator tools over the desk's activity bots: open orders, fills, hedge PnL/risk, and Pluto
bot status/config — all read-and-render wrappers over ``scripts/orders_bot.py``,
``scripts/hedge_bot.py``, ``scripts/botstatus_bot.py`` and ``scripts/desk_trades.py``.

Every tool is desk-aware through ``ctx.desk``: CoinRoutes-backed tools build a
``CoinRoutesClient`` on that desk's key via ``src.the_operator.desk.cr_token``; the others pass
``ctx.desk`` straight through to the bot module, which already resolves it against
``config/desks.yaml``.

MUST NOT: place/modify/cancel an order, write a bot config, start/stop/reconfigure a Pluto bot,
send anything to Telegram, or open the data lake. Every handler here is read-only (``Risk.READ``).
"""

from __future__ import annotations

from scripts import botstatus_bot, desk_trades, hedge_bot, orders_bot
from src.adapters.coinroutes.client import CoinRoutesClient
from src.common.config import load_config
from src.the_operator.contract import Attachment, Context, Risk, ToolResult, ToolSpec
from src.the_operator.desk import cr_token

_TEXT_CAP = 6000  # keep the model-visible text well under a context-busting table dump

_ORDERS_DESK_CFG = {"mm3": orders_bot._MM3, "mm2": orders_bot._MM2, "treasury": orders_bot._TREASURY}
_HEDGE_DESKS = ("mm3", "mm2")
_BOT_DESKS = ("mm3", "mm2", "treasury")


def _cap(text: str) -> str:
    return text if len(text) <= _TEXT_CAP else text[:_TEXT_CAP] + "\n…(truncated)"


# --------------------------------------------------------------------------- #
# open_orders / recent_fills — CoinRoutes, orders_bot
# --------------------------------------------------------------------------- #
def _open_orders(ctx: Context, args: dict) -> ToolResult:
    cfg = _ORDERS_DESK_CFG.get(ctx.desk)
    if cfg is None:
        return ToolResult(text="❌ not available on this desk")
    scope = str(args["scope"]).strip()
    if not scope:
        return ToolResult(text="❌ scope must not be empty")
    tok = cr_token(ctx.desk)
    client = CoinRoutesClient(api_token=tok)
    low = scope.lower()
    if low == "all":
        text, png, caption = orders_bot.book_all(client, cfg, tok)
    elif low == "hedge":
        text, png, caption = orders_bot.book_hedge(client, cfg, tok)
    else:
        text, png, caption = orders_bot.book_asset(client, cfg, scope.upper())
    if png is None:
        return ToolResult(text=caption)
    attachments = [Attachment(filename=f"orders_{low}.png", data=png, kind="photo", caption=caption)]
    return ToolResult(text=_cap(f"{caption}\n\n{text}"), attachments=attachments)


def _recent_fills(ctx: Context, args: dict) -> ToolResult:
    cfg = _ORDERS_DESK_CFG.get(ctx.desk)
    if cfg is None:
        return ToolResult(text="❌ not available on this desk")
    asset = str(args["asset"]).strip().upper()
    if not asset:
        return ToolResult(text="❌ asset must not be empty")
    n = max(1, min(int(args["n"]), 50))
    client = CoinRoutesClient(api_token=cr_token(ctx.desk))
    df = orders_bot.recent_fills(client, asset, n)
    if df is None or df.empty:
        return ToolResult(text=f"No fills for {asset}.")
    caption = f"{asset} · {len(df)} most recent fill(s)"
    png = orders_bot.render_fills_image(asset, df)
    attachments = [Attachment(filename=f"fills_{asset}.png", data=png, kind="photo", caption=caption)]
    return ToolResult(text=_cap(f"{caption}\n\n{df.to_string(index=False)}"), attachments=attachments)


# --------------------------------------------------------------------------- #
# fills_today — desk_trades
# --------------------------------------------------------------------------- #
def _fills_today(ctx: Context, args: dict) -> ToolResult:
    caption, png = desk_trades.handle(ctx.desk, args.get("category"), args.get("token"))
    if png is None:
        return ToolResult(text=caption)
    attachments = [Attachment(filename="fills_today.png", data=png, kind="photo", caption=caption)]
    return ToolResult(text=caption, attachments=attachments)


# --------------------------------------------------------------------------- #
# hedge_pnl / hedge_risk — hedge_bot
# --------------------------------------------------------------------------- #
def _hedge_pnl(ctx: Context, args: dict) -> ToolResult:
    if ctx.desk not in _HEDGE_DESKS:
        return ToolResult(text="❌ not available on this desk")
    tok = cr_token(ctx.desk)
    report = hedge_bot.build_report(tok, ctx.desk)
    caption = f"Total PnL: {report.iloc[-1]['total_pnl']:,.0f}"
    png = hedge_bot.render_table_image(report, ctx.desk)
    attachments = [Attachment(filename="hedge_pnl.png", data=png, kind="photo", caption=caption)]
    return ToolResult(text=_cap(f"{caption}\n\n{report.to_string(index=False)}"),
                      attachments=attachments)


def _hedge_risk(ctx: Context, args: dict) -> ToolResult:
    if ctx.desk not in _HEDGE_DESKS:
        return ToolResult(text="❌ not available on this desk")
    tok = cr_token(ctx.desk)
    rdf, tot = hedge_bot.build_risk(tok, ctx.desk, load_config("data/binance.yaml"))
    if rdf.empty:
        return ToolResult(text="No futures-book positions.")
    caption = hedge_bot.risk_caption(ctx.desk, tot)
    png = hedge_bot.render_risk_image(rdf, tot, ctx.desk)
    attachments = [Attachment(filename="hedge_risk.png", data=png, kind="photo", caption=caption)]
    return ToolResult(text=_cap(f"{caption}\n\n{rdf.to_string(index=False)}"), attachments=attachments)


# --------------------------------------------------------------------------- #
# bot_status / bot_config — botstatus_bot (Pluto BotManager, read endpoints only)
# --------------------------------------------------------------------------- #
def _bot_status(ctx: Context, args: dict) -> ToolResult:
    if ctx.desk not in _BOT_DESKS:
        return ToolResult(text="❌ not available on this desk")
    df, png, caption = botstatus_bot.botstatus_report(ctx.desk)
    attachments = [Attachment(filename="bot_status.png", data=png, kind="photo", caption=caption)]
    return ToolResult(text=_cap(f"{caption}\n\n{df.to_string(index=False)}"), attachments=attachments)


def _bot_config(ctx: Context, args: dict) -> ToolResult:
    if ctx.desk not in _BOT_DESKS:
        return ToolResult(text="❌ not available on this desk")
    bot_query = str(args["bot"]).strip()
    if not bot_query:
        return ToolResult(text="❌ bot must not be empty")
    tab_query = args.get("tab") or ""
    # botstatus_bot.bot_config_report resolves `bot_query`/`tab_query` ONLY inside desk_roster(desk)
    # — it never reaches outside the calling desk's own Pluto bot roster.
    png, text = botstatus_bot.bot_config_report(ctx.desk, bot_query, tab_query)
    if png is None:
        return ToolResult(text=text if text.startswith("❌") else f"❌ {text}")
    return ToolResult(text=text, attachments=[Attachment(filename="bot_config.png", data=png,
                                                          kind="photo", caption=text)])


TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="open_orders",
        description=(
            "The desk's open CoinRoutes orders, as a text ladder/table plus a chart image. "
            "`scope`: \"all\" for the desk's whole open book (spot gamma tokens, or the hedge "
            "book for a prop/treasury desk); \"hedge\" for the futures-book open orders only; "
            "or a token symbol (e.g. \"ICP\") for just that token's gamma-j book. Use this to "
            "check what the desk is currently quoting/resting, not what has filled."
        ),
        input_schema={
            "type": "object",
            "properties": {"scope": {"type": "string",
                                     "description": "\"all\", \"hedge\", or a token symbol."}},
            "required": ["scope"],
            "additionalProperties": False,
        },
        risk=Risk.READ,
        handler=_open_orders,
        desks=frozenset({"mm3", "mm2", "treasury"}),
    ),
    ToolSpec(
        name="recent_fills",
        description=(
            "The N most recently-updated CoinRoutes orders for one token that have a non-zero "
            "fill (includes still-open partially-filled orders), as a text table plus a chart "
            "image. `asset`: the token symbol (e.g. \"ICP\"). `n`: how many fills, 1-50 "
            "(clamped)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "asset": {"type": "string", "description": "Token symbol, e.g. \"ICP\"."},
                "n": {"type": "integer", "description": "Number of fills to return, 1-50."},
            },
            "required": ["asset", "n"],
            "additionalProperties": False,
        },
        risk=Risk.READ,
        handler=_recent_fills,
        desks=frozenset({"mm3", "mm2", "treasury"}),
    ),
    ToolSpec(
        name="fills_today",
        description=(
            "What the desk actually traded today (from the trade service, not CoinRoutes "
            "`client_orders`), aggregated per token/category/side with quantity, notional-"
            "weighted VWAP and USD notional. `category`: one of \"gamma\" (default — the "
            "market-making inventory book), \"mm\", \"hedge\", or \"all\" (also accepts plural "
            "forms like \"hedges\"); null uses the default. `token`: restrict to one token symbol "
            "(e.g. \"ICP\"), or null for every token traded today."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "category": {"type": ["string", "null"],
                             "description": "\"gamma\" | \"mm\" | \"hedge\" | \"all\", or null."},
                "token": {"type": ["string", "null"],
                         "description": "A token symbol, or null for every token."},
            },
            "required": ["category", "token"],
            "additionalProperties": False,
        },
        risk=Risk.READ,
        handler=_fills_today,
        desks=frozenset({"mm3", "mm2", "treasury"}),
    ),
    ToolSpec(
        name="hedge_pnl",
        description=(
            "Today's per-symbol futures-hedge positional + trading PnL (with a TOTAL row), as a "
            "text table plus a chart image. mm3/mm2 only. No arguments."
        ),
        input_schema={"type": "object", "properties": {}, "required": [],
                      "additionalProperties": False},
        risk=Risk.READ,
        handler=_hedge_pnl,
        desks=frozenset({"mm3", "mm2"}),
    ),
    ToolSpec(
        name="hedge_risk",
        description=(
            "The futures-hedge book's risk panel: exposure, entry/mark/uPnL, the nearer adverse "
            "JASON pivot or 2σ target per symbol, risk-to-that-level, 1-day 95% historical VaR/ES, "
            "and wrong-way flags (position opposes both JASON mean-reversion and the trend "
            "score). As a text table plus a chart image. mm3/mm2 only. No arguments."
        ),
        input_schema={"type": "object", "properties": {}, "required": [],
                      "additionalProperties": False},
        risk=Risk.READ,
        handler=_hedge_risk,
        desks=frozenset({"mm3", "mm2"}),
    ),
    ToolSpec(
        name="bot_status",
        description=(
            "The calling desk's Pluto trading-bot roster (from config/desks.yaml): Bot / Type / "
            "Status / Operational / Conn, plus each bot's lastError, as a text table plus a chart "
            "image. A rostered bot the API doesn't return shows as MISSING rather than vanishing. "
            "Read-only — status only, never start/stop/reconfigure. No arguments."
        ),
        input_schema={"type": "object", "properties": {}, "required": [],
                      "additionalProperties": False},
        risk=Risk.READ,
        handler=_bot_status,
        desks=frozenset({"mm3", "mm2", "treasury"}),
    ),
    ToolSpec(
        name="bot_config",
        description=(
            "One Pluto bot's run parameters, resolved ONLY inside the calling desk's own bot "
            "roster (an ambiguous or unknown name lists the candidates/roster instead of "
            "guessing). Read-only — never writes, starts, or stops a bot. `bot`: a "
            "case-insensitive substring of the bot's name (e.g. \"jasondh\"). `tab`: \"tabs\" "
            "lists every config tab and its field count; a tab name (case-insensitive, unique "
            "substring OK) shows that tab's Field/Value/Type rows; null shows the bot's "
            "`Strategy<N>` summary rows (Biprod/Active/Hedge Instr/LP/Max Exec/Resting Δ/"
            "Deadband) — a bot with none of those points you at \"tabs\" instead."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "bot": {"type": "string", "description": "Bot name or a substring of it."},
                "tab": {"type": ["string", "null"],
                       "description": "\"tabs\", a tab name/substring, or null for the strategy summary."},
            },
            "required": ["bot", "tab"],
            "additionalProperties": False,
        },
        risk=Risk.READ,
        handler=_bot_config,
        desks=frozenset({"mm3", "mm2", "treasury"}),
    ),
]
