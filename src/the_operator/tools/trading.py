"""MONEY tools: ``place_order``, ``place_scalp``, ``cancel_order``, ``move_order``.

Each handler BUILDS its action through ``scripts/trade_bot.py``'s own ``prepare_*`` functions —
the same parsing, per-token/USD caps, venue rails and touch pricing a typed ``/trade`` goes
through — and returns a human-readable preview plus the built action in ``ToolResult.proposal``.
The handler is the same in every phase; whether the proposal is dry-run, held for the requester's
confirm, or executed is ``executor``/``execution``'s decision from ``policy``, never this module's.

MUST NOT: import or construct ``CoinRoutesTrader``, call ``place``/``cancel``/``modify``/``apply_ladder`` or
any other order-writing path, hold caps or rails (they live in trade_bot and are exercised, not
copied, here), or send anything to Telegram. A test scans this module's source for those names.
"""

from __future__ import annotations

import pandas as pd

from scripts import trade_bot as tb
from src.adapters.coinroutes.client import CoinRoutesClient
from src.the_operator import policy
from src.the_operator.contract import Context, Risk, ToolResult, ToolSpec
from src.the_operator.desk import cr_token

TRADING_DESKS = frozenset(tb.DESKS)


def _refusal(msg: str) -> ToolResult:
    return ToolResult(msg if msg.startswith("❌") else f"❌ {msg}")


def _client(ctx: Context) -> CoinRoutesClient:
    return CoinRoutesClient(api_token=cr_token(ctx.desk))


RULE = "━" * 22


def card(title: str, rows: list[tuple[str, str]], notes: list[str]) -> str:
    """The preview block the gateway delivers verbatim: bold headline, one "• **Label** value"
    bullet per row (no column padding — Telegram's font is proportional), then notes. Rendered to
    Telegram HTML by ``render.to_html``; the equivalent typed command lives in the proposal
    (journal), not on the card."""
    body = [f"• **{k}** {v}" for k, v in rows]
    return "\n".join([f"**{title}**", RULE, *body, *([RULE, *notes] if notes else [])])


def _usd(v: float) -> str:
    return f"${v:,.0f}" if v >= 100 else f"${v:,.2f}"


def _limit_note(order: dict) -> str:
    if order["limit_price"] is not None:
        return "your limit"
    buf = tb.SLIP.get(order["aggression"], 0.0)
    return "at touch" if buf == 0 else f"{buf:.1%} through the touch"


# --------------------------------------------------------------------------- #
# place_order
# --------------------------------------------------------------------------- #
def order_command(args: dict) -> str:
    """The exact ``/trade`` text a trader would have typed for these arguments, so the preview is
    produced by the very parser the live bot uses (parity by construction, not by re-implementation)."""
    parts = ["/trade", args["token"], args["side"]]
    if args["aggression"]:
        parts.append(args["aggression"])
    if args["perp"]:
        parts.append("perp")
    if args["size"]:
        parts.append(f"size={args['size']}")
    if args["limit"] is not None:
        parts.append(f"limit={args['limit']}")
    if args["max_post"] is not None:
        parts.append(f"maxpost={args['max_post']}")
    if args["post_only"]:
        parts.append("post")
    if args["staged"]:
        parts.append("staged")
    return " ".join(parts)


def _place_order(ctx: Context, args: dict) -> ToolResult:
    desk = tb.DESKS.get(ctx.desk)
    if desk is None:
        return _refusal(f"{ctx.desk} is not a trading desk")
    text = order_command(args)
    plan, err = tb.prepare_order(text, _client(ctx), desk)
    if err is not None:
        return _refusal(err)
    order, kw, base, touch = plan["order"], plan["place_kwargs"], plan["base"], plan["touch"]
    perp = kw["currency_pair"].upper().endswith(".PERP")
    size = (_usd(order["quantity"]) if order["quantity_currency"] == "USDT"
            else f"{order['quantity']:,.10g} {base} (~{_usd(order['quantity'] * touch)})")
    headline = (f"{'🟢 BUY' if order['side'] == 'buy' else '🔴 SELL'} {base} · {size} · "
                f"{order['aggression']}")
    rows = [("Book", f"{order['strategy']} ({'perp' if perp else 'spot'})"),
            ("Pair", kw["currency_pair"]),
            ("Limit", f"`{plan['limit']}` ({_limit_note(order)})"),
            ("Touch", f"`{touch:.10g}`")]
    if kw["max_post_size"] is not None:
        mp = kw["max_post_size"]
        rows.append(("Max post", f"{mp:,.0f} {base} (~{_usd(mp * touch)})" if mp >= 100
                     else f"{mp:,.6g} {base} (~{_usd(mp * touch)})"))
    if kw["post_only"]:
        rows.append(("Post only", "yes"))
    if kw["extra"]:
        rows.append(("Venues", ", ".join(kw["extra"]["exchanges"])))
    # What will actually happen on confirm: a phase may force a status, else what /trade would do.
    status = policy.MONEY_STATUS or plan["status"]
    rows.append(("Status", f"{status} → manual release in CoinRoutes" if status == "staged"
                 else f"{status} → live into the market"))
    notional = order["quantity"] if order["quantity_currency"] == "USDT" else order["quantity"] * touch
    notes = []
    if perp and not args["perp"]:
        notes.append(f"⚠ {base} is not in the {ctx.desk} gamma roster, so this routes to the "
                     f"futures book. Say 'perp' if that is intended.")
    if (note := policy.four_eyes_note(notional, ctx.desk)) is not None:
        notes.append(note)
    proposal = {"kind": "place_order", "headline": headline, "command": text, "place_kwargs": kw,
                "notional_usd": notional, "reason": args["reason"]}
    return ToolResult(card(headline, rows, notes), proposal=proposal)


PLACE_ORDER = ToolSpec(
    name="place_order",
    description=(
        "Build one CoinRoutes order on the desk's book for a token, exactly as /trade would: "
        "resolves the strategy (gamma spot book for a desk token, else the futures book), applies "
        "the desk caps, prices the limit off the live touch and returns a preview. token: ticker, "
        "e.g. 'DNX'. side: 'buy'|'sell'. size: null = the token's desk default quantity; "
        "'15k'/'1000usd' = USD notional; '250000tok' = base tokens (only for tokens with a "
        "default). aggression: 'passive' (limit at touch), 'neutral' (0.5% through), 'aggressive' "
        "(1% through, the default) or null. limit: explicit limit price or null. max_post: max "
        "child post size in BASE TOKENS or null (default $500 worth). perp: true forces the "
        "futures book. post_only: post-only flag. staged: true parks the order for manual approval "
        "in the CoinRoutes UI instead of the bot default. reason: the requester's words for why, "
        "kept in the journal. Only call this when a trader explicitly asks to place/buy/sell; "
        "never on your own initiative."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "token": {"type": "string"},
            "side": {"type": "string", "enum": ["buy", "sell"]},
            "size": {"type": ["string", "null"]},
            "aggression": {"anyOf": [{"type": "string", "enum": sorted(tb._AGGR)}, {"type": "null"}]},
            "limit": {"type": ["number", "null"]},
            "max_post": {"type": ["number", "null"]},
            "perp": {"type": "boolean"},
            "post_only": {"type": "boolean"},
            "staged": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["token", "side", "size", "aggression", "limit", "max_post", "perp",
                     "post_only", "staged", "reason"],
        "additionalProperties": False,
    },
    risk=Risk.MONEY,
    handler=_place_order,
    desks=TRADING_DESKS,
)


# --------------------------------------------------------------------------- #
# place_scalp
# --------------------------------------------------------------------------- #
def scalp_command(args: dict) -> str:
    parts = ["/scalper", args["token"], str(args["usd_per_order"]), str(args["levels_per_side"])]
    if args["lookback"]:
        parts.append(args["lookback"])
    if args["vol"]:
        parts.append(f"vol={args['vol']}")
    return " ".join(parts)


def _place_scalp(ctx: Context, args: dict) -> ToolResult:
    desk = tb.DESKS.get(ctx.desk)
    if desk is None:
        return _refusal(f"{ctx.desk} is not a trading desk")
    text = scalp_command(args)
    plan, err = tb.prepare_scalper(text, _client(ctx), desk)
    if err is not None:
        return _refusal(err)
    cmd = plan["cmd"]
    headline = f"🪜 SCALP {plan['base']} · {_usd(cmd['size'])} × {cmd['n']} a side"
    src = (f"vol={cmd['vol'] * 100:.4g}% override" if cmd["vol"] is not None else f"{cmd['days']}d JASON")
    rows = [("Book", plan["strategy"]), ("Pair", plan["pair"]),
            ("Mid", f"{plan['mid']:g} · σ {plan['sigma'] * 100:.2f}%/day ({src})")]
    for side, mark in (("sell", "🔴"), ("buy", "🟢")):
        for px, usd, _unit in sorted(plan["want"][side], reverse=True):
            rows.append((f"{mark} {side}", f"`{px:g}` · {_usd(usd)} · {plan['kmap'][px]}σ"))
    # `size` is per order in whatever unit parse_size read; scalp_command only ever sends USD
    # today, but a notional that is silently in base tokens would mis-size the four-eyes gate.
    per_order_usd = cmd["size"] if cmd["unit"] == "usd" else cmd["size"] * plan["mid"]
    notional = per_order_usd * cmd["n"] * 2
    notes = [f"Replaces the whole resting {plan['base']} ladder: levels not listed are cancelled, "
             "matching ones are kept."]
    if (note := policy.four_eyes_note(notional, ctx.desk)) is not None:
        notes.append(note)
    proposal = {"kind": "place_scalp", "headline": headline, "command": text, "base": plan["base"],
                "pair": plan["pair"], "strategy": plan["strategy"], "pxv": plan["pxv"],
                "venues": plan["venues"], "mid": plan["mid"], "sigma": plan["sigma"],
                "want": plan["want"], "kmap": plan["kmap"], "head": plan["head"],
                "notional_usd": notional, "reason": args["reason"]}
    return ToolResult(card(headline, rows, notes), proposal=proposal)


PLACE_SCALP = ToolSpec(
    name="place_scalp",
    description=(
        "Build a symmetric scalper ladder for a desk token, exactly as /scalper would: n levels a "
        "side at 1..n daily sigma from the live mid, each of usd_per_order, using the token's "
        "JASON realized vol. Returns the levels as a preview. token: ticker. usd_per_order: USD "
        "size of EACH order. levels_per_side: 1-4. lookback: vol window like '7d', '2w', '1m' or "
        "null (default 7d). vol: annualised vol override such as '85%' or null. reason: the "
        "requester's words. Applying a scalp replaces the token's whole resting ladder with live "
        "quotes. Only call when a trader explicitly asks for a scalp/ladder."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "token": {"type": "string"},
            "usd_per_order": {"type": "number"},
            # no minimum/maximum: strict mode rejects them; trade_bot.parse_scalper enforces 1..LADDER_MAX_LEVELS
            "levels_per_side": {"type": "integer"},
            "lookback": {"type": ["string", "null"]},
            "vol": {"type": ["string", "null"]},
            "reason": {"type": "string"},
        },
        "required": ["token", "usd_per_order", "levels_per_side", "lookback", "vol", "reason"],
        "additionalProperties": False,
    },
    risk=Risk.MONEY,
    handler=_place_scalp,
    desks=TRADING_DESKS,
)


# --------------------------------------------------------------------------- #
# cancel_order
# --------------------------------------------------------------------------- #
def _cancel_order(ctx: Context, args: dict) -> ToolResult:
    if ctx.desk not in tb.DESKS:
        return _refusal(f"{ctx.desk} is not a trading desk")
    oid = str(args["order_id"]).strip()
    if not oid:
        return _refusal("order_id must not be empty")
    headline = f"🗑 CANCEL order {oid}"
    return ToolResult(card(headline, [("Desk", ctx.desk), ("Order", oid)], []),
                      proposal={"kind": "cancel_order", "headline": headline, "order_id": oid,
                                "notional_usd": 0.0, "reason": args["reason"]})


CANCEL_ORDER = ToolSpec(
    name="cancel_order",
    description=(
        "Cancel one of the desk's client orders by its CoinRoutes order id (as shown by "
        "open_orders or a previous placement). order_id: the id string. reason: the requester's "
        "words. Only call when a trader explicitly asks to cancel/pull a specific order."
    ),
    input_schema={
        "type": "object",
        "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["order_id", "reason"],
        "additionalProperties": False,
    },
    risk=Risk.MONEY,
    handler=_cancel_order,
    desks=TRADING_DESKS,
)


# --------------------------------------------------------------------------- #
# move_order
# --------------------------------------------------------------------------- #
def _desk_books(desk: dict) -> set[str]:
    names = {*desk["strategy"].values(), desk.get("perp_strategy"), desk.get("spot_strategy")}
    return {str(n).casefold() for n in names if n}


def _open_order(client: CoinRoutesClient, oid: str) -> dict | None:
    df = client.client_orders(order_status="open")
    if df is None or df.empty:
        return None
    for rec in df.to_dict("records"):
        if str(tb._order_ref({k: v for k, v in rec.items() if pd.notna(v)})) == oid:
            return rec
    return None


def _move_order(ctx: Context, args: dict) -> ToolResult:
    desk = tb.DESKS.get(ctx.desk)
    if desk is None:
        return _refusal(f"{ctx.desk} is not a trading desk")
    oid = str(args["order_id"]).strip()
    if not oid:
        return _refusal("order_id must not be empty")
    client = _client(ctx)
    order = _open_order(client, oid)
    if order is None:
        return _refusal(f"no open order `{oid}`: it filled, was cancelled, or the id is wrong")
    if str(order["strategy"]).casefold() not in _desk_books(desk):
        return _refusal(f"order `{oid}` is on {order['strategy']}, not a {ctx.desk} book")
    pair, side, strategy = str(order["currency_pair"]), str(order["side"]), str(order["strategy"])
    base = pair.split("-")[0]
    old = str(order["limit_price"])
    venues = tb.price_venues(client, strategy, base, pair.upper().endswith(".PERP"))
    if not venues:
        return _refusal(f"no venues known for {strategy}; refusing to price blind")
    # Our own side of the book: a bid joins the best bid (touch_on "sell" side), an offer the best ask.
    best = tb.touch_on(client, pair, "sell" if side == "buy" else "buy", venues)
    if best is None:
        return _refusal(f"no price for {pair} on {', '.join(venues)}")
    best_label = "best bid" if side == "buy" else "best offer"
    join = args["limit"] is None
    limit = f"{best:.10g}" if join else f"{args['limit']:.10g}"
    if float(limit) == float(old):
        return _refusal(f"order `{oid}` already rests at `{old}`")
    qty, unit = float(order["quantity"]), str(order["quantity_currency"])
    notional = qty if unit in tb.QUOTE_CURRENCIES else qty * float(limit)
    size = _usd(qty) if unit in tb.QUOTE_CURRENCIES else f"{qty:,.10g} {base} (~{_usd(notional)})"
    headline = f"↕ MOVE {'BID' if side == 'buy' else 'OFFER'} {base} · {size}"
    rows = [("Order", f"`{oid}`"), ("Book", strategy), ("Pair", pair),
            ("From", f"`{old}`"),
            ("To", f"`{limit}` ({f'joins the {best_label}' if join else 'your limit'})"),
            (best_label.capitalize(), f"`{best:.10g}`")]
    notes = []
    if (note := policy.four_eyes_note(notional, ctx.desk)) is not None:
        notes.append(note)
    proposal = {"kind": "move_order", "headline": headline, "order_id": oid, "pair": pair,
                "side": side, "venues": venues, "from_price": old, "limit_price": limit,
                "join": join, "base": base, "notional_usd": notional, "reason": args["reason"]}
    return ToolResult(card(headline, rows, notes), proposal=proposal)


MOVE_ORDER = ToolSpec(
    name="move_order",
    description=(
        "Move one of the desk's resting CoinRoutes orders to a new limit price, in place (same "
        "order id, size and book). order_id: the Id column from open_orders. limit: the new price, "
        "or null to JOIN the best price on the order's own side (a bid goes to the best bid, an "
        "offer to the best offer); null is the default for 'move it to market / to the touch / up "
        "to best'. Give an explicit limit only when the trader names a price or asks to cross or "
        "take. reason: the requester's words. Call open_orders first to get the id when the "
        "trader describes the order instead of naming it (e.g. 'the closest bid')."
    ),
    input_schema={
        "type": "object",
        "properties": {"order_id": {"type": "string"}, "limit": {"type": ["number", "null"]},
                       "reason": {"type": "string"}},
        "required": ["order_id", "limit", "reason"],
        "additionalProperties": False,
    },
    risk=Risk.MONEY,
    handler=_move_order,
    desks=TRADING_DESKS,
)


TOOLS: list[ToolSpec] = [PLACE_ORDER, PLACE_SCALP, CANCEL_ORDER, MOVE_ORDER]
