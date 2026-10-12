"""Read tools over the rapid-engine quoting boxes: ``mm_status`` and ``mm_overrides``.

``mm_status`` answers "is the book actually quoting?" — the question nothing on the desk watches
today. It separates a strategy stopped by a FAILURE from one somebody deliberately disabled,
because only the first is a call to action.

``mm_overrides`` answers "why is this strategy not behaving like its config says?" — an override is
a hand-set value sitting on top of the deployed configuration, and it outlives whoever set it.

``mm_rebound`` is the write: a strategy stops when the price leaves its ``price_boundaries`` band,
and rebounding moves the band back around the price. It only ever PROPOSES — the push itself
happens in ``execution`` after a human confirms.

MUST NOT: reach a box's write route from a handler, cross into another desk's program without the
caller naming it, or present a DISABLED strategy as a fault.
"""

from __future__ import annotations

import math

from src.common.logging import get_logger
from src.the_operator import rapid, volume
from src.the_operator.contract import Context, Risk, ToolResult, ToolSpec
from src.the_operator.tools.trading import card

_log = get_logger("operator")
_MAX_LISTED = 25


def _program(ctx: Context, args: dict) -> str:
    return (args.get("program") or ctx.desk or "").strip().lower()


def _line(s: rapid.Strategy) -> str:
    bits = [f"• `{s.exchange}` {s.ccypair} — **{s.status}**"]
    if s.resetting:
        bits.append("(resetting)")
    if s.last_error and s.last_error != s.status:
        bits.append(f"last error: {s.last_error}")
    bits.append(f"[{s.box}]")
    return " ".join(bits)


def _mm_status(ctx: Context, args: dict) -> ToolResult:
    program = _program(ctx, args)
    token = (args.get("token") or "").strip().upper()
    warnings: list[str] = []
    found = rapid.scan(program, warnings)
    if not found:
        note = " ".join(warnings) or f"no strategies returned for program {program!r}."
        return ToolResult(f"❌ No quoting status for {program!r}. {note}")
    if token:
        found = [s for s in found if s.token == token]
        if not found:
            return ToolResult(f"No {token} strategy in program {program!r} "
                              f"(it may be quoted by another program).")

    errored = [s for s in found if s.errored]
    disabled = [s for s in found if not s.healthy and not s.errored]
    stale = [s for s in found if s.healthy and s.stale()]
    trading = len(found) - len(errored) - len(disabled)

    head = f"**MM status · {program}**" + (f" · {token}" if token else "")
    out = [head, f"{trading} trading · {len(errored)} errored · {len(disabled)} disabled "
                 f"(of {len(found)} strategies)"]
    if errored:
        out += ["", f"**Stopped in error ({len(errored)})**"]
        out += [_line(s) for s in errored[:_MAX_LISTED]]
        if len(errored) > _MAX_LISTED:
            out.append(f"• …and {len(errored) - _MAX_LISTED} more")
    if stale:
        out += ["", f"**Stale heartbeat ({len(stale)})** — trading, but no status for "
                    f"over {rapid.STALE_S // 60} min"]
        out += [_line(s) for s in stale[:_MAX_LISTED]]
    if disabled and (token or not errored):
        # A full desk scan has dozens of deliberately-off strategies; only list them when the
        # caller asked about one token, or when nothing is actually broken.
        out += ["", f"**Disabled ({len(disabled)})** — deliberate stops, not faults"]
        out += [_line(s) for s in disabled[:_MAX_LISTED]]
        if len(disabled) > _MAX_LISTED:
            out.append(f"• …and {len(disabled) - _MAX_LISTED} more")
    elif disabled:
        out.append(f"({len(disabled)} disabled strategies not listed — ask for a token to see them.)")
    if warnings:
        out += ["", "**Not covered**"] + [f"• {w}" for w in warnings[:5]]
    return ToolResult("\n".join(out))


def _mm_overrides(ctx: Context, args: dict) -> ToolResult:
    program = _program(ctx, args)
    token = (args.get("token") or "").strip().upper()
    warnings: list[str] = []
    all_found = rapid.overrides(program, warnings)
    # Every box carries a `cancel_untracked_orders` override on each ExchangeOrderCancellator.
    # That is deployment boilerplate, hundreds of rows of it, and it buries the quoting strategies
    # that are the actual question.
    found = {calc: params for calc, params in all_found.items()
             if calc.split(":")[0] == "LM2Strategy"}
    skipped = len(all_found) - len(found)
    if token:
        found = {calc: params for calc, params in found.items()
                 if rapid.calculator_token(calc) == token}
    if not found:
        note = " ".join(warnings)
        return ToolResult(f"No quoting-strategy overrides on {program}"
                          + (f" for {token}" if token else "")
                          + (f" ({skipped} non-strategy overrides skipped)" if skipped else "")
                          + (f". {note}" if note else "."))
    head = f"**Live overrides · {program}**" + (f" · {token}" if token else "")
    out = [head, f"{len(found)} quoting strategy/strategies running a hand-set value on top of "
                 f"deployed config" + (f" ({skipped} non-strategy overrides skipped)" if skipped
                                       else "")]
    if not token:
        # Overriding is how this desk configures a strategy, so nearly every one carries some:
        # a desk-wide dump is the whole live config. Summarise, and let the caller name a token.
        by_attr: dict[str, int] = {}
        for params in found.values():
            for k in params:
                by_attr[k] = by_attr.get(k, 0) + 1
        out += ["", "**Most overridden parameters**"]
        out += [f"• `{k}` on {n} strateg{'y' if n == 1 else 'ies'}"
                for k, n in sorted(by_attr.items(), key=lambda kv: -kv[1])[:10]]
        out.append("")
        out.append("Name a token to see its actual values.")
    else:
        out.append("")
        for calc, params in sorted(found.items())[:_MAX_LISTED]:
            out.append(f"• `{calc}`")
            out += [f"    {k} = {v}" for k, v in sorted(params.items())]
        if len(found) > _MAX_LISTED:
            out.append(f"• …and {len(found) - _MAX_LISTED} more")
    if warnings:
        out += ["", "**Not covered**"] + [f"• {w}" for w in warnings[:5]]
    return ToolResult("\n".join(out))


_PROGRAM_ARG = ("program: the quoting program to scan (mm1-mm5, perp). Omit (null) for the "
                "requesting desk's own program — a coin is quoted by ONE program, which is not "
                "always the desk that asked, so name another program when the desk's own scan "
                "does not find the coin.")

MM_STATUS = ToolSpec(
    name="mm_status",
    description=(
        "Live state of the market-making strategies on the rapid-engine quoting boxes: how many "
        "are trading, which are STOPPED IN ERROR (a failure — this is the actionable one), which "
        "are DISABLED (somebody's deliberate off), and which are trading but have gone quiet. Use "
        "it for 'is the MM running', 'is anything down', 'are we quoting X', or before advising on "
        "a coin whose numbers look wrong. token: one ticker to narrow to, or null for the whole "
        "desk. " + _PROGRAM_ARG + " Takes ~10-20s (it scans every box)."
    ),
    input_schema={"type": "object",
                  "properties": {"token": {"type": ["string", "null"]},
                                 "program": {"type": ["string", "null"]}},
                  "required": ["token", "program"], "additionalProperties": False},
    risk=Risk.READ,
    handler=_mm_status,
)

MM_OVERRIDES = ToolSpec(
    name="mm_overrides",
    description=(
        "Parameter overrides currently sitting on top of the deployed configuration of the "
        "quoting strategies — hand-set values that outlive whoever set them. Use it when a "
        "strategy behaves unlike its configuration says it should, when a ZP's settings look "
        "wrong, or when asked what has been changed by hand. token: one ticker, or null for all. "
        + _PROGRAM_ARG
    ),
    input_schema={"type": "object",
                  "properties": {"token": {"type": ["string", "null"]},
                                 "program": {"type": ["string", "null"]}},
                  "required": ["token", "program"], "additionalProperties": False},
    risk=Risk.READ,
    handler=_mm_overrides,
)



# --------------------------------------------------------------------------- #
# mm_rebound (MONEY) — re-set an instrument's price band so quoting resumes
# --------------------------------------------------------------------------- #
DEFAULT_HALF_WIDTH_PCT = 2.5     # the desk's observed convention when a band gives no width


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.8g}"


def _decimals(v: float) -> int:
    text = repr(float(v))
    return len(text.split(".")[1]) if "." in text and "e" not in text else 8


def _new_band(b: rapid.Boundary, price_min: float | None,
              price_max: float | None) -> tuple[float, float] | None:
    """The band to push: the trader's numbers when given, else the current band recentred on the
    current mid keeping its own half-width. None when there is no mid to centre on."""
    if price_min is not None and price_max is not None:
        return price_min, price_max
    if b.mid is None:
        return None
    half = b.half_width_pct or DEFAULT_HALF_WIDTH_PCT
    lo, hi = b.mid * (1 - half / 100), b.mid * (1 + half / 100)
    # Desk bands are round numbers, so match the precision of the one being replaced. Rounded
    # OUTWARD, so rounding can only widen the band and never push the mid back outside it.
    if b.price_min is not None and b.price_max is not None:
        f = 10 ** max(_decimals(b.price_min), _decimals(b.price_max))
        lo, hi = math.floor(lo * f) / f, math.ceil(hi * f) / f
    return lo, hi


def _mm_rebound(ctx: Context, args: dict) -> ToolResult:
    token = (args["token"] or "").strip().upper()
    if not token:
        return ToolResult("❌ token must not be empty")
    price_min, price_max = args["price_min"], args["price_max"]
    if (price_min is None) != (price_max is None):
        return ToolResult("❌ give both price_min and price_max, or neither (and I recentre the "
                          "current band on the current mid).")
    if price_min is not None and price_min >= price_max:
        return ToolResult(f"❌ price_min ({price_min}) must be below price_max ({price_max}).")

    program = _program(ctx, args)
    warnings: list[str] = []
    found = rapid.boundaries(program, token, warnings)
    if not found:
        note = " ".join(warnings) or "no strategy with a price band found."
        return ToolResult(f"❌ No {token} price bands on program {program!r}. {note}")

    targets, skipped = [], []
    for b in found:
        band = _new_band(b, price_min, price_max)
        if band is None:
            skipped.append((b, "no current mid — cannot centre a band on nothing"))
            continue
        lo, hi = band
        if b.mid is not None and not (lo <= b.mid <= hi):
            # Pushing a band the price is already outside leaves the strategy stopped. This is the
            # rail that stops one quote's numbers being applied to another quote's scale.
            skipped.append((b, f"mid {_fmt(b.mid)} falls outside the proposed "
                               f"[{_fmt(lo)}, {_fmt(hi)}]"))
            continue
        targets.append((b, lo, hi))
    if not targets:
        lines = [f"• `{b.exchange}` {b.ccypair} — {why}" for b, why in skipped]
        return ToolResult("❌ Nothing to rebound.\n" + "\n".join(lines))

    how = "explicit" if price_min is not None else "recentred on mid"
    headline = f"🔧 REBOUND {token} · {len(targets)} strateg{'y' if len(targets) == 1 else 'ies'}"
    rows = [("Desk", f"{ctx.desk} (program {program})"), ("Band", how)]
    rows += [(f"{b.exchange} {b.ccypair}",
              f"`{_fmt(b.price_min)}–{_fmt(b.price_max)}` → `{_fmt(lo)}–{_fmt(hi)}` "
              f"(mid {_fmt(b.mid)}, {b.status})") for b, lo, hi in targets]
    notes = [f"⚠️ skipped `{b.exchange}` {b.ccypair}: {why}" for b, why in skipped]
    notes += [f"⚠️ {w}" for w in warnings]
    return ToolResult(
        card(headline, rows, notes),
        proposal={"kind": "mm_rebound", "headline": headline, "token": token, "program": program,
                  "how": how, "notional_usd": 0.0, "reason": args["reason"],
                  "targets": [{"box": b.box, "url": b.url, "calculator": b.calculator,
                               "exchange": b.exchange, "ccypair": b.ccypair,
                               "old": [b.price_min, b.price_max], "price_min": lo,
                               "price_max": hi} for b, lo, hi in targets]})


MM_REBOUND = ToolSpec(
    name="mm_rebound",
    description=(
        "Re-set the PRICE BAND (price_boundaries) of a token's market-making strategies on the "
        "rapid-engine boxes, so quoting resumes. A strategy stops (STOP (DISABLED)) when the price "
        "leaves its band; rebounding moves the band back around the price. Call it when a trader "
        "says to rebound/re-bound a token, or asks to restart quoting on one. token: the ticker. "
        "price_min/price_max: the trader's explicit band — pass BOTH, or BOTH null to recentre "
        "each strategy's current band on its current mid (that is what a bare 'rebound XYZ' "
        "means). It covers every venue for the token at once. reason: the requester's words. "
        + _PROGRAM_ARG
    ),
    input_schema={
        "type": "object",
        "properties": {
            "token": {"type": "string"},
            "price_min": {"type": ["number", "null"]},
            "price_max": {"type": ["number", "null"]},
            "program": {"type": ["string", "null"]},
            "reason": {"type": "string"},
        },
        "required": ["token", "price_min", "price_max", "program", "reason"],
        "additionalProperties": False,
    },
    risk=Risk.MONEY,
    handler=_mm_rebound,
)




# --------------------------------------------------------------------------- #
# mm_set_param (MONEY) — one field inside one parameter group, on a live strategy
# --------------------------------------------------------------------------- #
# The groups a quoting review actually recommends changing. price_boundaries is NOT here:
# mm_rebound owns it, because it carries the "is the price inside the new band" guard.
SETTABLE = ("ob_dynamic", "ob_implicit", "posting_policy", "descale_trigger", "price_range_rule")


def _mm_set_param(ctx: Context, args: dict) -> ToolResult:
    attribute, field = args["attribute"], str(args["field"]).strip()
    if attribute not in SETTABLE:
        return ToolResult(f"❌ {attribute} is not settable here. Allowed: {', '.join(SETTABLE)}."
                          + (" Use mm_rebound for price bands." if attribute == "price_boundaries"
                             else ""))
    if not field:
        return ToolResult("❌ field must not be empty")
    token = (args["token"] or "").strip().upper()
    program = _program(ctx, args)
    exchange = (args["exchange"] or "").strip().upper()
    value = args["value"]

    warnings: list[str] = []
    rows = rapid.strategy_params(program, token, warnings)
    if not rows:
        note = " ".join(warnings) or "no strategy found."
        return ToolResult(f"❌ No {token} strategies on program {program!r}. {note}")
    if exchange:
        rows = [r for r in rows if r["exchange"] == exchange]
        if not rows:
            return ToolResult(f"❌ No {token} strategy on {exchange} in program {program!r}.")

    targets, skipped = [], []
    for r in rows:
        group = r["params"].get(attribute)
        if not isinstance(group, dict):
            # Writing a group a strategy does not run switches its quoting mode. That is a
            # different decision from changing a number, and it is not this tool's job.
            skipped.append((r, f"runs no {attribute} (writing one would switch its quoting mode)"))
            continue
        if field not in group:
            skipped.append((r, f"{attribute} has no field {field!r} "
                               f"(it has {', '.join(sorted(group))})"))
            continue
        if type(group[field]) is bool and not isinstance(value, bool):
            skipped.append((r, f"{field} is a true/false setting; {value!r} is not"))
            continue
        targets.append((r, group[field]))
    if not targets:
        return ToolResult("❌ Nothing to set.\n"
                          + "\n".join(f"• `{r['exchange']}` {r['ccypair']}: {why}"
                                      for r, why in skipped))

    headline = (f"⚙️ SET {attribute}.{field} = {value} · {token} · "
                f"{len(targets)} strateg{'y' if len(targets) == 1 else 'ies'}")
    rows_out = [("Desk", f"{ctx.desk} (program {program})")]
    rows_out += [(f"{r['exchange']} {r['ccypair']}", f"`{old}` → `{value}`")
                 for r, old in targets]
    notes_out = [f"⚠️ skipped `{r['exchange']}` {r['ccypair']}: {why}" for r, why in skipped]
    notes_out += [f"⚠️ {w}" for w in warnings]
    notes_out.append("This changes live quoting immediately.")
    return ToolResult(
        card(headline, rows_out, notes_out),
        proposal={"kind": "mm_set_param", "headline": headline, "token": token, "program": program,
                  "attribute": attribute, "field": field, "value": value, "notional_usd": 0.0,
                  "reason": args["reason"],
                  "targets": [{"box": r["box"], "url": r["url"], "calculator": r["calculator"],
                               "exchange": r["exchange"], "ccypair": r["ccypair"], "old": old}
                              for r, old in targets]})


MM_SET_PARAM = ToolSpec(
    name="mm_set_param",
    description=(
        "Change ONE field of ONE parameter group on a token's live market-making strategies — the "
        "levers a quoting review recommends (base_min_spread_bps, top_level_weight, total_value_usd, "
        "levels, the scaling factors...). attribute: the group, one of " + ", ".join(SETTABLE) +
        ". field: the field inside it, exactly as the strategy spells it. value: a number, or "
        "true/false for a switch. exchange: one venue, or null for every venue quoting the token. "
        "Call it only when a trader explicitly asks to change a setting; the ZP's recommendations "
        "are advice until a trader says to apply one. Price bands are NOT here — use mm_rebound. "
        "reason: the requester's words. " + _PROGRAM_ARG
    ),
    input_schema={
        "type": "object",
        "properties": {
            "token": {"type": "string"},
            "attribute": {"type": "string", "enum": list(SETTABLE)},
            "field": {"type": "string"},
            "value": {"type": ["number", "boolean"]},
            "exchange": {"type": ["string", "null"]},
            "program": {"type": ["string", "null"]},
            "reason": {"type": "string"},
        },
        "required": ["token", "attribute", "field", "value", "exchange", "program", "reason"],
        "additionalProperties": False,
    },
    risk=Risk.MONEY,
    handler=_mm_set_param,
)




# --------------------------------------------------------------------------- #
# volume_check (READ) — market share against the desk's 5%-50% rule of thumb
# --------------------------------------------------------------------------- #
def _pct(text: str) -> float | None:
    try:
        return float(str(text).strip().rstrip("%").replace(",", "")) / 100
    except (TypeError, ValueError):
        return None


def _desk_shares(desk_name: str) -> list[tuple[str, float | None]]:
    """``(coin, share)`` from the desk's aggregator sheet — the number the desk already reads."""
    from scripts.agg_bot import _AGG_CFG, _MM2, _MM3, read_dashboard   # deferred: gspread import

    cfg = {"mm3": _MM3, "mm2": _MM2}.get(desk_name)
    if cfg is None or desk_name not in _AGG_CFG:
        return []
    df = read_dashboard(cfg)
    col = "Mkt Share"
    if col not in df.columns or "Coin" not in df.columns:
        return []
    return [(str(r["Coin"]), _pct(r[col])) for _, r in df.iterrows()
            if str(r["Coin"]).upper() not in {"ALL", ""}]


def _desk_card(ctx: Context, program: str, rows: list[tuple[str, float | None]]) -> ToolResult:
    """The desk-wide breach list: one line per coin, reasons from a single quoting scan.

    A per-venue diagnosis costs a box scan and a kline call per venue, which is far too much for
    every coin. One scan of the program gives every strategy's status at once, and "not quoting"
    is the reason behind almost every 0%."""
    stopped: dict[str, tuple[int, int]] = {}
    try:
        for st in rapid.scan(program, []):
            total, down = stopped.get(st.token, (0, 0))
            stopped[st.token] = (total + 1, down + (0 if st.healthy else 1))
    except Exception:  # noqa: BLE001 - without it the card simply carries no reasons
        _log.exception("volume card could not scan %s", program)

    def reason(token: str, too_high: bool) -> tuple[str, str | None]:
        """Why, and which command. A coin ABOVE the band is never sent to rebound: restarting a
        stopped venue there would push its share further up, not back into the band."""
        total, down = stopped.get(token, (0, 0))
        if not total:
            return "", None
        if down == total:
            why = f"not quoting ({total} venue{'s' if total > 1 else ''})"
        elif down:
            why = f"{down} of {total} venues stopped"
        else:
            why = "quoting on every venue"
        return why, ("check" if too_high or not down else "rebound")

    # A sheet row with no strategy on this program is not something the desk quotes — a
    # non-token row, or another program's coin. Listing it is noise: nothing here can act on it.
    quoted = [(c, sh) for c, sh in rows if sh is not None and stopped.get(c)]
    high = sorted((r for r in quoted if r[1] > volume.HIGH), key=lambda r: -r[1])
    low = sorted((r for r in quoted if r[1] < volume.LOW), key=lambda r: r[1])
    out = [f"**📊 Market share · {ctx.desk}** — {len(high)} above {volume.HIGH:.0%}, "
           f"{len(low)} below {volume.LOW:.0%}"]
    todo: dict[str, list[str]] = {"rebound": [], "check": []}
    for title, group, too_high in ((f"Above {volume.HIGH:.0%}", high, True),
                                   (f"Below {volume.LOW:.0%}", low, False)):
        if not group:
            continue
        out += ["", f"**{title}**"]
        for coin, share in group[:_MAX_LISTED]:
            why, action = reason(coin, too_high)
            out.append(f"• **{coin}** {share:.0%} — {why}")
            if action:
                todo[action].append(coin)
        if len(group) > _MAX_LISTED:
            out.append(f"• …and {len(group) - _MAX_LISTED} more")
    snoozed = volume.snoozed_today(ctx.desk)
    if snoozed:
        out += ["", f"Snoozed today: {', '.join(snoozed)}"]
    if any(todo.values()):
        out.append("")
        if todo["rebound"]:
            out.append(f"`/operator rebound <coin>` — {', '.join(todo['rebound'])}")
        if todo["check"]:
            out.append(f"`/operator volume check <coin>` — {', '.join(todo['check'])}")
    card_text = "\n".join(out)
    return ToolResult(f"The share card for {ctx.desk} has been sent to the chat as-is. Do not "
                      f"repeat it, do not restate the band or the caveats, and do not comment on "
                      f"coins it did not list. Add nothing unless the requester asked something "
                      f"the card does not answer.\n\n{card_text}",
                      verbatim=True, card=card_text)


def _volume_check(ctx: Context, args: dict) -> ToolResult:
    token = (args["token"] or "").strip().upper()
    program = _program(ctx, args)
    if not token:
        try:
            rows = _desk_shares(ctx.desk)
        except Exception as exc:  # noqa: BLE001 - the sheet is a dependency, not a certainty
            return ToolResult(f"❌ could not read the {ctx.desk} aggregator sheet: "
                              f"{type(exc).__name__}: {str(exc)[:120]}")
        if not rows:
            return ToolResult(f"No aggregator sheet share column for desk {ctx.desk}.")
        return _desk_card(ctx, program, rows)

    warnings: list[str] = []
    venues = volume.shares(program, token, warnings)
    if not venues:
        note = " ".join(warnings) or "no strategy found."
        return ToolResult(f"❌ No {token} strategies on program {program!r}. {note}")
    ours = sum(v.ours for v in venues)
    market = sum(v.market for v in venues if v.market)
    out = [f"**Market share · {token}** (last {volume.HOURS}h, per venue)"]
    out.append(f"Ours {ours:,.0f} of {market:,.0f} quote traded"
               + (f" — **{ours / market:.1%}** across the venues we quote" if market else ""))
    out.append("")
    for v in sorted(venues, key=lambda x: -(x.market or 0)):
        cause, fix = volume.diagnose(v)
        share = f"{v.share:.1%}" if v.share is not None else "n/a"
        out.append(f"• `{v.exchange}` {v.ccypair} — **{share}** "
                   f"({v.ours:,.0f} of {(v.market or 0):,.0f})")
        out.append(f"    {cause}")
        if fix:
            out.append(f"    **fix:** {fix}")
    if warnings:
        out += ["", "**Not covered**"] + [f"• {w}" for w in warnings[:4]]
    out.append("")
    out.append("Nothing is changed by this. Say which fix to apply and I will preview it.")
    return ToolResult("\n".join(out))


VOLUME_CHECK = ToolSpec(
    name="volume_check",
    description=(
        "How much of a coin's traded volume is ours, and WHY it is where it is. The desk's rule of "
        "thumb is 5%-50%; outside that band somebody usually has to act, though some coins are "
        "deliberately low and the desk decides that day by day — never call a share wrong on your "
        "own. token: one ticker for the per-venue breakdown with a cause and a named parameter fix "
        "for each venue (not quoting / outside the touch / size too small / size throttled by vol "
        "scaling / we are the market). token null gives the desk-wide list from the aggregator "
        "sheet — the day's cumulative share, which can read high for a coin that has since stopped "
        "quoting. Use it when asked about volumes, market share, capture, or why we are not "
        "trading a coin. " + _PROGRAM_ARG + " A single token takes ~30s."
    ),
    input_schema={"type": "object",
                  "properties": {"token": {"type": ["string", "null"]},
                                 "program": {"type": ["string", "null"]}},
                  "required": ["token", "program"], "additionalProperties": False},
    risk=Risk.READ,
    handler=_volume_check,
)




def _volume_snooze(ctx: Context, args: dict) -> ToolResult:
    token = (args["token"] or "").strip().upper()
    if not token:
        return ToolResult("❌ token must not be empty")
    already = volume.snoozed_today(ctx.desk)
    headline = f"🔕 SNOOZE {token} volume alerts on {ctx.desk} for today"
    rows = [("Coin", token), ("Desk", ctx.desk), ("Why", args["reason"])]
    notes = [f"Already snoozed today: {', '.join(already)}" if already else
             "Nothing else is snoozed today.",
             "It covers the rest of the UTC day; tomorrow the coin is watched again."]
    return ToolResult(card(headline, rows, notes),
                      proposal={"kind": "volume_snooze", "headline": headline, "token": token,
                                "notional_usd": 0.0, "reason": args["reason"]})


VOLUME_SNOOZE = ToolSpec(
    name="volume_snooze",
    description=(
        "Stop the volume watch flagging one coin for the rest of today, because the desk means its "
        "share to be where it is. Call it when a trader answers a volume alert with 'that is "
        "intended', 'we want it low', 'leave it', or similar — never on your own reading of the "
        "numbers. It silences alerts only; it changes no quoting parameter and no order. "
        "token: the ticker. reason: the requester's words, so the next person knows why."
    ),
    input_schema={"type": "object",
                  "properties": {"token": {"type": "string"}, "reason": {"type": "string"}},
                  "required": ["token", "reason"], "additionalProperties": False},
    risk=Risk.WRITE,
    handler=_volume_snooze,
)


TOOLS: list[ToolSpec] = [MM_STATUS, MM_OVERRIDES, VOLUME_CHECK, VOLUME_SNOOZE,
                         MM_REBOUND, MM_SET_PARAM]
