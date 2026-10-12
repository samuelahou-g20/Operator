"""The ``zp`` tool: a Zeyi Process quoting review for one token over a window.

Deterministic facts (headline P&L split, venue scorecard, markouts, capture, depth compliance)
are computed by ``src.the_operator.zp`` and delivered to the chat verbatim as a card plus an HTML
attachment; the model receives the same card, the checks packet and the standing open questions,
and writes the findings, recommendations and open questions the report ends with.

MUST NOT: place anything, re-derive a metric here (``zp.compute`` owns the arithmetic), or let the
model author a number — a section the run could not measure stays "not measured".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.common.config import load_config
from src.the_operator import cache
from src.the_operator.contract import Attachment, Context, Risk, ToolResult, ToolSpec
from src.the_operator.zp import checks as zp_checks
from src.the_operator.zp import report as zp_report
from src.the_operator.zp import run as zp_run

TTL_S = 30 * 60          # a review of the last N hours tolerates a half-hour-old answer; a run is ~1 min
MAX_HOURS = 72

_INSTRUCTION = (
    "Write the rest of the report from the checks below, in this order and nothing else:\n"
    "**Findings** — one bullet per check that fired, in severity order, each stating the number "
    "and what it means for the book (attribution, adverse selection, tick/min-spread, dynamic "
    "mode, ladder shape, compliance).\n"
    "**Recommendations** — numbered, each tagged High/Med/Low, each tied to a finding. The desk "
    "judges these against two objectives: (1) improve P&L and (2) get total capture to at least "
    "10%. Every recommendation must name a dynamic-mode PARAMETER, its current value and the "
    "value to try, and say what the formula (in the dynamic_settings check) makes that do — the "
    "desk tunes parameters, so anything that does not name one is not actionable. Say which of "
    "the two objectives it serves, and where the two conflict (tighter quotes buy capture and pay "
    "in markout) say so rather than picking silently. Never predict how much capture a change "
    "buys: that response is not measured here.\n"
    "**Open questions** — the ones this run could not settle, from the list below.\n"
    "Cite the numbers as given; never estimate a section marked not measured."
)


def _known_tokens() -> list[str]:
    return sorted(k for k in load_config("zp/tokens.yaml") if k.isupper())


def _parse_start(raw: str, now: datetime) -> datetime | None:
    """``"00:00"``, ``"2026-09-10T00:00"`` or ``"2026-09-10 00:00:00Z"`` -> a naive UTC datetime.

    A bare time means today UTC, and a time still ahead of the clock means yesterday, so "since
    23:00" just after midnight reads as the last hour rather than a window in the future."""
    text = raw.strip().replace("Z", "").replace("T", " ").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    for fmt in ("%H:%M:%S", "%H:%M", "%H"):
        try:
            t = datetime.strptime(text, fmt).time()
        except ValueError:
            continue
        candidate = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
        return candidate if candidate <= now else candidate - timedelta(days=1)
    return None


def _zp(ctx: Context, args: dict) -> ToolResult:
    token = str(args["token"]).upper().strip()
    known = _known_tokens()
    if token not in known:
        return ToolResult(f"❌ {token} is not in the ZP token registry (config/zp/tokens.yaml). "
                          f"Known: {', '.join(known[:40])}…")

    end = datetime.now(timezone.utc).replace(tzinfo=None, second=0, microsecond=0)
    raw_start = args.get("start")
    if raw_start:
        start = _parse_start(str(raw_start), end)
        if start is None:
            return ToolResult(f"❌ can't read start {raw_start!r}; use '00:00' (today UTC) or "
                              "'2026-09-10T00:00'")
        if start >= end:
            return ToolResult(f"❌ start {start:%Y-%m-%d %H:%M} UTC is not before now")
        hours = (end - start).total_seconds() / 3600
        if hours > MAX_HOURS:
            return ToolResult(f"❌ that window is {hours:.0f}h; the maximum is {MAX_HOURS}h")
    else:
        hours = int(args["hours"])
        if not 1 <= hours <= MAX_HOURS:
            return ToolResult(f"❌ hours must be between 1 and {MAX_HOURS}")
        start = end - timedelta(hours=hours)

    def build():
        result = zp_run.run_zp(token, start, end, ctx.desk)
        fired = zp_checks.checks(result, dmc=result.gamma_dash, ticks=result.ticks)
        return result, fired

    ttl = 0 if args.get("refresh") else TTL_S
    # An explicit start is its own cache key: "since 00:00" and "last 18h" are different windows.
    key = f"zp:{ctx.desk}:{token}:{start:%Y%m%d-%H%M}:{end:%Y%m%d-%H%M}"
    (result, fired), age = cache.cached(key, ttl, build)

    span = f"{hours:.0f}h"
    card = zp_report.card_text(result) + f"\n({cache.age_note(age)})"
    html = zp_report.html_report(result, fired).encode("utf-8")
    attachment = Attachment(filename=f"{token}_ZP_{span}_{result.end:%Y%m%d-%H%M}Z.html",
                            data=html, kind="document", mime="text/html",
                            caption=f"{token} ZP · {span} · full tables")
    notes = f"{card}\n\n{_INSTRUCTION}\n\n{zp_report.packet(result, fired)}"
    return ToolResult(notes, attachments=[attachment], verbatim=True, card=card)


ZP = ToolSpec(
    name="zp",
    description=(
        "Zeyi Process (ZP) quoting review for one of our market-made tokens: the desk's standard "
        "post-mortem of a quoting session. Computes, from our fills, exchange market data and the "
        "orderbook snapshots: total P&L split into passive market-making vs hedging, MM volume and "
        "net inventory, flow-level capture, vol-weighted markouts at 60/120/180/300/600s per venue "
        "(the adverse-selection read), our capture of each venue's volume, 2% depth compliance and "
        "uptime, plus a per-venue scorecard and deterministic checks (hedger attribution, tick "
        "dead zone, floored dynamic mode, ladder shape, compliance boundary). token: the ticker, "
        "which must be in the ZP registry. hours: window length back from now, 1-72 (the desk's "
        "default review is 9). start: use INSTEAD of hours for a window anchored to a time — "
        "'00:00' (that time today UTC, or yesterday if it is still ahead of the clock) or "
        "'2026-09-10T00:00'; null to use hours. refresh: true recomputes instead of reusing the "
        "30-minute cache. Takes about a "
        "minute. The card and the HTML attachment go to the chat as-is; you then write the "
        "findings, recommendations and open questions from the checks. Use when a trader asks for "
        "a ZP, a quoting review, or how our market making went on a token."
    ),
    input_schema={
        "type": "object",
        "properties": {"token": {"type": "string"}, "hours": {"type": "integer"},
                       "start": {"type": ["string", "null"]}, "refresh": {"type": "boolean"}},
        "required": ["token", "hours", "start", "refresh"],
        "additionalProperties": False,
    },
    risk=Risk.READ,
    handler=_zp,
    desks=frozenset({"mm3", "mm2"}),
)

TOOLS: list[ToolSpec] = [ZP]
