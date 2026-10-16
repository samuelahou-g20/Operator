"""``desk_view``: the evidence packet an advisory answer is written from, in the owner's order.

    1. macro (Claude web search, cached)   2. X sentiment digest (cached)
    2b. market positioning: Fear & Greed, Binance long/short and funding (free, cached 1h)
    2c. prediction markets: Polymarket odds on Fed, economy, crypto, geopolitics (free, cached 1h)
    3. our positioning: futures-book positions + gamma-book deltas
    4. JASON stretch, trend, vol, VS-RSD for every held name and the LS universe
    5. deterministic RULES: z >= STRETCH_Z up -> trim/sell candidate, down -> buy/cover candidate,
       read against the position; position sign vs LT trend; size vs vol-scaled RSD.

Every number comes from the existing bots (``hedge_bot``, ``ls_bot``, ``agg_bot``, ``macro_bot``,
``sentiment_bot``); this module only assembles and applies the rules, so the model's job is the
narrative and the justification, never the arithmetic. Sections fail independently ("unavailable")
and run in parallel because macro and the LS ranking each take tens of seconds.

MUST NOT: place or propose an order, invent a number a section did not return, or bypass the
caches on paid inputs unless the trader asked for a refresh.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

from scripts import agg_bot, hedge_bot, ls_bot
from src.common.config import load_config
from src.the_operator import cache
from src.the_operator.contract import Context, Risk, ToolResult, ToolSpec
from src.the_operator.desk import cr_token
from src.the_operator.tools import intel, market

STRETCH_Z = 2.0        # owner's rule: |z| >= 2 is a mean-reversion trigger
SIZE_HI = 1.5          # |position| > SIZE_HI x |VS-RSD|  -> oversized for the name's vol
SIZE_LO = 0.5          # |position| < SIZE_LO x |VS-RSD| -> undersized while regime agrees
LS_TTL_S = 30 * 60     # the ranking pulls bars for the whole universe; 30 min is fresh enough
RULE = "━" * 22


def _f(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def rules(positions: list[dict], stats: dict[str, dict]) -> list[dict]:
    """Deterministic flags per held name, then watch candidates among unheld names.

    ``positions``: [{sym, qty, notional, ...}]; ``stats``: {sym: {zup, zdn, lt, vsrsd, regime}}.
    Returns [{sym, held, side, notional, flags: [str]}] (only entries with at least one flag)."""
    out: list[dict] = []
    held = set()
    for p in positions:
        sym, s = p["sym"], stats.get(p["sym"], {})
        held.add(sym)
        long = p["qty"] > 0
        notional = abs(_f(p.get("notional")) or 0.0)
        flags: list[str] = []
        zup, zdn, lt, vsrsd = s.get("zup"), s.get("zdn"), s.get("lt"), s.get("vsrsd")
        if zup is not None and zup >= STRETCH_Z:
            flags.append(f"JASON z_up {zup:.1f} ≥ {STRETCH_Z:g}: stretched UP → "
                         + ("trim/sell the long" if long else "short is with the stretch; hold or add carefully"))
        if zdn is not None and zdn >= STRETCH_Z:
            flags.append(f"JASON z_dn {zdn:.1f} ≥ {STRETCH_Z:g}: stretched DOWN → "
                         + ("cover/reduce the short" if not long else "long is with the stretch; hold or add carefully"))
        if lt is not None and ((long and lt < 0) or (not long and lt > 0)):
            flags.append(f"{'long' if long else 'short'} against LT trend ({lt:+.0f})")
        if vsrsd is not None and vsrsd != 0:
            if (vsrsd > 0) != long:
                flags.append(f"direction against regime RSD ({'long' if vsrsd > 0 else 'short'} bias, "
                             f"VS-RSD ${vsrsd:,.0f})")
            else:
                ratio = notional / abs(vsrsd)
                if ratio > SIZE_HI:
                    flags.append(f"oversized {ratio:.1f}× the vol-scaled RSD (${abs(vsrsd):,.0f})")
                elif ratio < SIZE_LO:
                    flags.append(f"undersized {ratio:.1f}× the vol-scaled RSD (${abs(vsrsd):,.0f})")
        if flags:
            out.append({"sym": sym, "held": True, "side": "long" if long else "short",
                        "notional": notional, "flags": flags})
    for sym, s in stats.items():
        if sym in held:
            continue
        zup, zdn, lt = s.get("zup"), s.get("zdn"), s.get("lt")
        flags = []
        if zup is not None and zup >= STRETCH_Z:
            flags.append(f"not held, z_up {zup:.1f}: fade/short candidate"
                         + (" (LT trend agrees)" if lt is not None and lt < 0 else " (LT trend disagrees: caution)"))
        if zdn is not None and zdn >= STRETCH_Z:
            flags.append(f"not held, z_dn {zdn:.1f}: buy candidate"
                         + (" (LT trend agrees)" if lt is not None and lt > 0 else " (LT trend disagrees: caution)"))
        if flags:
            out.append({"sym": sym, "held": False, "side": "-", "notional": 0.0, "flags": flags})
    return out


# --------------------------------------------------------------------------- #
# sections (each returns text; failures become one line, never an exception)
# --------------------------------------------------------------------------- #
def _section_positions(desk: str, cfg: dict) -> tuple[str, list[dict], dict[str, dict]]:
    if desk not in hedge_bot.CR_STRATEGIES:
        return f"positions: no futures book configured for {desk}", [], {}
    pos = hedge_bot._agg_positions(cr_token(desk), desk)
    if pos.empty:
        return "positions: futures book is flat", [], {}
    syms = pos["sym"].tolist()
    piv, trend = hedge_bot.jason_pivots(syms), hedge_bot.hedge_trend(cfg, syms)
    stats: dict[str, dict] = {}
    lines = [f"{'sym':6} {'side':5} {'notional$':>11} {'entry':>11} {'mark':>11} {'uPnL$':>9} "
             f"{'z_up':>5} {'z_dn':>5} {'LT':>4}"]
    rows = []
    for _, p in pos.iterrows():
        hp, lp, zup, zdn, _u, _d = piv.get(p["sym"], (None,) * 6)
        lt = trend.get(p["sym"])
        stats[p["sym"]] = {"zup": zup, "zdn": zdn, "lt": lt, "hi_pivot": hp, "lo_pivot": lp}
        rows.append({"sym": p["sym"], "qty": float(p["qty"]), "notional": float(p["notional"])})
        lines.append(f"{p['sym']:6} {'long' if p['qty'] > 0 else 'short':5} {p['notional']:>11,.0f} "
                     f"{p['entry']:>11.6g} {p['mark']:>11.6g} {p['upnl']:>9,.0f} "
                     f"{'' if zup is None else f'{zup:.1f}':>5} {'' if zdn is None else f'{zdn:.1f}':>5} "
                     f"{'' if lt is None else f'{lt:+.0f}':>4}")
    return "\n".join(lines), rows, stats


def _section_gamma(desk: str) -> str:
    cfg = {"mm3": agg_bot._MM3, "mm2": agg_bot._MM2}.get(desk)
    if cfg is None:
        return "gamma book: no sheet for this desk"
    df = agg_bot.read_dashboard(cfg)
    col = next((c for c in df.columns if c.startswith("Net Δ")), None)
    if col is None or "Coin" not in df.columns:
        return "gamma book: delta column not found"
    d = df[df["Coin"] != "ALL"].copy()
    d["_v"] = pd.to_numeric(d[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
    d = d.dropna(subset=["_v"]).sort_values("_v", key=abs, ascending=False).head(12)
    total = df.loc[df["Coin"] == "ALL", col].astype(str).tolist()
    return ("gamma book net Δ$ (top |Δ|): " + ", ".join(f"{r['Coin']} {r['_v']:+,.0f}" for _, r in d.iterrows())
            + (f" · ALL {total[0]}" if total else ""))


def _section_universe(cfg: dict) -> tuple[str, dict[str, dict]]:
    def build():
        df, skipped, short = ls_bot.build_ranking(cfg, ls_bot.UNIVERSE)
        return df, skipped, short
    (df, skipped, short), age = cache.cached("ls_ranking:default", LS_TTL_S, build)
    stats: dict[str, dict] = {}
    for _, r in df.iterrows():
        stats[str(r["ticker"])] = {"zup": _f(r["Zup"]), "zdn": _f(r["Zdn"]), "lt": _f(r["LT6w"]),
                                   "st": _f(r["ST2w"]), "vol": _f(r["Vol2w"]), "regime": _f(r["Regime"]),
                                   "vsrsd": _f(r["VSRSD"]), "sig": str(r["Sig"] or "")}
    cols = ["#", "ticker", "Price", "LT6w", "ST2w", "Zup", "Zdn", "Vol2w", "Regime", "VSRSD", "Sig"]
    shown = df[cols].copy()
    for c in ("LT6w", "ST2w", "Regime"):
        shown[c] = shown[c].map(lambda v: "" if pd.isna(v) else f"{v:+.0f}" if c != "Regime" else f"{v:.1f}")
    for c in ("Zup", "Zdn"):
        shown[c] = shown[c].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
    shown["Vol2w"] = shown["Vol2w"].map(lambda v: "" if pd.isna(v) else f"{v * 100:.0f}%")
    shown["VSRSD"] = shown["VSRSD"].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}")
    shown["Price"] = shown["Price"].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
    text = (f"LS ranking ({cache.age_note(age)}; regime 1=max short … 5=max long; VS-RSD = $ resting delta "
            f"scaled by vol; Sig L/S = trend + level agree):\n" + shown.to_string(index=False))
    if skipped or short:
        text += f"\n(skipped: {', '.join(skipped) or '-'}; short history: {', '.join(short) or '-'})"
    return text, stats


def build_packet(desk: str, include_sentiment: bool, refresh: bool) -> str:
    cfg = load_config("data/binance.yaml")
    ttl = 0 if refresh else cache.DEFAULT_TTL_S
    jobs = {
        "macro": lambda: intel.macro_cached("global", ttl),
        "sentiment": (lambda: intel.sentiment_digest_cached(ttl)) if include_sentiment else (lambda: None),
        "market_pos": lambda: market.positioning_cached(0 if refresh else market.POSITIONING_TTL_S),
        "predictions": lambda: market.prediction_markets_cached(0 if refresh else market.PREDICTION_TTL_S),
        "positions": lambda: _section_positions(desk, cfg),
        "gamma": lambda: _section_gamma(desk),
        "universe": lambda: _section_universe(cfg),
    }
    results: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {name: pool.submit(fn) for name, fn in jobs.items()}
        for name, fut in futures.items():
            try:
                results[name] = fut.result()
            except Exception as exc:  # noqa: BLE001 - a section may fail; the packet must not
                results[name] = exc

    def txt(name: str, ok) -> str:
        r = results[name]
        return f"unavailable: {type(r).__name__}: {str(r)[:160]}" if isinstance(r, Exception) else ok(r)

    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    macro = txt("macro", lambda r: f"({cache.age_note(r[1])})\n{r[0][0][:2500]}")
    if include_sentiment:
        senti = txt("sentiment", lambda r: f"({cache.age_note(r[1])})\n{r[0][:2000]}")
    else:
        senti = "skipped by request"
    market_pos = txt("market_pos", lambda r: f"({cache.age_note(r[1])})\n{r[0]}")
    predictions = txt("predictions", lambda r: f"({cache.age_note(r[1])})\n{r[0]}")
    pos_text = txt("positions", lambda r: r[0])
    gamma = txt("gamma", lambda r: r)
    uni_text = txt("universe", lambda r: r[0])

    positions = results["positions"][1] if not isinstance(results["positions"], Exception) else []
    held_stats = results["positions"][2] if not isinstance(results["positions"], Exception) else {}
    uni_stats = results["universe"][1] if not isinstance(results["universe"], Exception) else {}
    stats = {**uni_stats, **{s: {**uni_stats.get(s, {}), **v} for s, v in held_stats.items()}}
    flagged = rules(positions, stats)
    rule_lines = [f"• {f['sym']} ({f['side']}{f', ${f['notional']:,.0f}' if f['held'] else ''}): " + "; ".join(f["flags"])
                  for f in flagged] or ["• nothing flagged by the rules"]

    return "\n".join([
        f"DESK VIEW · {desk} · {now}", RULE,
        "1) MACRO " + macro, RULE,
        "2) SENTIMENT " + senti, RULE,
        "2b) MARKET POSITIONING " + market_pos, RULE,
        "2c) PREDICTION MARKETS " + predictions, RULE,
        "3) POSITIONING\n" + pos_text, gamma, RULE,
        "4) UNIVERSE\n" + uni_text, RULE,
        f"5) RULES (z ≥ {STRETCH_Z:g} stretch; sign vs LT trend; size vs VS-RSD {SIZE_LO}-{SIZE_HI}×)\n"
        + "\n".join(rule_lines),
    ])


def _desk_view(ctx: Context, args: dict) -> ToolResult:
    return ToolResult(build_packet(ctx.desk, bool(args.get("include_sentiment", True)),
                                   bool(args.get("refresh", False))))


DESK_VIEW = ToolSpec(
    name="desk_view",
    description=(
        "The full evidence packet for an advisory answer, in the desk's reasoning order: 1) macro "
        "report, 2) X sentiment digest, 2b) market positioning (Fear & Greed, Binance retail vs "
        "top-trader long/short, funding, crowding flags), 2c) Polymarket odds (Fed, economy, crypto, "
        "geopolitics; 1-day movers), 3) our positioning (futures-book positions with JASON z and "
        "LT trend; gamma-book net deltas), 4) the LS universe ranking (trend, JASON z, vol, regime, "
        "vol-scaled resting-state delta VS-RSD, L/S signal), 5) deterministic rule flags: |z| ≥ 2 "
        "stretch read against our position, position against LT trend, size vs VS-RSD. Call this "
        "FIRST whenever a trader asks for ideas, a view, what to do, or whether the book is "
        "positioned right; then reason in that order and cite the section for every claim. "
        "include_sentiment: false skips the X digest (saves a paid call). refresh: true bypasses "
        "the 4h macro/sentiment caches (costs money; only if the trader asks for fresh). Takes "
        "30-90s. Never invent a number the packet marks unavailable."
    ),
    input_schema={"type": "object",
                  "properties": {"include_sentiment": {"type": "boolean"}, "refresh": {"type": "boolean"}},
                  "required": ["include_sentiment", "refresh"], "additionalProperties": False},
    risk=Risk.READ,
    handler=_desk_view,
    desks=frozenset({"mm3", "mm2", "treasury"}),
)

TOOLS: list[ToolSpec] = [DESK_VIEW]
