"""Client-facing execution report for one /jwap parent, built from its durable record.

One self-contained HTML page in the JWAP house style (Source Serif / IBM Plex, slate blue = the
JWAP and good for it, muted brown = bad, greys = the TWAP). The benchmark is a TWAP that buys the
same amount at every scheduled slice at the Binance spot 1-minute open of that minute, one tick
worse, paying the same fee rate the JWAP's fills paid. The per-unit difference splits into
placement (the opens at the JWAP's traded slices vs at all slices) plus the difference in
execution cost (JWAP paid vs its opens, against the TWAP's tick and fee), all in bps of the TWAP
price, so the parts add up.

Run ``python -m scripts.jwap_report <id> [out.html]`` for a local copy; trade_bot sends it on
``/jwap report <id>``; ``python -m scripts.jwap_report --publish`` (the jwap-report timer) rewrites
one page per OTC parent plus an index under ``results/jwap/``, which ``scripts.jwap_web`` serves.

MUST NOT: place, cancel or read orders, call any API other than Binance's public spot market data,
or show internal ids (order ids, record names, retries) -- the page is written for the client.
"""
from __future__ import annotations

import html
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scripts import jwap_record as jrec
from src.adapters.coinroutes.trade import TERMINAL
from src.common.paths import RESULTS_DIR
from src.data.fetching.binance.backfill import _get_json

_DAY = 86400
_SPOT = "https://api.binance.com"


def binance_opens(symbol: str, start: int, end: int) -> dict[int, float]:
    """Binance spot 1-minute opens keyed by the minute (epoch s), for ``start`` .. ``end``."""
    out: dict[int, float] = {}
    t = start * 1000
    while t <= end * 1000:
        rows = _get_json(_SPOT, "/api/v3/klines", {"symbol": symbol, "interval": "1m", "startTime": t,
                                                   "endTime": end * 1000, "limit": 1000})
        if not rows:
            break
        for r in rows:
            out[int(r[0]) // 1000] = float(r[1])
        t = int(rows[-1][0]) + 60_000
    return out


def binance_tick(symbol: str) -> float:
    info = _get_json(_SPOT, "/api/v3/exchangeInfo", {"symbol": symbol})
    pf = next(f for f in info["symbols"][0]["filters"] if f["filterType"] == "PRICE_FILTER")
    return float(pf["tickSize"])


# --------------------------------------------------------------------------- #
# numbers
# --------------------------------------------------------------------------- #
def _f(v) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def _harmonic(pairs: list[tuple[float, float]]) -> float | None:
    """Average price of buying ``notional`` at ``price`` for each pair: sum $ / sum units."""
    usd = sum(n for n, _p in pairs)
    units = sum(n / p for n, p in pairs if p > 0)
    return usd / units if units else None


def summarise(events: list[dict], now: float, opens: dict[int, float] | None = None,
              tick: float | None = None) -> dict:
    """Everything the page shows, from a parent's record events and the Binance 1-minute ``opens``
    (minute epoch -> open) with its ``tick``. Without opens the TWAP comparison is left empty. Pure."""
    created = next(e for e in events if e["kind"] == "created")
    p = created["params"]
    entry, every, n, child = int(p["entry"]), int(p["every"]), int(p["n"]), float(p["child"])
    sell = p["side"] == "sell"
    slices = {e["seq"]: e for e in events if e["kind"] == "slice"}
    placed = {e["cid"]: e for e in events if e["kind"] == "placed"}
    last: dict[str, dict] = {}
    for e in events:
        if e["kind"] == "fill":
            last[e["cid"]] = e["order"]
    stopped = any(e["kind"] == "stopped" for e in events)
    done = next((e for e in events if e["kind"] == "done"), None)

    kids = []                                   # one row per placed child
    for cid, pe in placed.items():
        o = last.get(cid) or pe.get("order") or {}
        qty, usd = _f(o.get("executed_qty")), abs(_f(o.get("net_consideration")))
        if not usd and qty:
            usd = qty * _f(o.get("avg_price"))
        status = str(o.get("order_status") or "")
        kids.append({"seq": pe["seq"], "due": entry + pe["seq"] * every, "qty": qty, "usd": usd,
                     "mid": (slices.get(pe["seq"]) or {}).get("mid"), "status": status,
                     "final": status in TERMINAL, "maker": _f(o.get("maker_quantity")),
                     "fee": _f(o.get("executed_fee_cost"))})

    notional = sum(k["usd"] for k in kids)
    fee_rate = sum(k["fee"] for k in kids) / notional if notional else 0.0   # the JWAP's own rate
    sgn = -1 if sell else 1                    # one tick and the fee always cost the trader
    opens = opens or {}

    def block(sl: list[dict], ks: list[dict]) -> dict:
        qty, usd = sum(k["qty"] for k in ks), sum(k["usd"] for k in ks)
        paid = usd / qty if qty else None
        bench = [opens[s["due"]] for s in sl if s["due"] in opens]
        o_avg = _harmonic([(child, o) for o in bench])
        twap = _harmonic([(child, (o + sgn * tick) * (1 + sgn * fee_rate)) for o in bench]) if tick else None
        chose = _harmonic([(k["usd"], opens[k["due"]]) for k in ks if k["usd"] and k["due"] in opens])
        out = {"slices": len(sl), "fired": sum(s["result"] == "fired" for s in sl),
               "decided": sum(s["result"] in ("fired", "gate_closed") for s in sl),
               "qty": qty, "usd": usd, "paid": paid, "twap": twap, "twap_open": o_avg, "chose": chose,
               "total": None, "placement": None, "execution": None, "exec_j": None, "exec_t": None}
        if paid and twap and chose:
            out["total"] = (paid - twap) / twap * 1e4
            out["placement"] = (chose - o_avg) / twap * 1e4
            out["exec_j"] = (paid - chose) / twap * 1e4
            out["exec_t"] = (twap - o_avg) / twap * 1e4
            out["execution"] = out["exec_j"] - out["exec_t"]
        return out

    allsl = [slices[k] for k in sorted(slices)]
    overall = block(allsl, kids)
    days = defaultdict(lambda: ([], []))
    for s in allsl:
        days[_day(s["due"])][0].append(s)
    for k in kids:
        days[_day(k["due"])][1].append(k)
    target = float(p["size"])
    unit_usd = p["unit"] == "usd"
    got = overall["usd"] if unit_usd else overall["qty"]
    rows, cum = [], 0.0
    for d in sorted(days):
        b = block(*days[d])
        cum += b["usd"] if unit_usd else b["qty"]
        rows.append({"day": d, **b, "cum": cum / target if target else 0.0})

    filled_kids = [k for k in kids if k["qty"]]
    working = [k for k in kids if not k["final"]]
    left = [k for k in kids if k["final"] and k["usd"] < child * 0.99]
    chart = {
        "t": [(s["due"] - entry) / 3600 for s in allsl],
        "mid": [s.get("mid") for s in allsl],
        "level": [s.get("level") for s in allsl],
        "state": [_state(s, sell, p["age"]) for s in allsl],
        "fills": [((k["due"] - entry) / 3600, k["usd"] if unit_usd else k["qty"]) for k in filled_kids],
    }
    return {
        "base": p["base"], "pair": p["pair"], "side": p["side"], "sell": sell, "target": target,
        "unit_usd": unit_usd, "child": child, "every": every, "n": n, "entry": entry,
        "end": entry + n * every, "now": now, "sigma": p["sigma"], "age": p["age"],
        "arrival": created.get("arrival_mid"), "got": got, "share": got / target if target else 0.0,
        "elapsed": len(allsl), "overall": overall, "days": rows, "today": rows[-1] if rows else None,
        "stopped": stopped, "done": done is not None,
        "maker_share": (sum(k["maker"] for k in kids) / overall["qty"]) if overall["qty"] else None,
        "fees": sum(k["fee"] for k in kids), "children": len(kids),
        "full": sum(1 for k in kids if k["final"] and k["usd"] >= child * 0.99),
        "partial": sum(1 for k in left if k["usd"] > 0), "empty": sum(1 for k in left if k["usd"] == 0),
        "working": len(working), "left_usd": sum(child - k["usd"] for k in left),
        "gate_now": _gate_now(allsl, p, sell), "chart": chart, "tick": tick, "fee_rate": fee_rate, "bench": bool(opens) and tick is not None,
        "no_open": sum(1 for s in allsl if s["due"] not in opens) if opens else 0,
    }


def _gate_now(allsl: list[dict], p: dict, sell: bool) -> dict | None:
    """The gate as the last priced slice saw it: price vs the level, and the pivot age vs the gate."""
    priced = [x for x in allsl if x.get("mid") and x.get("level")]
    if not priced:
        return None
    last = priced[-1]
    return {"due": last["due"], "mid": last["mid"], "level": last["level"], "days": last.get("days"),
            "age": p["age"], "price_ok": last["mid"] >= last["level"] if sell else last["mid"] <= last["level"],
            "age_ok": last.get("days") is not None and last["days"] >= p["age"]}


def _day(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%d")


def _state(s: dict, sell: bool, age: float) -> int:
    """2 = the slice traded, 1 = the price was past the level but the highs/lows were too young."""
    if s["result"] == "fired":
        return 2
    mid, lvl, days = s.get("mid"), s.get("level"), s.get("days")
    if mid is None or lvl is None:
        return 0
    past = mid >= lvl if sell else mid <= lvl
    return 1 if past and (days is None or days < age) else 0


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def bps(v: float | None) -> str:
    if v is None:
        return '<span class="muted">—</span>'
    r = round(v)
    return f"{'+' if r > 0 else '−' if r < 0 else ''}{abs(r):,}"


def usd(v: float) -> str:
    return f"${v:,.0f}"


def usd_short(v: float) -> str:
    if v >= 1e6:
        return f"${v / 1e6:.{0 if v % 1e6 == 0 else 1}f}M"
    if v >= 1e3:
        return f"${v / 1e3:,.{0 if v % 1e3 == 0 else 1}f}k"
    return usd(v)


def px(v: float | None) -> str:
    if not v:
        return '<span class="muted">—</span>'
    digits = len(str(int(abs(v)))) if abs(v) >= 1 else 0
    return f"{v:,.{max(2, 5 - digits)}f}"


def pct(v: float, dp: int = 2) -> str:
    return f"{100 * v:.{dp}f}%"


def _good(v: float | None, sell: bool) -> str:
    if v is None or round(v) == 0:
        return ""
    return "good" if (v > 0) == sell else "bad"


# --------------------------------------------------------------------------- #
# chart (inline SVG, house grammar)
# --------------------------------------------------------------------------- #
def chart_svg(s: dict) -> str:
    c, arr = s["chart"], s["arrival"]
    span = (s["end"] - s["entry"]) / 3600
    now_h = min(span, (s["now"] - s["entry"]) / 3600)
    W, H1, H2, L, R, T = 1140, 344, 140, 54, 16, 16
    x = lambda h: L + h / span * (W - L - R)  # noqa: E731
    mids, lvls = c["mid"], c["level"]
    avg_t, avg_j = s["overall"]["twap"], s["overall"]["paid"]
    vals = [v for v in mids + lvls + [avg_t, avg_j, arr] if v is not None]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.1 or abs(hi) * 0.01 or 1
    y = lambda v: T + (hi + pad - v) / (hi - lo + 2 * pad) * (H1 - T - 76)  # noqa: E731
    step_h = s["every"] / 3600
    g = []

    def runs(want) -> list[tuple[float, float]]:
        out, start = [], None
        for t, st in zip(c["t"], c["state"]):
            if want(st) and start is None:
                start = t
            if not want(st) and start is not None:
                out.append((start, t))
                start = None
        if start is not None:
            out.append((start, c["t"][-1] + step_h))
        return out

    for a, b in runs(lambda st: st == 2):
        g.append(f'<rect x="{x(a):.1f}" y="{T}" width="{max(1, x(b) - x(a)):.1f}" height="{H1 - T - 64}" '
                 f'style="fill:var(--up)" fill-opacity=".18"/>')
    raw = (hi - lo + 2 * pad) / 4 or 1
    mag = 10 ** math.floor(math.log10(raw))
    stp = next(k * mag for k in (1, 2, 2.5, 5, 10) if k * mag >= raw)
    dp = max(0, -math.floor(math.log10(stp) + 1e-9) + (1 if abs(stp / mag - 2.5) < 1e-9 else 0))
    v = math.ceil((lo - pad) / stp) * stp
    while v <= hi + pad + 1e-9:
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" style="stroke:var(--grid)"/>'
                 f'<text x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.{dp}f}</text>')
        v += stp
    if arr:
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{y(arr):.1f}" y2="{y(arr):.1f}" style="stroke:var(--hair)"/>'
                 f'<text x="{W - R - 4}" y="{y(arr) + 13:.1f}" text-anchor="end">arrival {px(arr)}</text>')
    dstep = 7 if span > 240 else 1
    d = 0
    while d * 24 <= span + 1e-9:
        xx = x(d * 24)
        g.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{T}" y2="{H1 - 64}" style="stroke:var(--grid)"/>'
                 f'<text x="{xx + 3:.1f}" y="{H1 - 30}">day {d}</text>')
        d += dstep
    if now_h < span:
        g.append(f'<line x1="{x(now_h):.1f}" x2="{x(now_h):.1f}" y1="{T}" y2="{H1 - 64}" '
                 f'style="stroke:var(--ink2)" stroke-dasharray="3 3"/>'
                 f'<text x="{x(now_h) + 3:.1f}" y="{T + 10}" style="fill:var(--ink2)">now</text>')

    def path(series) -> str:
        out, pen = [], False
        for t, v in zip(c["t"], series):
            if v is None:
                pen = False
                continue
            out.append(f"{'L' if pen else 'M'}{x(t):.1f} {y(v):.1f}")
            pen = True
        return "".join(out)

    lcol = "--down" if s["sell"] else "--up"
    g.append(f'<path d="{path(lvls)}" fill="none" style="stroke:var({lcol})" stroke-width="1.4" '
             f'stroke-dasharray="7 4" opacity=".95"/>')
    g.append(f'<path d="{path(mids)}" fill="none" style="stroke:var(--ink)" stroke-width="1.1" opacity=".8"/>')
    for val, col, lab, fr in ((avg_t, "--neutral2", "TWAP avg", .28), (avg_j, "--up", "JWAP avg paid", .88)):
        if val is None:
            continue
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{y(val):.1f}" y2="{y(val):.1f}" style="stroke:var({col})" '
                 f'stroke-dasharray="6 4" opacity=".9"/><text x="{L + (W - L - R) * fr:.1f}" y="{y(val) - 5:.1f}" '
                 f'text-anchor="middle" style="fill:var({col})">{lab} {px(val)}</text>')
    hourly: dict[int, float] = defaultdict(float)
    for t, q in c["fills"]:
        hourly[int(t)] += q
    mid_at = {}
    for t, m in zip(c["t"], mids):
        if m is not None:
            mid_at.setdefault(int(t), []).append(m)
    r0 = max(2.5, min(7, 1500 / span))
    for h, q in sorted(hourly.items()):
        share = q / s["target"]
        ms = mid_at.get(h)
        if not ms:
            continue
        m = sum(ms) / len(ms)
        amt = usd(q) if s["unit_usd"] else f"{q:,.4g} {s['base']}"
        tip = (f"day {h / 24:.1f} · {'sold' if s['sell'] else 'bought'} {amt} that hour · "
               f"mid {px(m)}")
        g.append(f'<circle data-tip="{html.escape(tip)}" cx="{x(h + .5):.1f}" cy="{y(m):.1f}" '
                 f'r="{r0 * math.sqrt(share * span):.2f}" style="fill:var(--up);stroke:var(--paper)" stroke-width=".5"/>')
    bw = max(1.0, x(step_h) - x(0))
    for t, st in zip(c["t"], c["state"]):
        if st:
            g.append(f'<rect x="{x(t):.1f}" y="{H1 - 58}" width="{bw:.2f}" height="7" '
                     f'style="fill:var({"--up" if st == 2 else "--muted"})" fill-opacity="{.9 if st == 2 else .35}"/>')
    # lower panel: cumulative share of the target
    CB = H1 + H2 - 30
    ys = lambda v: CB - v * (H2 - 60)  # noqa: E731
    word = "sold" if s["sell"] else "bought"
    g.append(f'<text x="{L}" y="{H1 + 22}">cumulative share of the target {word}</text>')
    for v in (0, .25, .5, .75, 1):
        col = "--hair" if v in (0, 1) else "--grid"
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" style="stroke:var({col})"/>'
                 f'<text x="{L - 6}" y="{ys(v) + 4:.1f}" text-anchor="end">{100 * v:.0f}%</text>')
    g.append(f'<path d="M{x(0):.1f} {ys(0):.1f}L{x(span):.1f} {ys(1):.1f}" fill="none" style="stroke:var(--neutral2)" '
             f'stroke-width="1.6"/><text x="{L + (W - L - R) * .45:.1f}" y="{ys(1) - 7:.1f}" text-anchor="middle" '
             f'style="fill:var(--neutral2)">TWAP 100%</text>')
    total, dd = 0.0, [f"M{x(0):.1f} {ys(0):.1f}"]
    for t, q in sorted(c["fills"]):
        dd.append(f"L{x(t):.2f} {ys(total):.2f}")
        total += q / s["target"]
        dd.append(f"L{x(t):.2f} {ys(total):.2f}")
    dd.append(f"L{x(now_h):.2f} {ys(total):.2f}")
    line = "".join(dd)
    g.append(f'<path d="{line} L{x(now_h):.1f} {ys(0):.1f} Z" style="fill:var(--up)" fill-opacity=".14" stroke="none"/>'
             f'<path d="{line}" fill="none" style="stroke:var(--up)" stroke-width="2.4"/>'
             f'<circle cx="{x(now_h):.1f}" cy="{ys(total):.1f}" r="3.2" style="fill:var(--up)"/>'
             f'<text x="{x(now_h) + 8:.1f}" y="{ys(total) - 7:.1f}" style="fill:var(--up)">JWAP {pct(total, 1)}</text>')
    return f'<svg viewBox="0 0 {W} {H1 + H2}" width="100%" role="img">{"".join(g)}</svg>'


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
_CSS = """
:root{--paper:#FAFAFA;--card:#F4F4F2;--ink:#1A1A19;--ink2:#4A4A48;--muted:#8A8A86;--hair:#DCDCD8;--grid:#EAEAE6;--up:#54697f;--down:#7d5f52;--neutral2:#9d9d9d;--serif:"Source Serif 4",Georgia,serif;--sans:"IBM Plex Sans",Arial,sans-serif;--mono:"IBM Plex Mono",Consolas,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--paper:#15171a;--card:#1d2024;--ink:#ecebe6;--ink2:#c3c2bc;--muted:#8d8d88;--hair:#33363b;--grid:#272a2f;--up:#afbbc8;--down:#c7b5ae;--neutral2:#818181}}
:root[data-theme="dark"]{--paper:#15171a;--card:#1d2024;--ink:#ecebe6;--ink2:#c3c2bc;--muted:#8d8d88;--hair:#33363b;--grid:#272a2f;--up:#afbbc8;--down:#c7b5ae;--neutral2:#818181}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 var(--sans)}
.wrap{max-width:1180px;margin:0 auto;padding:36px 16px 64px}.eyebrow{font:500 11px/1.4 var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
h1{font:600 clamp(28px,4vw,40px)/1.1 var(--serif);margin:8px 0 10px;text-wrap:balance}h2{font:600 22px/1.25 var(--serif);margin:0 0 8px}
p{color:var(--ink2);margin:0 0 10px}section{margin-top:44px;padding-top:22px;border-top:1px solid var(--hair)}.lede{font-size:16.5px;color:var(--ink)}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0}@media (max-width:1000px){.tiles{grid-template-columns:repeat(2,1fr)}}@media (max-width:620px){.tiles{grid-template-columns:1fr}}
.tiles.t2{grid-template-columns:repeat(2,1fr)}@media (max-width:620px){.tiles.t2{grid-template-columns:1fr}}
.tile{padding:14px 16px;border-top:2px solid var(--neutral2);background:var(--card)}.tile.hero{border-top-width:3px}
.tile .k{font:500 10.5px/1.4 var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}.tile .n{font:500 30px/1.1 var(--mono);color:var(--ink);margin:6px 0 2px;font-variant-numeric:tabular-nums}.tile .n small{font-size:12px;color:var(--muted);margin-left:3px}.tile .s{font-size:12.5px;color:var(--ink2)}
.good{color:var(--up)!important}.bad{color:var(--down)!important}.muted{color:var(--muted)}
.reshead{margin:6px 0 16px}.reshead .big{font:600 clamp(24px,3.2vw,34px)/1.2 var(--serif);color:var(--ink);text-wrap:balance}.reshead .sub{font-size:14.5px;color:var(--ink2);margin-top:6px}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}th,td{padding:9px 10px;text-align:right;border-bottom:1px solid var(--hair);white-space:nowrap;font-family:var(--mono)}th{font:500 10.5px/1.4 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}td:first-child,th:first-child{text-align:left}
.scroll{overflow-x:auto}.note{border-left:2px solid var(--hair);padding:2px 0 2px 14px;color:var(--ink2);font-size:14px;margin-top:12px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;color:var(--ink2);margin:6px 0 4px}.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px;background:var(--c)}.legend span.rect::before{border-radius:2px;width:16px;height:11px;opacity:.55}.legend span.dash::before{border-radius:0;height:0;width:16px;border-top:2px dashed var(--c);background:none;vertical-align:3px}
.chart{background:var(--card);padding:10px 12px;margin-top:8px}svg text{font:11px var(--mono);fill:var(--muted)}
.tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--paper);font:12px/1.4 var(--mono);padding:6px 8px;border-radius:4px;opacity:0;z-index:5;white-space:nowrap}
.foot{font-size:12.5px;color:var(--muted);border-top:1px solid var(--hair);padding-top:14px;margin-top:44px}
"""


def _tile(k: str, n: str, sub: str, cls: str = "", hero: bool = False) -> str:
    return (f'<div class="tile{" hero" if hero else ""}"><div class="k">{k}</div>'
            f'<div class="n {cls}">{n}</div><div class="s">{sub}</div></div>')


def render(s: dict) -> str:
    o, sell, base = s["overall"], s["sell"], html.escape(s["base"])
    verb, verbed = ("sell", "sold") if sell else ("buy", "bought")
    cheap = "higher" if sell else "cheaper"
    dear = "lower" if sell else "dearer"
    amount = (lambda v: usd(v)) if s["unit_usd"] else (lambda v: f"{v:,.4g} {base}")
    day_n = int((min(s["now"], s["end"]) - s["entry"]) // _DAY) + 1
    days_total = math.ceil((s["end"] - s["entry"]) / _DAY)
    asof = datetime.fromtimestamp(s["now"], timezone.utc).strftime("%Y-%m-%d %H:%M")
    start = datetime.fromtimestamp(s["entry"], timezone.utc).strftime("%Y-%m-%d %H:%M")
    status = "complete" if s["done"] and not s["stopped"] else "stopped" if s["stopped"] else f"day {day_n} of {days_total}"
    side_word = "above the weighted recent lows" if sell else "below the weighted recent highs"
    hl = "lows" if sell else "highs"
    lede = (f"A {usd_short(s['target']) if s['unit_usd'] else amount(s['target'])} {base} {verb} order over "
            f"{days_total} days from {start} UTC, in {amount(s['child'])} slices every {s['every'] // 60} minutes "
            f"({s['n']:,} slices). A slice trades only when the price is at least {s['sigma']:g} standard "
            f"deviations {side_word}, and those {hl} are at least {s['age']:g} days old. "
            f"<b>No catch-up:</b> a slice that does not trade is not made up later, so the amount {verbed} can "
            f"end below the target.")
    total = o["total"]
    if o["paid"] is None:
        head = f"Nothing {verbed} yet"
        sub = f"{s['elapsed']:,} of {s['n']:,} slices have come due; the price condition has not been met."
    elif total is None:
        head = (f"{verbed.capitalize()} {amount(s['got'])} ({pct(s['share'])} of the target) at "
                f"{px(o['paid'])}")
        sub = "The TWAP comparison is unavailable: the reference prices could not be loaded."
    else:
        signed = -total if not sell else total
        word = cheap if signed > 0 else dear
        head = (f"{verbed.capitalize()} {amount(s['got'])} ({pct(s['share'])} of the target) at "
                f"{px(o['paid'])}, <span class=\"{_good(total, sell)}\">{abs(total) / 100:.2f}% {word}</span> "
                f"than a TWAP over the same period")
        sub = (f"The TWAP {verbs(sell)} the same amount at every one of the {s['elapsed']:,} slices so far, "
               f"at the 1-minute open plus one tick and the same fees. It would hold "
               f"{pct(s['elapsed'] / s['n'])} of the target by now.")
    arr_diff = (o["paid"] / s["arrival"] - 1) * 1e4 if o["paid"] and s["arrival"] else None
    tiles1 = "".join([
        _tile(f"{verbed} so far", usd_short(s["got"]) if s["unit_usd"] else amount(s["got"]),
              f"{pct(s['share'])} of the {usd_short(s['target']) if s['unit_usd'] else amount(s['target'])} target",
              hero=True),
        _tile(f"{base} {verbed}", f"{o['qty']:,.3f}", f"in {s['children']:,} slices"),
        _tile("Average price", px(o["paid"]),
              f"arrival {px(s['arrival'])} · {bps(arr_diff)} bps from arrival"),
        _tile("Time used", pct((min(s["now"], s["end"]) - s["entry"]) / (s["end"] - s["entry"]), 1),
              f"{s['elapsed']:,} of {s['n']:,} slices due · ends "
              f"{datetime.fromtimestamp(s['end'], timezone.utc):%Y-%m-%d %H:%M} UTC"),
    ])
    tiles2 = "".join([
        _tile("Price difference per unit", f"{bps(total)}<small>bps</small>",
              "average paid vs the TWAP's all-in average, in bps of the TWAP price", _good(total, sell), hero=True),
        _tile("Placement", f"{bps(o['placement'])}<small>bps</small>",
              "the opens at the slices that traded vs at every slice", _good(o["placement"], sell)),
        _tile("Execution", f"{bps(o['exec_j'])} vs {bps(o['exec_t'])}<small>bps</small>",
              "cost against the open: this order vs the TWAP (one tick and fees)", _good(o["execution"], sell)),
        _tile("Slices traded", pct(o["fired"] / o["decided"], 1) if o["decided"] else "—",
              f"{o['fired']:,} of {o['decided']:,} slices met the condition"),
    ])
    t = s["today"]
    tiles3 = "".join([
        _tile(f"{verbed} today", amount(t["usd"] if s["unit_usd"] else t["qty"]) if t else "—",
              f"{t['fired']:,} of {t['slices']:,} slices traded" if t else ""),
        _tile("Today's average", px(t["paid"]) if t else "—", f"TWAP average today {px(t['twap']) if t else '—'}"),
        _tile("Today vs TWAP", f"{bps(t['total'])}<small>bps</small>" if t else "—", "per unit, in bps of the TWAP price",
              _good(t["total"], sell) if t else ""),
        _tile("Condition met today", pct(t["fired"] / t["decided"], 1) if t and t["decided"] else "—",
              "share of today's slices that traded"),
    ]) if t else ""
    gn = s["gate_now"]
    if gn:
        when = datetime.fromtimestamp(gn["due"], timezone.utc).strftime("%Y-%m-%d %H:%M")
        tiles_g = "".join([
            _tile("Price vs JASON level", f"{px(gn['mid'])} <small>vs {px(gn['level'])}</small>",
                  f"{'met' if gn['price_ok'] else 'not met'} · the price must be at or "
                  f"{'above' if sell else 'below'} the level · {when} UTC",
                  "good" if gn["price_ok"] else "", hero=True),
            _tile("Age vs gate", (f"{gn['days']:.1f} <small>vs {gn['age']:g} days</small>"
                                  if gn["days"] is not None else "—"),
                  f"{'met' if gn['age_ok'] else 'not met'} · the {hl} must be at least {gn['age']:g} days old",
                  "good" if gn["age_ok"] else "", hero=True),
        ])
        gate_sec = f'<section><h2>2 · JASON Gate</h2><div class="tiles t2">{tiles_g}</div></section>'
    else:
        gate_sec = ""
    rows = "".join(
        f"<tr><td>{r['day']}</td><td>{r['slices']:,}</td><td>{r['fired']:,}</td>"
        f"<td>{amount(r['usd'] if s['unit_usd'] else r['qty'])}</td><td>{r['qty']:,.3f}</td><td>{px(r['paid'])}</td>"
        f"<td>{px(r['twap'])}</td><td class=\"{_good(r['total'], sell)}\">{bps(r['total'])}</td>"
        f"<td>{pct(r['cum'])}</td></tr>" for r in s["days"])
    exec_note = (f"Every slice is a limit order placed on the consolidated book at most 1% through the best price "
                 f"and cancelled if still working after 5 minutes; what it did not fill is not re-sent. "
                 f"{s['full']:,} slices filled in full, {s['partial']:,} partly, {s['empty']:,} not at all"
                 + (f", {s['working']:,} still working" if s["working"] else "") + ".")
    tiles4 = "".join([
        _tile("Resting fills", pct(s["maker_share"], 1) if s["maker_share"] is not None else "—",
              f"share of the {base} {verbed} that filled as a resting (maker) order"),
        _tile("Fees", usd(s["fees"]) if s["fees"] else "$0", "exchange fees on the fills"),
        _tile("Not filled", amount(s["left_usd"]), "on slices that traded but did not fill in full"),
        _tile("Execution cost", f"{bps(o['exec_j'])}<small>bps</small>",
              "price paid vs the 1-minute open at each traded slice", _good(o["exec_j"], sell)),
    ])
    legend = ('<div class="legend"><span style="--c:var(--ink)">price</span>'
              f'<span class="dash" style="--c:var(--{"down" if sell else "up"})">the price condition level</span>'
              '<span class="rect" style="--c:var(--up)">slices traded</span>'
              f'<span style="--c:var(--up)">{verbed} that hour (area = amount)</span>'
              '<span class="dash" style="--c:var(--neutral2)">TWAP average, all-in</span></div>')
    gen = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    n0 = 4 if gate_sec else 3              # number of the chart section; later ones follow it
    sym = html.escape(f"{s['base']}USDT")
    foot = ("Chart prices are the consolidated mid on the execution venues, read at each slice; fills are the "
            "executed quantities and amounts reported for each order. ")
    if s["bench"]:
        foot += (f"The TWAP {verbs(sell)} the same amount at every scheduled slice at the Binance spot {sym} "
                 f"1-minute open of that minute, one tick ({s['tick']:g}) worse, with the same fee rate as this "
                 f"order's fills ({s['fee_rate'] * 1e4:.1f} bps). Placement compares the same Binance opens for "
                 f"both." + (f" {s['no_open']:,} slices without a Binance candle are left out." if s["no_open"] else ""))
    else:
        foot += "The TWAP comparison could not be computed: the Binance reference prices were unavailable."
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{base} Execution Report</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap">
<style>{_CSS}</style></head><body><div class="wrap">
<div class="eyebrow">Execution report · {html.escape(s['pair'])} · as of {asof} UTC</div>
<h1>{base} {verb} — {status}</h1>
<p class="lede">{lede}</p>
<section style="margin-top:28px"><h2>1 · Where the order stands</h2>
<div class="reshead"><div class="big">{head}</div><div class="sub">{sub}</div></div>
<div class="tiles">{tiles1}</div></section>
{gate_sec}
<section><h2>{3 if gate_sec else 2} · Price against the TWAP</h2>
<div class="tiles">{tiles2}</div>
<p class="note">Per unit, {"positive" if sell else "negative"} is better: it means the order {verbed} {cheap} than the TWAP. The difference is placement (when the order traded) plus the difference in execution cost (what this order paid against the 1-minute open, compared with the TWAP's one tick and fees); both are in bps of the TWAP price, so they add up. Always read it with the share of the target {verbed}.</p></section>
<section><h2>{4 if gate_sec else 3} · Price, condition level and fills</h2>
{legend}<div class="chart">{chart_svg(s)}</div>
<p class="note">Black: the mid at each slice; the hairline is the arrival price {px(s['arrival'])}. Dashed: the level the price must reach for a slice to trade. Shading: slices that traded. Strip under the chart: blue where a slice traded, grey where the price met the level but the {hl} were too young. Lower panel: share of the target {verbed}, against the TWAP's even schedule.</p></section>
{f'<section><h2>{n0 + 1} · Today ({t["day"]}, UTC)</h2><div class="tiles">{tiles3}</div></section>' if t else ''}
<section><h2>{n0 + (2 if t else 1)} · By day</h2><div class="scroll"><table><thead><tr><th>Day (UTC)</th><th>Slices due</th><th>Traded</th><th>{verbed.capitalize()}</th><th>{base}</th><th>Average</th><th>TWAP average</th><th>vs TWAP, bps</th><th>Cumulative</th></tr></thead><tbody>{rows}</tbody></table></div></section>
<section><h2>{n0 + (3 if t else 2)} · Execution</h2><div class="tiles">{tiles4}</div><p class="note">{exec_note}</p></section>
<div class="foot">{foot} Generated {gen} UTC.</div>
<div class="tip" id="tip"></div></div>
<script>(function(){{const tip=document.getElementById('tip');document.addEventListener('mousemove',e=>{{const t=e.target.closest&&e.target.closest('[data-tip]');if(!t){{tip.style.opacity=0;return}}tip.textContent=t.dataset.tip;tip.style.opacity=1;tip.style.left=Math.min(e.clientX+12,innerWidth-tip.offsetWidth-8)+'px';tip.style.top=(e.clientY+16)+'px'}})}})();</script>
</body></html>"""


def verbs(sell: bool) -> str:
    return "sells" if sell else "buys"


def build(pid: int, now: float | None = None) -> tuple[bytes, str, dict]:
    """(html bytes, filename, summary) for parent ``pid``. Raises FileNotFoundError if no record.
    If Binance's prices cannot be loaded the page says the TWAP comparison is unavailable."""
    now = time.time() if now is None else now
    events = jrec.load(pid)
    p = next(e for e in events if e["kind"] == "created")["params"]
    sym = f"{p['base']}USDT"
    try:
        end = min(int(now) // 60 * 60, int(p["entry"]) + int(p["n"]) * int(p["every"]))
        opens, tick = binance_opens(sym, int(p["entry"]), end), binance_tick(sym)
    except Exception:  # noqa: BLE001 - the page states the benchmark is missing; never guess one
        opens, tick = None, None
    s = summarise(events, now, opens, tick)
    name = f"{s['base']}_execution_{datetime.fromtimestamp(now, timezone.utc):%Y-%m-%d_%H%M}Z.html"
    return render(s).encode("utf-8"), name, s


PUBLISH_DIR = RESULTS_DIR / "jwap"


def _write_atomic(path: Path, data: bytes) -> None:
    """Write next to ``path`` and rename over it, so a reader never sees half a page."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def publish(desk: str = "otc", out_dir: Path | None = None, now: float | None = None) -> list[dict]:
    """Rewrite ``<id>.html`` for every ``desk`` parent, and ``index.html`` listing them, newest
    first. A finished parent whose page was written after it finished is not rebuilt."""
    now = time.time() if now is None else now
    out = out_dir or PUBLISH_DIR
    out.mkdir(parents=True, exist_ok=True)
    root = jrec.RECORD_DIR
    ids = sorted((int(f.stem) for f in root.glob("*.jsonl") if f.stem.isdigit()), reverse=True) \
        if root.exists() else []
    rows = []
    for pid in ids:
        events = jrec.load(pid)
        created = next((e for e in events if e["kind"] == "created"), None)
        if created is None or created["params"]["desk"] != desk:
            continue
        page = out / f"{pid}.html"
        done = next((e for e in events if e["kind"] == "done"), None)
        stale = done is None or not page.exists() or page.stat().st_mtime < datetime.fromisoformat(
            done["ts"]).timestamp()
        if stale:
            data, _name, summ = build(pid, now)
            _write_atomic(page, data)
        else:
            summ = summarise(events, now)
        rows.append({"id": pid, "s": summ, "updated": page.stat().st_mtime})
    _write_atomic(out / "index.html", render_index(rows, desk, now).encode("utf-8"))
    return rows


def render_index(rows: list[dict], desk: str, now: float) -> str:
    when = lambda e: datetime.fromtimestamp(e, timezone.utc).strftime("%Y-%m-%d %H:%M")  # noqa: E731
    body = "".join(
        f'<tr><td><a href="{r["id"]}.html">#{r["id"]}</a></td><td>{html.escape(r["s"]["base"])}</td>'
        f'<td>{r["s"]["side"]}</td><td>{when(r["s"]["entry"])}</td>'
        f'<td>{"done" if r["s"]["done"] and not r["s"]["stopped"] else "stopped" if r["s"]["stopped"] else "running"}</td>'
        f'<td>{usd(r["s"]["got"]) if r["s"]["unit_usd"] else r["s"]["got"]}</td><td>{pct(r["s"]["share"])}</td>'
        f'<td>{when(r["updated"])}</td></tr>' for r in rows) or '<tr><td colspan="8">No orders yet.</td></tr>'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="300"><title>Execution Reports</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap">
<style>{_CSS}a{{color:var(--up)}}</style></head><body><div class="wrap">
<div class="eyebrow">Execution reports · {html.escape(desk.upper())} desk · as of {when(now)} UTC</div>
<h1>JWAP execution reports</h1>
<p class="lede">One page per order, rebuilt every 5 minutes from the order's record. Times are UTC.</p>
<section style="margin-top:28px"><div class="scroll"><table><thead><tr><th>Order</th><th>Token</th><th>Side</th>
<th>Started</th><th>Status</th><th>Done so far</th><th>Of target</th><th>Page updated</th></tr></thead>
<tbody>{body}</tbody></table></div></section></div></body></html>"""


if __name__ == "__main__":
    if sys.argv[1:2] == ["--publish"]:
        published = publish()
        print(f"published {len(published)} report(s) -> {PUBLISH_DIR}")
        sys.exit(0)
    data, fname, summ = build(int(sys.argv[1]))
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(fname)
    out.write_bytes(data)
    print(json.dumps({k: summ[k] for k in ("got", "share", "elapsed", "n")}), "->", out)
