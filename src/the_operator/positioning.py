"""Market positioning snapshot: crowd mood (Fear & Greed) and Binance perp positioning (retail
account long/short, top-trader position long/short, funding) for a few majors, each read against
its own last 30 days. Free public endpoints, no key. Flags are fixed thresholds, no model.

Funding comes live from Binance, not the lake: the lake's funding is not kept current.

MUST NOT: call Telegram, read or write the data lake, place/propose a trade or a size, or call
an endpoint that needs a key.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import requests

from src.common.config import load_config

FNG_URL = "https://api.alternative.me/fng/"
SYMBOLS = ("BTC", "ETH", "SOL", "XRP", "DOGE")
DAYS = 30
HI, LO = 80, 20                  # percentile bands that count as stretched vs the last 30 days
GREED, FEAR = 75, 25             # Fear & Greed extremes


def _get(url: str, params: dict) -> object:
    r = requests.get(url, params=params, headers={"User-Agent": "operator-positioning"}, timeout=20)
    r.raise_for_status()
    return r.json()


def pctile(hist: list[float], now: float) -> int:
    """Share of ``hist`` strictly below ``now``, 0-100 (ties count half)."""
    if not hist:
        return 50
    below = sum(v < now for v in hist) + 0.5 * sum(v == now for v in hist)
    return round(100 * below / len(hist))


def daily_funding(settlements: list[tuple[datetime, float]], now: datetime) -> tuple[float, list[float]]:
    """``(last 24h sum, per-day sums of the complete prior UTC days)``. Venue intervals are
    dynamic, so funding is summed per day, never annualized from one rate."""
    last24 = sum(r for ts, r in settlements if ts > now - timedelta(hours=24))
    by_day: dict = {}
    for ts, r in settlements:
        if ts.date() < now.date():
            by_day[ts.date()] = by_day.get(ts.date(), 0.0) + r
    return last24, [by_day[d] for d in sorted(by_day)]


def fetch() -> dict:
    """Raw inputs, oldest first: ``{"fng": [{value, label}], "symbols": {SYM: {retail, top,
    funding}}}``. One symbol failing is recorded as ``{"error": ...}``, not raised."""
    rest = load_config("data/binance.yaml")["markets"]["perps"]["rest_url"]
    fng = _get(FNG_URL, {"limit": DAYS + 1})["data"][::-1]
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=DAYS + 1)).timestamp() * 1000)
    symbols: dict[str, dict] = {}
    for sym in SYMBOLS:
        pair = f"{sym}USDT"
        try:
            ratio = {"symbol": pair, "period": "1d", "limit": DAYS}
            retail = _get(f"{rest}/futures/data/globalLongShortAccountRatio", ratio)
            top = _get(f"{rest}/futures/data/topLongShortPositionRatio", ratio)
            fund = _get(f"{rest}/fapi/v1/fundingRate", {"symbol": pair, "startTime": start_ms, "limit": 1000})
        except (requests.exceptions.RequestException, ValueError) as e:
            symbols[sym] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
            continue
        symbols[sym] = {
            "retail": [float(r["longShortRatio"]) for r in sorted(retail, key=lambda r: r["timestamp"])],
            "top": [float(r["longShortRatio"]) for r in sorted(top, key=lambda r: r["timestamp"])],
            "funding": [(datetime.fromtimestamp(f["fundingTime"] / 1000, timezone.utc), float(f["fundingRate"]))
                        for f in fund],
        }
    return {"fng": [{"value": int(d["value"]), "label": d["value_classification"]} for d in fng],
            "symbols": symbols}


def summarize(raw: dict, now: datetime) -> dict:
    """Latest value and its percentile vs the prior days, per series, plus the flags."""
    fng = raw["fng"]
    vals = [d["value"] for d in fng]
    out: dict = {"fng": None, "symbols": {}, "flags": []}
    if vals:
        out["fng"] = {"now": vals[-1], "label": fng[-1]["label"],
                      "yday": vals[-2] if len(vals) > 1 else None,
                      "avg7": round(sum(vals[-7:]) / len(vals[-7:])),
                      "lo": min(vals[-DAYS:]), "hi": max(vals[-DAYS:])}
        if vals[-1] >= GREED:
            out["flags"].append(f"Fear & Greed {vals[-1]}: extreme greed")
        elif vals[-1] <= FEAR:
            out["flags"].append(f"Fear & Greed {vals[-1]}: extreme fear")

    for sym, d in raw["symbols"].items():
        if "error" in d or not d["retail"] or not d["top"]:
            out["symbols"][sym] = {"error": d.get("error", "no data")}
            continue
        f24, fdays = daily_funding(d["funding"], now)
        s = {"retail": d["retail"][-1], "retail_p": pctile(d["retail"][:-1], d["retail"][-1]),
             "top": d["top"][-1], "top_p": pctile(d["top"][:-1], d["top"][-1]),
             "fund_bp": f24 * 1e4, "fund_p": pctile(fdays, f24)}
        out["symbols"][sym] = s
        if s["retail_p"] >= HI and s["top_p"] <= LO:
            out["flags"].append(f"{sym}: retail long at a monthly high while top traders cut")
        elif s["retail_p"] <= LO and s["top_p"] >= HI:
            out["flags"].append(f"{sym}: retail short at a monthly low while top traders add longs")
        if s["retail_p"] >= HI and s["fund_p"] >= HI and f24 > 0:
            out["flags"].append(f"{sym}: crowded long (retail long and funding both at monthly highs)")
        elif s["retail_p"] <= LO and s["fund_p"] <= LO and f24 < 0:
            out["flags"].append(f"{sym}: crowded short (retail short and funding both at monthly lows)")
    return out


def render(summary: dict, now: datetime) -> str:
    lines = [f"POSITIONING · {now:%d %b %H:%M} UTC"]
    f = summary["fng"]
    if f:
        lines.append(f"Fear & Greed {f['now']} {f['label']} (yday {f['yday']} · 7d avg {f['avg7']} · "
                     f"{DAYS}d {f['lo']}–{f['hi']})")
    lines.append("Binance perps, pNN = percentile vs the last 30 days:")
    lines.append(f"{'':6}{'retail L/S':<14}{'top traders L/S':<18}funding 24h")
    for sym, s in summary["symbols"].items():
        if "error" in s:
            lines.append(f"{sym:<6}unavailable: {s['error']}")
            continue
        retail, top = f"{s['retail']:.2f} (p{s['retail_p']})", f"{s['top']:.2f} (p{s['top_p']})"
        lines.append(f"{sym:<6}{retail:<14}{top:<18}{s['fund_bp']:+.1f}bp (p{s['fund_p']})")
    lines += [f"⚠ {flag}" for flag in summary["flags"]] or ["No positioning flags."]
    return "\n".join(lines)


def snapshot() -> str:
    now = datetime.now(timezone.utc)
    return render(summarize(fetch(), now), now)
