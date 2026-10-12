"""The deterministic ZP judgement rules — ``docs/zp_spec.md`` §5 — as one ``checks()`` entry
point returning a list of finding dicts. Each rule is skipped (nothing emitted) when the inputs
it needs aren't available; nothing here ever raises out of ``checks()`` itself.

These are the numbers-only half of the PDF's "Findings"/"Recommendations" sections — the prose
narrative on top of them is written by a human (or the Operator's model), per
``docs/zp_spec.md`` §1.

MUST NOT: fetch data, place/modify/cancel orders, call Telegram, or write the lake.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.the_operator.zp import dynamic
from src.the_operator.zp.compute import ZPResult

_HEDGER_TAGS = {"HEDGERV2", "HEDGING"}
_COMPLIANCE_THRESHOLD_BPS = 400.0
_COMPLIANCE_BAND_BPS = 10.0
_BARBELL_L0_SHARE = 0.02
_BARBELL_NEXT_LEVEL_BPS = 30.0
_CAPTURE_HIGH = 0.10


def _collapse_hits(hits: list, sig, label, describe) -> str:
    """One line per venue, or a single line naming them all when the numbers are identical.

    A coin usually runs the same settings across every venue, so a per-venue line repeats the
    same sentence ten times and buries the finding."""
    if len(hits) > 1 and len({sig(h) for h in hits}) == 1:
        return f"all {len(hits)} venues ({', '.join(label(h) for h in hits)}): {describe(hits[0])}"
    return "; ".join(f"{label(h)}: {describe(h)}" for h in hits)


def _hedger_round_trip_bps(other_summary: pd.DataFrame | None) -> float | None:
    """(buy_avg - sell_avg)/mid in bps across the hedger-tagged rows of the "other" summary —
    negative means the hedger bought high and sold low ("trend chasing")."""
    if other_summary is None or other_summary.empty:
        return None
    rows = other_summary[other_summary["acct_type"].isin(_HEDGER_TAGS)]
    if rows.empty:
        return None
    buy_n, buy_q = rows["buy_notional"].sum(), rows["buy_quantity"].sum()
    sell_n, sell_q = rows["sell_notional"].sum(), rows["sell_quantity"].sum()
    if buy_q <= 0 or sell_q <= 0:
        return None
    buy_avg, sell_avg = buy_n / buy_q, sell_n / sell_q
    if buy_avg <= 0 or sell_avg <= 0:
        return None
    mid = (buy_avg + sell_avg) / 2
    return float((buy_avg - sell_avg) / mid * 1e4)


def _hedger_attribution(result: ZPResult) -> dict | None:
    """§5 "Attribution": hedging P&L vs MM P&L, plus whether the hedger's own round trip reads as
    trend-chasing (negative bps)."""
    hl = result.headline
    mm_pnl, hedge_pnl = hl.get("mm_pnl"), hl.get("hedging_pnl")
    if mm_pnl is None or hedge_pnl is None:
        return None
    rt_bps = _hedger_round_trip_bps(result.other_summary)
    ratio = (abs(hedge_pnl) / abs(mm_pnl)) if mm_pnl else None
    trend_chasing = rt_bps is not None and rt_bps < 0
    # Severity is about the COST: a hedger that made money needs no action however big it is.
    sev = "info" if hedge_pnl >= 0 else ("high" if (ratio is not None and ratio > 1) else "med")
    finding = f"Hedging P&L ${hedge_pnl:,.0f} vs MM P&L ${mm_pnl:,.0f}"
    if ratio is not None:
        finding += f" (|hedge|/MM = {ratio:.2f}x)"
    if rt_bps is not None:
        # Average prices conflate spread capture with inventory drift in a trending session (the
        # source report says so itself), so the round trip is indicative, not a cost measurement.
        finding += (f"; hedger round trip {rt_bps:+.1f} bps "
                    f"({'trend chasing' if trend_chasing else 'favourable'}, drift-contaminated)")
    return {
        "id": "hedger_attribution", "severity": sev, "finding": finding + ".",
        "evidence": {"mm_pnl": mm_pnl, "hedging_pnl": hedge_pnl, "ratio": ratio,
                    "hedger_round_trip_bps": rt_bps},
        "suggestion": ("Review hedger execution timing/sizing relative to the MM signal."
                      if trend_chasing else "Hedging P&L looks additive to MM; no action indicated."),
    }


def _adverse_selection(result: ZPResult) -> dict | None:
    """§5 "Adverse selection": markouts negative at 60-300s and recovering at 600s reads as
    momentum pick-off rather than informed flow; compared against flow-level capture."""
    mo = result.headline.get("markouts_bps")
    if not mo:
        return None
    short = [mo.get(str(w)) for w in (60, 120, 180, 300)]
    long_ = mo.get("600")
    if any(v is None for v in short) or long_ is None:
        return None
    pick_off = all(v < 0 for v in short) and long_ > min(short)
    flow_cap = result.headline.get("flow_capture_bps")
    finding = (f"Markouts negative 60-300s ({', '.join(f'{v:.1f}' for v in short)} bps) and "
               f"{'recovering' if long_ > min(short) else 'not recovering'} by 600s ({long_:.1f} bps)")
    if flow_cap is not None:
        finding += f"; flow-level capture {flow_cap:.1f} bps"
    finding += " — consistent with momentum pick-off, not informed flow." if pick_off else "."
    return {
        "id": "adverse_selection", "severity": "med" if pick_off else "info", "finding": finding,
        "evidence": {"markouts_bps": mo, "flow_capture_bps": flow_cap},
        "suggestion": ("Widen at short horizons or reduce size in high-momentum regimes."
                      if pick_off else "No clear adverse-selection pattern in the markouts."),
    }


def _tick_dead_zone(result: ZPResult, dmc: dict | None, ticks: dict | None) -> dict | None:
    """§5 "Tick dead zone": ``base_min_spread_bps / tick_bps < 1`` — scaling is cosmetic, the
    effective spread is one tick. Needs both the dynamic-mode settings (``dmc``) and per-venue
    tick sizes (``ticks``, ``{(exchange, pair): {'tick_bps': ...}}`` or ``{exchange: {...}}``);
    skipped when either is unavailable."""
    if not dmc or not ticks:
        return None
    hits = []
    for strategy_base, exchange, quote, dyn, _business_area, _levels in dmc.get("rows") or []:
        min_spread = (dyn or {}).get("base_min_spread_bps")
        if min_spread is None:
            continue
        # The row's first field is the strategy FLOW ("RENEAR"), not the ticker, so the token's
        # own pair is the key that matches `sources.fetch_ticksize`; the flow form is kept as a
        # fallback for a gamma-dash payload whose label happens to carry the base.
        tick_info = (ticks.get((exchange, f"{result.token}-{quote}"))
                     or ticks.get((exchange, f"{strategy_base}-{quote}"))
                     or ticks.get(exchange))
        tick_bps = (tick_info or {}).get("tick_bps") if isinstance(tick_info, dict) else None
        if not tick_bps:
            continue
        ratio = min_spread / tick_bps
        if ratio < 1:
            hits.append((exchange, quote, float(min_spread), float(tick_bps), float(ratio)))
    if not hits:
        return None
    lines = "; ".join(f"{e} {q}: min_spread {ms:.2f}bps / tick {tb:.2f}bps = {r:.2f}"
                      for e, q, ms, tb, r in hits)
    return {
        "id": "tick_dead_zone", "severity": "med", "finding": f"Tick dead zone on {lines}.",
        "evidence": {"hits": hits},
        "suggestion": ("Scaling is cosmetic at these venues — the effective spread is pinned to "
                      "one tick (first in queue). Raise base_min_spread_bps or accept queue "
                      "priority as the only lever."),
    }


def _floored_dynamic_mode(result: ZPResult, dmc: dict | None) -> dict | None:
    """§5 "Floored dynamic mode": realized vol below reference vol most of the window means min
    spread sits at base and size at max — dynamic mode effectively off. Uses gamma-dash's vol
    history when the dmc payload carries one; falls back to ``result.realized_vol_ann_pct`` (this
    window's realised vol off the reference klines, ``compute.realized_vol_ann_pct``) when it
    doesn't — e.g. the strategy-box source, which has no history, only current settings — and
    treats "for the window" as one comparison over the whole window rather than a fraction."""
    if not dmc:
        return None
    series_by_label = dict(dmc.get("series") or [])
    hits = []
    for strategy_base, exchange, quote, dyn, _business_area, _levels in dmc.get("rows") or []:
        ref_vol = (dyn or {}).get("reference_annualized_volatility_pct")
        if ref_vol is None:
            continue
        label = f"{exchange} {quote} @{exchange}:{strategy_base}"
        hist = series_by_label.get(label)
        if hist is None:
            # label building above is best-effort; fall back to any series for this exchange/quote
            hist = next((h for lbl, h in series_by_label.items()
                        if lbl.startswith(f"{exchange} {quote} ")), None)
        realized = [r.get("volatility_ann_pct") for r in (hist or []) if r.get("volatility_ann_pct") is not None]
        if realized:
            frac_below = sum(1 for v in realized if v < ref_vol) / len(realized)
            realized_val = float(np.median(realized))
        elif result.realized_vol_ann_pct is not None:
            realized_val = result.realized_vol_ann_pct
            frac_below = 1.0 if realized_val < ref_vol else 0.0
        else:
            continue
        if frac_below > 0.5:
            hits.append((exchange, quote, float(ref_vol), realized_val, float(frac_below),
                        realized_val / ref_vol))
    if not hits:
        return None
    lines = _collapse_hits(
        hits,
        sig=lambda h: (round(h[2], 3), round(h[3], 3), round(h[4], 3)),
        label=lambda h: f"{h[0]} {h[1]}",
        describe=lambda h: (f"ref {h[2]:.1f}% vs realized {h[3]:.1f}% (ratio {h[5]:.2f}, "
                            f"{h[4]:.0%} of window below ref)"))
    return {
        "id": "floored_dynamic_mode", "severity": "med",
        "finding": f"Dynamic mode likely floored on {lines}.",
        "evidence": {"hits": hits},
        "suggestion": ("Realized vol sits below reference most of the window — min spread and "
                      "size sit at their base/max settings; dynamic mode is effectively off."),
    }


def _scale_asymmetry(result: ZPResult, dmc: dict | None) -> dict | None:
    """§5 "Scale asymmetry": ``vol_min_scale > vol_max_scale``.

    Both spread multipliers are floored at 1 (``dynamic.applied``), so neither side of the ladder
    ever quotes tighter than base — rising vol only pushes quotes out. The asymmetry therefore
    means the INSIDE quote (TOB, min_spread) pulls away from mid faster than the OUTSIDE one (BOB,
    max_spread): the BOB expands at a smaller scale than the TOB, so the levels bunch relative to
    their distance from mid. It does NOT compress the ladder toward the inside.

    That only becomes a defect where the TOB actually catches the BOB, so severity is set by
    ``dynamic.collapse_ratio`` — how far above reference vol the ladder would run out of span."""
    if not dmc:
        return None
    hits = []
    for _strategy_base, exchange, quote, dyn, _business_area, _levels in dmc.get("rows") or []:
        d = dyn or {}
        vmin = d.get("volatility_min_spread_scaling_factor")
        vmax = d.get("volatility_max_spread_scaling_factor")
        if vmin is None or vmax is None or vmin <= vmax:
            continue
        spread_2x = dynamic.applied(d, 2 * d["reference_annualized_volatility_pct"]) \
            if d.get("reference_annualized_volatility_pct") else None
        hits.append((exchange, quote, float(vmin), float(vmax), dynamic.collapse_ratio(d),
                     (spread_2x or {}).get("min_spread_bps"), (spread_2x or {}).get("max_spread_bps")))
    if not hits:
        return None

    # One line per (min-scale, max-scale) pair, not per venue: a coin runs the same scales
    # everywhere and only the base spreads differ, so per-venue lines repeat one sentence ten times.
    groups: dict[tuple[float, float], list[tuple]] = {}
    for h in hits:
        groups.setdefault((h[2], h[3]), []).append(h)
    parts = []
    for (vmin, vmax), g in groups.items():
        s = (f"{', '.join(f'{h[0]} {h[1]}' for h in g)}: min-scale {vmin:g} > max-scale {vmax:g}, "
             f"so the TOB widens as vol^{vmin:g} and the BOB as vol^{vmax:g}")
        tob2x = [h[5] for h in g if h[5] is not None]
        bob2x = [h[6] for h in g if h[6] is not None]
        if tob2x and bob2x:
            s += (f" (at 2x ref vol the TOB runs {min(tob2x):.0f}-{max(tob2x):.0f}bps against a "
                  f"{max(bob2x):.0f}bps BOB)")
        ratios = [h[4] for h in g if h[4] is not None]
        if ratios:
            at = (f"{ratios[0]:.0f}x" if len(set(ratios)) == 1
                  else f"{min(ratios):.0f}-{max(ratios):.0f}x")
            s += f"; the TOB would only reach the BOB at {at} ref vol"
        parts.append(s)
    lines = "; ".join(parts)

    collapse = [h[4] for h in hits if h[4] is not None]
    # A ladder that runs out of span within reach of normal vol is a real defect; one that would
    # need a 20x vol event is a deliberate shape, and calling it a problem is what got flagged.
    reachable = bool(collapse) and min(collapse) <= 3
    return {
        "id": "scale_asymmetry", "severity": "high" if reachable else "info",
        "finding": f"Spread scaling is asymmetric on {lines}.",
        "evidence": {"hits": hits, "collapse_ratio_min": min(collapse) if collapse else None},
        "suggestion": ("The ladder loses its span within reach of normal vol — raise vol_max_scale "
                      "or lower vol_min_scale so the BOB keeps pace with the TOB."
                      if reachable else
                      "Deliberate shape, not a defect: the inside steps out faster than the outside, "
                      "and neither side ever tightens. No action unless the desk wants the TOB to "
                      "hold its distance in spikes (lower vol_min_scale)."),
    }


def _compliance_boundary(result: ZPResult, threshold_bps: float = _COMPLIANCE_THRESHOLD_BPS,
                          band_bps: float = _COMPLIANCE_BAND_BPS) -> dict | None:
    """§5 "Compliance on the boundary": BOB medians within a few bps of the 400bps compliance
    threshold mean compliance is a coin flip driven by vol scaling."""
    oc = result.ob_compliance
    if oc is None or oc.empty:
        return None
    near = oc[(~oc["nodata"]) & ((oc["bob_median"] - threshold_bps).abs() <= band_bps)]
    if near.empty:
        return None
    lines = "; ".join(f"{r['exchange']} {r['pair']}: BOB median {r['bob_median']:.0f}bps"
                      for _, r in near.iterrows())
    return {
        "id": "compliance_boundary", "severity": "med",
        "finding": f"Compliance sits on the {threshold_bps:.0f}bps boundary at {lines}.",
        "evidence": {"venues": near[["exchange", "pair", "bob_median", "compliance"]].to_dict("records")},
        "suggestion": (f"Compliance is a coin flip driven by vol scaling — tighten base max "
                      f"spread or cap the max multiplier so BOB sits clear of {threshold_bps:.0f}bps."),
    }


def _barbell_capture(result: ZPResult, dmc: dict | None) -> dict | None:
    """§5 "Barbell capture": L0 carrying under 2%% of ladder notional with the next level over
    30bps out means capture stays low despite volume — the book is barbell-shaped. Uses the real
    snapshot ladder (``ob_levels``) when present; falls back to the settings alone — L0's share is
    ``top_level_weight``, and the L0->L1 gap runs ``base_min_spread_bps`` to
    ``base_min_spread_bps + (base_max_spread_bps - base_min_spread_bps) / (levels - 1)`` — when
    ``ob_levels`` is empty (the strategy-box source, which serves settings, not live ladder state)."""
    if not dmc:
        return None
    hits = []
    for _strategy_base, exchange, quote, dyn, _business_area, ob_levels in dmc.get("rows") or []:
        if ob_levels:
            levels = sorted(ob_levels, key=lambda lv: lv.get("distance_bps", lv.get("level", 0)))
            total_w = sum(lv.get("weight", lv.get("notional_usd", 0)) or 0 for lv in levels)
            if not total_w:
                continue
            l0_share = (levels[0].get("weight", levels[0].get("notional_usd", 0)) or 0) / total_w
            next_dist = levels[1].get("distance_bps") if len(levels) > 1 else None
        else:
            d = dyn or {}
            l0_share = d.get("top_level_weight")
            base_min, base_max, n_levels = (d.get("base_min_spread_bps"), d.get("base_max_spread_bps"),
                                            d.get("levels"))
            if l0_share is None or base_min is None or base_max is None or not n_levels or n_levels <= 1:
                continue
            next_dist = base_min + (base_max - base_min) / (n_levels - 1)
        if l0_share is not None and next_dist is not None and \
                l0_share < _BARBELL_L0_SHARE and next_dist > _BARBELL_NEXT_LEVEL_BPS:
            hits.append((exchange, quote, float(l0_share), float(next_dist)))
    if not hits:
        return None
    cap_avg = result.headline.get("capture_avg")
    lines = "; ".join(f"{e} {q}: L0 {s:.1%} of notional, next level {d:.0f}bps out" for e, q, s, d in hits)
    finding = f"Barbell ladder shape on {lines}"
    if cap_avg is not None:
        finding += f" (capture avg {cap_avg:.1%})"
    return {
        "id": "barbell_capture", "severity": "low", "finding": finding + ".",
        "evidence": {"hits": hits, "capture_avg": cap_avg},
        "suggestion": "Add a level in the 10-25bps zone if the mandate carries volume/presence KPIs.",
    }


def _venue_actions(result: ZPResult) -> dict | None:
    """§5 "Per-venue actions": worst markouts + low volume -> widen or cut L0; best markouts on
    large volume -> increase weight; a small venue with high capture -> a modest increase only."""
    actions: list[tuple[str, str, str, str]] = []
    mo = result.markouts
    if mo is not None and not mo.empty and "pnl_180s_bps" in mo.columns:
        df = mo.reset_index()
        vol_median = df["volume_$"].median()
        for _, r in df.iterrows():
            mo180 = r["pnl_180s_bps"]
            if pd.isna(mo180):
                continue
            vol = r["volume_$"]
            if mo180 < -1 and vol < vol_median:
                actions.append((r["exchange"], r["inst"], "widen or cut L0",
                               f"markout {mo180:.1f}bps, volume ${vol:,.0f} (below median)"))
            elif mo180 > 1 and vol >= vol_median:
                actions.append((r["exchange"], r["inst"], "increase weight",
                               f"markout {mo180:.1f}bps, volume ${vol:,.0f} (at/above median)"))

    cap = result.captures
    if cap is not None and not cap.empty and "exchange" in cap.columns:
        small = cap[(cap["exchange"] != "TOTAL") & (cap["capture"] >= _CAPTURE_HIGH)]
        for _, r in small.iterrows():
            actions.append((r["exchange"], r.get("pair", ""), "modest increase only",
                           f"high capture {r['capture']:.0%} on this venue"))

    if not actions:
        return None
    lines = "; ".join(f"{e}/{p}: {a} ({why})" for e, p, a, why in actions)
    return {
        "id": "venue_actions", "severity": "info", "finding": f"Per-venue actions: {lines}.",
        "evidence": {"actions": actions},
        "suggestion": "Apply the listed per-venue weight/L0 adjustments.",
    }


def _dynamic_settings(result: ZPResult, dmc: dict | None) -> dict | None:
    """The live parameters and the ladder they currently produce — context, not a judgement.

    The desk tunes these daily, so a recommendation that does not name a parameter and a direction
    is not actionable. Carries the formulas too, so the reader can check the arithmetic."""
    if not dmc:
        return None
    vol = result.realized_vol_ann_pct
    venues, params_by_venue, applied_by_venue = [], {}, {}
    for _strategy_base, exchange, quote, dyn, _business_area, _levels in dmc.get("rows") or []:
        p = dynamic.params(dyn or {})
        if not p:
            continue
        venue = f"{exchange} {quote}"
        venues.append(venue)
        params_by_venue[venue] = p
        applied_by_venue[venue] = dynamic.applied(dyn or {}, vol)
    if not venues:
        return None
    # Split shared from per-venue so ten near-identical parameter dumps don't drown the packet.
    keys = set().union(*(p.keys() for p in params_by_venue.values()))
    shared = {k: params_by_venue[venues[0]][k] for k in sorted(keys)
              if len({str(p.get(k)) for p in params_by_venue.values()}) == 1}
    differs = {v: {k: p[k] for k in sorted(p) if k not in shared}
               for v, p in params_by_venue.items()}
    ladders = {v: {k: round(a[k], 2) for k in
                   ("min_spread_bps", "max_spread_bps", "size_usd", "tob_size_usd") if k in a}
               for v, a in applied_by_venue.items() if a}

    finding = f"Dynamic-mode parameters across {len(venues)} venues"
    a = next(iter(applied_by_venue.values()), None)
    if a:
        finding += (f"; at this window's realised vol {a['vol_ann_pct']:.0f}% vs ref "
                    f"{a['ref_vol']:.0f}% the multipliers are "
                    f"x{a.get('min_mult', 1):.2f} TOB / x{a.get('max_mult', 1):.2f} BOB")
    if ladders:
        lo = min(lad["min_spread_bps"] for lad in ladders.values() if "min_spread_bps" in lad)
        hi = max(lad["max_spread_bps"] for lad in ladders.values() if "max_spread_bps" in lad)
        finding += f", so the ladders run {lo:.1f}-{hi:.0f}bps"
    return {
        "id": "dynamic_settings", "severity": "info", "finding": finding + ".",
        "evidence": {"shared": shared, "per_venue": differs, "applied": ladders,
                     "formulas": dynamic.formulas()},
        "suggestion": ("These are the levers. Every recommendation should name one, its current "
                      "value and the value to try, and what the formula says that does."),
    }


def _capture_gap(result: ZPResult, dmc: dict | None,
                  target: float = _CAPTURE_HIGH) -> dict | None:
    """The desk's standing target: total capture >= 10%. Sizes the gap in volume and names the
    parameters that move it, plus the markout the trade-off has to respect.

    Deliberately does NOT predict how much capture a parameter change buys — that response is not
    measured anywhere in this data, and inventing a number is how a review loses its reader."""
    cap = result.captures
    total_capture = result.headline.get("capture_avg")
    if cap is None or cap.empty or total_capture is None:
        return None
    total = cap[cap["exchange"] == "TOTAL"]
    gap_usd = None
    if not total.empty and pd.notna(total["market_volume_usd"].iloc[0]):
        need = float(total["market_volume_usd"].iloc[0]) * target
        gap_usd = need - float(total["g20_volume_usd"].iloc[0])
    if total_capture >= target:
        return {
            "id": "capture_gap", "severity": "info",
            "finding": f"Total capture {total_capture:.1%} is at or above the {target:.0%} target.",
            "evidence": {"capture": total_capture, "target": target},
            "suggestion": "Target met — protect it; do not buy more capture with worse markouts.",
        }

    # Where the missing volume is: the venues with the most market volume we are least on.
    rows = cap[(cap["exchange"] != "TOTAL") & cap["capture"].notna()
               & (cap["capture"] < target)].copy()
    rows = rows.sort_values("market_volume_usd", ascending=False).head(3)
    where = "; ".join(f"{r['exchange']} {r.get('pair', '')}: {r['capture']:.1%} of "
                      f"${r['market_volume_usd']:,.0f}" for _, r in rows.iterrows())

    mo180 = (result.headline.get("markouts_bps") or {}).get("180")
    # Quote the parameters of the venue holding the most missing volume, not whichever row came
    # first: naming a $20k venue's settings while the gap sits on a $46m one is not a lever.
    lever_venue = rows["exchange"].iloc[0] if not rows.empty else None
    by_exchange = {exchange: dyn for _flow, exchange, _quote, dyn, _ba, _lv
                   in ((dmc or {}).get("rows") or [])}
    dyn = by_exchange.get(lever_venue) or next(iter(by_exchange.values()), None)
    levers = []
    if dyn:
        p = dynamic.params(dyn)
        prefix = f"on {lever_venue}, " if lever_venue in by_exchange else ""
        if "base_min_spread" in p:
            levers.append(f"{prefix}base_min_spread {p['base_min_spread']:g}bps (the inside quote "
                          f"— the direct capture lever)")
        if "top_lvl_wt" in p and p.get("levels"):
            levers.append(f"top_lvl_wt {p['top_lvl_wt']:g} over {p['levels']:g} levels (how much "
                          f"size actually sits at the top of book)")
        if "base_size" in p:
            levers.append(f"base_size ${p['base_size']:,.0f} (total ladder notional)")
    finding = (f"Total capture {total_capture:.1%} is below the {target:.0%} target")
    if gap_usd is not None:
        finding += f" — about ${gap_usd:,.0f} of extra volume over this window closes it"
    if where:
        finding += f". Most of the missing volume sits at {where}"
    room = ("markouts are positive at 180s (+%.1f bps), so there is room to quote tighter or "
            "bigger before selection costs bite" % mo180 if mo180 is not None and mo180 > 0 else
            "markouts are negative at 180s (%.1f bps), so buying capture with a tighter inside "
            "quote would pay for it in selection — take the volume through size at the back of "
            "the ladder instead" % mo180 if mo180 is not None else
            "no markout read this window, so the selection cost of tightening is unknown")
    return {
        "id": "capture_gap", "severity": "high",
        "finding": finding + ".",
        "evidence": {"capture": total_capture, "target": target, "gap_usd": gap_usd,
                     "markout_180s_bps": mo180,
                     "venues": rows[["exchange", "pair", "capture", "market_volume_usd"]].to_dict("records")},
        "suggestion": (f"{room}. Levers, in order: {'; '.join(levers)}."
                       if levers else f"{room}."),
    }


def checks(result: ZPResult, dmc: dict | None = None, ticks: dict | None = None) -> list[dict]:
    """The deterministic §5 judgement rules, each skipped (nothing emitted) when its inputs are
    unavailable or malformed. ``dmc`` defaults to ``result.gamma_dash`` (a payload shaped like
    ``sources.fetch_gamma_dash``'s return); ``ticks`` is optional per-venue tick-size data
    (``run.run_zp`` does not fetch it today — a future caller supplies it)."""
    dmc = dmc if dmc is not None else result.gamma_dash
    checkers: list[tuple[Any, tuple]] = [
        (_hedger_attribution, (result,)),
        (_adverse_selection, (result,)),
        (_tick_dead_zone, (result, dmc, ticks)),
        (_floored_dynamic_mode, (result, dmc)),
        (_scale_asymmetry, (result, dmc)),
        (_compliance_boundary, (result,)),
        (_barbell_capture, (result, dmc)),
        (_capture_gap, (result, dmc)),
        (_venue_actions, (result,)),
        (_dynamic_settings, (result, dmc)),
    ]
    out = []
    for fn, args in checkers:
        try:
            item = fn(*args)
        except Exception:  # noqa: BLE001 - malformed input is "unavailable", not a crash
            item = None
        if item is not None:
            out.append(item)
    return out
