"""The desk's dynamic-mode quoting formulas, hardcoded from the MM3_Aggregator "Dynamic mode" tab.

Hardcoded on the desk's own instruction: the formulas change rarely, the PARAMETERS change daily
and are what the desk actually tunes. Without them a ZP can only describe the settings; with them
it can say what a settings change would do, which is what a quoting review is for.

    vol        = max(ema_hl300(returns), ema_hl1800(returns))        annualised %
    ref_vol    = Garman-Klass over the reference bars                annualised %
    min_spread = base_min_spread * max(1, (vol / ref_vol) ** vol_min_scale)
    max_spread = base_max_spread * max(1, (vol / ref_vol) ** vol_max_scale)
    size       = base_size       * min(1, (ref_vol / vol) ** vol_size_scale)
    tob_size   = base_size * top_level_weight / levels

Both spread multipliers are floored at 1, so neither side of the ladder ever quotes TIGHTER than
its base: rising vol only pushes quotes out. ``vol_min_scale > vol_max_scale`` therefore means the
inside quote (TOB, ``min_spread``) pulls away from mid faster than the outside one (BOB,
``max_spread``) — the BOB expands at a smaller scale than the TOB — NOT that the ladder compresses
toward the inside.

Argument names are the quoting boxes' own ``ob_dynamic`` field names, so every function here takes
a box payload unchanged (``sources.fetch_strategy_params``); ``total_value_usd`` is the sheet's
``base_size``.

MUST NOT: fetch anything, read a sheet at runtime, judge a setting (that is ``checks``), or model
how capture or P&L responds to a parameter — these are the venue's own arithmetic, nothing more.
"""

from __future__ import annotations

_FORMULAS = (
    "min_spread = base_min_spread * max(1, (vol/ref_vol)^vol_min_scale)",
    "max_spread = base_max_spread * max(1, (vol/ref_vol)^vol_max_scale)",
    "size = base_size * min(1, (ref_vol/vol)^vol_size_scale)",
    "tob_size = base_size * top_level_weight / levels",
    "vol = max(ema_hl300(returns), ema_hl1800(returns)); ref_vol = Garman-Klass",
)


def params(dyn: dict) -> dict:
    """The box's ``ob_dynamic`` under the sheet's own names, dropping anything absent."""
    named = {
        "base_min_spread_bps": "base_min_spread",
        "base_max_spread_bps": "base_max_spread",
        "volatility_min_spread_scaling_factor": "vol_min_scale",
        "volatility_max_spread_scaling_factor": "vol_max_scale",
        "volatility_size_scaling_factor": "vol_size_scale",
        "reference_annualized_volatility_pct": "ref_vol",
        "total_value_usd": "base_size",
        "top_level_weight": "top_lvl_wt",
        "levels": "levels",
        "volatility_halflife_seconds_list": "vol_halflife_s",
        "volatility_interval_ms": "vol_interval_ms",
    }
    return {name: dyn[field] for field, name in named.items() if dyn.get(field) is not None}


def applied(dyn: dict, vol_ann_pct: float | None) -> dict | None:
    """The ladder the formulas produce at ``vol_ann_pct``: multipliers, spreads, size, ToB size.

    Returns None when the reference vol or the base spreads are missing (nothing can be said);
    individual outputs are simply absent when their own scaling factor is."""
    p = params(dyn)
    ref, base_min, base_max = p.get("ref_vol"), p.get("base_min_spread"), p.get("base_max_spread")
    if not ref or not vol_ann_pct or base_min is None or base_max is None:
        return None
    ratio = vol_ann_pct / ref
    out: dict[str, float] = {"vol_ann_pct": float(vol_ann_pct), "ref_vol": float(ref),
                             "vol_ratio": float(ratio)}
    if (s := p.get("vol_min_scale")) is not None:
        out["min_mult"] = max(1.0, ratio**s)
        out["min_spread_bps"] = base_min * out["min_mult"]
    if (s := p.get("vol_max_scale")) is not None:
        out["max_mult"] = max(1.0, ratio**s)
        out["max_spread_bps"] = base_max * out["max_mult"]
    if (s := p.get("vol_size_scale")) is not None and p.get("base_size") is not None:
        out["size_mult"] = min(1.0, (1 / ratio) ** s)
        out["size_usd"] = p["base_size"] * out["size_mult"]
    # base_size, NOT the vol-scaled size: the ToB slice is fixed, so rising vol shrinks the ladder
    # behind it while the top of book stays put. (The sheet's formula text said `size`; the desk
    # confirmed on 2026-09-17 that this was a typo and its OUTPUT cell the right one.) It follows
    # that ToB size needs no size scaling factor to be known.
    if p.get("base_size") is not None and p.get("levels") and p.get("top_lvl_wt") is not None:
        out["tob_size_usd"] = p["base_size"] * p["top_lvl_wt"] / p["levels"]
    return out


def collapse_ratio(dyn: dict) -> float | None:
    """The ``vol/ref_vol`` at which the faster-widening TOB reaches the BOB (min_spread >=
    max_spread) and the ladder has no span left. None unless ``vol_min_scale > vol_max_scale``,
    which is the only way the two can ever meet."""
    p = params(dyn)
    s_min, s_max = p.get("vol_min_scale"), p.get("vol_max_scale")
    base_min, base_max = p.get("base_min_spread"), p.get("base_max_spread")
    if None in (s_min, s_max, base_min, base_max) or s_min <= s_max or base_min <= 0:
        return None
    return float((base_max / base_min) ** (1.0 / (s_min - s_max)))


def formulas() -> str:
    """The formula set as one block, for the model packet."""
    return "\n".join(f"    {f}" for f in _FORMULAS)
