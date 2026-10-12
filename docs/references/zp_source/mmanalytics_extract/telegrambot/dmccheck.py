"""DMC sanity scanner: pull dynamic-mode settings and flag the suspect ones.

Coin-level overview — one row per coin, each setting cell showing the min–max across that coin's
venues; a cell is coloured if any of its venues trips a rule. Two tiers:
 - HARD rules — the desk's known bounds (flag when the condition is true; see _HARD).
 - PEER OUTLIERS — a value off the DESK-WIDE cluster (ratio to the desk median outside the bounds),
   so a single program's own norm can't skew its baseline. Backstop for every numeric setting.

`build_dmccheck_report(program, report_strategies, ref_strategies)` takes flat strategy dicts
{coin, venue, dyn}; the distribution comes from `ref_strategies` (pass the whole desk) so outliers
are desk-wide even when the report is one program. Reuses grid CSS + the shared sorter.
"""

import html as _html
import statistics as _st

from . import grid

# Hard rules — flag WHEN the condition is true (your signs). fn(value, whole_settings_dict) -> reason.
_HARD = {
    'top_level_weight':
        lambda v, d: f'{v:g} > 0.2' if v > 0.2 else None,
    'levels':
        lambda v, d: f'{v:g} < 2' if v < 2 else None,
    'base_min_spread_bps':
        lambda v, d: ('must be > 0' if v <= 0 else
                      (f'min {v:g} > max {d["base_max_spread_bps"]:g}'
                       if isinstance(d.get('base_max_spread_bps'), (int, float))
                       and v > d['base_max_spread_bps'] else None)),
    'base_max_spread_bps':
        lambda v, d: (f'{v:g} < 170' if v < 170 else f'{v:g} > 200' if v > 200 else None),
    'total_value_usd':
        lambda v, d: 'must be ≥ 0' if v < 0 else None,
    'reference_annualized_volatility_pct':
        lambda v, d: ('must be ≥ 0' if v < 0 else f'{v:g} > 250' if v > 250 else None),
    'volatility_min_spread_scaling_factor':
        lambda v, d: f'{v:g} < 0.25' if v < 0.25 else None,
    'volatility_max_spread_scaling_factor':
        lambda v, d: f'{v:g} < 0.25' if v < 0.25 else None,
    'volatility_size_scaling_factor':
        lambda v, d: f'{v:g} < 0.25' if v < 0.25 else None,
}

# Peer outlier = FAR from the desk median (ratio outside these bounds) AND rare (few peers share it).
# The rarity gate stops a legit second config (e.g. vol_interval 1000 vs 5000) from all flagging.
_OUT_HI, _OUT_LO = 3.0, 0.33
_LOOSE_HI, _LOOSE_LO = 5.0, 0.2    # size & ref-vol vary by coin — only flag when really far off
_RARE_FRAC = 0.08                  # a value shared by ≥8% of the desk is a config, not an outlier
_RARE_BAND = 1.25                  # "shared" = within ±25% of the value
_MIN_POP = 4                       # need this many desk values before outlier detection is trusted

_MONEY = lambda v: (f'${v/1e6:.1f}M' if abs(v) >= 1e6 else f'${v/1e3:.0f}k' if abs(v) >= 1e3
                    else f'${v:,.0f}')
# (key, column label, formatter, loose-outlier)
_SPEC = [
    ('top_level_weight', 'Top Lvl Wt', lambda v: f'{v:.2f}', False),
    ('levels', 'Levels', lambda v: f'{v:g}', False),
    ('base_min_spread_bps', 'Base Min', lambda v: f'{v:g}', False),
    ('base_max_spread_bps', 'Base Max', lambda v: f'{v:g}', False),
    ('total_value_usd', 'Total Value', _MONEY, True),
    ('reference_annualized_volatility_pct', 'Ref Vol %', lambda v: f'{v:g}', True),
    ('volatility_min_spread_scaling_factor', 'Vol Min Scale', lambda v: f'{v:g}', False),
    ('volatility_max_spread_scaling_factor', 'Vol Max Scale', lambda v: f'{v:g}', False),
    ('volatility_size_scaling_factor', 'Vol Size Scale', lambda v: f'{v:g}', False),
    ('volatility_interval_ms', 'Vol Interval', lambda v: f'{v:g}ms', False),
]

_HARD_BG, _OUT_BG = '#a83232', '#a3571f'   # red = hard rule, amber = peer outlier
_HARD_FG, _OUT_FG = '#ffd9d9', '#ffe2c4'

# Human-readable hard limit per setting, for the reference chips at the top of the report.
_HARD_DESC = {
    'top_level_weight': '> 0.2', 'levels': '< 2', 'base_min_spread_bps': '≤ 0 or > max',
    'base_max_spread_bps': '< 170 or > 200', 'total_value_usd': '< 0',
    'reference_annualized_volatility_pct': '< 0 or > 250',
    'volatility_min_spread_scaling_factor': '< 0.25',
    'volatility_max_spread_scaling_factor': '< 0.25', 'volatility_size_scaling_factor': '< 0.25',
}
def _hard_limits_tooltip():
    """Plain-text hard-limit reference shown on hover over the 'hard rule' legend swatch."""
    parts = [f'{l} {_HARD_DESC[k]}' for k, l, *_ in _SPEC if k in _HARD_DESC]
    parts.append('Halflife empty')
    return 'Hard limits — flag when:   ' + '   ·   '.join(parts)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _desk_dist(strategies):
    """{key: (median, values)} across every strategy (the desk) — the outlier baseline + population."""
    dist = {}
    for key, _lab, _fmt, _loose in _SPEC:
        xs = [s['dyn'].get(key) for s in strategies if _num(s['dyn'].get(key))]
        dist[key] = (_st.median(xs), xs) if xs else (None, [])
    return dist


def _outlier_reason(v, m, xs, loose, fmt):
    """Reason if v is FAR from the desk median (ratio outside bounds) AND rare (few peers within
    ±25%). The rarity gate keeps a common second config from flagging. Else ''."""
    if m == 0 or v == m:
        return ''
    hi, lo = (_LOOSE_HI, _LOOSE_LO) if loose else (_OUT_HI, _OUT_LO)
    ratio = v / m
    if lo <= ratio <= hi:
        return ''                                              # not far enough
    near = sum(1 for x in xs if v / _RARE_BAND <= x <= v * _RARE_BAND)
    if xs and near / len(xs) >= _RARE_FRAC:
        return ''                                              # common value → a config, not outlier
    return f'outlier: {ratio:.1f}× desk median {fmt(m)}'


def _venue_reasons(v, key, loose, fmt, dyn, dist, outliers=True):
    """Reasons ([(tier, text)]) for one venue's value of one setting: hard rule first, else outlier.
    outliers=False (single-coin / embedded views) runs the hard rules only."""
    reasons = []
    hard = _HARD.get(key)
    if hard:                       # a hard-ruled setting: the hard limit is the only trigger
        hr = hard(v, dyn)
        if hr:
            reasons.append(('hard', hr))
    elif outliers:                 # no hard rule → the statistical outlier is the backstop
        m, xs = dist.get(key, (None, []))
        if m is not None and len(xs) >= _MIN_POP:
            r = _outlier_reason(v, m, xs, loose, fmt)
            if r:
                reasons.append(('outlier', r))
    return reasons


def _val_cell(entry, fmt):
    """One value (a coin's min or max of a setting). The hover always names the venue(s) sitting at
    that value; coloured with the reason when the value trips a rule."""
    if entry is None:
        return '<td class="cell" style="background:#20262e;color:#5a636e">·</td>'
    v, reasons, venues = entry
    dv = f' data-v="{v:.6g}"'
    vtext = ', '.join(venues[:8]) + (f' +{len(venues) - 8} more' if len(venues) > 8 else '')
    if not reasons:
        return f'<td class="cell"{dv} title="{_html.escape(vtext)}">{fmt(v)}</td>'
    hard = any(t == 'hard' for t, _ in reasons)
    bg, fg = (_HARD_BG, _HARD_FG) if hard else (_OUT_BG, _OUT_FG)
    rtext = ' · '.join(dict.fromkeys(r for _, r in reasons))   # dedup, keep order
    tip = _html.escape(f'{rtext} — {vtext}')
    return (f'<td class="cell"{dv} title="{tip}" '
            f'style="background:{bg};color:{fg};cursor:help">{fmt(v)}</td>')


def build_dmccheck_report(program, report_strategies, ref_strategies=None, body_only=False,
                          outliers=True, title=True):
    dist = _desk_dist(ref_strategies if ref_strategies is not None else report_strategies)

    # Aggregate the reported strategies to one row per coin (min–max per setting + collected reasons).
    coins = {}
    for s in report_strategies:
        coins.setdefault(s['coin'], []).append(s)

    rows = []
    for coin, strs in coins.items():
        cell = {}       # key -> {'lo': (v, reasons, venues)|None, 'hi': ...}
        for key, _lab, fmt, loose in _SPEC:
            entries = [(st['dyn'].get(key), st['dyn'], st.get('venue', '?')) for st in strs
                       if _num(st['dyn'].get(key))]
            if not entries:
                cell[key] = {'lo': None, 'hi': None}
                continue
            lo_v = min(e[0] for e in entries)
            hi_v = max(e[0] for e in entries)

            def _side(target):     # (value, reasons, the venues that sit at this value)
                at = [e for e in entries if e[0] == target]
                reasons = _venue_reasons(target, key, loose, fmt, at[0][1], dist, outliers)
                return (target, reasons, [e[2] for e in at])

            cell[key] = {'lo': _side(lo_v), 'hi': _side(hi_v)}
        hl_bad = any('volatility_interval_ms' in st['dyn'] and not st['dyn'].get(
            'volatility_halflife_seconds_list') for st in strs)
        all_reasons = [r for c in cell.values() for side in (c['lo'], c['hi']) if side for r in side[1]]
        n_hard = sum(1 for t, _ in all_reasons if t == 'hard') + (1 if hl_bad else 0)
        n_flags = (sum(1 for c in cell.values()
                       if (c['lo'] and c['lo'][1]) or (c['hi'] and c['hi'][1])) + (1 if hl_bad else 0))
        rows.append({'coin': coin, 'cell': cell, 'hl_bad': hl_bad, 'nven': len(strs),
                     'n_hard': n_hard, 'n_flags': n_flags})
    rows.sort(key=lambda r: (-r['n_hard'], -r['n_flags'], r['coin']))
    total_flags = sum(r['n_flags'] for r in rows)
    n_bad = sum(1 for r in rows if r['n_flags'])

    if body_only:
        head, close = ([f'<h3>{program} — Dynamic settings</h3>'] if title else []), []
    else:
        head = ['<!doctype html><html><head><meta charset="utf-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1">',
                f'<title>{_html.escape(f"{program} Dynamic settings")}</title>',
                f'<style>{grid._CSS}td.cell[title]{{cursor:help}}</style></head><body>',
                f'<h1>{program} — Dynamic settings</h1>']
        close = ['</body></html>']
    if not rows:
        return ''.join(head + [f'<div class="headline">{program}: no dynamic strategies found.</div>']
                       + close)

    thead = ('<th>Coin</th><th>Venues</th>'
             + ''.join(f'<th>{l}<br><small>min</small></th><th>{l}<br><small>max</small></th>'
                       for _k, l, *_ in _SPEC))
    body_rows = ''
    for r in rows:
        ctip = ' title="empty volatility halflife list"' if r['hl_bad'] else ''
        cbg = f' style="background:{_HARD_BG};color:{_HARD_FG}"' if r['hl_bad'] else ''
        cells = ''.join(_val_cell(r['cell'][k]['lo'], fmt) + _val_cell(r['cell'][k]['hi'], fmt)
                        for k, _l, fmt, _lo in _SPEC)
        body_rows += (f'<tr><th{cbg}{ctip}>{_html.escape(r["coin"])}</th>'
                      f'<td class="cell" data-v="{r["nven"]}">{r["nven"]}</td>{cells}</tr>')

    grid_html = (f'<div class="gridwrap"><table class="cmp"><thead><tr>{thead}</tr></thead>'
                 f'<tbody>{body_rows}</tbody></table></div>')
    verdict = ('✓ nothing suspect' if not n_bad
               else f'{total_flags} flag{"" if total_flags == 1 else "s"} across {n_bad} '
                    f'coin{"" if n_bad == 1 else "s"}')
    headline = (f'<div class="headline">{program} · Dynamic settings · {len(rows)} coins · '
                f'{len(report_strategies)} strategies · {verdict}</div>')
    outlier_swatch = ('<span style="background:%s;color:%s;padding:2px 8px;border-radius:4px">'
                      'peer outlier</span>' % (_OUT_BG, _OUT_FG)) if outliers else ''
    hl_tip = _html.escape(_hard_limits_tooltip())
    legend = (f'<div class="legend"><span title="{hl_tip}" style="background:{_HARD_BG};'
              f'color:{_HARD_FG};padding:2px 8px;border-radius:4px;margin-right:6px;cursor:help">'
              f'hard rule ⓘ</span>{outlier_swatch}</div>')
    outlier_note = ('' if not outliers else
                    ' Peer outliers (amber) are a backstop for settings with no hard rule (e.g. vol '
                    f'interval): a value &gt;{_OUT_HI:g}× or &lt;{_OUT_LO:g}× the desk median AND '
                    f'shared by fewer than {_RARE_FRAC:.0%} of strategies.')
    note = ('<p class="note">One row per coin; each setting has a min and max column across the '
            'coin’s venues, coloured if that value trips a rule (hover for which venues). Settings '
            f'with a hard limit flag only via that limit (red).{outlier_note}</p>')
    body = [headline, legend, grid_html, note]
    if not body_only:
        from . import sortable
        body.append(sortable.SORT_JS)
    return ''.join(head + body + close)


def strategies_from_dmc_map(dmc_map):
    """Flatten {coin: DmcData} into the flat strategy dicts build_dmccheck_report expects."""
    out = []
    for coin, data in (dmc_map or {}).items():
        if not data or not getattr(data, 'rows', None):
            continue
        for base, exchange, dyn, *rest in data.rows:
            perp = rest[0] if rest else False
            venue = f'{exchange.split(":", 1)[0]}{".P" if perp else ""} {base}'
            out.append({'coin': coin, 'venue': venue, 'perp': perp, 'dyn': dyn or {}})
    return out


def build_from_dmc_map(program, dmc_map, outliers=False, body_only=False, title=True):
    """Convenience for the aiozp Summary / individual ZP: build the check straight from a computed
    {coin: DmcData}. Defaults to hard-rules-only (no desk-wide outlier population there)."""
    strats = strategies_from_dmc_map(dmc_map)
    return build_dmccheck_report(program, strats, ref_strategies=strats, body_only=body_only,
                                 outliers=outliers, title=title)
