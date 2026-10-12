"""HTML grid report for program-level captures (G20's share of market volume), a thin wrapper
over the shared grid renderer (compliance.grid). Per-asset × (exchange, quote) heatmap of
capture %, volume-weighted averages, worst-first rows. Built from
trade_analysis_core.fetch_captures_matrix().
"""

from . import grid

# capture % -> colour. high = good = green.
_BANDS = [(10, '#2f9e44', '#ffffff'), (8, '#e0b100', '#141414'),
          (5, '#e8833a', '#141414'), (0, '#d64545', '#ffffff')]
_BANDS_LEGEND = [('≥10%', '#2f9e44', '#fff'), ('8–10%', '#e0b100', '#141414'),
                 ('5–8%', '#e8833a', '#141414'), ('<5%', '#d64545', '#fff')]
_NOTE = ('Capture = our (G20) share of market volume on each venue/pair, over the venues where we '
         'run an active orderbook (MM-only). Averages are volume-weighted (total G20 ÷ total '
         'market), not a flat mean. Venues with no active book (or no market data) are greyed and '
         'excluded from the averages.')


def _km(x):
    x = x or 0
    ax = abs(x)
    if ax >= 1e6:
        return f'{x / 1e6:.2f}M'
    if ax >= 1e3:
        return f'{x / 1e3:.1f}k'
    return f'{x:.0f}'


def _hover(asset, col, r):
    cap = r.get('capture', float('nan'))
    cap = 'n/a' if cap != cap else f'{cap * 100:.1f}%'
    return (f'{asset} · {col}\n'
            f'capture {cap}\n'
            f'G20 ${_km(r.get("g20"))}  ·  market ${_km(r.get("market"))}')


def _active(cells):  # MM venues only: drop no-data and non-MM (no active orderbook)
    m = ~cells['nodata']
    if 'nonmm' in cells.columns:
        m &= ~cells['nonmm']
    return cells[m]


def _headline(program, cells):
    if cells is None or cells.empty:
        return f'{program}: no capture data in this period.'
    data = _active(cells)
    tot_g, tot_m = data['g20'].sum(), data['market'].sum()
    overall = 'n/a' if not tot_m else f'{tot_g / tot_m * 100:.1f}%'
    below = int((data['capture'] * 100 < 5).sum())
    return (f'{program} · Capture {overall} (vol-wtd, MM only) · {data["asset"].nunique()} assets · '
            f'{len(data)} venues · {below} below 5%')


def _avg_primary(cells):
    g = _active(cells).groupby('asset')
    return g['g20'].sum() / g['market'].sum()   # volume-weighted per asset, MM venues only


def build_captures_report(program, cells, ignored, tokens, start, end, period_label=None,
                          body_only=False):
    period = period_label or f'{start:%Y-%m-%d %H:%M}–{end:%H:%M} UTC'
    subtitle = f'{start:%Y-%m-%d %H:%M} → {end:%H:%M} UTC'
    headline = _headline(program, cells)
    if cells is not None and not cells.empty:
        avg_p = _avg_primary(cells)
        n_v = _active(cells).groupby('asset').size()
    else:
        avg_p = n_v = None
    has_nonmm = cells is not None and 'nonmm' in getattr(cells, 'columns', []) and bool(cells['nonmm'].any())

    return grid.build_grid(
        title=f'{program} captures {period}', h1=f'{program} — Captures (G20 share of market)',
        subtitle=subtitle, program=program, cells=cells, ignored=ignored,
        value_col='capture', bands=_BANDS, bands_legend=_BANDS_LEGEND,
        empty_msg=headline, headline_html=headline, note=_NOTE, hover=_hover,
        avg_primary=avg_p,
        avg_hover=(lambda a: f'{a} — volume-weighted capture across {int(n_v.get(a, 0))} venue(s)')
        if n_v is not None else None,
        dim=(dict(col='nonmm', label='Show all venues') if has_nonmm else None), body_only=body_only)


def write_captures_report(program, cells, ignored, tokens, start, end, path, period_label=None):
    html = build_captures_report(program, cells, ignored, tokens, start, end, period_label)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path
