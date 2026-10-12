"""HTML grid report for program-level markouts — one cell per asset × (exchange, quote),
coloured by the volume-weighted markout (bps) at a selectable time horizon. A dropdown toggles
the horizon (10s…600s); the default horizon is fully server-rendered so it still reads on a
phone with no JS, and the toggle just re-colours in place.

Positive = favourable (the market moved our way after the fill); negative = adverse selection.
Built from trade_analysis_core.fetch_markouts_matrix(). Unlike compliance/captures this metric
is signed, so it uses a diverging palette and an N-way horizon toggle rather than the shared
grid.build_grid — but it reuses grid's CSS, legend and no-data colour so the three reports match.
"""

import html as _html

import numpy as np

from . import grid

# Diverging bps bands, descending, saturating at ±5. Positive = favourable, negative = adverse
# selection. (threshold_bps, background, text).
_BANDS = [(5.0, '#1b6b3a', '#d7f5df'), (1.0, '#3f7a52', '#e8f7ee'),
          (-1.0, '#33383f', '#c7ccd4'), (-5.0, '#9a6327', '#ffe9cf'),
          (-1e9, '#a83232', '#ffd9d9')]
_BANDS_LEGEND = [('≥ +5', '#1b6b3a', '#d7f5df'), ('+1…+5', '#3f7a52', '#e8f7ee'),
                 ('±1', '#33383f', '#c7ccd4'), ('−5…−1', '#9a6327', '#ffe9cf'),
                 ('< −5', '#a83232', '#ffd9d9')]

_CSS_EXTRA = """
select{background:#1a1f26;color:#e6e9ef;border:1px solid #3a424e;border-radius:6px;
padding:6px 10px;font-size:13px;margin:6px 8px 6px 0}
label.hz{color:#8b94a3;font-size:13px}
td.cell{font-variant-numeric:tabular-nums;font-weight:600}
"""

_NOTE = ('Volume-weighted markout in bps at the selected horizon after each MM fill, per venue. '
         'Positive = the market moved in our favour after the trade; negative = adverse selection. '
         'Colour saturates at ±5 bps. A cell is greyed where the venue wasn’t traded or has no '
         'market data. Rows and columns are ordered by traded volume.')


def _fmt(v):
    if v is None or v != v:
        return '—'
    return '0.0' if abs(v) < 0.05 else f'{v:+.1f}'


def _band(v):
    for lo, bg, fg in _BANDS:
        if v >= lo:
            return bg, fg
    return _BANDS[-1][1], _BANDS[-1][2]


def _vw(sub, col):
    """Volume-weighted mean of `col` over rows of `sub`, ignoring NaN values/weights."""
    v = sub[col].to_numpy(dtype=float)
    w = sub['volume'].to_numpy(dtype=float)
    m = ~np.isnan(v) & ~np.isnan(w) & (w > 0)
    if not m.any():
        return float('nan')
    return float(np.average(v[m], weights=w[m]))


def _headline_text(program, hv, sec, n_coins, n_venues):
    val = 'n/a' if hv != hv else f'{_fmt(hv)} bps'
    return (f'{program} · markout {val} @ {sec}s · '
            f'{n_coins} coins · {n_venues} venues')


def _toggle_js(horizons, default):
    conds = ''.join(f"if(v>={lo})return['{bg}','{fg}'];" for lo, bg, fg in _BANDS)
    return (
        "<script>(function(){"
        f"var MISS=['{grid.MISSING[0]}','{grid.MISSING[1]}'];"
        f"function band(v){{{conds}return MISS;}}"
        "function fmt(v){var a=Math.abs(v)<0.05?0:v;return (a>0?'+':'')+a.toFixed(1);}"
        "function render(sec){"
        "document.querySelectorAll('td.cell').forEach(function(td){"
        "var v=td.getAttribute('data-v'+sec);"
        "if(v===null||v===''){td.textContent='\\u2014';td.style.background=MISS[0];"
        "td.style.color=MISS[1];return;}"
        "v=parseFloat(v);var b=band(v);td.textContent=fmt(v);"
        "td.style.background=b[0];td.style.color=b[1];});"
        "var h=document.getElementById('mo-headline');"
        "if(h){var hv=h.getAttribute('data-v'+sec);var txt=h.getAttribute('data-prog')"
        "+' \\u00b7 markout '+(hv===''||hv===null?'n/a':fmt(parseFloat(hv))+' bps')"
        "+' @ '+sec+'s \\u00b7 '+h.getAttribute('data-coins')+' coins \\u00b7 '"
        "+h.getAttribute('data-venues')+' venues';h.textContent=txt;"
        "var hb=(hv===''||hv===null)?MISS:band(parseFloat(hv));"
        "h.style.borderLeftColor=hb[0];}}"
        "var sel=document.getElementById('mohz');"
        "if(sel){sel.addEventListener('change',function(){render(sel.value);});}"
        f"render('{default}');"
        "})();</script>")


def build_markouts_report(program, cells, ignored, tokens, horizons, start, end, period_label=None,
                          body_only=False):
    period = period_label or f'{start:%Y-%m-%d %H:%M}–{end:%H:%M} UTC'
    subtitle = f'{start:%Y-%m-%d %H:%M} → {end:%H:%M} UTC'
    default = 180 if 180 in horizons else (horizons[len(horizons) // 2] if horizons else 0)

    if body_only:  # fragment (no shell, no horizon picker/script) — static default-horizon view
        head, close = [f'<h3>{program} — Markouts</h3>', f'<p class="sub">{subtitle}</p>'], []
    else:
        head = ['<!doctype html><html><head><meta charset="utf-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1">',
                f'<title>{_html.escape(f"{program} markouts {period}")}</title>',
                f'<style>{grid._CSS}{_CSS_EXTRA}</style></head><body>',
                f'<h1>{program} — Markouts</h1>', f'<p class="sub">{subtitle}</p>']
        close = ['</body></html>']
    ign = f'<p class="ignored">ignored (not in config): {", ".join(ignored)}</p>' if ignored else ''

    if cells is None or getattr(cells, 'empty', True) or not horizons:
        msg = f'{program}: no markout data in this period.'
        return ''.join(head + [f'<div class="headline">{msg}</div>', ign] + close)

    mo_cols = [f'mo_{s}' for s in horizons]
    valid = cells[~cells['nodata']]

    # Row/column order (stable across horizon toggles): by traded volume.
    keep = (cells.groupby('col_label')['volume'].sum()
            .sort_values(ascending=False).index.tolist())
    order = (cells.groupby('asset')['volume'].sum()
             .sort_values(ascending=False).index.tolist())

    metah = cells.drop_duplicates('col_label').set_index('col_label')
    info = cells.drop_duplicates(['asset', 'col_label']).set_index(['asset', 'col_label'])
    present = set(info.index)

    def header(col):
        r = metah.loc[col]
        top = f"{str(r['exchange']).title()}{'.P' if r['perp'] else ''}"
        return f'{_html.escape(top)}<br><small>{_html.escape(str(r["quote"]))}</small>'

    def dattrs(get):
        return ''.join(
            f' data-v{s}="{(lambda v: "" if v is None or v != v else f"{v:.2f}")(get(s))}"'
            for s in horizons)

    def td(get, extra_cls=''):
        cls = 'cell' + (f' {extra_cls}' if extra_cls else '')
        d = get(default)
        if d is None or d != d:
            bg, fg, txt = grid.MISSING[0], grid.MISSING[1], '—'
        else:
            bg, fg = _band(d)
            txt = _fmt(d)
        return f'<td class="{cls}"{dattrs(get)} style="background:{bg};color:{fg}">{txt}</td>'

    def empty_td():
        attrs = ''.join(f' data-v{s}=""' for s in horizons)
        return f'<td class="cell"{attrs} style="background:{grid.MISSING[0]};color:{grid.MISSING[1]}">—</td>'

    # Per-coin volume-weighted averages (one value per horizon), for the leading Avg column.
    avg = {a: {s: _vw(sub, f'mo_{s}') for s in horizons} for a, sub in valid.groupby('asset')}

    thead = ('<th>Asset</th><th class="avghead">Avg</th>'
             + ''.join(f'<th>{header(c)}</th>' for c in keep))
    rows_html = ''
    for asset in order:
        amap = avg.get(asset, {})
        avg_html = td((lambda s, m=amap: m.get(s, float('nan'))), extra_cls='avgcol')
        tds = ''
        for c in keep:
            if (asset, c) in present:
                r = info.loc[(asset, c)]
                if bool(r.get('nodata')):
                    tds += empty_td()
                else:
                    tds += td(lambda s, row=r: row.get(f'mo_{s}', float('nan')))
            else:
                tds += empty_td()
        rows_html += f'<tr><th>{_html.escape(asset)}</th>{avg_html}{tds}</tr>'

    grid_html = (f'<div class="gridwrap"><table class="cmp"><thead><tr>{thead}</tr></thead>'
                 f'<tbody>{rows_html}</tbody></table></div>')

    # Program headline: volume-weighted markout across every cell, per horizon.
    head_v = {s: _vw(valid, f'mo_{s}') for s in horizons}
    n_coins, n_venues = valid['asset'].nunique(), len(valid)
    hattrs = ''.join(
        f' data-v{s}="{(lambda v: "" if v != v else f"{v:.2f}")(head_v[s])}"' for s in horizons)
    headline_html = (
        f'<div class="headline" id="mo-headline" data-prog="{_html.escape(program)}" '
        f'data-coins="{n_coins}" data-venues="{n_venues}"{hattrs}>'
        f'{_html.escape(_headline_text(program, head_v[default], default, n_coins, n_venues))}</div>')

    legend = grid.swatch_legend(_BANDS_LEGEND)
    note_html = f'<p class="note">{_NOTE}</p>'
    body = [headline_html, ign]
    if not body_only:
        opts = ''.join(f'<option value="{s}"{" selected" if s == default else ""}>{s}s</option>'
                       for s in horizons)
        body.append(f'<label class="hz" for="mohz">Horizon</label><select id="mohz">{opts}</select>')
    body += [legend, grid_html, note_html]
    if not body_only:
        from . import sortable
        body.append(_toggle_js(horizons, default))
        body.append(sortable.SORT_JS)
    return ''.join(head + body + close)


def write_markouts_report(program, cells, ignored, tokens, horizons, start, end, path,
                          period_label=None):
    html = build_markouts_report(program, cells, ignored, tokens, horizons, start, end, period_label)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path
