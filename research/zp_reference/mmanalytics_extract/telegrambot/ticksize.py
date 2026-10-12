"""Desk grid: tick size across a program's configured exchange/pairs, and — for the dynamic-mode
venues — how many ticks the min-spread setting spans.

Tick size is a static market property (from ccxt), so this is a snapshot, not a windowed report.
Primary cell = tick size in bps (tick / mid). A desktop dropdown swaps to "min spread in ticks"
(base_min_spread_bps / tick_bps), populated only where a DMC dynamic strategy exists — the tick-
constraint check: < 1 tick means the min-spread setting is finer than the exchange allows (so it's
unachievable / tick-bound). Same progressive-enhancement pattern as the vol and markouts grids.
"""

import html as _html

from . import grid

# (threshold, bg, fg, legend label), descending — first band the value meets wins.
_TICK_BANDS = [(8, '#9e2f2f', '#ffdada', '≥ 8 bps'), (4, '#a3571f', '#ffe2c4', '4–8'),
               (2, '#7d6320', '#f2d79a', '2–4'), (0.8, '#22505f', '#bfe6f2', '0.8–2'),
               (-1e9, '#1c5f70', '#c7f2ff', '< 0.8 fine')]
_MINTICK_BANDS = [(5, '#1c5f70', '#c7f2ff', '≥ 5 ticks'), (2, '#22505f', '#bfe6f2', '2–5'),
                  (1, '#a3571f', '#ffe2c4', '1–2 floor'), (-1e9, '#9e2f2f', '#ffdada', '< 1 tick')]

# key -> (dropdown label, description, cell value fn, formatter, bands)
_METRICS = {
    'tickbps': ('Tick size (bps)',
                'Exchange tick size as bps of price (tick ÷ mid) — how coarse the price grid is; '
                'higher = chunkier ticks.',
                lambda c: c.get('tickbps'), lambda v: f'{v:.0f}', _TICK_BANDS),
    'minticks': ('Min spread (ticks)',
                 'How many ticks the dynamic-mode min-spread setting spans (base_min_spread_bps ÷ '
                 'tick bps). < 1 = finer than a tick (unachievable / tick-bound). Only shown for '
                 'venues with a dynamic strategy.',
                 lambda c: c.get('minticks'), lambda v: f'{v:.1f}', _MINTICK_BANDS),
}
_DEFAULT = 'tickbps'

_CSS_EXTRA = """
select#tickmetric{background:#1a1f26;color:#e6e9ef;border:1px solid #3a424e;border-radius:6px;
padding:6px 10px;font-size:13px;margin:6px 8px 6px 0}
label.tm{color:#8b94a3;font-size:13px}
td.tcell{font-variant-numeric:tabular-nums;font-weight:600}
"""
_SWATCH = ('display:inline-block;padding:2px 8px;border-radius:4px;margin-right:6px;'
           'background:%s;color:%s')
_NOTE = ('Tick size (minimum price increment) per configured exchange/pair, as bps of the current '
         'mid. The "Min spread (ticks)" view (dynamic-mode venues only) is base_min_spread_bps ÷ '
         'tick bps: below 1 the min-spread setting is finer than the exchange tick allows, so it '
         'can’t actually be quoted. Rows alphabetical; columns by how many coins trade the venue.')


def dmc_min_spreads(dmc_map):
    """{(coin, 'EXCHANGE QUOTE'): base_min_spread_bps} from a desk's DMC data."""
    out = {}
    for coin, data in (dmc_map or {}).items():
        if not data:
            continue
        for base, exchange, dyn, *rest in data.rows:
            perp = rest[0] if rest else False
            m = (dyn or {}).get('base_min_spread_bps')
            if m is not None:
                # .P so a perp's min-spread attaches to its perp tick column, not the spot one.
                col = f'{exchange.split(":", 1)[0]}{".P" if perp else ""} {base}'
                out.setdefault((coin, col), m)
    return out


def _merge_cells(tick_cells, dmc_min):
    """Combine tick data (all venues) with min-spread ticks (DMC venues only)."""
    cells = {}
    for key, t in tick_cells.items():
        tb = t.get('tick_bps')
        cells[key] = {'tick': t.get('tick'), 'mid': t.get('mid'), 'tickbps': tb,
                      'minticks': (dmc_min[key] / tb if key in dmc_min and tb else None)}
    return cells


def _band(bands, v):
    for thr, bg, fg, _lab in bands:
        if v >= thr:
            return bg, fg
    return bands[-1][1], bands[-1][2]


def _legend_html(bands):
    sw = ''.join(f'<span style="{_SWATCH % (bg, fg)}">{_html.escape(lab)}</span>'
                 for _thr, bg, fg, lab in bands)
    return f'<div class="legend" id="tick-legend">{sw}</div>'


def _toggle_js():
    import json
    bands = {k: [[t, bg, fg, lab] for t, bg, fg, lab in m[4]] for k, m in _METRICS.items()}
    descs = {k: m[1] for k, m in _METRICS.items()}
    return (
        "<script>(function(){"
        f"var BANDS={json.dumps(bands)},DESC={json.dumps(descs)};"
        "var FMT={tickbps:function(v){return v.toFixed(0);},minticks:function(v){return v.toFixed(1);}};"
        "var MISS=['" + grid.MISSING[0] + "','" + grid.MISSING[1] + "'];"
        "function band(bs,v){for(var i=0;i<bs.length;i++){if(v>=bs[i][0])return bs[i];}return bs[bs.length-1];}"
        "function render(m){var bs=BANDS[m];"
        "document.querySelectorAll('td.tcell').forEach(function(td){"
        "var raw=td.getAttribute('data-'+m);"
        "if(raw===null||raw===''){td.textContent='\\u00b7';td.style.background=MISS[0];td.style.color=MISS[1];return;}"
        "var v=parseFloat(raw),b=band(bs,v);td.textContent=FMT[m](v);"
        "td.style.background=b[1];td.style.color=b[2];});"
        "var lg=document.getElementById('tick-legend');"
        "if(lg){lg.innerHTML=bs.map(function(b){return '<span style=\\'display:inline-block;"
        "padding:2px 8px;border-radius:4px;margin-right:6px;background:'+b[1]+';color:'+b[2]"
        "+'\\'>'+b[3]+'</span>';}).join('');}"
        "var d=document.getElementById('tick-desc');if(d){d.textContent=DESC[m];}}"
        "var sel=document.getElementById('tickmetric');"
        "if(sel){sel.addEventListener('change',function(){render(sel.value);});}"
        f"render('{_DEFAULT}');"
        "})();</script>")


def _bandcell(v, bands, fmt):
    if v is None or v != v:
        return '<td class="cell" style="background:%s;color:%s">·</td>' % grid.MISSING
    bg, fg = _band(bands, v)
    return f'<td class="cell" style="background:{bg};color:{fg}">{fmt(v)}</td>'


def coin_table(tick_cells, dmc_min, coin):
    """A small per-venue tick table for one coin's ZP report: Venue | Tick bps | Min spread (ticks).
    `tick_cells`/`dmc_min` are the desk-level dicts keyed (coin, col); sliced to `coin` here."""
    cols = sorted(col for (c, col) in tick_cells if c == coin)
    if not cols:
        return ''
    rows = ''
    for col in cols:
        tb = tick_cells[(coin, col)].get('tick_bps')
        key = (coin, col)
        mt = (dmc_min[key] / tb) if (key in dmc_min and tb) else None
        rows += (f'<tr><th>{_html.escape(col)}</th>'
                 f'{_bandcell(tb, _TICK_BANDS, lambda v: f"{v:.0f}")}'
                 f'{_bandcell(mt, _MINTICK_BANDS, lambda v: f"{v:.1f}")}</tr>')
    return ('<div class="gridwrap"><table class="cmp"><thead><tr><th>Venue</th>'
            '<th>Tick bps</th><th>Min spread (ticks)</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')


def build_ticksize_report(program, tick_cells, dmc_min, body_only=False):
    cells = _merge_cells(tick_cells, dmc_min)
    if body_only:
        head, close = [f'<h3>{program} — Tick size</h3>'], []
    else:
        head = ['<!doctype html><html><head><meta charset="utf-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1">',
                f'<title>{_html.escape(f"{program} tick size")}</title>',
                f'<style>{grid._CSS}{_CSS_EXTRA}</style></head><body>',
                f'<h1>{program} — Tick size</h1>']
        close = ['</body></html>']

    if not cells:
        return ''.join(head + [f'<div class="headline">{program}: no tick-size data.</div>'] + close)

    coins = sorted({c for c, _v in cells})
    vfreq = {}
    for (_c, venue) in cells:
        vfreq[venue] = vfreq.get(venue, 0) + 1
    venues = sorted(vfreq, key=lambda v: (-vfreq[v], v))

    _, _desc0, _val0, _fmt0, _bands0 = _METRICS[_DEFAULT]
    thead = '<th>Coin</th>' + ''.join(f'<th>{v}</th>' for v in venues)
    rows_html = ''
    for coin in coins:
        tds = ''
        for v in venues:
            c = cells.get((coin, v))
            if c is None:
                tds += '<td class="cell" style="background:%s;color:%s">·</td>' % grid.MISSING
                continue
            attrs = ''
            for k, m in _METRICS.items():
                val = m[2](c)
                attrs += f' data-{k}="{val:.4g}"' if val is not None else f' data-{k}=""'
            tip = (f'{coin} · {v}  ·  tick {c["tick"]:g} · mid {c["mid"]:g}' if c.get('mid') else
                   f'{coin} · {v}  ·  tick {c["tick"]:g}')
            v0 = _val0(c)
            if v0 is None:
                tds += f'<td class="cell tcell"{attrs} style="background:%s;color:%s">·</td>' % grid.MISSING
            else:
                bg, fg = _band(_bands0, v0)
                tds += (f'<td class="cell tcell"{attrs} title="{_html.escape(tip)}" '
                        f'style="background:{bg};color:{fg}">{_html.escape(_fmt0(v0))}</td>')
        rows_html += f'<tr><th>{_html.escape(coin)}</th>{tds}</tr>'

    grid_html = (f'<div class="gridwrap"><table class="cmp"><thead><tr>{thead}</tr></thead>'
                 f'<tbody>{rows_html}</tbody></table></div>')
    headline = (f'<div class="headline">{program} · tick size · {len(coins)} coins · '
                f'{len(venues)} venues</div>')

    body = [headline]
    if not body_only:
        opts = ''.join(f'<option value="{k}"{" selected" if k == _DEFAULT else ""}>'
                       f'{_html.escape(m[0])}</option>' for k, m in _METRICS.items())
        body.append(f'<label class="tm" for="tickmetric">Metric</label>'
                    f'<select id="tickmetric">{opts}</select>')
    body.append(f'<p class="note" id="tick-desc">{_html.escape(_desc0)}</p>')
    body.append(_legend_html(_bands0))
    body.append(grid_html)
    body.append(f'<p class="note">{_NOTE}</p>')
    if not body_only:
        from . import sortable
        body.append(_toggle_js())
        body.append(sortable.SORT_JS)
    return ''.join(head + body + close)
