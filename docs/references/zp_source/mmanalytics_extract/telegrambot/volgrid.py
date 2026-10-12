"""Desk grid: realized volatility vs the dynamic-mode reference setting, one cell per coin x venue.

Built from the desk's DMC data (the same DmcData the ZP report already fetches per coin). Each cell
compares the median realized annualized vol over the period (`volatility_ann_pct`) to that
strategy's `reference_annualized_volatility_pct` — the ratio is exactly the multiplier (floored at
1) that scales both min and max spread in dynamic mode.

Primary cell metric = the ratio, fully server-rendered so it reads on a no-JS phone. On desktop a
dropdown swaps the cell to "% of period above ref" or "median vol (abs)" and re-colours in place —
same progressive-enhancement pattern as the markouts grid.
"""

import html as _html
import statistics as _stats

from . import dmc as _dmc
from . import grid

# Cell colour bands per metric: (threshold, bg, fg, legend label), descending. First band whose
# threshold the value meets wins. Diverging around 1 for the ratio; sequential for the others.
_RATIO_BANDS = [(1.6, '#9e2f2f', '#ffdada', '≥ 1.6× hot'), (1.3, '#a3571f', '#ffe2c4', '1.3–1.6'),
                (1.12, '#7d6320', '#f2d79a', '1.12–1.3'), (0.9, '#2b3038', '#c7ccd4', '0.9–1.12'),
                (0.8, '#22505f', '#bfe6f2', '0.8–0.9'), (-1e9, '#1c5f70', '#c7f2ff', '< 0.8 calm')]
_PCT_BANDS = [(90, '#9e2f2f', '#ffdada', '≥ 90%'), (70, '#a3571f', '#ffe2c4', '70–90'),
              (40, '#7d6320', '#f2d79a', '40–70'), (15, '#2b3038', '#c7ccd4', '15–40'),
              (-1e9, '#1c5f70', '#c7f2ff', '< 15%')]
_MED_BANDS = [(200, '#7a2547', '#ffd7e6', '≥ 200'), (130, '#7d3a5a', '#f2c8db', '130–200'),
              (100, '#5a4a6e', '#dccef0', '100–130'), (80, '#3a3f52', '#c9cfe0', '80–100'),
              (-1e9, '#2b3038', '#c7ccd4', '< 80')]

# metric key -> (dropdown label, one-line description, value fn on a cell, formatter, bands)
_METRICS = {
    'ratio': ('Ratio (median ÷ ref)',
              'Median realized vol ÷ reference setting — the scaling multiplier (floored at 1) for '
              'min & max spread. >1 = market ran hotter than assumed.',
              lambda c: c['ratio'], lambda v: f'{v:.2f}×', _RATIO_BANDS),
    'pct': ('% of period above ref',
            'How often realized vol sat above the reference over the period.',
            lambda c: c['pct'], lambda v: f'{round(v)}%', _PCT_BANDS),
    'med': ('Median vol (abs)',
            'Absolute median annualized vol — not vs setting; shown for contrast.',
            lambda c: c['med'], lambda v: f'{round(v)}', _MED_BANDS),
}
_DEFAULT = 'ratio'

_CSS_EXTRA = """
select#volmetric{background:#1a1f26;color:#e6e9ef;border:1px solid #3a424e;border-radius:6px;
padding:6px 10px;font-size:13px;margin:6px 8px 6px 0}
label.vm{color:#8b94a3;font-size:13px}
td.vcell{font-variant-numeric:tabular-nums;font-weight:600}
"""

_NOTE = ('Median realized annualized volatility over the period vs each strategy’s reference vol '
         'setting. The ratio (median ÷ ref, floored at 1) is what scales min & max spread in '
         'dynamic mode: >1 the market ran hotter than assumed, <1 calmer (spreads near the floor). '
         'A cell is one dynamic strategy (exchange × quote); greyed where there’s no strategy. '
         'Rows ordered by ratio, columns by how many coins trade the instrument.')


def _pctile(vals_sorted, q):
    return vals_sorted[min(len(vals_sorted) - 1, int(q * (len(vals_sorted) - 1)))]


def compute_vol_cells(dmc_map):
    """{token: DmcData} -> {(coin, col): cell dict} where col = 'EXCHANGE QUOTE' (one dynamic
    strategy per cell). cell keys: ref, med, p90, mx, ratio, pct, n. (n>1 only if two strategies
    share the same exchange+quote — then they're averaged.)"""
    strat = {}  # (coin, col) -> list of per-strategy dicts
    for coin, data in dmc_map.items():
        if data is None:
            continue
        series = {label: hist for label, hist in data.series}
        for base, exchange, dyn, *rest in data.rows:
            perp = rest[0] if rest else False
            ref = (dyn or {}).get('reference_annualized_volatility_pct')
            hist = series.get(_dmc._label(base, exchange, perp)) or []
            vals = [r['volatility_ann_pct'] for r in hist if r.get('volatility_ann_pct') is not None]
            if not ref or not vals:
                continue
            vs = sorted(vals)
            med = _stats.median(vals)
            col = f'{exchange.split(":", 1)[0]}{".P" if perp else ""} {base}'  # e.g. "BINANCE USDT" / "BINANCE.P USDT"
            strat.setdefault((coin, col), []).append({
                'ref': ref, 'med': med, 'p90': _pctile(vs, 0.9), 'mx': max(vals),
                'ratio': med / ref, 'pct': 100 * sum(1 for v in vals if v > ref) / len(vals)})
    cells = {}
    for key, lst in strat.items():
        n = len(lst)
        cells[key] = {'n': n, 'mx': max(x['mx'] for x in lst),
                      **{k: sum(x[k] for x in lst) / n for k in ('ref', 'med', 'p90', 'ratio', 'pct')}}
    return cells


def _band(bands, v):
    for thr, bg, fg, _lab in bands:
        if v >= thr:
            return bg, fg
    return bands[-1][1], bands[-1][2]


_SWATCH = ('display:inline-block;padding:2px 8px;border-radius:4px;margin-right:6px;'
           'background:%s;color:%s')  # inline so the legend styles itself in any embed context


def _legend_html(bands):
    sw = ''.join(f'<span style="{_SWATCH % (bg, fg)}">{_html.escape(lab)}</span>'
                 for _thr, bg, fg, lab in bands)
    return f'<div class="legend" id="vol-legend">{sw}</div>'


def _toggle_js():
    # bands + formatters per metric, then a render(metric) that recolours every cell from its
    # data-<metric> attribute, and swaps the legend. Default metric is already server-rendered.
    import json
    bands = {k: [[t, bg, fg, lab] for t, bg, fg, lab in m[4]] for k, m in _METRICS.items()}
    descs = {k: m[1] for k, m in _METRICS.items()}
    fmts = {'ratio': "v.toFixed(2)+'\\u00d7'", 'pct': "Math.round(v)+'%'", 'med': "''+Math.round(v)"}
    return (
        "<script>(function(){"
        f"var BANDS={json.dumps(bands)},DESC={json.dumps(descs)};"
        "var FMT={ratio:function(v){return " + fmts['ratio'] + ";},"
        "pct:function(v){return " + fmts['pct'] + ";},med:function(v){return " + fmts['med'] + ";}};"
        "function band(bs,v){for(var i=0;i<bs.length;i++){if(v>=bs[i][0])return bs[i];}return bs[bs.length-1];}"
        "function render(m){var bs=BANDS[m];"
        "document.querySelectorAll('td.vcell').forEach(function(td){"
        "var raw=td.getAttribute('data-'+m);if(raw===null||raw===''){return;}"
        "var v=parseFloat(raw),b=band(bs,v);td.textContent=FMT[m](v);"
        "td.style.background=b[1];td.style.color=b[2];});"
        "var lg=document.getElementById('vol-legend');"
        "if(lg){lg.innerHTML=bs.map(function(b){return '<span style=\\'display:inline-block;"
        "padding:2px 8px;border-radius:4px;margin-right:6px;background:'+b[1]+';color:'+b[2]"
        "+'\\'>'+b[3]+'</span>';}).join('');}"
        "var d=document.getElementById('vol-desc');if(d){d.textContent=DESC[m];}}"
        "var sel=document.getElementById('volmetric');"
        "if(sel){sel.addEventListener('change',function(){render(sel.value);});}"
        f"render('{_DEFAULT}');"
        "})();</script>")


def build_vol_report(program, cells, start, end, period_label=None, body_only=False):
    period = period_label or f'{start:%Y-%m-%d %H:%M}–{end:%H:%M} UTC'
    subtitle = f'{start:%Y-%m-%d %H:%M} → {end:%H:%M} UTC'

    if body_only:  # fragment (no shell, no toggle) — the static default-metric view
        head, close = [f'<h3>{program} — Vol vs setting</h3>', f'<p class="sub">{subtitle}</p>'], []
    else:
        head = ['<!doctype html><html><head><meta charset="utf-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1">',
                f'<title>{_html.escape(f"{program} vol vs setting {period}")}</title>',
                f'<style>{grid._CSS}{_CSS_EXTRA}</style></head><body>',
                f'<h1>{program} — Vol vs setting</h1>', f'<p class="sub">{subtitle}</p>']
        close = ['</body></html>']

    if not cells:
        msg = f'{program}: no dynamic-mode volatility data in this period.'
        return ''.join(head + [f'<div class="headline">{msg}</div>'] + close)

    coins_r = {}
    for (coin, _v), c in cells.items():
        coins_r.setdefault(coin, []).append(c['ratio'])
    coins = sorted(coins_r, key=lambda c: -sum(coins_r[c]) / len(coins_r[c]))
    vfreq = {}
    for (_c, venue) in cells:
        vfreq[venue] = vfreq.get(venue, 0) + 1
    venues = sorted(vfreq, key=lambda v: (-vfreq[v], v))

    _, _desc0, _val0, _fmt0, _bands0 = _METRICS[_DEFAULT]  # server-render the default metric
    thead = '<th>Coin</th><th class="avghead">Ref</th>' + ''.join(f'<th>{v}</th>' for v in venues)
    rows_html = ''
    for coin in coins:
        ref = next(c['ref'] for (cc, _v), c in cells.items() if cc == coin)
        tds = f'<td class="avgcol">{round(ref)}</td>'
        for v in venues:
            c = cells.get((coin, v))
            if c is None:
                tds += '<td class="cell" style="background:%s;color:%s">·</td>' % grid.MISSING
                continue
            attrs = ''.join(f' data-{k}="{_METRICS[k][2](c):.4g}"' for k in _METRICS)
            tip = (f'{coin} · {v}  ·  ref {round(c["ref"])} · med {round(c["med"])} · '
                   f'p90 {round(c["p90"])} · max {round(c["mx"])}  ·  {c["n"]} inst')
            bg, fg = _band(_bands0, _val0(c))
            tds += (f'<td class="cell vcell"{attrs} title="{_html.escape(tip)}" '
                    f'style="background:{bg};color:{fg}">{_html.escape(_fmt0(_val0(c)))}</td>')
        rows_html += f'<tr><th>{_html.escape(coin)}</th>{tds}</tr>'

    grid_html = (f'<div class="gridwrap"><table class="cmp"><thead><tr>{thead}</tr></thead>'
                 f'<tbody>{rows_html}</tbody></table></div>')
    n_coins, n_venues = len(coins), len(venues)
    headline = (f'<div class="headline">{program} · vol vs setting · '
                f'{n_coins} coins · {n_venues} venues</div>')

    body = [headline]
    if not body_only:
        opts = ''.join(f'<option value="{k}"{" selected" if k == _DEFAULT else ""}>'
                       f'{_html.escape(m[0])}</option>' for k, m in _METRICS.items())
        body.append(f'<label class="vm" for="volmetric">Metric</label>'
                    f'<select id="volmetric">{opts}</select>')
    body.append(f'<p class="note" id="vol-desc">{_html.escape(_desc0)}</p>')
    body.append(_legend_html(_bands0))
    body.append(grid_html)
    body.append(f'<p class="note">{_NOTE}</p>')
    if not body_only:
        from . import sortable
        body.append(_toggle_js())
        body.append(sortable.SORT_JS)
    return ''.join(head + body + close)
