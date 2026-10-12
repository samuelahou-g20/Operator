"""Generic program × (exchange, quote) heatmap grid — shared by the compliance and captures
reports so the table/colouring/ordering isn't duplicated. Renders server-side (works on a
phone with no JS); an optional metric toggle + JS is added only when a second metric is given.

`cells` is a tidy DataFrame with columns: asset, exchange, quote, perp, col_label, plus the
metric column(s). Metric values are fractions (0..1). `bands` is a descending list of
(min_pct, background, text) colour stops.
"""

import html as _html

MISSING = ('#20262e', '#5a636e')  # no-data cell (background, text)

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:18px;
color:#e6e9ef;background:#0f1216;max-width:100%}
h1{margin:0 0 2px;font-size:20px}
.sub{color:#8b94a3;margin:2px 0;font-size:13px}
.headline{font-size:15px;background:#161a20;border:1px solid #262b33;border-left:3px solid #2f9e44;
border-radius:6px;padding:10px 14px;margin:10px 0}
.ignored{color:#8b94a3;font-size:12.5px;margin:6px 0}
.note{color:#8b94a3;font-size:12.5px;margin:8px 0;max-width:760px}
.legend{margin:10px 0;font-size:12px}
.legend span{display:inline-block;padding:2px 8px;border-radius:4px;margin-right:6px;color:#141414}
button{background:#1a1f26;color:#e6e9ef;border:1px solid #3a424e;border-radius:6px;padding:6px 12px;
font-size:13px;cursor:pointer;margin:6px 0}
.gridwrap{overflow:auto;max-height:82vh;border:1px solid #262b33;border-radius:8px;margin-top:8px}
table.cmp{border-collapse:separate;border-spacing:0;font-size:12px}
table.cmp th,table.cmp td{border-right:1px solid #262b33;border-bottom:1px solid #262b33;
padding:5px 7px;white-space:nowrap;text-align:center}
table.cmp thead th{position:sticky;top:0;background:#1a1f26;z-index:2;font-weight:600;line-height:1.25}
table.cmp thead th small{color:#8b94a3;font-weight:400}
table.cmp tbody th{position:sticky;left:0;background:#161a20;text-align:left;z-index:1;font-weight:600}
table.cmp tbody th,table.cmp thead th:first-child{width:78px;min-width:78px;max-width:78px;
overflow:hidden;text-overflow:ellipsis}
table.cmp thead th:first-child{left:0;z-index:3}
table.cmp thead th.avghead{position:sticky;left:78px;z-index:3;background:#1a1f26}
table.cmp td.avgcol{position:sticky;left:78px;z-index:1;border-right:2px solid #3a424e}
td.cell{font-variant-numeric:tabular-nums;font-weight:600}
td.cell[title]{cursor:help}
td.cell.dim{opacity:.28}
.gridwrap.show-all td.cell.dim{opacity:1}
.osmark{font-size:9px;vertical-align:super;opacity:.85;margin-left:1px}
"""


def band(bands, v):
    """(background, text) colour for a value v (in %) against descending `bands`."""
    for lo, bg, fg in bands:
        if v >= lo:
            return bg, fg
    return bands[-1][1], bands[-1][2]


def swatch_legend(bands_legend, extra=''):
    """Build the coloured band legend from [(label, bg, fg), ...] plus a 'no data' swatch."""
    spans = ''.join(f'<span style="background:{bg};color:{fg}">{_html.escape(lbl)}</span>'
                    for lbl, bg, fg in bands_legend)
    spans += f'<span style="background:{MISSING[0]};color:{MISSING[1]}">no data</span>'
    return f'<div class="legend">{spans}{extra}</div>'


def _toggle_js(bands, glyph):
    conds = ''.join(f"if(v>={lo})return['{bg}','{fg}'];" for lo, bg, fg in bands)
    return (f"<script>(function(){{"
            f"function band(v){{{conds}return['{MISSING[0]}','{MISSING[1]}'];}}"
            f"function render(m){{document.querySelectorAll('td.cell').forEach(function(td){{"
            f"var v=td.getAttribute('data-'+m);"
            f"if(v===null||v===''){{td.textContent='—';td.style.background='{MISSING[0]}';"
            f"td.style.color='{MISSING[1]}';return;}}"
            f"v=parseFloat(v);var b=band(v);var mk=td.getAttribute('data-mark');"
            f"td.innerHTML=Math.round(v)+'%'+(mk?' <span class=\\\"osmark\\\">{glyph}</span>':'');"
            f"td.style.background=b[0];td.style.color=b[1];}});}}"
            f"var metric='p';var btn=document.getElementById('toggle');"
            f"if(btn){{btn.addEventListener('click',function(){{metric=metric==='p'?'s':'p';"
            f"btn.textContent=metric==='p'?btn.getAttribute('data-tsec'):btn.getAttribute('data-tpri');"
            f"render(metric);}});}}}})();</script>")


def _layout(cells, value_col, row_key=None):
    """Shared pivot + ordering used by both the HTML grid and the PNG image, so they match.
    Rows are ordered ascending by `row_key` (a Series asset->value, e.g. the Avg column) when
    given, else by the row's flat mean. Returns (val, mask, keep_columns, row_order)."""
    val = cells.pivot(index='asset', columns='col_label', values=value_col)
    nod = cells.pivot(index='asset', columns='col_label', values='nodata') if 'nodata' in cells.columns else None
    mask = (val.notna() & (nod != True)) if nod is not None else val.notna()
    n_coins = mask.sum(axis=0)
    tie = val.where(mask).mean(axis=0)
    keep = [c for c in n_coins.index if n_coins[c] > 0]
    keep.sort(key=lambda c: (int(n_coins[c]), 0.0 if tie[c] != tie[c] else float(tie[c])), reverse=True)
    key = row_key.reindex(val.index) if row_key is not None else val.where(mask).mean(axis=1)
    order = key.sort_values(ascending=True, na_position='last').index.tolist()
    return val, mask, keep, order


def build_grid(*, title, h1, subtitle, program, cells, ignored, value_col, bands, bands_legend,
               empty_msg, headline_html, note='', hover=None, avg_primary=None,
               avg_secondary=None, avg_hover=None, toggle=None, mark=None, legend_extra='',
               dim=None, body_only=False):
    """Render the full HTML grid.
    toggle: None, or dict(col, tsec, tpri) — a second metric column + button labels.
    mark:   None, or dict(col, threshold, glyph) — a corner glyph when col >= threshold.
    avg_primary/avg_secondary: Series asset->fraction for the leading Avg column (None omits it).
    body_only: return just the section fragment (no page shell, no toggle button/script) — the
               static default view, for embedding (e.g. the no-JS aiozp base). Needs grid._CSS.
    """
    if body_only:
        head, close = [f'<h3>{h1}</h3>', f'<p class="sub">{subtitle}</p>'], []
    else:
        head = ['<!doctype html><html><head><meta charset="utf-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1">',
                f'<title>{_html.escape(title)}</title>',
                f'<style>{_CSS}</style></head><body>',
                f'<h1>{h1}</h1>', f'<p class="sub">{subtitle}</p>']
        close = ['</body></html>']
    ign = f'<p class="ignored">ignored (not in config): {", ".join(ignored)}</p>' if ignored else ''
    note_html = f'<p class="note">{note}</p>' if note else ''
    if cells is None or getattr(cells, 'empty', True):
        return ''.join(head + [f'<div class="headline">{empty_msg}</div>', ign, note_html] + close)

    val, mask, keep, order = _layout(cells, value_col, row_key=avg_primary)
    sec = cells.pivot(index='asset', columns='col_label', values=toggle['col']) if toggle else None
    markm = cells.pivot(index='asset', columns='col_label', values=mark['col']) if mark else None
    dimm = cells.pivot(index='asset', columns='col_label', values=dim['col']) if dim else None

    metah = cells.drop_duplicates('col_label').set_index('col_label')

    def header(col):
        r = metah.loc[col]
        top = f"{str(r['exchange']).title()}{'.P' if r['perp'] else ''}"
        return f'{_html.escape(top)}<br><small>{_html.escape(str(r["quote"]))}</small>'

    def cell(pv, sv=float('nan'), title='', extra_cls='', marked=False, dimmed=False):
        cls = 'cell' + (f' {extra_cls}' if extra_cls else '') + (' dim' if dimmed else '')
        if pv != pv:
            return (f'<td class="{cls}" data-p="" data-s="" data-mark="" '
                    f'style="background:{MISSING[0]};color:{MISSING[1]}">—</td>')
        v = pv * 100
        bg, fg = band(bands, v)
        ds = f'{sv * 100:.0f}' if sv == sv else ''
        t = f' title="{_html.escape(title)}"' if title else ''
        g = f'<span class="osmark">{mark["glyph"]}</span>' if marked else ''
        return (f'<td class="{cls}" data-p="{v:.0f}" data-s="{ds}" data-mark="{"1" if marked else ""}"'
                f'{t} style="background:{bg};color:{fg}">{v:.0f}%{g}</td>')

    show_avg = avg_primary is not None
    thead = ('<th>Asset</th>' + ('<th class="avghead">Avg</th>' if show_avg else '')
             + ''.join(f'<th>{header(c)}</th>' for c in keep))
    info = cells.set_index(['asset', 'col_label'])
    rows_html = ''
    for asset in order:
        avg_html = ''
        if show_avg:
            ap = avg_primary.get(asset, float('nan'))
            asx = avg_secondary.get(asset, float('nan')) if avg_secondary is not None else float('nan')
            at = avg_hover(asset) if callable(avg_hover) else ''
            avg_html = cell(ap, asx, title=at, extra_cls='avgcol')
        tds = ''
        for c in keep:
            if bool(mask.at[asset, c]) and val.at[asset, c] == val.at[asset, c]:
                sv = sec.at[asset, c] if sec is not None else float('nan')
                marked = bool(markm is not None and float(markm.at[asset, c] or 0) >= mark['threshold'])
                dimmed = bool(dimm is not None and dimm.at[asset, c])
                title = hover(asset, c, info.loc[(asset, c)]) if callable(hover) else ''
                tds += cell(val.at[asset, c], sv, title=title, marked=marked, dimmed=dimmed)
            else:
                tds += cell(float('nan'))
        rows_html += f'<tr><th>{_html.escape(asset)}</th>{avg_html}{tds}</tr>'

    grid = (f'<div class="gridwrap"><table class="cmp"><thead><tr>{thead}</tr></thead>'
            f'<tbody>{rows_html}</tbody></table></div>')
    parts = [f'<div class="headline">{headline_html}</div>', ign]
    if toggle and not body_only:
        parts.append(f'<button id="toggle" data-tsec="{_html.escape(toggle["tsec"])}" '
                     f'data-tpri="{_html.escape(toggle["tpri"])}">{_html.escape(toggle["tsec"])}</button>')
    if dim and not body_only:  # reveal the dimmed (non-MM) venues; headline/averages stay MM-only
        parts.append(f'<button id="showall">{_html.escape(dim.get("label", "Show all"))}</button>')
    parts += [swatch_legend(bands_legend, legend_extra), grid, note_html]
    if toggle and not body_only:
        parts.append(_toggle_js(bands, mark['glyph'] if mark else '◐'))
    if dim and not body_only:
        parts.append(
            '<script>(function(){var b=document.getElementById("showall"),'
            'w=document.querySelector(".gridwrap");if(!b||!w)return;var on=false;'
            'b.addEventListener("click",function(){on=!on;w.classList.toggle("show-all",on);'
            'b.textContent=on?"MM only":"' + _html.escape(dim.get("label", "Show all")) + '";});})();</script>')
    if not body_only:
        from . import sortable
        parts.append(sortable.SORT_JS)
    return ''.join(head + parts + close)
