"""The aiozp desk 'headline' table — one row per coin with the numbers that matter at a glance:
net traded, PnL (MM / Gamma / Total), 2% compliance, capture and 180s markout. Everything is pulled
from the per-coin ZPResults the bundle already computed (no extra fetching); rows are sorted worst
Total PnL first. A plain table, so it reads on a no-JS phone like the other Summary grids.
"""

import html as _html

import numpy as np
import pandas as pd

_GREEN, _RED = '#3fb950', '#d9534f'
_MO_SEC = 180  # markout horizon shown (canonical default)
# Background bands matching the compliance/captures grids (value %, descending): (thr, bg, fg).
_COMP_BANDS = [(95, '#2f9e44', '#fff'), (90, '#e0b100', '#141414'),
               (75, '#e8833a', '#141414'), (0, '#d64545', '#fff')]
_CAP_BANDS = [(10, '#2f9e44', '#fff'), (8, '#e0b100', '#141414'),
              (5, '#e8833a', '#141414'), (0, '#d64545', '#fff')]
# markout grid's diverging ±5 bps scale — used for both Markout and PnL bps.
_MO_BANDS = [(5.0, '#1b6b3a', '#d7f5df'), (1.0, '#3f7a52', '#e8f7ee'), (-1.0, '#33383f', '#c7ccd4'),
             (-5.0, '#9a6327', '#ffe9cf'), (-1e9, '#a83232', '#ffd9d9')]
# /vol grid's ratio bands (diverging around 1).
_VOLREF_BANDS = [(1.6, '#9e2f2f', '#ffdada'), (1.3, '#a3571f', '#ffe2c4'), (1.12, '#7d6320', '#f2d79a'),
                 (0.9, '#2b3038', '#c7ccd4'), (0.8, '#22505f', '#bfe6f2'), (-1e9, '#1c5f70', '#c7f2ff')]


def _band(bands, v):
    for thr, bg, fg in bands:
        if v >= thr:
            return bg, fg
    return bands[-1][1], bands[-1][2]


def _total_pnl(df, account):
    if df is None or getattr(df, 'empty', True):
        return None
    m = df[df['account'] == account]
    return float(m.iloc[0]['pnl_$']) if not m.empty else None


def _row(coin, res):
    if res is None:
        return {'coin': coin}
    net = vol = None
    ts = res.total_summary
    if ts is not None and not ts.empty:
        r = ts[ts['account'] == 'TOTAL']
        if not r.empty:
            net = float(r.iloc[0]['net_notional'])
            vol = float(r.iloc[0]['buy_notional']) + float(r.iloc[0]['sell_notional'])  # gross traded
    comp = uptime = None
    breakdown = {}  # wide / one_sided / empty composition (flat avg %), for the compliance hover
    oc = res.ob_compliance
    if oc is not None and not oc.empty:
        valid = oc[~oc['nodata']] if 'nodata' in oc.columns else oc
        if not valid.empty:
            comp = float(valid['compliance'].mean()) * 100
            uptime = float(valid['uptime'].mean()) * 100
            for k in ('wide', 'one_sided', 'empty'):
                if k in valid.columns:
                    breakdown[k] = float(valid[k].mean()) * 100
    cap = market = g20 = cap_all = None
    c = res.captures
    if c is not None and not c.empty:
        wm = c[(c.index.get_level_values(0) != 'TOTAL') & (c['Market Volume $'].fillna(0) > 0)]
        if not wm.empty:
            ga, ma = float(wm['G20 Volume $'].sum()), float(wm['Market Volume $'].sum())
            cap_all = ga / ma * 100 if ma > 0 else None          # all configured venues
            act = wm[wm['ob_active'].fillna(True)] if 'ob_active' in wm.columns else wm  # MM only
            g20, market = float(act['G20 Volume $'].sum()), float(act['Market Volume $'].sum())
            cap = g20 / market * 100 if market > 0 else None
    mo_by_h = {}  # {horizon_sec: volume-weighted markout bps} across the coin's venues
    mk = res.markouts
    if mk is not None and not mk.empty and 'volume_$' in mk.columns:
        w = mk['volume_$'].to_numpy(float)
        for c in mk.columns:
            if c.startswith('pnl_') and c.endswith('s_bps'):
                y = mk[c].to_numpy(float)
                keep = ~np.isnan(y) & ~np.isnan(w) & (w > 0)
                if keep.any():
                    mo_by_h[int(c[4:-5])] = float(np.average(y[keep], weights=w[keep]))
    pmove = None
    if res.price_ohlc and res.price_ohlc.get('pct_change') is not None:
        pmove = float(res.price_ohlc['pct_change'])
    total = _total_pnl(res.total_summary, 'TOTAL')
    return {'coin': coin, 'net': net, 'vol': vol, 'pmove': pmove,
            'mm': _total_pnl(res.mm_summary, 'MM TOTAL'),
            'gamma': _total_pnl(res.other_summary, 'GAMMA TOTAL'), 'total': total,
            'pnl_bps': (total / vol * 10000) if (total is not None and vol) else None,
            'uptime': uptime, 'compliance': comp, 'breakdown': breakdown,
            'capture': cap, 'market': market, 'g20': g20, 'cap_all': cap_all,
            'markout': mo_by_h.get(_MO_SEC), 'mo_by_h': mo_by_h}


def summary_rows(results, vol_info=None):
    """{coin: ZPResult|None} -> list of row dicts, highest volume first (no-volume coins last).
    `vol_info` {coin: {'ratio','ref','med','p90','mx'}} feeds the Vol/ref column + its hover."""
    vol_info = vol_info or {}
    rows = []
    for coin in results:
        r = _row(coin, results.get(coin))
        r['volinfo'] = vol_info.get(coin)
        rows.append(r)
    return sorted(rows, key=lambda r: (r.get('vol') is None, -(r.get('vol') or 0.0)))


_NUM = 'font-variant-numeric:tabular-nums'


def _dv(v):  # numeric sort key on the cell (empty -> sorts to the bottom)
    return f' data-v="{v:.6g}"' if v is not None and v == v else ' data-v=""'


def _compact(v):
    if v is None or v != v:
        return '—'
    a = abs(v)
    return f'${v/1e9:.1f}B' if a >= 1e9 else f'${v/1e6:.1f}M' if a >= 1e6 else \
        f'${v/1e3:.0f}k' if a >= 1e3 else f'${v:,.0f}'


def _plain_money(x):
    return '—' if x is None or x != x else f'{"−" if x < 0 else ""}${abs(x):,.0f}'


def _money(v):
    if v is None or v != v:
        return '<td data-v="">—</td>'
    col = _GREEN if v > 0 else (_RED if v < 0 else '')
    sty = (f'color:{col};' if col else '') + _NUM
    return f'<td{_dv(v)} style="{sty}">{_plain_money(v)}</td>'


def _pct(v, bands=None):
    if v is None or v != v:
        return '<td data-v="">—</td>'
    if bands:
        bg, fg = _band(bands, v)
        return f'<td{_dv(v)} style="background:{bg};color:{fg};{_NUM}">{v:.0f}%</td>'
    return f'<td{_dv(v)}>{v:.0f}%</td>'


def _banded(v, bands, fmt):
    """Background-banded numeric cell (markout / PnL bps / Vol-ref)."""
    if v is None or v != v:
        return '<td data-v="">—</td>'
    bg, fg = _band(bands, v)
    return f'<td{_dv(v)} style="background:{bg};color:{fg};{_NUM}">{fmt(v)}</td>'


def _total_cell(total, mm, gamma):
    """Total PnL $ (signed green/red), with the MM / Gamma split on hover."""
    if total is None or total != total:
        return '<td data-v="">—</td>'
    col = _GREEN if total > 0 else (_RED if total < 0 else '')
    tip = f'MM {_plain_money(mm)} · Gamma {_plain_money(gamma)}'
    sty = (f'color:{col};' if col else '') + _NUM + ';cursor:help'
    return f'<td{_dv(total)} title="{tip}" style="{sty}">{_plain_money(total)}</td>'


def _capture_cell(cap, market, g20, cap_all=None):
    """Capture % (MM venues, banded), with market/G20 volumes and the all-venues capture on hover."""
    if cap is None or cap != cap:
        return '<td data-v="">—</td>'
    bg, fg = _band(_CAP_BANDS, cap)
    bits = []
    if market:
        bits.append(f'MM vol {_compact(market)} · G20 {_compact(g20)}')
    if cap_all is not None and cap_all == cap_all:
        bits.append(f'all venues {cap_all:.0f}%')
    tip = f' title="{" · ".join(bits)}"' if bits else ''
    cur = ';cursor:help' if bits else ''
    return f'<td{_dv(cap)}{tip} style="background:{bg};color:{fg};{_NUM}{cur}">{cap:.0f}%</td>'


def _compliance_cell(comp, breakdown):
    """Compliance % (banded), with the wide / one-sided / empty composition on hover."""
    if comp is None or comp != comp:
        return '<td data-v="">—</td>'
    bg, fg = _band(_COMP_BANDS, comp)
    labels = (('wide', 'wide'), ('one_sided', 'one-sided'), ('empty', 'empty'))
    bits = [f'{lab} {breakdown[k]:.0f}%' for k, lab in labels if breakdown and breakdown.get(k) is not None]
    tip = f' title="{" · ".join(bits)}"' if bits else ''
    cur = ';cursor:help' if bits else ''
    return f'<td{_dv(comp)}{tip} style="background:{bg};color:{fg};{_NUM}{cur}">{comp:.0f}%</td>'


def _markout_cell(mo, mo_by_h):
    """180s markout (banded on the markout scale), with every horizon's decay on hover."""
    if mo is None or mo != mo:
        return '<td data-v="">—</td>'
    bg, fg = _band(_MO_BANDS, mo)
    bits = [f'{h}s {mo_by_h[h]:+.1f}' for h in sorted(mo_by_h)] if mo_by_h else []
    tip = f' title="{" · ".join(bits)}"' if bits else ''
    cur = ';cursor:help' if bits else ''
    return f'<td{_dv(mo)}{tip} style="background:{bg};color:{fg};{_NUM}{cur}">{mo:+.1f}</td>'


def _volref_cell(info):
    """Vol/ref ratio (banded on the vol-grid scale), with ref + realized median/p90/max on hover."""
    if not info or info.get('ratio') is None:
        return '<td data-v="">—</td>'
    ratio = info['ratio']
    bg, fg = _band(_VOLREF_BANDS, ratio)
    bits = [f'{lab} {info[k]:.0f}%' for k, lab in
            (('ref', 'ref'), ('med', 'realized median'), ('p90', 'p90'), ('mx', 'max'))
            if info.get(k) is not None]
    tip = f' title="{" · ".join(bits)}"' if bits else ''
    cur = ';cursor:help' if bits else ''
    return f'<td{_dv(ratio)}{tip} style="background:{bg};color:{fg};{_NUM}{cur}">{ratio:.2f}×</td>'


def _vol(v):  # gross traded, unsigned magnitude — compact so the wide table stays scannable
    if v is None or v != v:
        return '<td data-v="">—</td>'
    return f'<td{_dv(v)} style="{_NUM}">{_compact(v)}</td>'


def _pmove(v):  # coin's price change over the window — simple green up / red down
    if v is None or v != v:
        return '<td data-v="">—</td>'
    col = _GREEN if v > 0 else (_RED if v < 0 else '')
    sty = (f'color:{col};' if col else '') + _NUM
    return f'<td{_dv(v)} style="{sty}">{v:+.1f}%</td>'


def _desk_pnl_curve(results):
    """Sum every coin's cumulative Total PnL onto one shared time grid. Returns (index, series) or
    None. Each coin's series is forward-filled onto the union of all timestamps; before a coin's
    first mark it holds no position, so it contributes $0 (fillna(0)) rather than dragging the sum."""
    series = []
    for res in results.values():
        dp = getattr(res, 'delta_pnl_ts', None)
        if dp is None or getattr(dp, 'empty', True) or 'cum_total_pnl' not in getattr(dp, 'columns', []):
            continue
        s = dp.set_index('timestamp')['cum_total_pnl']
        s = s[~s.index.duplicated(keep='last')].sort_index()
        if not s.empty:
            series.append(s)
    if not series:
        return None
    idx = series[0].index
    for s in series[1:]:
        idx = idx.union(s.index)
    total = None
    for s in series:
        r = s.reindex(idx).ffill().fillna(0.0)
        total = r if total is None else total.add(r, fill_value=0.0)
    return idx, total


_SVG_W, _SVG_H = 760, 200
_PAD_L, _PAD_R, _PAD_T, _PAD_B = 10, 74, 16, 22


def desk_pnl_svg(results):
    """Inline SVG (no chart lib) of desk-wide cumulative Total PnL over the window — the boss view.
    Green/red by final sign, with a zero baseline, min/max on the axis and the running end value."""
    cur = _desk_pnl_curve(results)
    if cur is None:
        return ''
    idx, total = cur
    ys = total.to_numpy(float)
    n = len(ys)
    if n < 2:
        return ''
    if n > 420:                                   # downsample (even stride) to keep the SVG small
        step = int(np.ceil(n / 420))
        sel = np.arange(0, n, step)
        if sel[-1] != n - 1:
            sel = np.append(sel, n - 1)
        ys, times = ys[sel], idx[sel]
    else:
        times = idx
    m = len(ys)
    lo, hi = min(float(np.min(ys)), 0.0), max(float(np.max(ys)), 0.0)  # always frame the zero line
    span = (hi - lo) or 1.0
    x0, x1, y0, y1 = _PAD_L, _SVG_W - _PAD_R, _PAD_T, _SVG_H - _PAD_B
    X = lambda i: x0 + (x1 - x0) * i / (m - 1)
    Y = lambda v: y1 - (y1 - y0) * (v - lo) / span
    pts = ' '.join(f'{X(i):.1f},{Y(ys[i]):.1f}' for i in range(m))
    zy = Y(0.0)
    final = ys[-1]
    color = _GREEN if final >= 0 else _RED
    fx = X(m - 1)
    try:
        t_lo, t_hi = pd.Timestamp(times[0]), pd.Timestamp(times[-1])
        xlab_l, xlab_r = t_lo.strftime('%H:%M'), t_hi.strftime('%H:%M')
    except Exception:
        xlab_l = xlab_r = ''
    return (
        f'<div class="ds-pnl"><div class="ds-pnl-h">Desk cumulative PnL '
        f'<b style="color:{color}">{_plain_money(final)}</b></div>'
        f'<svg viewBox="0 0 {_SVG_W} {_SVG_H}" preserveAspectRatio="none" class="ds-pnl-svg" '
        f'role="img" aria-label="Desk cumulative PnL over the window">'
        f'<polygon points="{X(0):.1f},{zy:.1f} {pts} {fx:.1f},{zy:.1f}" '
        f'fill="{color}" fill-opacity="0.14"/>'
        f'<line x1="{x0}" y1="{zy:.1f}" x2="{x1}" y2="{zy:.1f}" stroke="#3a424e" '
        f'stroke-width="1" stroke-dasharray="3 3"/>'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6" '
        f'stroke-linejoin="round"/>'
        f'<circle cx="{fx:.1f}" cy="{Y(final):.1f}" r="2.6" fill="{color}"/>'
        f'<text x="{x1 + 6}" y="{Y(hi):.1f}" class="ds-pnl-ax" dy="3">{_compact(hi)}</text>'
        f'<text x="{x1 + 6}" y="{Y(lo):.1f}" class="ds-pnl-ax" dy="3">{_compact(lo)}</text>'
        f'<text x="{x0}" y="{_SVG_H - 6}" class="ds-pnl-ax">{xlab_l}</text>'
        f'<text x="{x1}" y="{_SVG_H - 6}" class="ds-pnl-ax" text-anchor="end">{xlab_r} UTC</text>'
        f'</svg></div>')


_PNL_CSS = ('<style>.ds-pnl{background:#12161c;border:1px solid #262b33;border-radius:8px;'
            'padding:10px 12px 4px;margin:8px 0}.ds-pnl-h{font-size:13px;color:#c7ccd4;margin:0 0 2px}'
            '.ds-pnl-svg{width:100%;height:auto;display:block}'
            '.ds-pnl-ax{fill:#8b94a3;font-size:11px;font-variant-numeric:tabular-nums}</style>')


_CTRL_CSS = ('<style>.ds-ctrls{display:flex;gap:10px;align-items:center;margin:8px 0 4px;flex-wrap:wrap}'
             '.ds-q{background:#161a20;color:#e6e9ef;border:1px solid #3a424e;border-radius:6px;'
             'padding:6px 10px;font-size:13px;min-width:150px}'
             '.ds-cw{position:relative}#ds-colbtn{background:#161a20;color:#e6e9ef;border:1px solid '
             '#3a424e;border-radius:6px;padding:6px 10px;font-size:13px;cursor:pointer}'
             '.ds-menu{position:absolute;top:112%;left:0;z-index:20;background:#161a20;border:1px solid '
             '#3a424e;border-radius:8px;padding:8px 10px;max-height:60vh;overflow:auto;'
             'box-shadow:0 6px 20px rgba(0,0,0,.45)}.ds-menu label{display:block;font-size:12.5px;'
             'color:#c7ccd4;padding:3px 0;white-space:nowrap;cursor:pointer}'
             'th.ccell{cursor:pointer}th.ccell:hover{color:#6ea8ff;text-decoration:underline}</style>')


def _controls_html(headers):
    opts = ''.join(f'<label><input type="checkbox" data-col="{i}" checked> {_html.escape(h)}</label>'
                   for i, h in enumerate(headers) if i > 0)  # col 0 (Coin) always shown
    return (_CTRL_CSS + '<div class="ds-ctrls">'
            '<input id="ds-q" class="ds-q" placeholder="Filter coins…" autocomplete="off">'
            '<div class="ds-cw"><button id="ds-colbtn" type="button">Columns ▾</button>'
            f'<div id="ds-colmenu" class="ds-menu" hidden>{opts}</div></div></div>')


_INTERACT_JS = """<script>(function(){
var t=document.querySelector('table.cmp'); if(!t) return;
var q=document.getElementById('ds-q');
if(q) q.addEventListener('input',function(){var s=q.value.trim().toUpperCase();
  t.tBodies[0].querySelectorAll('tr[data-coin]').forEach(function(r){
    r.hidden = !!s && r.getAttribute('data-coin').toUpperCase().indexOf(s)<0; });});
var b=document.getElementById('ds-colbtn'), m=document.getElementById('ds-colmenu');
if(b&&m){ b.addEventListener('click',function(e){e.stopPropagation();m.hidden=!m.hidden;});
  document.addEventListener('click',function(){m.hidden=true;});
  m.addEventListener('click',function(e){e.stopPropagation();});
  m.querySelectorAll('input[data-col]').forEach(function(cb){cb.addEventListener('change',function(){
    var i=+cb.getAttribute('data-col');
    t.querySelectorAll('tr').forEach(function(r){ if(r.cells[i]) r.cells[i].style.display=cb.checked?'':'none'; });
  });});}
t.tBodies[0].querySelectorAll('th.ccell').forEach(function(th){
  function go(e){if(e)e.stopPropagation();var r=th.closest('tr'),c=r&&r.getAttribute('data-coin');
    if(c) try{window.parent.postMessage({t:'aiozp-goto',coin:c},'*');}catch(e){}}
  th.addEventListener('click',go);
  th.addEventListener('keydown',function(e){if(e.key==='Enter')go(e);});});
// Row-expand: click anywhere on a coin row (except its name, which opens the ZP) to reveal that
// coin's breakdown — the same numbers as the desktop hovers, but tappable on a phone.
t.tBodies[0].addEventListener('click',function(e){
  if(e.target.closest('th.ccell')) return;
  var row=e.target.closest('tr[data-coin]'); if(!row) return;
  var c=row.getAttribute('data-coin');
  var d=t.querySelector('tr.ds-detail[data-detail-of="'+c+'"]'); if(!d) return;
  d.hidden=!d.hidden; row.classList.toggle('ds-open',!d.hidden);
});
})();</script>"""


_EXP_CSS = ('<style>table.cmp tbody tr[data-coin]{cursor:pointer}'
            'table.cmp tbody tr[data-coin].ds-open td,table.cmp tbody tr[data-coin]:hover td'
            '{background:rgba(110,168,255,.07)}'
            'tr.ds-detail>td{background:#0e1319;text-align:left;white-space:normal;padding:8px 12px;'
            'border-bottom:2px solid #303844}'
            '.ds-chip{display:inline-block;margin:2px 12px 2px 0;font-size:12px;color:#c7ccd4}'
            '.ds-chip em{color:#8b94a3;font-style:normal;margin-right:4px}</style>')


def _detail_cells(r):
    """The expandable breakdown for one coin — same content as the per-cell hovers, as chips."""
    chips = []

    def chip(label, value):
        if value:
            chips.append(f'<span class="ds-chip"><em>{_html.escape(label)}</em> {value}</span>')

    mm, gamma = r.get('mm'), r.get('gamma')
    if mm is not None or gamma is not None:
        chip('PnL split', f'MM {_plain_money(mm)} · Gamma {_plain_money(gamma)}')
    bd = r.get('breakdown') or {}
    parts = [f'{lab} {bd[k]:.0f}%' for k, lab in
             (('wide', 'wide'), ('one_sided', 'one-sided'), ('empty', 'empty')) if bd.get(k) is not None]
    if parts:
        chip('Non-compliant', ' · '.join(parts))
    if r.get('market'):
        cap = f'MM vol {_compact(r["market"])} · G20 {_compact(r.get("g20"))}'
        if r.get('cap_all') is not None and r['cap_all'] == r['cap_all']:
            cap += f' · all venues {r["cap_all"]:.0f}%'
        chip('Capture', cap)
    mo = r.get('mo_by_h') or {}
    if mo:
        chip('Markout decay', ' · '.join(f'{h}s {mo[h]:+.1f}' for h in sorted(mo)))
    info = r.get('volinfo') or {}
    vparts = [f'{lab} {info[k]:.0f}%' for k, lab in
              (('ref', 'ref'), ('med', 'realized median'), ('p90', 'p90'), ('mx', 'max'))
              if info.get(k) is not None]
    if vparts:
        chip('Vol/ref', ' · '.join(vparts))
    return ''.join(chips)


def build_summary_table(program, rows, body_only=False, chart_html=''):
    from . import grid
    if body_only:
        head, close = [f'<h3>{program} — Desk summary</h3>'], []
    else:
        head = ['<!doctype html><html><head><meta charset="utf-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1">',
                f'<title>{_html.escape(f"{program} desk summary")}</title>',
                f'<style>{grid._CSS}</style></head><body>', f'<h1>{program} — Desk summary</h1>']
        close = ['</body></html>']
    rows = [r for r in rows if r.get('coin')]
    if not rows:
        return ''.join(head + ['<div class="headline">No desk data.</div>'] + close)

    ncols = 11
    body_rows = ''
    tot = {k: 0.0 for k in ('net', 'vol', 'mm', 'gamma', 'total')}
    _bps = lambda x: f'{x:+.1f}'
    for r in rows:
        for k in tot:
            if r.get(k) is not None:
                tot[k] += r[k]
        coin = _html.escape(r["coin"])
        body_rows += (
            f'<tr data-coin="{coin}"><th class="ccell" tabindex="0" title="Open {coin} ZP">{coin}</th>'
            f'{_total_cell(r.get("total"), r.get("mm"), r.get("gamma"))}'
            f'{_pmove(r.get("pmove"))}{_money(r.get("net"))}{_vol(r.get("vol"))}'
            f'{_pct(r.get("uptime"), _COMP_BANDS)}'
            f'{_compliance_cell(r.get("compliance"), r.get("breakdown"))}'
            f'{_capture_cell(r.get("capture"), r.get("market"), r.get("g20"), r.get("cap_all"))}'
            f'{_markout_cell(r.get("markout"), r.get("mo_by_h"))}'
            f'{_banded(r.get("pnl_bps"), _MO_BANDS, _bps)}'
            f'{_volref_cell(r.get("volinfo"))}</tr>')
        if not body_only:  # tappable breakdown row (desktop interactive; phones use the per-coin views)
            detail = _detail_cells(r)
            if detail:
                body_rows += (f'<tr class="ds-detail" data-detail-of="{coin}" hidden>'
                              f'<td colspan="{ncols}">{detail}</td></tr>')
    desk_bps = (tot['total'] / tot['vol'] * 10000) if tot['vol'] else None
    body_rows += (f'<tr class="total"><th>TOTAL</th>'
                  f'{_total_cell(tot["total"], tot["mm"], tot["gamma"])}'
                  f'<td>—</td>{_money(tot["net"])}{_vol(tot["vol"])}'
                  f'<td>—</td><td>—</td><td>—</td><td>—</td>'
                  f'{_banded(desk_bps, _MO_BANDS, _bps)}<td>—</td></tr>')

    headers = ['Coin', 'Total PnL', 'Price Δ%', 'Net traded', 'Volume', 'Uptime', 'Compliance',
               'Capture', f'Markout {_MO_SEC}s', 'PnL bps', 'Vol/ref']
    thead = ''.join(f'<th>{h}</th>' for h in headers)
    grid_html = (f'<div class="gridwrap"><table class="cmp"><thead><tr>{thead}</tr></thead>'
                 f'<tbody>{body_rows}</tbody></table></div>')
    controls = '' if body_only else (_EXP_CSS + _controls_html(headers))
    chart = (_PNL_CSS + chart_html) if chart_html else ''
    expand_hint = '' if body_only else ' Click a row for its breakdown; click a coin to open its ZP.'
    note = (f'<p class="note">Per coin over the period: net $ traded, gross volume, Total PnL (hover '
            f'for the MM / Gamma split; MM + Gamma = Total) and PnL in bps of volume; two-sided '
            f'uptime and 2% compliance (flat avg across venues); capture (G20 ÷ market volume); '
            f'volume-weighted {_MO_SEC}s markout; Vol/ref (median realized vol ÷ reference); and the '
            f'coin’s Price Δ% over the window. '
            f'Markout &amp; PnL bps use the markout ±5 bps scale; Vol/ref the vol-grid scale. '
            f'Sorted by volume (highest first).{expand_hint}</p>')
    from . import sortable
    tail = [] if body_only else [sortable.SORT_JS, _INTERACT_JS]
    return ''.join(head + [f'<div class="headline">{program} · desk summary · {len(rows)} coins</div>',
                           chart, controls, grid_html, note] + tail + close)
