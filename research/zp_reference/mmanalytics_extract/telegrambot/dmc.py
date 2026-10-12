"""Dynamic mode charts (DMC) — a token's current dynamic mode (MM strategy) settings + volatility/
spread history, from the gamma-dash "price-bias" API. Rendered as a ZP section and by /dmc.
VPN-only (no auth). Fetching happens here; the ZP report (zp_html) renders it.

The token's own *dynamic* strategies (nodes with `ob_dynamic` set — the RE<coin> family, not the
shared REQASIM reference which is static/implicit) are auto-discovered, so only the ZP token is
needed. Endpoints:
  GET /api/price-bias?token=T -> {"data": {T: {BASE: {"EXCH:INSTR": {ob_dynamic, ob_implicit,...}}}}}
  GET /api/price-bias/history?token=T&base=B&exchange=E&hours=H -> {"history": [{...}]}
"""

import html as _html
import json
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime

# gamma-dash "price-bias" API bases, tried in order (first that responds wins). On the bot box the
# service is exposed locally on :8618; the public host is the fallback. Set DMC_BASE_URL to force a
# single base (comma-separated for an explicit ordered list).
_DEFAULT_BASES = ['http://127.0.0.1:8618/api/price-bias',
                  'https://gamma-dash.ext.g20api.com/api/price-bias']
_BASES = ([b.strip().rstrip('/') for b in os.environ['DMC_BASE_URL'].split(',') if b.strip()]
          if os.environ.get('DMC_BASE_URL') else _DEFAULT_BASES)
_working_base = None  # cache the base that last worked, so we don't re-probe every request

# The four chart panels: (history field, title). Overlaid across exchanges (colour per exchange).
# vt_* are the live (volatility-adjusted) values; size, min & max spread all scale with vol now.
SERIES = [('vt_total_value_usd', 'Size ($)'),          # live order size (scales via size factor)
          ('volatility_ann_pct', 'Volatility (Ann %)'),
          ('vt_min_spread_bps', 'Min Spread (bps)'),   # min now scales too (was static); vt_ is live
          ('vt_max_spread_bps', 'Max Spread (bps)')]

# "Current settings" table columns: (header, ob_dynamic key, formatter).
_SETTINGS_COLS = [
    ('Levels', 'levels', lambda v: f'{v:g}'),
    ('Base Min Spread', 'base_min_spread_bps', lambda v: f'{v:g}'),
    ('Base Max Spread', 'base_max_spread_bps', lambda v: f'{v:g}'),
    ('Total Value', 'total_value_usd', lambda v: f'${v:,.0f}'),
    ('Top Lvl Wt', 'top_level_weight', lambda v: f'{v:g}'),
    ('Vol Min Scale', 'volatility_min_spread_scaling_factor', lambda v: f'{v:g}'),  # min now scales
    ('Vol Max Scale', 'volatility_max_spread_scaling_factor', lambda v: f'{v:g}'),
    ('Vol Size Scale', 'volatility_size_scaling_factor', lambda v: f'{v:g}'),       # new: size scaling
    ('Ref Vol (%)', 'reference_annualized_volatility_pct', lambda v: f'{v:g}'),
    ('Vol Interval', 'volatility_interval_ms', lambda v: f'{v:g}ms'),
    ('Vol Halflife', 'volatility_halflife_seconds_list',
     lambda v: ' / '.join(f'{x:g}s' for x in v) if isinstance(v, (list, tuple)) else f'{v:g}s'),
]

PALETTE = ['#e0a13a', '#e0559a', '#3fb950', '#5a6ff0', '#3fa7ff', '#d9534f', '#26c6a8',
           '#b07de0', '#c9a227', '#8bc34a', '#ff8a65', '#7e9ce0', '#f0c674', '#9ac0e0']


@dataclass
class DmcData:
    token: str
    rows: list      # [(base, exchange, ob_dynamic dict)]  — the settings table
    series: list    # [(label, [history record dicts])]    — the charts


def _get(rel, timeout=20):
    """GET a relative path (e.g. '?token=X' or '/history?...') against each candidate base, trying
    the last-known-good base first. Raises the last error if all bases fail."""
    global _working_base
    ordered = ([_working_base] if _working_base else []) + [b for b in _BASES if b != _working_base]
    last = None
    for base in ordered:
        try:
            with urllib.request.urlopen(base + rel, timeout=timeout) as r:
                data = json.load(r)
            _working_base = base
            return data
        except Exception as e:
            last = e
    raise last if last is not None else RuntimeError('no DMC base URL configured')


def _label(base, exchange, perp=False):
    return f'{exchange.split(":", 1)[0]}{".P" if perp else ""} {base} @{exchange}'


def _in_window(recs, start, end):
    """Keep only records whose timestamp falls in [start, end] (both UTC-naive). The history API
    only takes 'hours from now', so the caller fetches back far enough to cover `start` and we trim
    to the actual report window here — so a backdated period lines up with the other grids."""
    if start is None or end is None:
        return recs
    out = []
    for r in recs:
        ts = r.get('timestamp')
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(ts).replace(tzinfo=None)  # strip the +00:00 -> UTC-naive
        except ValueError:
            continue
        if start <= t <= end:
            out.append(r)
    return out


def fetch_settings(tokens, max_workers=12):
    """Settings-only fetch (no history) for /dmccheck: one `?token=` call per gamma name, returning
    a flat list of dynamic strategies as dicts {token, base, exchange, perp, business_area, dyn}.
    Deduped by (token, base, exchange). Failed tokens are skipped (best-effort)."""
    if isinstance(tokens, str):
        tokens = [tokens]

    def _one(gt):
        out = []
        try:
            data = _get(f'?token={urllib.parse.quote(gt)}').get('data', {}).get(gt, {})
        except Exception:
            return out
        for base, exs in data.items():
            for exchange, node in exs.items():
                dyn = node.get('ob_dynamic')
                if dyn:
                    out.append({'token': gt, 'base': base, 'exchange': exchange,
                                'perp': gt.startswith('P.'), 'dyn': dyn,
                                'business_area': (node.get('business_area') or '').lower()})
        return out

    out, seen = [], set()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for recs in ex.map(_one, list(tokens)):
            for r in recs:
                k = (r['token'], r['base'], r['exchange'])
                if k not in seen:
                    seen.add(k)
                    out.append(r)
    return out


def fetch_dmc(tokens, hours=24, max_workers=8, start=None, end=None):
    """Current settings + per-exchange vol/spread history for a coin's dynamic strategies, across
    ALL of its gamma-dash names. `tokens` is a list (a bare str is accepted too) — a coin can trade
    under more than one name and gamma-dash keys data by name (e.g. AI's binance strategy lives
    under 'AIGENSYN'). Dynamic strategies are merged, deduped by (base, exchange), and each history
    call uses the gamma-dash token that strategy came from. When `start`/`end` (UTC-naive) are
    given, history is trimmed to that window (fetch `hours` far enough back to cover it). Returns a
    DmcData, or None if there are no dynamic strategies across any name. **Raises** if a settings
    fetch fails (e.g. gamma-dash unreachable) so the caller can surface the real reason."""
    if isinstance(tokens, str):
        tokens = [tokens]
    strategies, seen = [], set()  # strategies: (gamma_token, base, exchange, ob_dynamic)
    for gt in tokens:
        data = _get(f'?token={urllib.parse.quote(gt)}').get('data', {}).get(gt, {})
        for base, exs in data.items():
            for exchange, node in exs.items():
                if node.get('ob_dynamic') and (base, exchange) not in seen:
                    seen.add((base, exchange))
                    strategies.append((gt, base, exchange, node['ob_dynamic']))
    if not strategies:
        return None

    def _one(s):
        gt, base, exchange, _ = s
        try:
            q = urllib.parse.urlencode({'token': gt, 'base': base, 'exchange': exchange, 'hours': hours})
            recs = _get(f'/history?{q}').get('history') or []
            return (_label(base, exchange, gt.startswith('P.')), _in_window(recs, start, end))
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        series = [s for s in ex.map(_one, strategies) if s and s[1]]
    # 4th field = perp: a perp strategy comes from the prefixed gamma token (P.PARTI), so its
    # settings/min-spread attach to the perp market, not the same-quote spot one.
    rows = [(base, exchange, dyn, gt.startswith('P.')) for gt, base, exchange, dyn in strategies]
    return DmcData(tokens[0], rows, series)


# --------------------------------------------------------------------------------------
# Rendering pieces (pure) — the ZP report (zp_html) wraps these in its chart/PNG framework.
# --------------------------------------------------------------------------------------
def settings_table_html(g):
    from . import dmccheck   # hard-rule check reused here so a breach is highlighted in-place
    th = ''.join(f'<th>{_html.escape(h)}</th>' for h, _, _ in _SETTINGS_COLS)
    body = []
    for base, exchange, d, *rest in g.rows:
        d = d or {}
        perp = rest[0] if rest else False
        pill = (' <span style="padding:1px 6px;border:1px solid #3a6ea5;border-radius:4px;'
                'color:#6ea8ff;font-size:10px;font-weight:700;letter-spacing:.3px;'
                'vertical-align:middle">PERP</span>') if perp else ''
        tds = [f'<td class="gstrat"><b>{_html.escape(exchange.split(":", 1)[0])} {_html.escape(base)}</b>'
               f'{pill} <span class="gex">@{_html.escape(exchange)}</span></td>']
        for _, key, fmt in _SETTINGS_COLS:
            v = d.get(key)
            cell = '—' if v is None else _html.escape(fmt(v))
            reason = None
            hard = dmccheck._HARD.get(key)
            if hard is not None and isinstance(v, (int, float)) and not isinstance(v, bool):
                reason = hard(v, d)                 # numeric hard rules (incl. base_min > max cross-field)
            elif key == 'volatility_halflife_seconds_list' and 'volatility_interval_ms' in d and not v:
                reason = 'empty halflife list'
            if reason:
                tds.append(f'<td style="background:{dmccheck._HARD_BG};color:{dmccheck._HARD_FG};'
                           f'cursor:help" title="{_html.escape(reason)}">{cell}</td>')
            else:
                tds.append(f'<td>{cell}</td>')
        body.append('<tr>' + ''.join(tds) + '</tr>')
    note = (f'<p style="color:#8b94a3;font-size:12px;margin:4px 0 0;cursor:help" '
            f'title="{_html.escape(dmccheck._hard_limits_tooltip())}">'
            f'Red cell = breaches a hard limit (hover the cell for the rule).</p>')
    return (f'<div class="gtablewrap"><table class="gtable"><thead><tr>'
            f'<th class="gstrat">Strategy</th>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>{note}')


def build_dmc_fig(g):
    """2x2 Plotly grid, every exchange overlaid (colour per exchange, legendgroup per exchange)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=[t for _, t in SERIES],
                        vertical_spacing=0.12, horizontal_spacing=0.07)
    for li, (label, hist) in enumerate(g.series):
        recs = hist[::max(1, len(hist) // 800)]  # downsample so the overlay stays light
        x = [r['timestamp'] for r in recs]
        color = PALETTE[li % len(PALETTE)]
        for si, (fieldname, _t) in enumerate(SERIES):
            fig.add_trace(go.Scatter(x=x, y=[r.get(fieldname) for r in recs], mode='lines',
                                     name=label, legendgroup=label, showlegend=(si == 0),
                                     line=dict(color=color, width=1.3)),
                          row=si // 2 + 1, col=si % 2 + 1)
    fig.update_xaxes(showgrid=True, gridcolor='#20262e')
    fig.update_yaxes(showgrid=True, gridcolor='#20262e', rangemode='tozero')
    fig.update_layout(height=680, template='plotly_dark', paper_bgcolor='#161a20',
                      plot_bgcolor='#161a20', margin=dict(t=40, l=48, r=16, b=36),
                      legend=dict(orientation='h', y=-0.1, font=dict(size=10)))
    for ann in fig.layout.annotations:
        ann.font = dict(color='#3fa7ff', size=13)
    return fig


def mpl_dmc_build(g):
    """Return build(fig) drawing the 4 panels (all exchanges overlaid) for the PNG fallback."""
    from datetime import datetime

    def build(fig):
        axes = fig.subplots(2, 2).ravel()
        for li, (label, hist) in enumerate(g.series):
            recs = hist[::max(1, len(hist) // 600)]
            x = [datetime.fromisoformat(r['timestamp']) for r in recs]
            color = PALETTE[li % len(PALETTE)]
            for si, (fieldname, _t) in enumerate(SERIES):
                axes[si].plot(x, [r.get(fieldname) for r in recs], color=color, lw=0.9, label=label)
        for si, (_f, title) in enumerate(SERIES):
            axes[si].set_title(title, color='#3fa7ff', fontsize=9)
            axes[si].grid(alpha=0.15)
            axes[si].tick_params(labelsize=6)
            for lbl in axes[si].get_xticklabels():
                lbl.set_rotation(30)
                lbl.set_ha('right')
        if g.series:
            axes[0].legend(fontsize=5, ncol=2, loc='upper right')
    return build


def select_div_html(g, div_id):
    """Native <select> (readable in dark mode) — the exchange filter. Desktop-only: it lives in the
    interactive layer, so a no-JS phone just sees the static PNG (all exchanges overlaid)."""
    opts = '<option value="all">All</option>' + ''.join(
        f'<option value="{i}">{_html.escape(lbl)}</option>' for i, (lbl, _h) in enumerate(g.series))
    return (f'<div class="gctrl"><label for="{div_id}-sel">Exchange:</label>'
            f'<select id="{div_id}-sel">{opts}</select></div>')


def select_script(g, div_id):
    """Restyle handler — resolves the graph div inside the handler, so it works wherever it sits."""
    return (f'<script>(function(){{var sel=document.getElementById("{div_id}-sel");if(!sel)return;'
            f'var nSer={len(SERIES)},nExch={len(g.series)};'
            f'sel.addEventListener("change",function(){{'
            f'var gd=document.getElementById("{div_id}");if(!gd||!window.Plotly)return;'
            f'var total=nExch*nSer,vis=[],i;'
            f'if(sel.value==="all"){{for(i=0;i<total;i++)vis.push(true);}}'
            f'else{{var e=+sel.value;for(i=0;i<total;i++)vis.push(Math.floor(i/nSer)===e);}}'
            f'Plotly.restyle(gd,{{visible:vis}});}});}})();</script>')
