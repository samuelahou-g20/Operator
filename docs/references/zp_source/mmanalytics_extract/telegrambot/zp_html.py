"""Interactive single-page HTML ZP report — Plotly charts + styled tables.

Each chart is a static PNG (rendered by matplotlib's Agg backend — pure Python, no browser,
works on any CPU arch) with the interactive Plotly chart layered on top and revealed by JS
once it renders. So a desktop browser gets interactive charts; a phone's no-JS Telegram
viewer shows the static image. The numeric detail (summary + captures) is also emitted as
plain HTML tables, which render everywhere regardless of JS or image support. Plotly's JS
loads from its CDN.
"""

import base64
import html as _html
import io
import json
import sys
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from .zp_format import format_report
from . import dmc as _dmc
from . import logo as _logo
from . import timecut as _timecut

_LAYOUT = dict(margin=dict(t=36, l=48, r=16, b=32), template='plotly_dark',
               paper_bgcolor='#161a20', plot_bgcolor='#161a20',
               legend=dict(font=dict(size=10)), font=dict(size=12))


def _fig_html(fig, include_js=False, div_id=None):
    return pio.to_html(fig, include_plotlyjs=('cdn' if include_js else False),
                       full_html=False, config={'displayModeBar': True, 'responsive': True},
                       **({'div_id': div_id} if div_id else {}))


def _chart_block(fig, png_bytes=None, include_js=False, div_id=None):
    """Static matplotlib PNG (default, for the phone) + interactive Plotly (revealed by JS on
    desktop). If the PNG couldn't be rendered we degrade to interactive-only. `div_id` gives the
    Plotly div a stable id (so the time brush can re-range it)."""
    img = (f'<img class="fallback" src="data:image/png;base64,{base64.b64encode(png_bytes).decode()}">'
           if png_bytes else '')
    return (f'<div class="chart-wrap">{img}<div class="interactive">'
            f'{_fig_html(fig, include_js=include_js, div_id=div_id)}</div></div>')


# --------------------------------------------------------------------------------------
# Static fallback charts — matplotlib (Agg backend, no browser, arch-independent). These
# mirror the interactive Plotly charts below and are what a phone's no-JS viewer shows.
# --------------------------------------------------------------------------------------
_MPL_RC = {
    'figure.facecolor': '#161a20', 'axes.facecolor': '#161a20', 'savefig.facecolor': '#161a20',
    'axes.edgecolor': '#3a424e', 'axes.labelcolor': '#e6e9ef', 'text.color': '#e6e9ef',
    'axes.titlecolor': '#e6e9ef', 'xtick.color': '#8b94a3', 'ytick.color': '#8b94a3',
    'grid.color': '#262b33', 'legend.fontsize': 7, 'font.size': 9, 'figure.dpi': 110,
}


def _mpl_png(build, figsize):
    """Run `build(fig)` under the dark theme and return PNG bytes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context(_MPL_RC):
        fig = plt.figure(figsize=figsize)
        try:
            build(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
        finally:
            plt.close(fig)
    return buf.getvalue()


def _rotate_x(ax):
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha('right')


def _safe_png(build_fn, *args):
    """Render a fallback PNG, or log + return None so the chart degrades to interactive-only."""
    try:
        return build_fn(*args)
    except Exception as e:
        print(f'[zp_html] static fallback PNG failed ({getattr(build_fn, "__name__", build_fn)}): '
              f'{e!r}', file=sys.stderr, flush=True)
        return None


def _mpl_flow_png(trades):
    price_col = 'price_usd' if 'price_usd' in trades.columns else 'price'
    nmax = float(trades['notional_usd'].max()) if 'notional_usd' in trades.columns else 0.0

    def _sz(sub):  # marker area (pt^2) proportional to USD notional; ~10..200 (2x the old default)
        if not (nmax > 0):
            return 20
        return 10 + 190 * (sub['notional_usd'].fillna(0) / nmax)

    def build(fig):
        ax1, ax2 = fig.subplots(2, 1)  # stacked: By side above, By exchange below
        for side, color, label in [('B', '#26a269', 'Buy'), ('S', '#e01b24', 'Sell')]:
            s = trades[trades['side'] == side]
            if not s.empty:
                ax1.scatter(s['exch_ts'], s[price_col], s=_sz(s), c=color, alpha=0.5, label=label)
        ax1.set_title('By side'); ax1.legend()
        for exch, g in trades.groupby('exchange'):
            ax2.scatter(g['exch_ts'], g[price_col], s=_sz(g), alpha=0.5, label=str(exch))
        ax2.set_title('By exchange'); ax2.legend()
        for ax in (ax1, ax2):
            ax.set_ylabel('Price (USD)')
            ax.tick_params(labelsize=7); ax.grid(alpha=0.15); ax.set_xlabel('UTC'); _rotate_x(ax)
    return _mpl_png(build, (7, 7.6))


def _mpl_delta_pnl_png(dp):
    def build(fig):
        ax1, ax2, ax3 = fig.subplots(3, 1, sharex=True)
        t = dp['timestamp']
        ax1.plot(t, dp['price'], color='#4c8dff', lw=1.0)
        ax1.set_title('Price (USD)'); ax1.set_ylabel('Price (USD)')
        ax2.plot(t, dp['cum_delta'], color='#4c8dff', lw=1.2)
        ax2.set_title('Delta ($)'); ax2.set_ylabel('Delta ($)')
        ax3.plot(t, dp['cum_total_pnl'], color='#e06666', lw=1.4, label='Total P&L')
        ax3.plot(t, dp['cum_realised_pnl'], color='#e01b24', lw=1.0, ls='--', label='Realised P&L')
        ax3.plot(t, dp['cum_unrealised_pnl'], color='#26a269', lw=1.0, ls=':', label='Unrealised P&L')
        ax3.set_title('P&L ($)'); ax3.set_ylabel('P&L ($)'); ax3.legend(fontsize=6)
        for ax in (ax1, ax2, ax3):
            ax.tick_params(labelsize=7); ax.grid(alpha=0.15)
        ax3.set_xlabel('UTC'); _rotate_x(ax3)
    return _mpl_png(build, (10, 8))


def _mpl_markouts_png(mk):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('rwg', ['#d73027', '#ffffff', '#1a9850'])
    if 'volume_$' in mk.columns:
        mk = mk.sort_values('volume_$', ascending=False)  # highest volume at top (imshow origin upper)
    bps_cols = [c for c in mk.columns if c.endswith('_bps')]
    labels = [f'{e} {i}' for e, i in mk.index]
    xs = [c.replace('pnl_', '').replace('s_bps', 's') for c in bps_cols]
    z = mk[bps_cols].values.astype(float)
    vmax = (np.nanmax(np.abs(z)) if z.size else 1.0) or 1.0

    def build(fig):
        ax = fig.subplots()
        im = ax.imshow(z, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                v = z[i, j]
                ax.text(j, i, 'n/a' if v != v else f'{v:.1f}', ha='center', va='center',
                        color='#222', fontsize=7)
        ax.set_xlabel('markout window')
        fig.colorbar(im, ax=ax, label='bps', fraction=0.046, pad=0.04)
    return _mpl_png(build, (10, max(2.4, 0.42 * len(labels) + 1.4)))


def _mm_venue_filter(captures_ts, trades, mm_keys):
    """(mm_ts, mm_trades) restricted to MM venues — (exchange, quote) in mm_keys (active orderbook).
    The quote key matches an aliased venue (config pair vs canonical trade inst). Falsy mm_keys ->
    everything (no filter)."""
    if not mm_keys:
        return captures_ts, trades
    q = lambda p: str(p).rsplit('-', 1)[-1].upper()
    cm = [(str(e), q(p)) in mm_keys for e, p in zip(captures_ts['exchange'], captures_ts['pair'])]
    mm_ts = captures_ts[cm]
    if trades is None or trades.empty:
        mm_tr = trades
    else:
        tm = [(str(e), q(i)) in mm_keys for e, i in zip(trades['exchange'], trades['inst'])]
        mm_tr = trades[tm]
    return mm_ts, mm_tr


def _mpl_captures_png(captures_ts, trades, mm_keys=None):
    ts, tr = _mm_venue_filter(captures_ts, trades, mm_keys)   # default view = MM venues
    mkt = ts.set_index('timestamp')['market_volume_usd'].groupby(level=0).sum().sort_index()
    g20 = tr.set_index('exch_ts')['notional_usd'].groupby(level=0).sum().sort_index()
    m = mkt.resample('5min').sum()
    g = g20.resample('5min').sum()
    idx = m.index.union(g.index)
    m = m.reindex(idx, fill_value=0)
    g = g.reindex(idx, fill_value=0)
    cum = (g.cumsum() / m.cumsum().replace(0, np.nan)) * 100

    def build(fig):
        ax = fig.subplots()
        ax.plot(cum.index, cum.values, color='#4c9be8')
        ax.set_ylabel('Capture %'); ax.grid(alpha=0.15)
        ax.set_title('Cumulative capture — MM venues' if mm_keys else 'Cumulative capture')
        ax.set_xlabel('UTC'); _rotate_x(ax)
    return _mpl_png(build, (10, 3.2))


def _mpl_widths_png(ts):
    has_sm = 'tob_bps_smooth' in ts.columns  # default view is smoothed (matches the interactive)
    tcol, bcol = ('tob_bps_smooth', 'bob_bps_smooth') if has_sm else ('tob_bps', 'bob_bps')

    def build(fig):
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
        for (exch, pair), g in ts.groupby(['exchange', 'pair']):
            g = g.sort_values('timestamp')
            label = f'{pair}:{exch}'
            ax1.plot(g['timestamp'], g[tcol], label=label, linewidth=1)
            ax2.plot(g['timestamp'], g[bcol], label=label, linewidth=1)
        ax1.set_title('Top of Book'); ax1.set_ylabel('bps'); ax1.grid(alpha=0.15); ax1.legend()
        ax2.set_title('Back of Book'); ax2.set_ylabel('bps'); ax2.grid(alpha=0.15)
        ax2.set_xlabel('UTC'); _rotate_x(ax2)
    return _mpl_png(build, (10, 5.2))


def _mpl_compliance_png(df):
    rows = list(df.iterrows())
    labels = [f"{r['exchange']} {r['pair']}" for _, r in rows]

    def build(fig):
        ax = fig.subplots()
        yv = list(range(len(labels)))
        left = [0.0] * len(labels)
        for name, col, color in _COMPLIANCE_SEGS:
            vals = [0 if r['nodata'] else r[col] * 100 for _, r in rows]
            ax.barh(yv, vals, left=left, color=color, label=name)
            left = [l + v for l, v in zip(left, vals)]
        nd = [100 if r['nodata'] else 0 for _, r in rows]
        ax.barh(yv, nd, left=left, color='#2b3038', edgecolor='#3a424e', label='No data')
        ax.set_yticks(yv); ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()  # best compliance on top
        ax.set_xlabel('% of window'); ax.set_xlim(0, 100); ax.grid(axis='x', alpha=0.15)
        ax.legend(fontsize=6, ncol=5, loc='lower center', bbox_to_anchor=(0.5, -0.28))
    return _mpl_png(build, (10, max(2.4, 0.5 * len(labels) + 1.6)))


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------
# Flow-chart bubble sizing: markers are scaled by USD notional (area ∝ notional_usd). A
# continuous HTML range slider rescales all of them at once by restyling the shared
# marker.sizeref (see _flow_chart_block). These bound/seed the "largest bubble" diameter in px.
_FLOW_PX_MIN, _FLOW_PX_MAX, _FLOW_PX_DEFAULT = 6, 100, 44
_FLOW_DIV_ID = 'zpflow'


def _area_sizeref(values, max_px):
    """Plotly sizemode='area' sizeref so the largest bubble ≈ max_px px across the figure."""
    m = float(np.nanmax(values)) if len(values) else 0.0
    return (2.0 * m / (max_px ** 2)) if m > 0 else None


def _flow_fig(trades):
    price_col = 'price_usd' if 'price_usd' in trades.columns else 'price'
    notional = trades['notional_usd'] if 'notional_usd' in trades.columns else None
    arr = notional.to_numpy(dtype=float) if notional is not None else np.array([])
    finite = arr[np.isfinite(arr)]
    nmax = float(finite.max()) if finite.size else 0.0
    sized = nmax > 0
    sizeref = _area_sizeref(notional, _FLOW_PX_DEFAULT) if sized else None

    def _marker(sub, base):
        base = dict(base, opacity=0.45)
        if sized:
            base.update(size=sub['notional_usd'], sizemode='area', sizeref=sizeref, sizemin=3)
        else:
            base.update(size=6)
        return base

    def _hover(sub):
        if notional is None:
            return sub['exchange']
        return [f'{e} · ${n:,.0f}' for e, n in zip(sub['exchange'], sub['notional_usd'].fillna(0))]

    # Stacked (rows=2) so each panel is full width — reads better on a phone.
    fig = make_subplots(rows=2, cols=1, subplot_titles=('By side', 'By exchange'),
                        vertical_spacing=0.13)
    for side, color, label in [('B', 'green', 'Buy'), ('S', 'red', 'Sell')]:
        s = trades[trades['side'] == side]
        if not s.empty:
            fig.add_trace(go.Scattergl(x=s['exch_ts'], y=s[price_col], mode='markers',
                                       marker=_marker(s, dict(color=color)),
                                       name=label, hovertext=_hover(s)), row=1, col=1)
    for exch, g in trades.groupby('exchange'):
        fig.add_trace(go.Scattergl(x=g['exch_ts'], y=g[price_col], mode='markers',
                                   marker=_marker(g, dict()), name=str(exch),
                                   hovertext=_hover(g)), row=2, col=1)
    fig.update_yaxes(title_text='Price (USD)')
    fig.update_xaxes(title_text='UTC')
    fig.update_layout(height=680, **_LAYOUT)
    return fig


def _flow_chart_block(trades, png_bytes=None, include_js=False):
    """Flow chart with a continuous 'Bubble size' slider (interactive view only). The slider
    restyles the shared marker.sizeref on the Plotly div so every bubble rescales at once."""
    fig = _flow_fig(trades)
    notional = trades['notional_usd'] if 'notional_usd' in trades.columns else None
    arr = notional.to_numpy(dtype=float) if notional is not None else np.array([])
    finite = arr[np.isfinite(arr)]
    nmax = float(finite.max()) if finite.size else 0.0

    img = (f'<img class="fallback" src="data:image/png;base64,{base64.b64encode(png_bytes).decode()}">'
           if png_bytes else '')
    plot = _fig_html(fig, include_js=include_js, div_id=_FLOW_DIV_ID)
    control = ''
    if nmax > 0:  # slider + a tiny restyle handler; nmax baked in so sizeref = 2*nmax/px^2
        control = (
            f'<div class="flow-size"><label for="{_FLOW_DIV_ID}-size">Bubble size</label>'
            f'<input type="range" id="{_FLOW_DIV_ID}-size" min="{_FLOW_PX_MIN}" '
            f'max="{_FLOW_PX_MAX}" value="{_FLOW_PX_DEFAULT}" step="1"></div>'
            f'<script>(function(){{var nmax={nmax:.6g};'
            f'var gd=document.getElementById("{_FLOW_DIV_ID}");'
            f'var sl=document.getElementById("{_FLOW_DIV_ID}-size");'
            f'if(!gd||!sl)return;sl.addEventListener("input",function(){{'
            f'var px=+sl.value;if(window.Plotly)Plotly.restyle(gd,'
            f'{{"marker.sizeref":2*nmax/(px*px)}});}});}})();</script>')
    return f'<div class="chart-wrap">{img}<div class="interactive">{plot}{control}</div></div>'


def _delta_pnl_fig(dp):
    """Stacked Price / Delta ($) / P&L ($) — net position marked to the 1-min MD mid."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=('Price (USD)', 'Delta ($)', 'P&L ($)'))
    t = dp['timestamp']
    fig.add_trace(go.Scatter(x=t, y=dp['price'], mode='lines', name='Price',
                             line=dict(color='#4c8dff', width=1.3), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=dp['cum_delta'], mode='lines', name='Delta',
                             line=dict(color='#4c8dff', width=1.6), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=dp['cum_total_pnl'], mode='lines', name='Total P&L',
                             line=dict(color='#e06666', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=dp['cum_realised_pnl'], mode='lines', name='Realised P&L',
                             line=dict(color='#e01b24', width=1.3, dash='dash')), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=dp['cum_unrealised_pnl'], mode='lines', name='Unrealised P&L',
                             line=dict(color='#26a269', width=1.3, dash='dot')), row=3, col=1)
    fig.update_yaxes(title_text='Price (USD)', row=1, col=1)
    fig.update_yaxes(title_text='Delta ($)', tickformat=',d', row=2, col=1)
    fig.update_yaxes(title_text='P&L ($)', tickformat=',d', row=3, col=1)
    fig.update_xaxes(title_text='UTC', row=3, col=1)
    fig.update_layout(height=760, hovermode='x unified', **_LAYOUT)
    return fig


def _markouts_fig(mk):
    if 'volume_$' in mk.columns:
        mk = mk.sort_values('volume_$', ascending=False)
    bps_cols = [c for c in mk.columns if c.endswith('_bps')]
    labels = [f'{e} {i}' for e, i in mk.index]
    x = [c.replace('pnl_', '').replace('s_bps', 's') for c in bps_cols]
    z = mk[bps_cols].values.astype(float)
    vmax = np.nanmax(np.abs(z)) if z.size else 1.0
    vmax = vmax or 1.0
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=labels, zmid=0, zmin=-vmax, zmax=vmax,
        colorscale=[[0, '#d73027'], [0.5, '#ffffff'], [1, '#1a9850']],
        text=np.round(z, 1), texttemplate='%{text}', hoverongaps=False,
        colorbar=dict(title='bps')))
    fig.update_layout(height=max(280, 34 * len(labels) + 120), **_LAYOUT)
    fig.update_xaxes(title_text='markout window')
    fig.update_yaxes(autorange='reversed')  # highest-volume venue on top (matches the table)
    return fig, labels, z


def _markouts_chart_block(mk, png_bytes=None, include_js=False):
    """Markout heatmap (venue × window) with a Venue <select> to focus one venue's row — 'All'
    shows every venue (desktop). Static PNG (all venues) for the no-JS phone. Mirrors captures."""
    fig, labels, z = _markouts_fig(mk)
    img = (f'<img class="fallback" src="data:image/png;base64,{base64.b64encode(png_bytes).decode()}">'
           if png_bytes else '')
    fig_html = _fig_html(fig, include_js=include_js, div_id='zpmarkouts')
    ctrl = script = ''
    if len(labels) > 1:  # a single-venue coin needs no filter
        text = np.round(z, 1)
        text = [['' if v != v else v for v in row] for row in text.tolist()]  # NaN -> '' in labels
        zj = json.dumps([[None if v != v else v for v in row] for row in z.tolist()])
        vopts = '<option value="all">All venues</option>' + ''.join(
            f'<option value="{i}">{_html.escape(v)}</option>' for i, v in enumerate(labels))
        ctrl = (f'<div class="gctrl"><label for="zpmarkouts-venue">Venue:</label>'
                f'<select id="zpmarkouts-venue">{vopts}</select></div>')
        script = (
            '<script>(function(){'
            f'var ven=document.getElementById("zpmarkouts-venue");'
            'function gd(){return document.getElementById("zpmarkouts");}'
            # Shared so the time brush (window.__tcMk.set) and the Venue <select> both drive the
            # heatmap: set() swaps the values (venue axis order fixed), apply() re-applies the filter.
            f'var M=window.__tcMk={{Z:{zj},Y:{json.dumps(labels)},T:{json.dumps(text)},'
            'set:function(z,t){this.Z=z;this.T=t;this.apply();},'
            'apply:function(){var g=gd();if(!g||!window.Plotly)return;var v=ven?ven.value:"all",z,y,t;'
            'if(v==="all"){z=this.Z;y=this.Y;t=this.T;}else{var e=+v;z=[this.Z[e]];y=[this.Y[e]];t=[this.T[e]];}'
            'Plotly.restyle(g,{z:[z],y:[y],text:[t]});Plotly.relayout(g,{height:Math.max(200,34*y.length+120)});}};'
            'if(ven)ven.addEventListener("change",function(){M.apply();});})();</script>')
    return f'<div class="chart-wrap">{img}<div class="interactive">{ctrl}{fig_html}{script}</div></div>'


def _cum_capture(mkt_series, g20_series):
    """Cumulative capture % over time = cumulative our-volume / cumulative market-volume, on a
    shared 5-min grid. Returns (index, values%)."""
    m = mkt_series.resample('5min').sum()
    g = g20_series.resample('5min').sum()
    idx = m.index.union(g.index)
    m = m.reindex(idx, fill_value=0)
    g = g.reindex(idx, fill_value=0)
    return idx, ((g.cumsum() / m.cumsum().replace(0, np.nan)) * 100).values


def _captures_fig(captures_ts, trades, mm_keys=None):
    """Cumulative capture %: trace 0 = 'All MM venues' (visible default), trace 1 = 'All venues'
    (hidden), then one hidden trace per (exchange, pair). The Venue <select> in
    _captures_chart_block toggles which is shown."""
    def _agg(ts, tr):
        return _cum_capture(
            ts.set_index('timestamp')['market_volume_usd'].groupby(level=0).sum().sort_index(),
            tr.set_index('exch_ts')['notional_usd'].groupby(level=0).sum().sort_index())

    fig = go.Figure()
    mm_ts, mm_tr = _mm_venue_filter(captures_ts, trades, mm_keys)
    idx, cum = _agg(mm_ts, mm_tr)
    fig.add_trace(go.Scatter(x=idx, y=cum, mode='lines', name='All MM venues'))
    idx, cum = _agg(captures_ts, trades)
    fig.add_trace(go.Scatter(x=idx, y=cum, mode='lines', name='All venues', visible=False))
    venues = []
    for (exch, pair), grp in captures_ts.groupby(['exchange', 'pair']):
        venues.append(f'{pair}:{exch}')
        our = trades[(trades['inst'] == pair) & (trades['exchange'] == exch)]
        idx, cum = _cum_capture(
            grp.set_index('timestamp')['market_volume_usd'].sort_index(),
            our.set_index('exch_ts')['notional_usd'].groupby(level=0).sum().sort_index())
        fig.add_trace(go.Scatter(x=idx, y=cum, mode='lines', name=venues[-1], visible=False))
    fig.update_yaxes(title_text='Capture %')
    fig.update_xaxes(title_text='UTC')
    fig.update_layout(height=320, title='Cumulative capture', **_LAYOUT)
    return fig, venues


def _captures_chart_block(captures_ts, trades, include_js=False, fallback=True, mm_keys=None):
    """Cumulative capture with a Venue <select> (All MM venues [default] / All venues / per
    exchange:pair) that toggles trace visibility (desktop). Static PNG (the MM-venues aggregate)
    for the no-JS phone."""
    fig, venues = _captures_fig(captures_ts, trades, mm_keys)
    png = _safe_png(_mpl_captures_png, captures_ts, trades, mm_keys) if fallback else None
    img = (f'<img class="fallback" src="data:image/png;base64,{base64.b64encode(png).decode()}">'
           if png else '')
    fig_html = _fig_html(fig, include_js=include_js, div_id='zpcaptures')
    vopts = ('<option value="mm" selected>All MM venues</option>'
             '<option value="all">All venues</option>'
             + ''.join(f'<option value="{i}">{_html.escape(v)}</option>' for i, v in enumerate(venues)))
    ctrl = (f'<div class="gctrl"><label for="zpcaptures-venue">Venue:</label>'
            f'<select id="zpcaptures-venue">{vopts}</select></div>')
    script = (
        '<script>(function(){'
        f'var nV={len(venues)};'  # trace 0 = MM agg, 1 = All agg, 2..nV+1 = per venue
        'var ven=document.getElementById("zpcaptures-venue");'
        'function gd(){return document.getElementById("zpcaptures");}'
        'if(ven)ven.addEventListener("change",function(){var g=gd();if(!g||!window.Plotly)return;'
        'var total=nV+2,vis=[],i;for(i=0;i<total;i++)vis.push(false);'
        'if(ven.value==="mm")vis[0]=true;else if(ven.value==="all")vis[1]=true;'
        'else vis[2+(+ven.value)]=true;'
        'Plotly.restyle(g,{visible:vis});});})();</script>')
    return f'<div class="chart-wrap">{img}<div class="interactive">{ctrl}{fig_html}{script}</div></div>'


def _widths_fig(ts):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('Top of Book', 'Back of Book'))
    raw_ys, smooth_ys, venues = [], [], []  # per-trace y-data (Raw/Smoothed) + venue labels
    has_sm = 'tob_bps_smooth' in ts.columns
    for (exch, pair), g in ts.groupby(['exchange', 'pair']):
        g = g.sort_values('timestamp')
        label = f'{pair}:{exch}'
        venues.append(label)  # each venue = 2 traces (TOB row1, BOB row2)
        smt = g['tob_bps_smooth'] if has_sm else g['tob_bps']
        smb = g['bob_bps_smooth'] if has_sm else g['bob_bps']
        # Default view is smoothed — initialise the traces with the smoothed series.
        fig.add_trace(go.Scatter(x=g['timestamp'], y=smt, mode='lines',
                                 name=label, legendgroup=label), row=1, col=1)
        raw_ys.append(list(g['tob_bps'])); smooth_ys.append(list(smt))
        fig.add_trace(go.Scatter(x=g['timestamp'], y=smb, mode='lines',
                                 name=label, legendgroup=label, showlegend=False), row=2, col=1)
        raw_ys.append(list(g['bob_bps'])); smooth_ys.append(list(smb))
    fig.update_yaxes(title_text='bps', row=1, col=1)
    fig.update_yaxes(title_text='bps', row=2, col=1)
    fig.update_xaxes(title_text='UTC', row=2, col=1)
    fig.update_layout(height=560, **_LAYOUT)
    # Traces default to smoothed; the Line (Smoothed/Raw) and Venue controls are HTML <select>s
    # driven by _widths_chart_block (readable in dark mode, unlike Plotly's own dropdown).
    return fig, venues, raw_ys, smooth_ys


def _widths_chart_block(ts, include_js=False, fallback=True):
    """Orderbook widths with two native <select>s (desktop): Line (Smoothed/Raw) restyles the
    y-data, Venue (All / per venue) restyles trace visibility. Static PNG (smoothed) for the phone."""
    fig, venues, raw_ys, smooth_ys = _widths_fig(ts)
    png = _safe_png(_mpl_widths_png, ts) if fallback else None
    img = (f'<img class="fallback" src="data:image/png;base64,{base64.b64encode(png).decode()}">'
           if png else '')
    fig_html = _fig_html(fig, include_js=include_js, div_id='zpwidths')
    has_sm = 'tob_bps_smooth' in ts.columns
    line_ctrl = ('<label for="zpwidths-line">Line:</label>'
                 '<select id="zpwidths-line"><option value="smoothed">Smoothed</option>'
                 '<option value="raw">Raw</option></select>') if has_sm else ''
    vopts = '<option value="all">All</option>' + ''.join(
        f'<option value="{i}">{_html.escape(v)}</option>' for i, v in enumerate(venues))
    ctrl = (f'<div class="gctrl">{line_ctrl}<label for="zpwidths-venue">Venue:</label>'
            f'<select id="zpwidths-venue">{vopts}</select></div>')
    script = (
        '<script>(function(){'
        f'var RAW={json.dumps(raw_ys)},SM={json.dumps(smooth_ys)},nV={len(venues)};'
        'function gd(){return document.getElementById("zpwidths");}'
        'var line=document.getElementById("zpwidths-line"),ven=document.getElementById("zpwidths-venue");'
        'if(line)line.addEventListener("change",function(){var g=gd();'
        'if(g&&window.Plotly)Plotly.restyle(g,{y:line.value==="raw"?RAW:SM});});'
        'if(ven)ven.addEventListener("change",function(){var g=gd();if(!g||!window.Plotly)return;'
        'var total=nV*2,vis=[],i;'
        'if(ven.value==="all"){for(i=0;i<total;i++)vis.push(true);}'
        'else{var e=+ven.value;for(i=0;i<total;i++)vis.push(Math.floor(i/2)===e);}'
        'Plotly.restyle(g,{visible:vis});});})();</script>')
    return f'<div class="chart-wrap">{img}<div class="interactive">{ctrl}{fig_html}{script}</div></div>'


_COMPLIANCE_SEGS = [('Compliant (<400bps)', 'compliance', '#3fb950'),
                    ('Too wide (≥400bps)', 'wide', '#d9a441'),
                    ('One-sided', 'one_sided', '#c9683a'),
                    ('Empty book', 'empty', '#55606e')]


def _compliance_fig(df):
    """Stacked horizontal composition bar per venue (best compliance on top); no-data venues
    a single grey bar."""
    rows = list(df.iterrows())
    y = [f"{r['exchange']} {r['pair']}" for _, r in rows]
    fig = go.Figure()
    for name, col, color in _COMPLIANCE_SEGS:
        fig.add_bar(y=y, x=[0 if r['nodata'] else r[col] * 100 for _, r in rows], name=name,
                    orientation='h', marker_color=color,
                    hovertemplate='%{x:.1f}%<extra>' + name + '</extra>')
    fig.add_bar(y=y, x=[100 if r['nodata'] else 0 for _, r in rows], name='No data (uncollected)',
                orientation='h', marker_color='#2b3038', marker_line_color='#3a424e',
                marker_line_width=1, hovertemplate='no data<extra></extra>')
    fig.update_layout(barmode='stack', height=max(240, 40 * len(y) + 130), **_LAYOUT)
    fig.update_xaxes(title_text='% of window', range=[0, 100])
    fig.update_yaxes(autorange='reversed')
    return fig


# --------------------------------------------------------------------------------------
# Page assembly
# --------------------------------------------------------------------------------------
_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:18px;
color:#e6e9ef;background:#0f1216;max-width:1100px;margin:auto}
h1{margin:0 0 2px;font-size:22px}h2{font-size:16px;margin:26px 0 8px;border-bottom:1px solid #262b33;padding-bottom:4px}
.sub{color:#8b94a3;margin:0 0 14px;font-size:13px}
h3{font-size:13px;margin:12px 0 4px;color:#8b94a3;font-weight:600}
.tablewrap{overflow-x:auto;margin:4px 0 10px}
table.grid{border-collapse:collapse;font-size:12.5px}
table.grid th,table.grid td{border:1px solid #262b33;padding:5px 10px;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
table.grid thead th{background:#1a1f26;text-align:center;font-weight:600}
table.grid td.l,table.grid th.l{text-align:left}
table.grid tbody tr:nth-child(even):not(.total) td{background:#13171d}
table.grid tr.total td{font-weight:700;background:#1c222b;border-top:2px solid #3a424e}
.pos{color:#3fb950}.neg{color:#e5534b}
.card{background:#161a20;border:1px solid #262b33;border-radius:8px;padding:14px 16px;margin-bottom:8px}
pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px;line-height:1.5;white-space:pre-wrap;margin:0}
pre.warn{color:#f0b429}
.mut{color:#6b7480}
img.fallback{display:block;width:100%;height:auto;border-radius:4px}
.interactive{display:none}
.flow-size{display:flex;align-items:center;gap:10px;margin:2px 4px 4px;color:#8b94a3;font-size:12px}
.flow-size input[type=range]{flex:1;max-width:320px;accent-color:#4c8dff}
:root{color-scheme:dark}
.gctrl{display:flex;align-items:center;gap:8px;margin:2px 0 8px}
.gctrl label{color:#8b94a3;font-size:12px}
.gctrl select{background:#1a1f26;color:#e6e9ef;border:1px solid #3a424e;border-radius:6px;
padding:5px 9px;font-size:12.5px;min-width:230px}
.gtablewrap{overflow-x:auto;border:1px solid #262b33;border-radius:8px;margin-bottom:8px}
table.gtable{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}
table.gtable th,table.gtable td{padding:6px 10px;text-align:right;border-bottom:1px solid #1c2128}
table.gtable th{color:#8b94a3;font-weight:600;background:#161a20}
table.gtable th.gstrat,table.gtable td.gstrat{text-align:left}
td.gstrat .gex{color:#6ea8ff;font-size:11px}
table.gtable tbody tr:hover{background:#12161c}
.zp-strip{display:flex;flex-wrap:wrap;gap:8px;margin:2px 0 12px}
.zp-strip .st{background:#161a20;border:1px solid #262b33;border-radius:8px;padding:6px 12px;
min-width:96px}
.zp-strip .st .k{color:#8b94a3;font-size:11px;line-height:1.3}
.zp-strip .st .v{font-size:16px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.35}
.zp-toc{position:sticky;top:0;z-index:20;display:flex;flex-wrap:wrap;align-items:center;gap:4px 6px;
padding:8px 10px;margin:0 0 12px;background:rgba(15,18,22,.92);backdrop-filter:blur(6px);
border:1px solid #262b33;border-radius:8px}
.zp-toc a{color:#c7ccd4;text-decoration:none;font-size:12px;padding:3px 8px;border-radius:5px;
white-space:nowrap}
.zp-toc a:hover{background:#1e242c;color:#6ea8ff}
"""

# Swap the static image for the interactive chart ONLY once Plotly has actually rendered it
# (so a viewer that runs a little JS but can't load Plotly keeps the static image).
_SWAP_SCRIPT = """<script>
window.addEventListener('load', function(){
  if (!window.Plotly) return;
  setTimeout(function(){
    document.querySelectorAll('.chart-wrap').forEach(function(w){
      var gd = w.querySelector('.js-plotly-plot');
      if (gd && gd.querySelector('.main-svg')){
        var img = w.querySelector('img.fallback'); if (img) img.style.display='none';
        var itv = w.querySelector('.interactive'); if (itv) itv.style.display='block';
        try { Plotly.Plots.resize(gd); } catch(e){}
      }
    });
  }, 400);
});
</script>"""


def _num(x, kind='money'):
    """Format a cell; NaN/None -> 'n/a'."""
    if x is None or x != x:
        return 'n/a'
    if kind == 'money':
        x = 0.0 if -0.5 < x < 0.5 else x  # avoid an ugly "-0"
        return f'{x:,.0f}'
    if kind == 'price':
        return f'{x:,.6g}'
    if kind == 'pct':
        return f'{x:.0%}'
    return _html.escape(str(x))


def _signed(x):
    """Money cell colour-coded green/red by sign (n/a for NaN)."""
    if x is None or x != x:
        return 'n/a'
    v = 0.0 if -0.5 < x < 0.5 else x
    cls = 'pos' if v > 0 else ('neg' if v < 0 else '')
    txt = f'{v:,.0f}'
    return f'<span class="{cls}">{txt}</span>' if cls else txt


def _grid(headers, rows, left_cols=(0, 1)):
    """Build a scrollable HTML table. `rows` is a list of (cells, is_total) tuples or plain
    cell-lists. `left_cols` are the column indices to left-align (labels)."""
    def cls(i):
        return ' class="l"' if i in left_cols else ''
    thead = ''.join(f'<th{cls(i)}>{h}</th>' for i, h in enumerate(headers))
    body = ''
    for r in rows:
        cells, is_total = r if isinstance(r, tuple) else (r, False)
        tr = ' class="total"' if is_total else ''
        body += f'<tr{tr}>' + ''.join(f'<td{cls(i)}>{c}</td>' for i, c in enumerate(cells)) + '</tr>'
    return (f'<div class="tablewrap"><table class="grid"><thead><tr>{thead}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def _summary_tables_html(result):
    """MM / Gamma / Total per-account buy/sell/net notional, avg prices and PnL. The Gamma table is
    actually everything that isn't MM (gamma, hedgers, any tag), so MM + Gamma reconciles to Total;
    the Strategy column shows each account's tag (MM / GAMMA / HEDGERV2 / …)."""
    out = []
    for title, df in [('MM', result.mm_summary), ('Gamma', result.other_summary),
                      ('Total', result.total_summary)]:
        if df is None or getattr(df, 'empty', True):
            continue
        rows = []
        for _, r in df.iterrows():
            acct = str(r.get('account', ''))
            rows.append(([
                _html.escape(acct), _html.escape(str(r.get('acct_type', '') or '')),
                _html.escape(str(r.get('inst', ''))),
                _num(r.get('buy_notional')), _num(r.get('sell_notional')), _signed(r.get('net_notional')),
                _num(r.get('buy_avg_price'), 'price'), _num(r.get('sell_avg_price'), 'price'),
                _signed(r.get('pnl_$')),
            ], acct.endswith('TOTAL')))
        out.append(f'<h3>{title}</h3>' + _grid(
            ['Account', 'Strategy', 'Inst', 'Buy $', 'Sell $', 'Net $', 'Buy avg', 'Sell avg', 'PnL $'],
            rows, left_cols=(0, 1, 2)))
    return ''.join(out)


def _markouts_table_html(result):
    """Per-venue volume-weighted markout bps per window, plus the venue's volume — sorted by
    volume descending."""
    mk = result.markouts
    if mk is None or mk.empty:
        return ''
    bps_cols = [c for c in mk.columns if c.endswith('_bps')]
    d = mk.sort_values('volume_$', ascending=False) if 'volume_$' in mk.columns else mk
    wins = [c.replace('pnl_', '').replace('s_bps', 's') for c in bps_cols]
    rows = []
    for idx, r in d.iterrows():
        exch, inst = idx if isinstance(idx, tuple) else (idx, '')
        cells = [_html.escape(str(exch)), _html.escape(str(inst)), _num(r.get('volume_$'))]
        cells += ['n/a' if r[c] != r[c] else f'{r[c]:+.1f}' for c in bps_cols]
        rows.append(cells)
    return _grid(['Exchange', 'Inst', 'Vol $'] + wins, rows)


# Capture % bands (same as the desk grid / summary table): green ≥10 · amber 8 · orange 5 · red <5.
_CAP_BANDS = [(10, '#2f9e44', '#fff'), (8, '#e0b100', '#141414'),
              (5, '#e8833a', '#141414'), (0, '#d64545', '#fff')]


def _cap_band(pct):
    for thr, bg, fg in _CAP_BANDS:
        if pct >= thr:
            return bg, fg
    return _CAP_BANDS[-1][1], _CAP_BANDS[-1][2]


def _captures_table_html(result):
    """Per-venue G20 vs market volume and capture %. Matches the aiozp summary definition: capture is
    coloured only on MM venues (where we run an active orderbook, per ob_active) — G20 there is our
    total share (all accounts). Non-MM and no-market venues are shown but greyed. TOTAL is MM-only,
    with an all-venues total below for reference."""
    cap = result.captures
    if cap is None or cap.empty:
        return ''
    has_active = 'ob_active' in cap.columns
    body = ''
    tmm_g = tmm_m = tall_g = tall_m = 0.0
    for idx, r in cap.iterrows():
        exch, pair = idx if isinstance(idx, tuple) else (idx, '')
        if str(exch) == 'TOTAL':
            continue                          # skip fetch_captures' all-venues total; recompute below
        g, mk, c = r.get('G20 Volume $'), r.get('Market Volume $'), r.get('Capture')
        active = bool(r.get('ob_active', True)) if has_active else True   # MM = active orderbook
        nodata = mk is None or mk != mk or c is None or c != c
        gf = float(g) if (g is not None and g == g) else 0.0
        if not nodata:
            tall_g += gf; tall_m += float(mk)
            if active:
                tmm_g += gf; tmm_m += float(mk)
        if nodata:
            capcell = '<td class="l mut">no market</td>'
        elif active:
            bg, fg = _cap_band(c * 100)
            capcell = f'<td style="background:{bg};color:{fg};font-weight:700">{c * 100:.1f}%</td>'
        else:
            capcell = f'<td style="color:#8b94a3">{c * 100:.1f}%</td>'
        rcls = '' if (active and not nodata) else ' style="opacity:.45"'
        body += (f'<tr{rcls}><td class="l">{_html.escape(str(exch))}</td>'
                 f'<td class="l">{_html.escape(str(pair))}</td>'
                 f'<td>{_num(g)}</td><td>{_num(mk)}</td>{capcell}</tr>')
    if tmm_m:
        mmc = tmm_g / tmm_m * 100
        body += (f'<tr class="total"><td class="l">TOTAL (MM Venues)</td><td></td><td>{_num(tmm_g)}</td>'
                 f'<td>{_num(tmm_m)}</td><td style="font-weight:700">{mmc:.1f}%</td></tr>')
    if tall_m and abs(tall_g - tmm_g) > 0.5:      # only add the all-venues row when it differs
        allc = tall_g / tall_m * 100
        body += (f'<tr class="total" style="opacity:.5"><td class="l">TOTAL (all venues)</td><td></td>'
                 f'<td>{_num(tall_g)}</td><td>{_num(tall_m)}</td><td>{allc:.1f}%</td></tr>')
    thead = ('<th class="l">Exchange</th><th class="l">Pair</th><th>G20 $</th>'
             '<th>Market $</th><th>Capture</th>')
    note = ('<p class="sub" style="margin:2px 0 0">Capture coloured only where we run an active '
            'orderbook (MM); other venues greyed. TOTAL is MM-only (our share on the venues we make).</p>')
    return (f'<div class="tablewrap"><table class="grid"><thead><tr>{thead}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>{note}')


def _compliance_table_html(result):
    """Per-venue 2% depth: uptime, compliance, avg BOB. No-data venues shown greyed. (Avg, not
    median, so the time-brush can re-cut it exactly from the per-minute buckets.)"""
    df = result.ob_compliance
    if df is None or df.empty:
        return ''
    has_mean = 'bob_mean' in df.columns
    rows = []
    for _, r in df.iterrows():
        venue = _html.escape(f"{r['exchange']} {r['pair']}")
        if r['nodata']:
            rows.append([f'<span class="mut">{venue}</span>', '—',
                         '<span class="mut">no data</span>', '—'])
        else:
            bob = r['bob_mean'] if has_mean else r['bob_median']
            bobs = 'n/a' if bob != bob else f"{bob:.0f}"
            rows.append([venue, f"{r['uptime'] * 100:.0f}%", f"{r['compliance'] * 100:.0f}%", bobs])
    return _grid(['Venue', 'Uptime', 'Compliance', 'Avg BOB'], rows, left_cols=(0,))


def _mpl_dmc_png(g):
    return _mpl_png(_dmc.mpl_dmc_build(g), (10, 7))


def _dmc_chart_block(g, include_js=False, fallback=True):
    """Static PNG (phone) + interactive 2x2 overlay with the exchange <select> (desktop-only)."""
    png = _safe_png(_mpl_dmc_png, g) if fallback else None
    img = (f'<img class="fallback" src="data:image/png;base64,{base64.b64encode(png).decode()}">'
           if png else '')
    fig_html = _fig_html(_dmc.build_dmc_fig(g), include_js=include_js, div_id='zpdmc')
    interactive = _dmc.select_div_html(g, 'zpdmc') + fig_html + _dmc.select_script(g, 'zpdmc')
    return f'<div class="chart-wrap">{img}<div class="interactive">{interactive}</div></div>'


_NAV_PLACEHOLDER = '\x00ZPTOC\x00'  # replaced with the TOC once all sections are known


def _stat_strip(result):
    """Compact headline stats atop the ZP — the coin's own desk-summary numbers, so the figures
    that matter are there before you scroll. Reuses summary_table's row extraction + colour bands."""
    from . import summary_table as st
    r = st._row(result.token, result)
    if not r or all(r.get(k) is None for k in ('total', 'compliance', 'capture', 'markout', 'pmove')):
        return ''
    cells = []

    def add(label, value, color=None, key=None):
        col = f'color:{color};' if color else ''
        dk = f' data-tc="{key}"' if key else ''   # data-tc: the time brush re-cuts these chips
        cells.append(f'<div class="st"{dk}><div class="k">{label}</div>'
                     f'<div class="v" style="{col}">{value}</div></div>')

    tot = r.get('total')
    if tot is not None and tot == tot:
        add('Total PnL', st._plain_money(tot),
            st._GREEN if tot > 0 else (st._RED if tot < 0 else None), key='total')
    pm = r.get('pmove')
    if pm is not None and pm == pm:
        add('Price Δ%', f'{pm:+.1f}%', st._GREEN if pm > 0 else (st._RED if pm < 0 else None))
    for label, key, bands, fmt in (
            ('Compliance', 'compliance', st._COMP_BANDS, lambda v: f'{v:.0f}%'),
            ('Capture', 'capture', st._CAP_BANDS, lambda v: f'{v:.0f}%'),
            (f'Markout {st._MO_SEC}s', 'markout', st._MO_BANDS, lambda v: f'{v:+.1f}')):
        v = r.get(key)
        if v is not None and v == v:
            add(label, fmt(v), st._band(bands, v)[0], key=key)
    return f'<div class="zp-strip">{"".join(cells)}</div>' if cells else ''


def _toc_html(sections):
    """Sticky jump-nav for the ZP's sections. Links use real #anchors (no-JS fallback); the nav
    script upgrades them to a cross-frame smooth scroll so they also work when the ZP is an aiozp
    iframe. (The 'back to Desk summary' control lives in the aiozp bar, not here.)"""
    if not sections:
        return ''
    links = ''.join(f'<a class="zpj" href="#{sid}">{_html.escape(lbl)}</a>' for sid, lbl in sections)
    return f'<nav class="zp-toc">{links}</nav>'


_NAV_JS = """<script>(function(){
  document.querySelectorAll('a.zpj').forEach(function(a){a.addEventListener('click',function(e){
    var el=document.getElementById(a.getAttribute('href').slice(1));
    if(el){e.preventDefault();try{el.scrollIntoView({behavior:'smooth',block:'start'});}
      catch(_){el.scrollIntoView();}}});});
})();</script>"""


def build_html_report(result, capture_threshold=0.10, period_label=None, inline_js=False,
                      dmc_data=None, fallback=True, tick_html=''):
    """inline_js=True embeds Plotly (~4.7 MB, self-contained, opens offline);
    False references the CDN (~50 KB, needs internet to render)."""
    period = period_label or f'{result.start:%Y-%m-%d %H:%M}–{result.end:%H:%M} UTC'
    text = format_report(result, capture_threshold, period_label=period_label)

    # Split the text report: summary (goes on top), orderbook-widths section (goes under
    # its chart), and warnings (bottom).
    ob_idx = text.find('Orderbook width')
    warn_idx = text.find('\nWarnings:')
    end = warn_idx if warn_idx != -1 else len(text)
    summary_text = (text[:ob_idx] if ob_idx != -1 else text[:end])
    ob_text = (text[ob_idx:end] if ob_idx != -1 else '').strip('\n')
    warn_text = (text[warn_idx + 1:] if warn_idx != -1 else '').strip('\n')
    # drop the leading "<TICKER> ZP <period>" line — it's already the <h1>
    summary_text = summary_text.split('\n', 1)[-1].strip('\n')

    def _pre(t, cls=''):
        return f'<div class="card"><pre class="{cls}">{_html.escape(t)}</pre></div>'

    head_js = f'<script>{get_plotlyjs()}</script>' if inline_js else ''
    parts = ['<!doctype html><html><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             _logo.FAVICON,
             f'<title>{result.token} ZP {period}</title>',
             f'<style>{_CSS}{_timecut.CSS}</style>{head_js}</head><body>',
             f'<h1>{_logo.mark()}{result.token} — ZP {period}</h1>',
             f'<p class="sub">{result.start:%Y-%m-%d %H:%M} → {result.end:%H:%M} UTC</p>',
             _stat_strip(result),
             f'<div class="tc-sticky">{_timecut.brush_html()}{_NAV_PLACEHOLDER}</div>']

    # Each section header registers itself for the sticky TOC and gets a stable anchor id.
    sections = []

    def _sec(html_label, toc_label):
        sid = f'zpsec-{len(sections) + 1}'
        sections.append((sid, toc_label))
        return f'<h2 id="{sid}">{html_label}</h2>'

    has_trades = result.trades is not None and not result.trades.empty
    # The first Plotly chart that actually renders must carry the CDN <script> (when not inlining).
    # Which section is first varies — a no-trades report starts at compliance/widths, not flows —
    # so hand the include-JS flag out dynamically to whichever chart renders first.
    _js_pending = [not inline_js]

    def _cjs():
        v = _js_pending[0]
        _js_pending[0] = False
        return v

    # Static PNG fallbacks: rendered when `fallback` (the default — a standalone /zp opened on a
    # no-JS phone needs them). aiozp passes fallback=False for its interactive iframes (desktop has
    # JS+Plotly; the PNGs live once in the static layer), which halves the file and skips the render.
    _fb = _safe_png if fallback else (lambda *a: None)

    parts.append(f'<div id="tc-textsum">'
                 f'{_pre(summary_text if has_trades else (summary_text or "No trades in this period."))}'
                 f'</div>')

    if has_trades:
        # Summary detail table (renders on any device — no JS/images needed)
        summary_tbl = _summary_tables_html(result)
        if summary_tbl:
            parts.append(_sec('Summary', 'Summary')
                         + '<div class="card"><div id="tc-summary">' + summary_tbl + '</div></div>')

        # Price, delta & cumulative P&L (net position marked to the MD mid)
        if result.delta_pnl_ts is not None and not result.delta_pnl_ts.empty:
            parts.append(_sec('Price, delta &amp; P&amp;L', 'P&L') + '<div class="card">'
                         + _chart_block(_delta_pnl_fig(result.delta_pnl_ts),
                                        _fb(_mpl_delta_pnl_png, result.delta_pnl_ts),
                                        include_js=_cjs(), div_id='zppnl') + '</div>')

        # Flows chart
        parts.append(_sec('Flows', 'Flows') + '<div class="card">'
                     + _flow_chart_block(result.trades, _fb(_mpl_flow_png, result.trades),
                                         include_js=_cjs()) + '</div>')
        if result.markouts is not None and not result.markouts.empty:
            parts.append(_sec('Markouts (bps)', 'Markouts') + '<div class="card">'
                         + _markouts_chart_block(result.markouts,
                                                 _fb(_mpl_markouts_png, result.markouts),
                                                 include_js=_cjs()) + '</div>')
            markouts_tbl = _markouts_table_html(result)
            if markouts_tbl:
                parts.append('<div class="card"><div id="tc-markouts">' + markouts_tbl + '</div></div>')

        # Captures: chart, then the full per-venue table below it
        parts.append(_sec('Captures', 'Captures'))
        if result.captures_ts is not None and not result.captures_ts.empty:
            # MM venues (active orderbook) as (exchange, quote) — the chart defaults to their aggregate.
            mm_keys = set()
            cap = result.captures
            if cap is not None and not cap.empty and 'ob_active' in cap.columns:
                for (e, p), act in zip(cap.index, cap['ob_active']):
                    if str(e) != 'TOTAL' and bool(act):
                        mm_keys.add((str(e), str(p).rsplit('-', 1)[-1].upper()))
            parts.append('<div class="card">'
                         + _captures_chart_block(result.captures_ts, result.trades,
                                                 include_js=_cjs(), fallback=fallback, mm_keys=mm_keys)
                         + '</div>')
        captures_tbl = _captures_table_html(result)
        if captures_tbl:
            parts.append('<div class="card"><div id="tc-captures">' + captures_tbl + '</div></div>')

    # 2% depth compliance — independent of trades, so it renders even for a no-trades coin
    if result.ob_compliance is not None and not result.ob_compliance.empty:
        parts.append(_sec('2% depth compliance', 'Compliance'))
        parts.append('<p class="sub">Note: this only measures whether the total spread '
                     'between our back-of-book orders is under 4% (±2% each side of mid). '
                     'It does not take specific KPIs (e.g. required depth or per-venue '
                     'obligations) into account.</p>')
        parts.append('<div class="card">'
                     + _chart_block(_compliance_fig(result.ob_compliance),
                                    _fb(_mpl_compliance_png, result.ob_compliance),
                                    include_js=_cjs(), div_id='zpcompliance') + '</div>')
        ctable = _compliance_table_html(result)
        if ctable:
            parts.append('<div class="card"><div id="tc-compliance-tbl">' + ctable + '</div></div>')

    # Orderbook widths — independent of trades
    if (result.ob_widths_ts is not None and not result.ob_widths_ts.empty) or ob_text:
        parts.append(_sec('Orderbook widths', 'Widths'))
        if result.ob_widths_ts is not None and not result.ob_widths_ts.empty:
            parts.append('<div class="card">'
                         + _widths_chart_block(result.ob_widths_ts, include_js=_cjs(),
                                               fallback=fallback) + '</div>')
        if ob_text:
            parts.append(_pre(ob_text))

    # Dynamic mode (gamma-dash): current MM settings + volatility/spread history (independent of trades)
    if dmc_data is not None and dmc_data.rows:
        parts.append(_sec('Dynamic mode settings &amp; volatility', 'Dynamic mode'))
        parts.append(_timecut.SETTINGS_NOTE)
        parts.append('<div class="card">' + _dmc.settings_table_html(dmc_data) + '</div>')
        if dmc_data.series:
            parts.append('<div class="card">'
                         + _dmc_chart_block(dmc_data, include_js=_cjs(), fallback=fallback) + '</div>')

    if tick_html:  # per-venue tick size (bps) + min-spread-in-ticks (ccxt; independent of trades)
        parts.append(_sec('Tick sizes', 'Ticks') + '<div class="card">' + tick_html + '</div>')

    if warn_text:
        parts.append(_pre(warn_text, cls='warn'))

    from . import sortable
    parts.append(_SWAP_SCRIPT + sortable.SORT_JS + _NAV_JS
                 + _timecut.engine_js(result.start, result.end)
                 + _timecut.recompute_js(result) + '</body></html>')
    return ''.join(parts).replace(_NAV_PLACEHOLDER, _toc_html(sections))


def build_static_report(result, dmc_data=None, period_label=None, fallback=True):
    """Script-free static body fragment (tables + text + PNG charts, no interactive charts, no
    <script>) for embedding — e.g. one coin inside the no-JS aiozp base. Uses the same _mpl_*_png
    builders and table helpers as the interactive report. Needs zp_html._CSS on the host page.
    fallback=False drops the PNG charts (tables + text only) — used when the aiozp file is too big
    to send, so it still fits (the phone then shows no charts)."""
    text = format_report(result, period_label=period_label)
    ob_idx, warn_idx = text.find('Orderbook width'), text.find('\nWarnings:')
    end = warn_idx if warn_idx != -1 else len(text)
    summary_text = (text[:ob_idx] if ob_idx != -1 else text[:end]).split('\n', 1)[-1].strip('\n')
    ob_text = (text[ob_idx:end] if ob_idx != -1 else '').strip('\n')
    warn_text = (text[warn_idx + 1:] if warn_idx != -1 else '').strip('\n')
    has_trades = result.trades is not None and not result.trades.empty

    def _pre(t, cls=''):
        return f'<div class="card"><pre class="{cls}">{_html.escape(t)}</pre></div>'

    def _img(png):
        return (f'<div class="card"><img class="fallback" src="data:image/png;base64,'
                f'{base64.b64encode(png).decode()}"></div>') if png else ''

    _render = _safe_png if fallback else (lambda *a: None)  # skip PNG charts when disabled

    parts = [_pre(summary_text if has_trades else (summary_text or 'No trades in this period.'))]
    if has_trades:
        summary_tbl = _summary_tables_html(result)
        if summary_tbl:
            parts.append('<h3>Summary</h3><div class="card">' + summary_tbl + '</div>')
        if result.delta_pnl_ts is not None and not result.delta_pnl_ts.empty:
            parts.append('<h3>Price, delta &amp; P&amp;L</h3>'
                         + _img(_render(_mpl_delta_pnl_png, result.delta_pnl_ts)))
        parts.append('<h3>Flows</h3>' + _img(_render(_mpl_flow_png, result.trades)))
        if result.markouts is not None and not result.markouts.empty:
            parts.append('<h3>Markouts (bps)</h3>' + _img(_render(_mpl_markouts_png, result.markouts)))
            mt = _markouts_table_html(result)
            if mt:
                parts.append('<div class="card">' + mt + '</div>')
        parts.append('<h3>Captures</h3>')
        if result.captures_ts is not None and not result.captures_ts.empty:
            parts.append(_img(_render(_mpl_captures_png, result.captures_ts, result.trades)))
        ct = _captures_table_html(result)
        if ct:
            parts.append('<div class="card">' + ct + '</div>')
    if result.ob_compliance is not None and not result.ob_compliance.empty:
        parts.append('<h3>2% depth compliance</h3>'
                     + _img(_render(_mpl_compliance_png, result.ob_compliance)))
        cc = _compliance_table_html(result)
        if cc:
            parts.append('<div class="card">' + cc + '</div>')
    if (result.ob_widths_ts is not None and not result.ob_widths_ts.empty) or ob_text:
        parts.append('<h3>Orderbook widths</h3>')
        if result.ob_widths_ts is not None and not result.ob_widths_ts.empty:
            parts.append(_img(_render(_mpl_widths_png, result.ob_widths_ts)))
        if ob_text:
            parts.append(_pre(ob_text))
    if dmc_data is not None and dmc_data.rows:
        parts.append('<h3>Dynamic mode settings &amp; volatility</h3>')
        parts.append('<div class="card">' + _dmc.settings_table_html(dmc_data) + '</div>')
        if dmc_data.series:
            parts.append(_img(_render(_mpl_dmc_png, dmc_data)))
    if warn_text:
        parts.append(_pre(warn_text, cls='warn'))
    return ''.join(parts)


def write_html_report(result, out_path, capture_threshold=0.10, period_label=None, inline_js=False,
                      dmc_data=None, tick_html=''):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(build_html_report(result, capture_threshold, period_label, inline_js=inline_js,
                                  dmc_data=dmc_data, tick_html=tick_html))
    return out_path


def build_dmc_report(g, token, period_label=None):
    """Standalone gamma-dash page (/dmc): settings table + vol/spread charts, same chart framework
    as the ZP (static PNG for the phone, interactive Plotly + exchange <select> on desktop)."""
    sub = 'gamma-dash · dynamic strategies' + (f' · {_html.escape(period_label)}' if period_label else '')
    parts = ['<!doctype html><html><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             _logo.FAVICON,
             f'<title>{_html.escape(token)} dynamic mode settings &amp; vol</title>',
             f'<style>{_CSS}</style></head><body>',
             f'<h1>{_logo.mark()}{_html.escape(token)} — Dynamic mode settings &amp; volatility</h1>',
             f'<p class="sub">{sub}</p>',
             '<div class="card">' + _dmc.settings_table_html(g) + '</div>']
    if g.series:
        parts.append('<div class="card">' + _dmc_chart_block(g, include_js=True) + '</div>')
    else:
        parts.append('<p class="sub">No history available for these strategies.</p>')
    parts.append(_SWAP_SCRIPT + '</body></html>')
    return ''.join(parts)


def write_dmc_report(g, out_path, token, period_label=None):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(build_dmc_report(g, token, period_label))
    return out_path
