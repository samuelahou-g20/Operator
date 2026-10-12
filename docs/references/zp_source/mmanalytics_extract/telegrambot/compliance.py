"""HTML grid report for program-level 2% depth compliance — a thin wrapper over the shared
grid renderer (compliance.grid). Per-asset × (exchange, quote) heatmap of compliance %,
with a JS toggle to uptime %, a ◐ one-sided-book marker, and worst-first rows.
Built from trade_analysis_core.fetch_compliance_matrix().
"""

from . import grid

_BANDS = [(95, '#2f9e44', '#ffffff'), (90, '#e0b100', '#141414'),
          (75, '#e8833a', '#141414'), (0, '#d64545', '#ffffff')]
_BANDS_LEGEND = [('≥95%', '#2f9e44', '#fff'), ('90–95%', '#e0b100', '#141414'),
                 ('75–90%', '#e8833a', '#141414'), ('<75%', '#d64545', '#fff')]
_ONESIDED_MARK = 0.10  # flag a cell when >= this share of snapshots had a one-sided book

_NOTE = ('Note: this only measures whether the total spread between our back-of-book orders '
         'is under 4% (±2% each side of mid). It does not take specific KPIs (e.g. required '
         'depth or per-venue obligations) into account. Empty book counts as a breach; '
         'uncollected venues are shown greyed.')


def _hover(asset, col, r):
    comp, up = r.get('compliance', float('nan')) * 100, r.get('uptime', float('nan')) * 100
    med, lo, hi = r.get('bob_median'), r.get('bob_lo'), r.get('bob_hi')
    bob = 'n/a' if med != med else f'{med:.0f} [{lo:.0f}–{hi:.0f}]'
    return (f'{asset} · {col}\n'
            f'compliance {comp:.0f}%  ·  uptime {up:.0f}%\n'
            f'snapshots {int(r.get("snaps", 0))}\n'
            f'breakdown: compliant {comp:.0f}% · wide {float(r.get("wide", 0) or 0) * 100:.0f}% · '
            f'one-sided {float(r.get("one_sided", 0) or 0) * 100:.0f}% · '
            f'empty {float(r.get("empty", 0) or 0) * 100:.0f}%\n'
            f'BOB width bps, median [min–max]: {bob}')


def _headline(program, cells):
    if cells is None or cells.empty:
        return f'{program}: no orderbook data in this period.'
    data = cells[~cells['nodata']] if 'nodata' in cells.columns else cells
    vals = data['compliance'] * 100
    overall = 'n/a' if vals.empty else f'{vals.mean():.0f}%'
    return (f'{program} · Compliance {overall} · {data["asset"].nunique()} assets · '
            f'{len(data)} cells · {int((vals < 75).sum())} below 75%')


def build_compliance_report(program, cells, ignored, tokens, start, end, period_label=None,
                            body_only=False):
    period = period_label or f'{start:%Y-%m-%d %H:%M}–{end:%H:%M} UTC'
    subtitle = f'{start:%Y-%m-%d %H:%M} → {end:%H:%M} UTC'
    headline = _headline(program, cells)
    if cells is not None and not cells.empty:
        data = cells[~cells['nodata']] if 'nodata' in cells.columns else cells
        avg_p = data.groupby('asset')['compliance'].mean()
        avg_s = data.groupby('asset')['uptime'].mean()
        n_v = data.groupby('asset').size()
    else:
        avg_p = avg_s = n_v = None

    legend_extra = ('<span style="background:transparent;color:#e6e9ef">◐ one-sided book '
                    f'(≥{_ONESIDED_MARK * 100:.0f}% of snapshots)</span>')
    return grid.build_grid(
        title=f'{program} 2% compliance {period}', h1=f'{program} — Within-2% compliance',
        subtitle=subtitle, program=program, cells=cells, ignored=ignored,
        value_col='compliance', bands=_BANDS, bands_legend=_BANDS_LEGEND, legend_extra=legend_extra,
        empty_msg=headline, headline_html=headline, note=_NOTE, hover=_hover,
        avg_primary=avg_p, avg_secondary=avg_s,
        avg_hover=(lambda a: f'{a} — flat average across {int(n_v.get(a, 0))} venue(s) with data') if n_v is not None else None,
        toggle=dict(col='uptime', tsec='Show uptime', tpri='Show compliance'),
        mark=dict(col='one_sided', threshold=_ONESIDED_MARK, glyph='◐'), body_only=body_only)


def write_compliance_report(program, cells, ignored, tokens, start, end, path, period_label=None):
    html = build_compliance_report(program, cells, ignored, tokens, start, end, period_label)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path
