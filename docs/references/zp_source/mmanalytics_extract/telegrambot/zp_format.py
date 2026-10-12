"""Objective-facts text formatter for the Zeyi Process report.

Descriptive only — states numbers, signs and threshold splits. No judgement vocabulary
("good/bad/strong/leaking"); the human adds that.
"""


def _km(x):
    """Abbreviate a magnitude: 73303 -> 73.3k, 1_250_000 -> 1.25M, 950 -> 950."""
    ax = abs(x)
    if ax >= 1_000_000:
        return f'{x / 1_000_000:.2f}M'
    if ax >= 1_000:
        return f'{x / 1_000:.1f}k'
    return f'{x:.0f}'


def _sk(x):
    """Signed abbreviated dollars, e.g. -$41.6k, +$6.8k."""
    try:
        return f'{"-" if x < 0 else "+"}${_km(abs(x))}'
    except (TypeError, ValueError):
        return 'n/a'


def _price(x):
    try:
        return f'{x:,.6g}'
    except (TypeError, ValueError):
        return 'n/a'


def _total_row(df, account):
    if df is None or getattr(df, 'empty', True):
        return None
    m = df[df['account'] == account]
    return m.iloc[0] if not m.empty else None


def compliance_headline(result):
    """One-line 2% depth summary: flat (unweighted) average of uptime & compliance across the
    venues that have data. Returns the line, or None if no compliance data."""
    df = getattr(result, 'ob_compliance', None)
    if df is None or df.empty:
        return None
    n_total = len(df)
    data = df[~df['nodata']]
    if data.empty:
        return f'2% Depth compliance: no venues reporting ({n_total} tracked)'
    comp = data['compliance'].mean() * 100
    up = data['uptime'].mean() * 100
    return (f'2% Depth compliance {comp:.0f}% · Uptime {up:.0f}% · '
            f'{len(data)}/{n_total} venues reporting')


def _ob_widths_lines(result):
    """Orderbook-width text block (median [min-max] per exchange/pair). Trade-independent, so it's
    emitted on both the normal and the no-trades path."""
    lines = ['', 'Orderbook width (bps, TOB / BOB; median [min-max]):']
    w = result.ob_widths
    if w is None or w.empty:
        lines.append('  n/a')
        return lines
    by_pair = {}
    for _, r in w.iterrows():
        by_pair.setdefault(r['pair'], []).append(
            f'    {r["exchange"]}: '
            f'TOB {r["tob_median"]:.0f} [{r["tob_lo"]:.0f}-{r["tob_hi"]:.0f}]  '
            f'BOB {r["bob_median"]:.0f} [{r["bob_lo"]:.0f}-{r["bob_hi"]:.0f}]')
    for pair in sorted(by_pair):
        lines.append(f'  {pair}:')
        lines += by_pair[pair]
    return lines


def format_report(result, capture_threshold=0.10, period_label=None):
    period = period_label or f'{result.start:%Y-%m-%d %H:%M} to {result.end:%Y-%m-%d %H:%M} UTC'
    lines = [f'{result.token} ZP {period}', '']

    if result.trades is None or result.trades.empty:
        lines.append('No trades in this period.')
        headline = compliance_headline(result)  # orderbook sections are trade-independent
        if headline:
            lines += ['', headline]
        lines += _ob_widths_lines(result)
        if result.warnings:
            lines += ['', 'Warnings:'] + [f'  - {w}' for w in result.warnings]
        return '\n'.join(lines)

    # 2% depth compliance headline (top of the report)
    headline = compliance_headline(result)
    if headline:
        lines += [headline, '']

    # Price OHLC (whole-period bar)
    ohlc = result.price_ohlc
    if ohlc:
        lines.append(
            f'Price (period OHLC): O {_price(ohlc["open"])}  H {_price(ohlc["high"])}  '
            f'L {_price(ohlc["low"])}  C {_price(ohlc["close"])}  ({ohlc["pct_change"]:+.2f}%)'
        )
    else:
        lines.append('Price (period OHLC): n/a')
    lines.append('')

    books = [
        ('MM', _total_row(result.mm_summary, 'MM TOTAL')),
        ('Gamma', _total_row(result.other_summary, 'GAMMA TOTAL')),
        ('Total', _total_row(result.total_summary, 'TOTAL')),
    ]

    # Volume + PnL per book — one line each. Buy/Sell breakdown is in summary.png.
    lines.append('Volume / PnL:')
    for name, row in books:
        if row is None:
            lines.append(f'  {name:<7} n/a')
        else:
            volume = row['buy_notional'] + row['sell_notional']
            lines.append(f'  {name:<7} ${_km(volume)} ({_sk(row["net_notional"])} net) | PnL {_sk(row["pnl_$"])}')
    lines.append('')

    # Flows: MM sell avg vs buy avg
    mm = books[0][1]
    if mm is not None:
        buy_avg, sell_avg = mm['buy_avg_price'], mm['sell_avg_price']
        rel = '>' if sell_avg > buy_avg else ('<' if sell_avg < buy_avg else '=')
        lines.append(f'Flows (MM): sell avg {_price(sell_avg)} {rel} buy avg {_price(buy_avg)}')
    else:
        lines.append('Flows (MM): n/a')
    lines.append('')

    # Markouts: volume-weighted average bps per window (bucket), across all venues
    mk = result.markouts
    if mk is None or mk.empty:
        lines.append('Markouts: n/a')
    else:
        bps_cols = [c for c in mk.columns if c.startswith('pnl_') and c.endswith('_bps')]
        vol = mk['volume_$'] if 'volume_$' in mk.columns else None
        parts = []
        for c in bps_cols:
            sec = c.replace('pnl_', '').replace('s_bps', '')
            col = mk[c]
            mask = col == col  # non-NaN
            if not mask.any():
                parts.append(f'{sec}s n/a')
            elif vol is not None and vol[mask].sum():
                parts.append(f'{sec}s {(vol[mask] * col[mask]).sum() / vol[mask].sum():+.1f}')
            else:
                parts.append(f'{sec}s {col[mask].mean():+.1f}')
        lines.append('Markouts (vol-wtd bps): ' + '  '.join(parts))
    lines.append('')

    # Captures: one overall summary — traded venues beating the threshold; Avg is the
    # sheet's TOTAL row (volume-weighted overall capture = total G20 / total market).
    cap = result.captures
    caps = []
    total_capture = float('nan')
    if cap is not None and not cap.empty:
        for idx, row in cap.iterrows():
            exch, pair = idx if isinstance(idx, tuple) else (idx, '')
            if exch == 'TOTAL':
                total_capture = row['Capture']
                continue
            g20 = row.get('G20 Volume $', 0)
            if not (g20 and g20 > 0):  # only venues we actually trade on
                continue
            c = row['Capture']
            if c == c:  # non-NaN
                caps.append(c)
    if caps:
        x = sum(1 for c in caps if c > capture_threshold)
        avg = f'{total_capture:.0%}' if total_capture == total_capture else 'n/a'
        lines.append(f'Captures: {x}/{len(caps)} > {capture_threshold:.0%}. Avg {avg}')
    else:
        lines.append('Captures: n/a')

    # Orderbook widths: median [min-max] per exchange/pair
    lines += _ob_widths_lines(result)

    if result.warnings:
        lines += ['', 'Warnings:'] + [f'  - {w}' for w in result.warnings]

    return '\n'.join(lines)

