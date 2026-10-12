"""Streamlit-free shared compute core for the Trade Analysis dashboard and the headless
report suite (the `telegrambot` package: ZP, compliance, captures).

The trade/summary/markout/capture compute lives here so both the Streamlit app (via
`data.py`) and the Telegram bot run one code path. This module must NEVER import
`data.py` — that would trigger data.py's import-time service block (secrets + DB).

Callers wire in services/config once via `init(...)`; the dashboard passes Streamlit-backed
callbacks, the bot passes plain `print`.
"""

import calendar
import os
import threading
import time
import yaml
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from market_data import CCXT_EXCHANGE_NAMES
from matplotlib.colors import LinearSegmentedColormap

_CONFIG_DIR = Path(__file__).parent


MD_CONFIG = {
    'binance': {
        'interval': '1s',
        'markout_windows': [60, 120, 180, 300, 600]
    },
    'gate': {
        'interval': '10s',
        'markout_windows': [60, 120, 180, 300, 600]
    },
    'binanceusdm': {
        'interval': '1m',
        'markout_windows': [60, 120, 180, 300, 600]
    },
    'bybit': {
        'interval': '1m',
        'markout_windows': [60, 120, 180, 300, 600]
    },
    'bitget': {
        'interval': '1m',
        'markout_windows': [60, 120, 180, 300, 600]
    },
    'kucoin': {
        'interval': '1m',
        'markout_windows': [60, 120, 180, 300, 600]
    },
    'coinbase': {
        'interval': '1m',
        'markout_windows': [60, 120, 180, 300, 600]
    },
}

_HOURLY_CAPTURES_EXCHANGES = {'BITVAVO', 'LCX'}  # Exchanges with unreliable 1m kline data
_USD_QUOTES = ('USDT', 'USDC', 'USD', 'FDUSD')


# --------------------------------------------------------------------------------------
# Injected state (set once via init) — mirrors how data.py uses module-level globals.
# --------------------------------------------------------------------------------------
_trade_service = None
_ref_service = None
_md_client = None
_fx_cache = None
_ob_db_engine = None        # SQLAlchemy engine for the orderbook DB (optional; ZP widths)
_config_provider = lambda: {}
_loc_meta = None            # cached location_id_metadata DataFrame
_warn_cb = print            # UI warning sink
_tls = threading.local()    # per-thread compute state, so a parallel desk doesn't cross wires
                            # (_tls.warnings: when a list, compute_zp collects warnings into it)


class _PrintStatus:
    """Headless stand-in for st.status(): prints progress labels."""
    def __init__(self, label=None):
        if label:
            print(label)

    def update(self, label=None, state=None):
        if label:
            print(label)


_status_factory = _PrintStatus  # data.py injects st.status


def init(config_provider, trade_service, ref_service, md_client, fx_cache,
         status_factory=None, warn=None, ob_db_engine=None):
    """Wire the core to services and config. `config_provider` is a callable returning the
    config dict (e.g. `lambda: st.session_state.config`). `status_factory(label)` returns a
    handle with `.update(label=, state=)`; `warn(msg)` reports a non-fatal issue.
    `ob_db_engine` is an optional SQLAlchemy engine for the orderbook DB (ZP widths)."""
    global _trade_service, _ref_service, _md_client, _fx_cache, _config_provider
    global _status_factory, _warn_cb, _ob_db_engine
    _config_provider = config_provider
    _trade_service = trade_service
    _ref_service = ref_service
    _md_client = md_client
    _fx_cache = fx_cache
    if status_factory is not None:
        _status_factory = status_factory
    if warn is not None:
        _warn_cb = warn
    if ob_db_engine is not None:
        _ob_db_engine = ob_db_engine


def _status(label):
    return _status_factory(label)


def _warn(msg):
    w = getattr(_tls, 'warnings', None)   # thread-local: only the running compute_zp collects
    if w is not None:
        w.append(str(msg))
    _warn_cb(msg)


@contextmanager
def _timer(name, sink):
    """Time a compute stage; record it in `sink` and print when it finishes (so a slow or
    hanging stage shows up in the log immediately, not only in the final summary)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        sink[name] = dt
        print(f'[zp timing] {name}: {dt:.2f}s', flush=True)


# --------------------------------------------------------------------------------------
# Config loading (moved from app.py; Streamlit-free)
# --------------------------------------------------------------------------------------
def _expand_exchanges(exchanges_dict):
    """Convert the compact dict format {pair: [exchanges]} to a flat list [{pair, exchange}]."""
    flat = []
    if isinstance(exchanges_dict, dict):
        for pair, exch_list in exchanges_dict.items():
            for exch in exch_list:
                flat.append({'pair': pair, 'exchange': exch})
    elif isinstance(exchanges_dict, list):
        flat = exchanges_dict  # already in flat format
    return flat


def load_config():
    config_path = _CONFIG_DIR / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    for token, token_config in config.items():
        token_config.setdefault('g20_name', token.lower())
        token_config['exchanges'] = _expand_exchanges(token_config.get('exchanges', {}))
        token_config.setdefault('deployment_names', [token_config.get('g20_name', token.lower()).upper()])
    return config


# --------------------------------------------------------------------------------------
# Reference / trade enrichment
# --------------------------------------------------------------------------------------
def get_pairs(token):
    """Derive unique pairs from a token's exchanges list."""
    return sorted(set(e['pair'] for e in _config_provider()[token].get('exchanges', [])))


def build_location_metadata():
    """Fetch and index location_id metadata from the reference service."""
    location_id_metadata = pd.json_normalize(_ref_service.get_all_locations(), sep='.')
    location_id_metadata = location_id_metadata.set_index('location_id')
    location_id_metadata['tags.strategy'] = location_id_metadata['tags.strategy'].str[0]
    return location_id_metadata


def _get_loc_meta():
    global _loc_meta
    if _loc_meta is None:
        _loc_meta = build_location_metadata()
    return _loc_meta


def apply_usdt_fx_rates(df, pair_col, ts_col):
    df[['base_ccy', 'quote_ccy']] = df[pair_col].str.split('-', expand=True)
    df['fx_rate'] = 1.0
    missing_ccys = []
    for quote_ccy in df['quote_ccy'].unique():
        if quote_ccy in ('USDT', 'USDC', 'USD'):
            continue
        fx_series = _fx_cache.get_rate_series(quote_ccy, '1m', df[ts_col].min(), df[ts_col].max())
        if fx_series is None or fx_series.empty:
            missing_ccys.append(quote_ccy)
            continue  # fx_rate stays at default 1.0
        mask = df['quote_ccy'] == quote_ccy
        subset = df.loc[mask, [ts_col]].copy()
        subset = pd.merge_asof(
            subset.sort_values(ts_col),
            fx_series.rename('_fx').to_frame(),
            left_on=ts_col,
            right_index=True,
            direction='nearest',
            tolerance=pd.Timedelta(minutes=5),
        )
        df.loc[subset.index, 'fx_rate'] = subset['_fx'].ffill().fillna(1.0).values
    if missing_ccys:
        _warn(f'FX rates unavailable for: {", ".join(missing_ccys)}. Defaulting to 1.0.')
    return df


def _enrich_trades(trades, reverse_map):
    """Shared tail of the trades pipeline: remap inst names back to config pairs, attach
    exchange/account from location metadata, and add fx-adjusted USD price/notional. Used by both
    the per-coin fetch_trades and the desk-batched _desk_trades_for, so a coin's frame is identical
    however it was sourced."""
    if trades.empty:
        return trades
    # Remap trade instrument names back to config pair names
    if reverse_map:
        trades['inst'] = trades['inst'].replace(reverse_map)
    loc_meta = _get_loc_meta()
    trades['exchange'] = trades['location_id'].map(loc_meta['exchange_uid'].to_dict())
    trades['acct_type'] = trades['location_id'].map(loc_meta['tags.strategy'].to_dict())
    trades['account'] = trades['exchange'] + ' ' + trades['acct_type']
    unmapped = trades[trades['exchange'].isna()]['location_id'].unique()
    if len(unmapped) > 0:
        _warn(f'Unknown location_id(s): {", ".join(unmapped)}. These trades will have missing exchange/account info.')
    trades = apply_usdt_fx_rates(trades, pair_col='inst', ts_col='exch_ts')
    trades['price_usd'] = trades['price'] * trades['fx_rate']
    trades['notional_usd'] = trades['notional'] * trades['fx_rate']
    return trades


def fetch_trades(token, start_dt, end_dt):
    trades = pd.DataFrame()
    trade_pair_map = _config_provider()[token].get('trade_pairs', {})
    reverse_map = {v: k for k, v in trade_pair_map.items()}
    for pair in get_pairs(token):
        query_pair = trade_pair_map.get(pair, pair)
        trades = pd.concat([trades, _trade_service.get_trades(query_pair, start_dt, end_dt)], ignore_index=True)
    return _enrich_trades(trades, reverse_map)


def _token_query_pairs(token):
    """The inst names trade-streaming reports for a token (config pairs after trade_pairs remap),
    with the reverse map to turn them back into config pair names."""
    tpm = _config_provider()[token].get('trade_pairs', {})
    return {tpm.get(p, p) for p in get_pairs(token)}, {v: k for k, v in tpm.items()}


def _desk_trades_for(token, raw_all):
    """This coin's enriched trades, sliced from one desk-wide no-inst pull (`raw_all`, already
    row-shaped by get_all_trades). Same rows and same enrichment as fetch_trades(token)."""
    qpairs, reverse_map = _token_query_pairs(token)
    sliced = raw_all[raw_all['inst'].isin(qpairs)].copy()
    return _enrich_trades(sliced, reverse_map)


# --------------------------------------------------------------------------------------
# Summary (pure)
# --------------------------------------------------------------------------------------
def get_total_row(trade_summary, total_prefix=''):
    total_row = {
        'account': f'{total_prefix}TOTAL',
        'acct_type': '',
        'inst': '',
        'buy_quantity': trade_summary['buy_quantity'].sum(),
        'sell_quantity': trade_summary['sell_quantity'].sum(),
        'net_quantity': trade_summary['net_quantity'].sum(),
        'buy_notional': trade_summary['buy_notional'].sum(),
        'sell_notional': trade_summary['sell_notional'].sum(),
        'net_notional': trade_summary['net_notional'].sum(),
        'buy_avg_price': trade_summary['buy_notional'].sum()/trade_summary['buy_quantity'].sum(),
        'sell_avg_price': trade_summary['sell_notional'].sum()/trade_summary['sell_quantity'].sum(),
        'net_avg_price': trade_summary['net_notional'].sum()/trade_summary['net_quantity'].sum(),
        'pnl_$': trade_summary['pnl_$'].sum()
    }
    return total_row


def fetch_summary(trades):
    last_px = trades[trades['exch_ts']==trades['exch_ts'].max()]['price_usd'].mean()
    trade_summary = trades.copy()
    trade_summary['buy_quantity'] = np.where(trade_summary['side']=='B', trade_summary['qty'], 0)
    trade_summary['sell_quantity'] = np.where(trade_summary['side']=='S', trade_summary['qty'], 0)
    trade_summary['net_quantity'] = trade_summary['buy_quantity'] - trade_summary['sell_quantity']
    trade_summary['buy_notional'] = np.where(trade_summary['side']=='B', trade_summary['notional_usd'], 0)
    trade_summary['sell_notional'] = np.where(trade_summary['side']=='S', trade_summary['notional_usd'], 0)
    trade_summary['net_notional'] = trade_summary['buy_notional'] - trade_summary['sell_notional']
    # Keep acct_type (the strategy tag, e.g. MM / GAMMA / HEDGERV2) as a column so the report can
    # show it and so the split below is by that tag, not a brittle name substring.
    trade_summary = trade_summary.groupby(['account','acct_type','inst'], as_index=False)[['buy_quantity','sell_quantity','net_quantity','buy_notional','sell_notional','net_notional']].sum()
    trade_summary['buy_avg_price'] = np.where(trade_summary['buy_notional']==0, 0, trade_summary['buy_notional'] / trade_summary['buy_quantity'])
    trade_summary['sell_avg_price'] = np.where(trade_summary['sell_notional']==0, 0, trade_summary['sell_notional'] / trade_summary['sell_quantity'])
    trade_summary['net_avg_price'] = trade_summary['net_notional'] / trade_summary['net_quantity']
    trade_summary['pnl_$'] = (last_px - trade_summary['buy_avg_price']) * trade_summary['buy_quantity'] + (trade_summary['sell_avg_price'] - last_px) * trade_summary['sell_quantity']
    # MM = market-making accounts (name contains ' MM', unchanged); Other = everything else (gamma,
    # any hedger like HEDGERV2, and any future tag). Complementary masks, so MM + Other always ==
    # Total — no trades can silently vanish from the sub-tables the way HEDGERV2 did before.
    is_mm = trade_summary['account'].str.contains(' MM', na=False)
    mm_trade_summary = trade_summary[is_mm]
    other_trade_summary = trade_summary[~is_mm]
    mm_trade_summary = pd.concat([mm_trade_summary, pd.DataFrame([get_total_row(mm_trade_summary, total_prefix='MM ')])], ignore_index=True)
    other_trade_summary = pd.concat([other_trade_summary, pd.DataFrame([get_total_row(other_trade_summary, total_prefix='GAMMA ')])], ignore_index=True)
    total_summary = pd.DataFrame([get_total_row(trade_summary)])
    return mm_trade_summary, other_trade_summary, total_summary


# --------------------------------------------------------------------------------------
# Markouts
# --------------------------------------------------------------------------------------
def fetch_markouts(token, trades):
    mm_trades = trades[trades['acct_type']=='MM'].copy()
    mm_trades['price_usdt'] = mm_trades['price_usd']
    mm_trades['volume_$'] = mm_trades['price_usd'] * mm_trades['qty']
    md_pair = _config_provider()[token]['md_pair']
    md_exchange = _config_provider()[token]['md_exchange']
    interval = MD_CONFIG[md_exchange]['interval']
    markout_windows = MD_CONFIG[md_exchange]['markout_windows']
    progress = _status(f'Fetching markouts: {md_pair} from {md_exchange}...')
    prices = _md_client.get_klines_df(
        symbol = md_pair.replace('-','/'),
        exchange = md_exchange,
        timeframe = interval,
        start_dt = mm_trades['exch_ts'].min(),
        end_dt = mm_trades['exch_ts'].max() + pd.Timedelta(seconds=max(markout_windows)),
        request_limit = 1000,  # fetch big pages: 1s candles over a long window = far fewer round-trips
    ).set_index('timestamp')
    pd_interval = interval.replace('m','min')
    total = len(markout_windows)
    for i, sec in enumerate(markout_windows, 1):
        progress.update(label=f'Computing markouts: {sec}s window ({i}/{total})...')
        mm_trades[f'last_px_{sec}s'] = mm_trades['exch_ts'].apply(
            lambda ts: prices.loc[(ts + pd.Timedelta(seconds=sec)).ceil(pd_interval), 'open'] if (ts + pd.Timedelta(seconds=sec)).ceil(pd_interval) in prices.index else np.nan
        )
        mm_trades[f'pnl_{sec}s_$'] = (mm_trades[f'last_px_{sec}s'] - mm_trades['price_usdt']) * mm_trades['qty'] * np.where(mm_trades['side']=='B', 1, -1)
        mm_trades[f'pnl_{sec}s_bps'] = 10000 * (mm_trades[f'last_px_{sec}s'] / mm_trades['price_usdt'] - 1) * np.where(mm_trades['side']=='B', 1, -1)
    weighted_avg = lambda x: np.average(x[~np.isnan(x)], weights=mm_trades.loc[x.index[~np.isnan(x)], 'qty']) if not np.isnan(x).all() else np.nan
    markouts_summary = pd.pivot_table(
        mm_trades,
        index = ['exchange','inst'],
        values = ['volume_$'] + [f'pnl_{sec}s_bps' for sec in markout_windows],
        aggfunc={
            'volume_$': 'sum',
            **{col: weighted_avg for col in [f'pnl_{sec}s_bps' for sec in markout_windows]}
        }
    )
    markouts_summary = markouts_summary[['volume_$']+[f'pnl_{sec}s_bps' for sec in markout_windows if f'pnl_{sec}s_bps' in markouts_summary.columns]]
    progress.update(label=f'Markouts loaded ({total} windows)', state='complete')
    return markouts_summary, mm_trades


# --------------------------------------------------------------------------------------
# Delta & P&L time series
# --------------------------------------------------------------------------------------
def _compute_pnl(trades_df, price_col):
    """Cumulative net position, avg entry price, and realised P&L for a trades frame sorted by
    exch_ts (single net book across all pairs). Pure — ported from the dashboard's charts."""
    signed_qty = np.where(trades_df['side'] == 'B', trades_df['qty'], -trades_df['qty'])
    position = 0.0
    avg_price = 0.0
    cum_realized = 0.0
    positions, avg_prices, cum_realized_pnl = [], [], []
    for qty, price in zip(signed_qty, trades_df[price_col]):
        if position == 0 or np.sign(position) == np.sign(qty):
            new_position = position + qty
            avg_price = price if position == 0 else (avg_price * abs(position) + price * abs(qty)) / abs(new_position)
            position = new_position
        else:
            closing_qty = min(abs(position), abs(qty)) * np.sign(qty)
            cum_realized += (avg_price - price) * closing_qty
            position += qty
            if position != 0 and np.sign(position) == np.sign(qty):
                avg_price = price
        positions.append(position)
        avg_prices.append(avg_price)
        cum_realized_pnl.append(cum_realized)
    trades_df = trades_df.copy()
    trades_df['position'] = positions
    trades_df['avg_price'] = avg_prices
    trades_df['cum_realised_pnl'] = cum_realized_pnl
    trades_df['cum_unrealised_pnl'] = (trades_df[price_col] - trades_df['avg_price']) * trades_df['position']
    trades_df['cum_delta'] = trades_df['position'] * trades_df[price_col]
    return trades_df


def fetch_delta_pnl(token, trades, start_dt, end_dt):
    """Cumulative dollar delta and realised/unrealised/total P&L ($) over time for the token's net
    position (across all accounts, matching the dashboard's default view), marked to the 1-minute
    market-data mid where available (else trade prices). Returns a tidy frame with columns
    [timestamp, price, cum_delta, cum_realised_pnl, cum_unrealised_pnl, cum_total_pnl] or None."""
    ft = trades.sort_values('exch_ts')
    price_col = 'price_usd' if 'price_usd' in ft.columns else 'price'
    if ft.empty:
        return None
    pnl_df = _compute_pnl(ft, price_col)
    md = None
    try:
        cfg = _config_provider()[token]
        md = _md_client.get_klines_df(
            symbol=cfg['md_pair'].replace('-', '/'), exchange=cfg['md_exchange'],
            timeframe='1m', start_dt=start_dt, end_dt=end_dt).set_index('timestamp')
    except Exception as e:
        _warn(f'Delta/PnL market prices failed: {e}')
        md = None
    if md is not None and not md.empty:
        # Forward-fill the book state onto the 1-minute MD grid and re-mark to the MD mid, so
        # delta and unrealised P&L are smooth between trades.
        md_mid = (md['open'] + md['close']) / 2
        state = pnl_df.set_index('exch_ts')[['position', 'avg_price', 'cum_realised_pnl']]
        state = state[~state.index.duplicated(keep='last')]
        combined = state.index.union(md_mid.index).sort_values()
        filled = state.reindex(combined).ffill().fillna(0)
        pos = filled.loc[md_mid.index, 'position']
        avgp = filled.loc[md_mid.index, 'avg_price']
        agg = pd.DataFrame(index=md_mid.index)
        agg['price'] = md_mid.values
        agg['cum_delta'] = (pos * md_mid).values
        agg['cum_realised_pnl'] = filled.loc[md_mid.index, 'cum_realised_pnl'].values
        agg['cum_unrealised_pnl'] = ((md_mid - avgp) * pos).values
    else:
        agg = pnl_df.set_index('exch_ts')[[price_col, 'cum_delta', 'cum_realised_pnl',
                                           'cum_unrealised_pnl']].rename(columns={price_col: 'price'})
    agg['cum_total_pnl'] = agg['cum_realised_pnl'] + agg['cum_unrealised_pnl']
    agg.index.name = 'timestamp'
    return agg.reset_index()


def fetch_lead_scores(trades, window_ms=100):
    # LEAD SCORES TOTAL FILLS VOLUME WEIGHTED
    mm_trades = trades[(trades['acct_type']=='MM') & (trades['taker_or_maker']=='M')]
    trade_sequences = mm_trades.groupby(['exch_ts','exchange','side'], as_index=False)['qty'].sum()
    trade_sequences["event_id"] = ((trade_sequences["side"].map({'buy':0,'sell':1}).diff().abs() > 0) | (trade_sequences["exch_ts"].diff().dt.total_seconds() * 1000 > window_ms)).cumsum()
    leaders = trade_sequences.groupby('event_id').agg({'exchange': 'first', 'qty': 'sum' })
    weighted_scores = leaders.groupby("exchange")["qty"].sum()
    lead_scores = (weighted_scores / weighted_scores.sum()).reset_index()
    lead_scores = lead_scores.rename(columns={'qty':'lead_score'})
    lead_scores = lead_scores.sort_values('lead_score', ascending=False).reset_index(drop=True)
    return lead_scores


# --------------------------------------------------------------------------------------
# Captures
# --------------------------------------------------------------------------------------
def _ccxt_symbol_candidates(token, pair):
    """ccxt unified symbols to try for a config pair, in order: the pair's own symbol first, then
    the coin's OTHER base aliases for the same quote. ccxt (public market data) may list a venue
    under a different base than config/OB — e.g. BSV's kucoin book is BCHSV in config but ccxt's
    kucoin lists it as BSV — so we fall back through this coin's own aliases. Only aliases of THIS
    coin are tried, so a fallback can't accidentally match an unrelated token."""
    p = str(pair)
    perp = p.startswith('P.')
    quote = p.rsplit('-', 1)[-1]

    def sym(base):
        return f'{base}/{quote}:{quote}' if perp else f'{base}/{quote}'

    order = [p.replace('P.', '').split('-', 1)[0]]  # the config pair's own base first
    if token:
        try:
            for gp in get_pairs(token):
                b = gp.replace('P.', '').split('-', 1)[0]
                if gp.rsplit('-', 1)[-1] == quote and b not in order:
                    order.append(b)
        except Exception:
            pass
    return [sym(b) for b in order]


def fetch_captures(start_dt, end_dt, trades, token=None, existing=None, existing_ts=None, no_perp_capture=False):
    """Fetch capture data for all exchange/pair combos. Includes config exchanges with zero G20 volume.
    Skips (exchange, pair) combos already in `existing` DataFrame.
    no_perp_capture=True drops perp pairs (P.XXX-YYY) from the capture set.

    One row per PHYSICAL market. Config is authoritative for each venue's symbol (we trade the coin
    as AIGENSYN on binance, BCHSV on kucoin) while the trade feed tags fills with the canonical
    instrument (AI-USDT), so we key markets by (exchange, perp?, quote) and attribute G20 across
    whatever alias the trades carry. Keying by the exact inst instead splits an aliased venue into
    two rows: our volume vs the WRONG ccxt market (binance's unrelated 'AI'), and the real market
    with zero volume — so its capture is never computed. The perp flag keeps a spot and a perp of
    the same quote apart."""
    def _mkt(exch, pair):
        p = str(pair)
        return (str(exch), p.startswith('P.'), p.rsplit('-', 1)[-1].upper())

    # G20 (our) notional per market, summed over every inst alias that maps to it.
    g20_by_mkt = {}
    if not trades.empty:
        for inst, exch, notl in trades[['inst', 'exchange', 'notional_usd']].itertuples(index=False):
            k = _mkt(exch, inst)
            g20_by_mkt[k] = g20_by_mkt.get(k, 0.0) + (float(notl) if notl == notl else 0.0)

    # Combos = config venue symbols (authoritative), then any traded market NOT covered by config
    # (so volume on an unconfigured venue is never silently dropped, and never double-counted).
    config_combos, covered = [], set()
    if token:
        for entry in _config_provider().get(token, {}).get('exchanges', []):
            config_combos.append((entry['pair'], entry['exchange']))
            covered.add(_mkt(entry['exchange'], entry['pair']))
    extra_combos = {(str(inst), str(exch)) for inst, exch
                    in trades[['inst', 'exchange']].drop_duplicates().to_records(index=False)
                    if _mkt(exch, inst) not in covered}
    combos = sorted(set(config_combos) | extra_combos)
    if no_perp_capture:
        combos = [(p, e) for p, e in combos if not p.startswith('P.')]
    # Determine which combos need fetching
    if existing is not None and not existing.empty:
        cached_keys = set(existing.index.droplevel(-1) if 'TOTAL' in existing.index.get_level_values(0) else existing.index)
        cached_keys.discard(('TOTAL', ''))
        new_combos = [(p, e) for p, e in combos if (e, p) not in cached_keys]
    else:
        new_combos = list(combos)
        existing = pd.DataFrame(columns=['Exchange','Pair','G20 Volume $','Market Volume $'])
        existing = existing.set_index(['Exchange', 'Pair'])

    # existing_ts: previously cached timeseries data
    existing_ts = existing_ts if existing_ts is not None else []

    if new_combos:
        total = len(new_combos)
        progress = _status(f'Fetching captures (0/{total})...')

        def _fetch_combo(pair, exch):
            """Fetch one (pair, exchange) market volume. Returns (row, ts_df_or_None,
            failed_msg_or_None); the caller aggregates, so this touches no shared state. G20 is the
            physical market's total (across inst aliases), market volume uses the config symbol."""
            g20_vol = g20_by_mkt.get(_mkt(exch, pair), 0.0)
            try:
                quote_ccy = str(pair).rsplit('-', 1)[-1]
                ccxt_name = CCXT_EXCHANGE_NAMES.get(exch)
                if not ccxt_name:
                    return [exch, pair, g20_vol, np.nan], None, f'{exch} {pair} (unknown exchange)'
                # Use 1h candles for exchanges with unreliable 1m data
                kline_tf = '1h' if exch in _HOURLY_CAPTURES_EXCHANGES else '1m'
                # Try the config symbol, then the coin's other base aliases (ccxt may list the venue
                # under a different base, e.g. kucoin BCHSV -> BSV). First non-empty wins.
                klines, last = None, 'no data'
                for symbol in _ccxt_symbol_candidates(token, pair):
                    try:
                        k = _md_client.get_klines_df(symbol, ccxt_name, kline_tf, start_dt, end_dt)
                    except Exception as e:
                        last = e
                        continue
                    if not k.empty:
                        klines = k
                        break
                if klines is None:
                    return [exch, pair, g20_vol, np.nan], None, f'{exch} {pair} ({last})'
                klines['volume_quote_ccy'] = klines['volume'] * (klines['open'] + klines['close'] + klines['high'] + klines['low']) / 4
                if quote_ccy not in _USD_QUOTES:
                    fx_series = _fx_cache.get_rate_series(quote_ccy, kline_tf, start_dt, end_dt)
                    if fx_series is not None and not fx_series.empty:
                        klines = pd.merge_asof(
                            klines.sort_values('timestamp'),
                            fx_series.rename('fx_rate').to_frame(),
                            left_on='timestamp', right_index=True,
                            direction='nearest', tolerance=pd.Timedelta(hours=2) if kline_tf == '1h' else pd.Timedelta(minutes=5),
                        )
                        klines['volume_quote_ccy'] = klines['volume_quote_ccy'] * klines['fx_rate'].ffill()
                total_volume_usd = klines['volume_quote_ccy'].sum()
                ts = klines[['timestamp', 'volume_quote_ccy']].assign(exchange=exch, pair=pair).rename(
                    columns={'volume_quote_ccy': 'market_volume_usd'})
                return [exch, pair, g20_vol, total_volume_usd], ts, None
            except Exception as e:
                return [exch, pair, g20_vol, np.nan], None, f'{exch} {pair} ({e})'

        # Parallelise ACROSS exchanges, but keep pairs on the same exchange sequential — one
        # ccxt connector is shared per exchange and isn't thread-safe. Pre-create connectors
        # single-threaded first so workers never race to lazily build the same one.
        by_exch = {}
        for pair, exch in new_combos:
            by_exch.setdefault(exch, []).append((pair, exch))
        for exch in by_exch:
            name = CCXT_EXCHANGE_NAMES.get(exch)
            if name:
                _md_client.ensure_connector(name)

        def _fetch_exchange(exch):
            return [_fetch_combo(pair, e) for pair, e in by_exch[exch]]

        new_rows, failed, done = [], [], 0
        with ThreadPoolExecutor(max_workers=min(8, len(by_exch))) as pool:
            futs = {pool.submit(_fetch_exchange, exch): exch for exch in by_exch}
            for fut in as_completed(futs):
                for row, ts, fail in fut.result():
                    new_rows.append(row)
                    if ts is not None:
                        existing_ts.append(ts)
                    if fail:
                        failed.append(fail)
                done += len(by_exch[futs[fut]])
                progress.update(label=f'Fetching captures ({done}/{total})...')

        status_label = f'Captures loaded ({total} pairs)'
        if failed:
            status_label += f'. Failed: {len(failed)}'
        progress.update(label=status_label, state='complete')
        if failed:
            _warn(f'Failed to fetch market data for: {", ".join(failed)}')
        new_df = pd.DataFrame(new_rows, columns=['Exchange','Pair','G20 Volume $','Market Volume $']).set_index(['Exchange', 'Pair'])
        existing = existing[existing.index.get_level_values('Exchange') != 'TOTAL']
        # Skip empty/all-NA frames in the concat (avoids a pandas FutureWarning)
        parts = [f for f in (existing, new_df) if not f.empty]
        captures = pd.concat(parts) if parts else new_df
    else:
        captures = existing[existing.index.get_level_values('Exchange') != 'TOTAL']
    # Sort by G20 volume descending, then add total row
    captures = captures.sort_values('G20 Volume $', ascending=False)
    captures.loc[('TOTAL',''),:] = [captures['G20 Volume $'].sum(), captures['Market Volume $'].sum()]
    captures['Capture'] = captures['G20 Volume $'] / captures['Market Volume $']
    timeseries = pd.concat(existing_ts, ignore_index=True) if existing_ts else pd.DataFrame()
    return captures, timeseries


# --------------------------------------------------------------------------------------
# Period OHLC (new) — one bar spanning the whole window from the token's md_pair/md_exchange
# --------------------------------------------------------------------------------------
def _ohlc_timeframe(start_dt, end_dt):
    hrs = (end_dt - start_dt).total_seconds() / 3600
    if hrs <= 6:
        return '5m'
    if hrs <= 24 * 3:
        return '1h'
    return '1d'


def fetch_period_ohlc(token, start_dt, end_dt):
    cfg = _config_provider()[token]
    md_pair = cfg.get('md_pair')
    md_exchange = cfg.get('md_exchange')
    if not md_pair or not md_exchange:
        _warn(f'No md_pair/md_exchange configured for {token}; skipping price OHLC.')
        return None
    try:
        klines = _md_client.get_klines_df(
            symbol=md_pair.replace('-', '/'),
            exchange=md_exchange,
            timeframe=_ohlc_timeframe(start_dt, end_dt),
            start_dt=start_dt,
            end_dt=end_dt,
            request_limit=500,
        )
    except Exception as e:
        _warn(f'Price OHLC fetch failed for {md_pair} on {md_exchange}: {e}')
        return None
    if klines.empty:
        _warn(f'No price data for {md_pair} on {md_exchange}.')
        return None
    o = float(klines['open'].iloc[0])
    h = float(klines['high'].max())
    lo = float(klines['low'].min())
    c = float(klines['close'].iloc[-1])
    pct = (c / o - 1) * 100 if o else float('nan')
    return {'open': o, 'high': h, 'low': lo, 'close': c, 'pct_change': pct}


# --------------------------------------------------------------------------------------
# Orderbook widths (TOB / BOB) — needs the injected orderbook DB engine
# --------------------------------------------------------------------------------------
_OB_WIDTH_WINDOW = 15   # rolling-median window in minutes (widths chart only)
_COMPLIANCE_BPS = 400   # BOB width under this (vs mid) == compliant with the +/-2% depth obligation
_UPTIME_FLOOR = 0.05    # venues below this uptime are "no data" (uncollected): excluded from the headline avg


def _ob_query(pairs, start_dt, end_dt, product_area=None):
    pair_list = "','".join(pairs)
    # start/end are naive UTC (like the rest of the ZP path); timegm treats them as UTC,
    # unlike datetime.timestamp() which would interpret them as local time.
    start_micros = calendar.timegm(start_dt.timetuple()) * 1_000_000
    end_micros = calendar.timegm(end_dt.timetuple()) * 1_000_000
    # product_area scopes to one deployment/program (e.g. 'mm2'); several programs write
    # snapshots for the same ccypair, and the others' rows carry no depth entries — without
    # this filter they'd be misread as an empty book.
    pa = f"AND ob.product_area = '{product_area}'\n            " if product_area else ''
    return f"""SELECT ob.id AS book_id, ob.ccypair, ob.exchange, ob.timestamp_micros,
                   obe.price, obe.size, obe.type
            FROM exch_order_books ob
            LEFT JOIN exch_order_book_entries obe
            ON obe.order_book_id = ob.id
            WHERE ob.ccypair IN ('{pair_list}')
            {pa}AND ob.timestamp_micros >= {start_micros}
            AND ob.timestamp_micros < {end_micros};
        """


def _ob_widths_query(token, start_dt, end_dt):
    return _ob_query(get_pairs(token), start_dt, end_dt)


def _fetch_ob_raw_pairs(pairs, start_dt, end_dt, product_area=None):
    """Run the orderbook snapshot query for an explicit ccypair list (raw levels: book_id,
    exchange, ccypair, timestamp_micros, price, size, type). Returns the DataFrame or None.
    product_area scopes to one program/deployment (e.g. 'mm2')."""
    if _ob_db_engine is None:
        _warn('No orderbook DB configured; skipping widths/compliance.')
        return None
    if not pairs:
        return None
    try:
        ob = pd.read_sql(_ob_query(pairs, start_dt, end_dt, product_area), _ob_db_engine)
    except Exception as e:
        _warn(f'Orderbook query failed: {e}')
        return None
    if ob.empty and product_area:
        _warn(f"No orderbook snapshots for product_area='{product_area}' — check the program name.")
    return None if ob.empty else ob


def _fetch_ob_raw(token, start_dt, end_dt, product_area=None):
    """Raw orderbook snapshots for one token (widths + compliance share this one round-trip)."""
    return _fetch_ob_raw_pairs(get_pairs(token), start_dt, end_dt, product_area)


def _gap_breaks(series, factor=2, floor=pd.Timedelta('2min')):
    """Insert a NaN row at the midpoint of any time gap larger than `factor` × the median
    sampling interval (min `floor`), so a line chart BREAKS across missing data instead of
    drawing a flat segment straight across it. `series` all share one index; returns them
    with the same break rows added to each (so they stay aligned)."""
    idx = series[0].index
    if len(idx) < 3:
        return series
    gaps = idx.to_series().diff().dropna()
    thr = max(gaps.median() * factor, floor)
    breaks = [idx[i - 1] + (idx[i] - idx[i - 1]) / 2
              for i in range(1, len(idx)) if (idx[i] - idx[i - 1]) > thr]
    if not breaks:
        return series
    bidx = pd.DatetimeIndex(breaks)
    return [pd.concat([s, pd.Series(np.nan, index=bidx)]).sort_index() for s in series]


def _ob_widths_from_raw(ob, rolling_window=_OB_WIDTH_WINDOW):
    """Per exchange/pair TOB/BOB width in bps at 1-minute resolution (raw), plus a
    rolling-median-smoothed version for the chart's optional toggle.
    Returns (summary_df, series_df) or (None, None). series_df columns: timestamp, exchange,
    pair, tob_bps, bob_bps (raw), tob_bps_smooth, bob_bps_smooth."""
    ob = ob.copy()
    ob['timestamp'] = pd.to_datetime(ob['timestamp_micros'], unit='us').dt.round('1min')
    win = f'{int(rolling_window)}min'
    summary_rows, series_frames = [], []
    for (exch, pair), grp in ob.groupby(['exchange', 'ccypair']):
        tob = grp.groupby(['timestamp', 'type'])['price'].apply(lambda px: px.max() if px.name[1] == 'BID' else px.min()).unstack()
        bob = grp.groupby(['timestamp', 'type'])['price'].apply(lambda px: px.min() if px.name[1] == 'BID' else px.max()).unstack()
        if tob.empty or 'ASK' not in tob.columns or 'BID' not in tob.columns:
            continue
        tob_raw = (10000 * (tob['ASK'] - tob['BID']) / tob['ASK']).sort_index()
        bob_raw = (10000 * (bob['ASK'] - bob['BID']) / bob['ASK']).sort_index()
        tob_sm = tob_raw.rolling(win, min_periods=1).median()
        bob_sm = bob_raw.rolling(win, min_periods=1).median()
        summary_rows.append({
            'exchange': exch, 'pair': pair,
            'tob_median': tob_raw.median(), 'tob_lo': tob_raw.min(), 'tob_hi': tob_raw.max(),
            'bob_median': bob_raw.median(), 'bob_lo': bob_raw.min(), 'bob_hi': bob_raw.max(),
        })
        # break the plotted line across real data gaps instead of flat-lining across them
        # (all four share the same index, so the same break rows keep them aligned)
        tr, br, ts_, bs = _gap_breaks([tob_raw, bob_raw, tob_sm, bob_sm])
        series_frames.append(pd.DataFrame({'timestamp': tr.index, 'exchange': exch, 'pair': pair,
                                           'tob_bps': tr.values, 'bob_bps': br.values,
                                           'tob_bps_smooth': ts_.values, 'bob_bps_smooth': bs.values}))
    if not summary_rows:
        return None, None
    summary = pd.DataFrame(summary_rows).sort_values('tob_median').reset_index(drop=True)
    series = pd.concat(series_frames, ignore_index=True)
    return summary, series


def _ob_compliance_from_raw(ob):
    """Per exchange/pair 2% depth compliance from RAW snapshots (no smoothing/rounding):
    uptime = %% of snapshots with a two-sided book; compliance = %% of ALL snapshots that are
    two-sided AND back-of-book width < 400bps vs mid (empty/one-sided snapshots count as
    breaches). Composition fractions (compliant/wide/one_sided/empty) sum to 1. Returns a
    DataFrame sorted best-compliance-first (no-data venues last) or None.
    Columns: exchange, pair, snaps, uptime, compliance, wide, one_sided, empty, bob_median, nodata."""
    total = ob.groupby(['exchange', 'ccypair'])['book_id'].nunique()  # incl. empty books
    lvl = ob.dropna(subset=['type'])  # empty-book rows have NULL type -> excluded here, kept in `total`
    if lvl.empty:
        return None
    px = lvl.groupby(['exchange', 'ccypair', 'book_id', 'type'])['price'].agg(['min', 'max']).unstack('type')

    def side(agg, s):
        return px[(agg, s)] if (agg, s) in px.columns else pd.Series(np.nan, index=px.index)

    snap = pd.DataFrame({'best_bid': side('max', 'BID'), 'worst_bid': side('min', 'BID'),
                         'best_ask': side('min', 'ASK'), 'worst_ask': side('max', 'ASK')})
    snap['two_sided'] = snap[['best_bid', 'best_ask']].notna().all(axis=1)
    mid = (snap['best_bid'] + snap['best_ask']) / 2
    with np.errstate(invalid='ignore', divide='ignore'):  # mid is NaN for one-sided/empty snaps
        snap['bob_bps'] = 10000 * (snap['worst_ask'] - snap['worst_bid']) / mid
    snap['compliant'] = snap['two_sided'] & (snap['bob_bps'] < _COMPLIANCE_BPS)

    g = snap.groupby(level=['exchange', 'ccypair'])
    in_snap = g.size()            # snapshots with >= 1 entry
    two_cnt = g['two_sided'].sum()
    comp_cnt = g['compliant'].sum()
    two_bob = snap[snap['two_sided']].groupby(level=['exchange', 'ccypair'])['bob_bps']
    bob_med, bob_lo, bob_hi = two_bob.median(), two_bob.min(), two_bob.max()
    bob_mean = two_bob.mean()  # count-weighted mean is reproducible from the per-minute buckets
                               # (bob_sum/bob_cnt), so the time-brush windowed table stays consistent

    rows = []
    for (exch, pair), n in total.items():
        n = int(n)
        ins = int(in_snap.get((exch, pair), 0))
        tc = int(two_cnt.get((exch, pair), 0))
        cc = int(comp_cnt.get((exch, pair), 0))
        uptime = tc / n if n else float('nan')
        rows.append({
            'exchange': exch, 'pair': pair, 'snaps': n,
            'uptime': uptime, 'compliance': (cc / n if n else float('nan')),
            'wide': (tc - cc) / n if n else float('nan'),        # two-sided but >= 400bps
            'one_sided': (ins - tc) / n if n else float('nan'),  # one side only
            'empty': (n - ins) / n if n else float('nan'),       # no book at all
            'bob_median': bob_med.get((exch, pair), float('nan')),
            'bob_mean': bob_mean.get((exch, pair), float('nan')),
            'bob_lo': bob_lo.get((exch, pair), float('nan')),
            'bob_hi': bob_hi.get((exch, pair), float('nan')),
            'nodata': uptime < _UPTIME_FLOOR,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    return pd.concat([df[~df['nodata']].sort_values('compliance', ascending=False),
                      df[df['nodata']]]).reset_index(drop=True)


def _ob_compliance_buckets(ob, freq='1min'):
    """Per (exchange, pair, time-bucket) 2% depth compliance COUNTS, so a client can re-aggregate
    compliance for any sub-window by summing the buckets that fall in it. Summing ALL buckets for a
    venue reproduces `_ob_compliance_from_raw` exactly for the count-based stats (uptime and the
    compliant/wide/one_sided/empty composition) and for bob_lo/bob_hi; only the central BoB differs
    (we carry sum+count → a mean, not the full-window median). Returns a tidy frame with columns
    [timestamp, exchange, pair, snaps, in_snap, two_cnt, comp_cnt, bob_sum, bob_cnt, bob_lo, bob_hi]
    or None. `snaps` counts every book_id (incl. empty, NULL-type); the rest cover entried books."""
    ob = ob.copy()
    ob['tbucket'] = pd.to_datetime(ob['timestamp_micros'], unit='us').dt.floor(freq)
    # a book_id is one snapshot (its rows share a timestamp) -> one bucket; min is that bucket
    book_bkt = ob.groupby(['exchange', 'ccypair', 'book_id'])['tbucket'].min()
    book = book_bkt.reset_index()  # exchange, ccypair, book_id, tbucket
    total = (book.groupby(['exchange', 'ccypair', 'tbucket'])['book_id'].nunique()
             .rename('snaps'))  # every book_id, incl. empty
    lvl = ob.dropna(subset=['type'])  # empty-book rows (NULL type) excluded here, kept in `total`
    if lvl.empty:
        return None
    px = lvl.groupby(['exchange', 'ccypair', 'book_id', 'type'])['price'].agg(['min', 'max']).unstack('type')

    def side(agg, s):
        return px[(agg, s)] if (agg, s) in px.columns else pd.Series(np.nan, index=px.index)

    snap = pd.DataFrame({'best_bid': side('max', 'BID'), 'worst_bid': side('min', 'BID'),
                         'best_ask': side('min', 'ASK'), 'worst_ask': side('max', 'ASK')})
    snap['two_sided'] = snap[['best_bid', 'best_ask']].notna().all(axis=1)
    mid = (snap['best_bid'] + snap['best_ask']) / 2
    with np.errstate(invalid='ignore', divide='ignore'):
        snap['bob_bps'] = 10000 * (snap['worst_ask'] - snap['worst_bid']) / mid
    snap['compliant'] = snap['two_sided'] & (snap['bob_bps'] < _COMPLIANCE_BPS)
    snap = snap.join(book_bkt.rename('tbucket'))  # both keyed by (exchange, ccypair, book_id)
    grp = snap.reset_index().groupby(['exchange', 'ccypair', 'tbucket'])
    agg = grp.agg(in_snap=('two_sided', 'size'), two_cnt=('two_sided', 'sum'),
                  comp_cnt=('compliant', 'sum'))
    two = snap[snap['two_sided']].reset_index().groupby(['exchange', 'ccypair', 'tbucket'])['bob_bps']
    bob = pd.DataFrame({'bob_sum': two.sum(), 'bob_cnt': two.count(),
                        'bob_lo': two.min(), 'bob_hi': two.max()})
    out = total.to_frame().join(agg, how='left').join(bob, how='left').reset_index()
    for c in ['in_snap', 'two_cnt', 'comp_cnt', 'bob_sum', 'bob_cnt']:
        out[c] = out[c].fillna(0)
    return out.rename(columns={'ccypair': 'pair', 'tbucket': 'timestamp'})


def fetch_ob_widths(token, start_dt, end_dt, rolling_window=_OB_WIDTH_WINDOW):
    """Per exchange/pair TOB/BOB width in bps, rolling-median smoothed. Returns
    (summary_df, series_df) or (None, None). (Dashboard entry point; the ZP path uses
    _fetch_ob_raw directly to share the query with compliance.)"""
    progress = _status(f'Fetching orderbook widths for {token}...')
    ob = _fetch_ob_raw(token, start_dt, end_dt)
    if ob is None:
        progress.update(label='No orderbook data for widths.', state='complete')
        return None, None
    summary, series = _ob_widths_from_raw(ob, rolling_window)
    progress.update(label=f'Orderbook widths loaded ({0 if summary is None else len(summary)} combos)',
                    state='complete')
    return summary, series


# --------------------------------------------------------------------------------------
# Program grid cell-shapers (pure) — turn a per-coin/per-batch compute frame into the tidy
# per-(asset, exchange, quote) cells the grid reports consume. Shared by the *_matrix
# functions and by fetch_desk_bundle (so the single-pass desk report reuses them, no re-fetch).
# --------------------------------------------------------------------------------------
def _add_asset_labels(df):
    """Add asset/quote/perp/col_label from a `pair` column (base = pair before the first '-')."""
    base = df['pair'].str.split('-').str[0]
    df['perp'] = base.str.startswith('P.')
    df['asset'] = base.str.replace('P.', '', regex=False)
    df['quote'] = df['pair'].str.split('-').str[1].fillna('')
    df['col_label'] = (df['exchange'].astype(str)
                       + df['perp'].map({True: '.P', False: ''}) + ' ' + df['quote'])
    return df


def _compliance_cells(comp):
    """Shape an `_ob_compliance_from_raw` frame (cols incl. pair, exchange, nodata) into grid
    cells, or None if empty."""
    if comp is None or comp.empty:
        return None
    return _add_asset_labels(comp.copy())


def _venue_key(exchange, pair):
    """Physical-market key for the captures<->orderbook join: (exchange, perp?, quote-ccy), dropping
    the base symbol. Per coin there's one market per exchange+perp+quote, so a venue alias in the
    base (AI/AIGENSYN, BSV/BCHSV) — captures uses the config symbol, the OB DB the canonical one —
    collapses to the same key and still matches. The perp flag keeps a spot and a perp of the same
    quote apart (e.g. PARTI runs both PARTI-USDT and P.PARTI-USDT on binance) so one's active book
    doesn't flag the other. Quote upper-cased; exchange left as-is (both sides already agree on it)."""
    p = str(pair)
    return (str(exchange), p.startswith('P.'), p.rsplit('-', 1)[-1].upper())


def _captures_cells(cap):
    """Shape a `fetch_captures` per-venue frame (MultiIndex (Exchange, Pair)) into grid cells,
    or None if empty."""
    if cap is None or cap.empty:
        return None
    cap = cap[cap.index.get_level_values('Exchange') != 'TOTAL'].reset_index()
    if cap.empty:
        return None
    cap = cap.rename(columns={'Exchange': 'exchange', 'Pair': 'pair', 'G20 Volume $': 'g20',
                              'Market Volume $': 'market', 'Capture': 'capture'})
    cap = _add_asset_labels(cap)
    # nodata = truly no market data (blank cell). nonmm = has data but no active orderbook — kept in
    # the grid (dimmed, toggle-revealable) but excluded from the MM-only headline/averages.
    if 'ob_active' in cap.columns:
        cap['nonmm'] = (~cap['ob_active'].fillna(False)) & cap['capture'].notna()
        cap['nodata'] = cap['capture'].isna()
    else:  # no OB info (standalone matrix path): old behaviour, nothing dimmed
        cap['nonmm'] = False
        cap['nodata'] = (cap['g20'].fillna(0) <= 0) | cap['capture'].isna()
    return cap


def _markouts_cells(mk):
    """Shape a `fetch_markouts` pivot ((exchange, inst) index; volume_$ + pnl_<s>s_bps) into grid
    cells. Returns (cells_df, horizons_list) — (None, []) if empty."""
    if mk is None or mk.empty:
        return None, []
    summary = mk.reset_index()
    hz = sorted(int(c[4:-5]) for c in summary.columns
                if c.startswith('pnl_') and c.endswith('s_bps'))
    summary = summary.rename(columns={'volume_$': 'volume', 'inst': 'pair',
                                      **{f'pnl_{s}s_bps': f'mo_{s}' for s in hz}})
    cells = _add_asset_labels(summary)
    mo_cols = [f'mo_{s}' for s in hz]
    cells['nodata'] = (cells['volume'].fillna(0) <= 0) | cells[mo_cols].isna().all(axis=1)
    return cells, hz


def fetch_compliance_matrix(coins, start_dt, end_dt, product_area=None):
    """2% depth compliance for a set of coins (e.g. one program), as a tidy per-cell frame.
    Coins not in config are dropped (returned as `ignored`). One OB query across all pairs,
    scoped to `product_area` (the program, e.g. 'mm2') so other programs' entry-less snapshots
    aren't misread as an empty book. Returns (cells_df, ignored, tokens). cells_df has one row
    per asset × (exchange, quote[,perp]) cell with columns: asset, exchange, quote, perp,
    col_label, snaps, uptime, compliance, wide, one_sided, empty, bob_median, nodata."""
    cfg = _config_provider()
    coins = [c.upper() for c in coins]
    tokens = [c for c in coins if c in cfg]
    ignored = [c for c in coins if c not in cfg]
    if not tokens:
        return None, ignored, tokens
    pairs = sorted({p for t in tokens for p in get_pairs(t)})
    ob = _fetch_ob_raw_pairs(pairs, start_dt, end_dt, product_area)
    if ob is None:
        return None, ignored, tokens
    cells = _compliance_cells(_ob_compliance_from_raw(ob))
    if cells is None:
        return None, ignored, tokens
    return cells, ignored, tokens


def fetch_captures_matrix(coins, start_dt, end_dt, no_perp_capture=True):
    """Per-(asset, exchange, quote) capture % for a set of coins (e.g. one program), reusing
    the existing fetch_trades + fetch_captures per token (no new capture math). Coins not in
    config are dropped (returned as `ignored`). Returns (cells_df, ignored, tokens); cells
    columns: asset, exchange, quote, perp, col_label, capture, g20, market, nodata. Slow —
    it fetches trades + market data per token."""
    cfg = _config_provider()
    coins = [c.upper() for c in coins]
    tokens = [c for c in coins if c in cfg]
    ignored = [c for c in coins if c not in cfg]
    frames = []
    total = len(tokens)
    for i, token in enumerate(tokens, 1):
        progress = _status(f'Captures: {token} ({i}/{total})...')
        try:
            trades = fetch_trades(token, start_dt, end_dt)
            if trades.empty:
                continue
            cap, _ = fetch_captures(start_dt, end_dt, trades, token=token, no_perp_capture=no_perp_capture)
            cells = _captures_cells(cap)
            if cells is not None:
                frames.append(cells)
        except Exception as e:
            _warn(f'Captures for {token} failed: {e}')
        finally:
            progress.update(label=f'Captures: {token} ({i}/{total})', state='complete')
    if not frames:
        return None, ignored, tokens
    return pd.concat(frames, ignore_index=True), ignored, tokens


def fetch_markouts_matrix(coins, start_dt, end_dt):
    """Per-(asset, exchange, pair) markouts for a set of coins (e.g. one program), reusing the
    existing fetch_trades + fetch_markouts per token (no new markout math). Each cell carries the
    volume-weighted markout (bps, positive = favourable) at every horizon the coin's md_exchange
    computes; horizons vary per exchange, so the returned `horizons` is their sorted union and a
    cell is NaN where its exchange didn't compute that horizon. Coins not in config are dropped
    (returned as `ignored`). Returns (cells_df, ignored, tokens, horizons); cells columns: asset,
    exchange, quote, perp, col_label, volume, mo_<sec> per horizon, nodata. Slow — it fetches
    trades + market-data klines per token."""
    cfg = _config_provider()
    coins = [c.upper() for c in coins]
    tokens = [c for c in coins if c in cfg]
    ignored = [c for c in coins if c not in cfg]
    frames = []
    horizons = set()
    total = len(tokens)
    for i, token in enumerate(tokens, 1):
        progress = _status(f'Markouts: {token} ({i}/{total})...')
        try:
            trades = fetch_trades(token, start_dt, end_dt)
            if trades.empty:
                continue
            summary, _ = fetch_markouts(token, trades)
            cells, hz = _markouts_cells(summary)
            if cells is not None:
                frames.append(cells)
                horizons.update(hz)
        except Exception as e:
            _warn(f'Markouts for {token} failed: {e}')
        finally:
            progress.update(label=f'Markouts: {token} ({i}/{total})', state='complete')
    if not frames:
        return None, ignored, tokens, sorted(horizons)
    df = pd.concat(frames, ignore_index=True)  # missing-horizon columns align to NaN
    return df, ignored, tokens, sorted(horizons)


@dataclass
class DeskBundle:
    """One-pass desk compute: the per-coin ZP results plus the three program grids derived from
    them (so nothing is re-fetched)."""
    results: dict                              # token -> ZPResult (order = tokens)
    tokens: list
    ignored: list
    compliance_cells: Optional[pd.DataFrame]
    captures_cells: Optional[pd.DataFrame]
    markouts_cells: Optional[pd.DataFrame]
    horizons: list


def _ccxt_symbol_for(pair):
    base, quote = pair.split('-', 1)
    if base.startswith('P.'):
        return f'{base[2:]}/{quote}:{quote}'  # perp
    return f'{base}/{quote}'


def fetch_ticksize(coins, *, no_perp=True):
    """Per (coin, 'EXCHANGE QUOTE') tick size + mid + tick-in-bps for every CONFIGURED venue of the
    given coins, via ccxt (one load_markets + fetch_tickers per exchange, parallel across exchanges).
    Tick size is a static market property, so no time window. Returns
    {(coin, col): {'tick', 'mid', 'tick_bps'}}; cells with no resolvable tick are dropped."""
    cfg = _config_provider()
    want, by_ex = [], {}  # want: (coin, col, ccxt_name, symbol); by_ex: ccxt_name -> {symbols}
    for coin in coins:
        cu = coin.upper()
        for e in cfg.get(cu, {}).get('exchanges', []):
            pair, exch = e['pair'], e['exchange']
            if (no_perp and pair.startswith('P.')) or '-' not in pair:
                continue
            prefix = exch.split(':', 1)[0]
            ccxt_name = CCXT_EXCHANGE_NAMES.get(prefix)
            if not ccxt_name:
                continue
            # .P marks a perp so it doesn't collide with the same-exchange/quote spot column.
            perp = '.P' if pair.startswith('P.') else ''
            col = f'{prefix}{perp} {pair.split("-", 1)[1]}'   # EXCHANGE[.P] QUOTE
            sym = _ccxt_symbol_for(pair)
            want.append((cu, col, ccxt_name, sym))
            by_ex.setdefault(ccxt_name, set()).add(sym)
    if not by_ex:
        return {}

    def _one(ccxt_name):
        try:
            return ccxt_name, _md_client.get_ticks_and_mids(ccxt_name, sorted(by_ex[ccxt_name]))
        except Exception as e:
            _warn(f'Tick sizes for {ccxt_name} failed: {e}')
            return ccxt_name, {}

    with ThreadPoolExecutor(max_workers=min(8, len(by_ex))) as pool:
        results = dict(pool.map(_one, by_ex))
    cells = {}
    for coin, col, ccxt_name, sym in want:
        tick, mid = results.get(ccxt_name, {}).get(sym, (None, None))
        if tick is None:
            continue
        cells.setdefault((coin, col), {
            'tick': tick, 'mid': mid, 'tick_bps': (tick / mid * 10000) if mid else None})
    return cells


def _desk_fetch_workers(default=4):
    """How many coins to compute at once in fetch_desk_bundle. Per-coin compute now only hits the
    public exchanges (trades/OB are batched up front), so this doesn't touch the internal services.
    Tunable via DESK_FETCH_WORKERS; 1 restores fully-serial behaviour."""
    try:
        return max(1, int(os.environ.get('DESK_FETCH_WORKERS', default)))
    except (TypeError, ValueError):
        return default


def fetch_desk_bundle(coins, start_dt, end_dt, *, product_area, no_perp_capture=True,
                      usd_only=False, progress=None):
    """Single-pass desk compute for a program: run `compute_zp` once per coin and derive the three
    program grids (compliance/captures/markouts) from those per-coin results — no re-fetching, so
    it's ~half the work of calling the three *_matrix functions plus a ZP per coin. Both the
    orderbook snapshots and the trades are fetched for the whole desk in ONE call each and sliced
    per coin (far fewer round-trips than one per coin/pair, and lighter on the services); either
    batch falls back to per-coin fetching if it fails. `product_area` scopes the orderbook queries (the
    program, e.g. 'mm2'). `progress(i, n, token)` is called before each coin if given (for a live
    status line). Coins not in config are dropped (`ignored`). No-trades coins still contribute
    compliance cells (orderbook-only). Returns a DeskBundle."""
    cfg = _config_provider()
    coins = [c.upper() for c in coins]
    tokens = [c for c in coins if c in cfg]
    ignored = [c for c in coins if c not in cfg]
    results = {}
    comp_frames, cap_frames, mo_frames = [], [], []
    horizons = set()
    n = len(tokens)
    fetch_times = {}  # per-coin compute wall time — to size the parallel-fetch payoff

    # Batch the orderbook query for the WHOLE desk into one round-trip, then hand each coin its
    # slice — the same rows a per-coin query would return (WHERE ccypair IN coin_pairs), but in one
    # indexed scan and one connection instead of `n`. None => no OB data for the desk (or no DB).
    t_ob0 = time.perf_counter()
    ob_all = _fetch_ob_raw_pairs(sorted({p for t in tokens for p in get_pairs(t)}),
                                 start_dt, end_dt, product_area)
    print(f'[desk timing] {product_area}: orderbook batch query {time.perf_counter() - t_ob0:.1f}s '
          f'({0 if ob_all is None else len(ob_all)} rows)', flush=True)

    # Batch the trades the same way: ONE no-inst pull for the whole window (far fewer requests than
    # paginating each pair), sliced per coin below. If the pull errors or comes back empty (which an
    # 8h firm-wide window never legitimately does), fall back to per-coin fetching so a bad pull can
    # never mask the desk as "no trades". Also degrades gracefully if the service lacks the method.
    t_tr0 = time.perf_counter()
    raw_trades = None
    if hasattr(_trade_service, 'get_all_trades'):
        try:
            raw_trades = _trade_service.get_all_trades(start_dt, end_dt)
        except Exception as e:
            _warn(f'Desk trades batch failed ({e}); falling back to per-coin fetch.')
    use_trade_batch = raw_trades is not None and not raw_trades.empty
    print(f'[desk timing] {product_area}: trades batch pull {time.perf_counter() - t_tr0:.1f}s '
          f'({0 if raw_trades is None else len(raw_trades)} rows, batch={use_trade_batch})', flush=True)

    # With trades + orderbook now fetched up front, per-coin compute_zp only hits the PUBLIC
    # exchanges (markout/capture/ohlc klines) — nothing internal — so coins can run concurrently
    # without adding any load to trade-streaming or the OB DB. The klines go through _md_client's
    # per-exchange lock (same exchange serialises, different exchanges parallelise). Workers do the
    # I/O; cell-shaping/appends happen here as each finishes (kept single-threaded). Pre-warm the
    # location metadata so N workers don't race to build it. DESK_FETCH_WORKERS tunes/serialises it.
    try:
        _get_loc_meta()
    except Exception:
        pass  # a real failure will resurface (and be warned) inside the per-coin fetch
    workers = _desk_fetch_workers()

    def _one(token):
        c0 = time.perf_counter()
        # this coin's slice of the batched orderbook (None passes through -> widths skipped, no
        # re-query); explicit so compute_zp never falls back to a per-coin query here
        coin_ob = None if ob_all is None else ob_all[ob_all['ccypair'].isin(get_pairs(token))]
        coin_trades = _desk_trades_for(token, raw_trades) if use_trade_batch else _UNSET
        res = compute_zp(token, start_dt, end_dt, product_area=product_area,
                         no_perp_capture=no_perp_capture, usd_only=usd_only,
                         ob_raw=coin_ob, trades=coin_trades)
        return res, time.perf_counter() - c0

    def _collect(token, res, dt):
        fetch_times[token] = dt
        results[token] = res
        c = _compliance_cells(res.ob_compliance)
        if c is not None:
            comp_frames.append(c)
        cap = _captures_cells(res.captures)
        if cap is not None:
            cap_frames.append(cap)
        mo, hz = _markouts_cells(res.markouts)
        if mo is not None:
            mo_frames.append(mo)
            horizons.update(hz)

    t_bundle0 = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, token): token for token in tokens}
        for fut in as_completed(futures):
            token = futures[fut]
            done += 1
            if progress:
                try:
                    progress(done, n, token)  # completion-order under concurrency
                except Exception:
                    pass
            try:
                res, dt = fut.result()
            except Exception as e:
                _warn(f'Desk compute for {token} failed: {e}')
                continue
            _collect(token, res, dt)
    slow = ', '.join(f'{t} {d:.1f}s' for t, d in
                     sorted(fetch_times.items(), key=lambda kv: -kv[1])[:5])
    print(f'[desk timing] {product_area}: {len(fetch_times)}/{n} coins computed ({workers} workers) '
          f'in {time.perf_counter() - t_bundle0:.1f}s | slowest: {slow}', flush=True)
    return DeskBundle(
        results=results, tokens=tokens, ignored=ignored,
        compliance_cells=pd.concat(comp_frames, ignore_index=True) if comp_frames else None,
        captures_cells=pd.concat(cap_frames, ignore_index=True) if cap_frames else None,
        markouts_cells=pd.concat(mo_frames, ignore_index=True) if mo_frames else None,
        horizons=sorted(horizons))


# --------------------------------------------------------------------------------------
# Stylers (pure) — used by the dashboard tables
# --------------------------------------------------------------------------------------
def style_summary(summary):
    return summary.style.format({
        'buy_quantity': '{:,.0f}',
        'sell_quantity': '{:,.0f}',
        'net_quantity': '{:,.0f}',
        'buy_notional': '{:,.0f}',
        'sell_notional': '{:,.0f}',
        'net_notional': '{:,.0f}',
        'pnl_$': '{:,.0f}'
    })


def style_markouts(markouts):
    markout_cols = [c for c in markouts.columns if c != 'volume_$']
    styled_markouts = (
        markouts
        .style
        .background_gradient(
            cmap = LinearSegmentedColormap.from_list('', ['red', 'white', 'green']),
            subset = markout_cols,
            vmin = -markouts[markout_cols].abs().max().max(),
            vmax = markouts[markout_cols].abs().max().max()
        )
        .format({col: '{:,.0f}' if col == 'volume_$' else '{:,.1f}' for col in markouts.columns})
    )
    return styled_markouts


def style_captures(captures):
    return captures.style.format({
        'Market Volume $':'{:,.0f}',
        'G20 Volume $':'{:,.0f}',
        'Capture':'{:.0%}'
    })


# --------------------------------------------------------------------------------------
# ZP orchestrator
# --------------------------------------------------------------------------------------
_UNSET = object()  # compute_zp(ob_raw=): distinguishes "not supplied → fetch it" from "None → none"


@dataclass
class ZPResult:
    token: str
    start: datetime
    end: datetime
    trades: pd.DataFrame
    mm_summary: pd.DataFrame
    other_summary: pd.DataFrame        # everything that isn't MM (gamma, hedgers, any other tag)
    total_summary: pd.DataFrame
    price_ohlc: Optional[dict]
    markouts: Optional[pd.DataFrame]
    markout_windows: Optional[list]
    captures: Optional[pd.DataFrame]
    captures_ts: Optional[pd.DataFrame]
    ob_widths: Optional[pd.DataFrame]
    ob_widths_ts: Optional[pd.DataFrame]
    lead_scores: Optional[pd.DataFrame]
    warnings: list
    ob_compliance: Optional[pd.DataFrame] = None  # per-venue 2% depth uptime/compliance
    delta_pnl_ts: Optional[pd.DataFrame] = None    # cumulative delta ($) + realised/unrealised/total P&L over time
    markout_trades: Optional[pd.DataFrame] = None  # per-trade MM fills w/ pnl_<s>s_bps (for windowed re-cut)
    ob_compliance_ts: Optional[pd.DataFrame] = None  # per venue-per-minute compliance counts (windowed re-cut)


def _filter_usd(trades):
    return trades[trades['inst'].str.split('-').str[1].isin(_USD_QUOTES)]


def compute_zp(token, start_dt, end_dt, *, want_markouts=True, want_captures=True,
               want_lead_scores=False, want_widths=True, want_pnl=True, usd_only=False,
               no_perp_capture=True, product_area=None, ob_raw=_UNSET, trades=_UNSET):
    """Run the full ZP compute path headless (requires init()). Non-fatal failures are
    collected into ZPResult.warnings so a partial report is still produced.

    `ob_raw` and `trades` let a desk-level caller (fetch_desk_bundle) inject this coin's data from
    one batched fetch instead of querying per coin: `ob_raw` = the coin's raw orderbook slice (or
    None if none), `trades` = the coin's enriched trades slice. Left unset, each is fetched here as
    usual (the standalone /zp path)."""
    _tls.warnings = []          # thread-local, so parallel coins never clobber each other
    timings = {}
    t_start = time.perf_counter()
    try:
        with _timer('trades', timings):
            if trades is _UNSET:
                trades = fetch_trades(token, start_dt, end_dt)  # standalone /zp: fetch per coin
            # else: a desk-batched slice was injected (already enriched) — use it as-is
        has_trades = not trades.empty
        if has_trades and usd_only:
            trades = _filter_usd(trades)

        # Trade-dependent sections. A coin with no trades still gets a report — the orderbook
        # sections below are computed regardless (they don't depend on trades).
        mm_summary = other_summary = total_summary = pd.DataFrame()
        price_ohlc = markouts = markout_windows = captures = captures_ts = lead_scores = None
        delta_pnl_ts = markout_trades = None
        if has_trades:
            with _timer('summary', timings):
                mm_summary, other_summary, total_summary = fetch_summary(trades)
            with _timer('ohlc', timings):
                price_ohlc = fetch_period_ohlc(token, start_dt, end_dt)

            if want_pnl:
                try:
                    with _timer('delta_pnl', timings):
                        delta_pnl_ts = fetch_delta_pnl(token, trades, start_dt, end_dt)
                except Exception as e:
                    _warn(f'Delta/PnL failed: {e}')

            if want_markouts:
                try:
                    with _timer('markouts', timings):
                        markouts, markout_trades = fetch_markouts(token, trades)
                        md_exchange = _config_provider()[token]['md_exchange']
                        markout_windows = MD_CONFIG[md_exchange]['markout_windows']
                except Exception as e:
                    _warn(f'Markouts failed: {e}')

            if want_captures:
                try:
                    with _timer('captures', timings):
                        captures, captures_ts = fetch_captures(start_dt, end_dt, trades, token=token,
                                                               no_perp_capture=no_perp_capture)
                except Exception as e:
                    _warn(f'Captures failed: {e}')

            if want_lead_scores:
                try:
                    with _timer('lead_scores', timings):
                        lead_scores = fetch_lead_scores(trades)
                except Exception as e:
                    _warn(f'Lead scores failed: {e}')

        # Orderbook widths + compliance — trade-independent (config pairs + OB DB only), so they
        # run even when there are no trades.
        ob_widths = ob_widths_ts = ob_compliance = ob_compliance_ts = None
        if want_widths:
            try:
                with _timer('widths', timings):
                    # one query -> widths + compliance; or a desk-batched slice passed by the caller
                    raw = (_fetch_ob_raw(token, start_dt, end_dt, product_area)
                           if ob_raw is _UNSET else ob_raw)
                    if raw is not None and not raw.empty:
                        ob_widths, ob_widths_ts = _ob_widths_from_raw(raw)
                        ob_compliance = _ob_compliance_from_raw(raw)
                        ob_compliance_ts = _ob_compliance_buckets(raw)
            except Exception as e:
                _warn(f'Orderbook widths/compliance failed: {e}')

        # Flag which capture venues we actually make markets on (an active 2% book, per the OB
        # compliance ~nodata set). Lets captures default to MM-only. Fallback: if there's no OB
        # data at all, mark everything active so captures never comes back empty.
        # The join is on (exchange, quote-ccy), NOT the full pair: some coins trade under a venue
        # alias (AI→AIGENSYN on binance, BSV→BCHSV on kucoin) so captures uses the config symbol
        # while the OB DB logs the book under the coin's canonical symbol. Per coin there's one
        # market per exchange+quote, so dropping the base sidesteps the alias entirely. See
        # cap_ob_join_probe.py.
        if captures is not None and not captures.empty:
            active = set()
            if ob_compliance is not None and not ob_compliance.empty:
                active = {_venue_key(e, p) for e, p, nd in zip(ob_compliance['exchange'],
                          ob_compliance['pair'], ob_compliance['nodata']) if not nd}
            captures = captures.copy()
            captures['ob_active'] = ([_venue_key(e, p) in active
                                      for e, p in captures.index] if active else True)

        return ZPResult(token, start_dt, end_dt, trades, mm_summary, other_summary,
                        total_summary, price_ohlc, markouts, markout_windows,
                        captures, captures_ts, ob_widths, ob_widths_ts, lead_scores,
                        list(_tls.warnings or []), ob_compliance=ob_compliance,
                        delta_pnl_ts=delta_pnl_ts, markout_trades=markout_trades,
                        ob_compliance_ts=ob_compliance_ts)
    finally:
        _tls.warnings = None
        total = time.perf_counter() - t_start
        ranked = ', '.join(f'{k} {v:.1f}s' for k, v in sorted(timings.items(),
                                                               key=lambda kv: -kv[1]))
        print(f'[zp timing] TOTAL {total:.1f}s  ({ranked})', flush=True)
