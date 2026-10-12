import threading
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from market_data import MarketDataClient


# Explicit overrides for specific currencies
FX_OVERRIDES = {
    'FDUSD': ('FDUSD/USDT', 'binance'),
    'GBP': ('GBP/USD', 'bitstamp'),
}

CACHE_DAYS = 7
REFRESH_INTERVAL_SECONDS = 60
_TF_TO_PANDAS = {'1s': '1s', '1m': '1min', '5m': '5min', '10s': '10s', '15m': '15min', '1h': '1h', '1d': '1D'}
_RESAMPLE_AGG = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}


@dataclass
class FxPairSpec:
    quote_ccy: str
    symbol: str
    exchange: str
    invert: bool
    is_direct_usd: bool
    denominator_symbol: str | None = None  # For derived pairs: rate = symbol / denominator_symbol


def derive_required_fx_pairs(config: dict, binance_symbols: set[str] = None) -> list[FxPairSpec]:
    quote_currencies = set()
    for token_config in config.values():
        for entry in token_config.get('exchanges', []):
            parts = entry['pair'].split('-') if isinstance(entry, dict) else entry.split('-')
            if len(parts) == 2:
                quote_currencies.add(parts[1])

    specs = [
        # Always include USDC/USDT for premium analysis USDT->USD conversion
        FxPairSpec(quote_ccy='_USDC_USDT', symbol='USDC/USDT', exchange='binance',
                   invert=False, is_direct_usd=False),
    ]

    for ccy in sorted(quote_currencies):
        if ccy in ('USDT', 'USDC', 'USD'):
            continue

        if ccy in FX_OVERRIDES:
            symbol, exchange = FX_OVERRIDES[ccy]
            specs.append(FxPairSpec(
                quote_ccy=ccy, symbol=symbol, exchange=exchange,
                invert=False, is_direct_usd=True,
            ))
        elif binance_symbols and f'{ccy}/USDT' in binance_symbols:
            specs.append(FxPairSpec(
                quote_ccy=ccy, symbol=f'{ccy}/USDT', exchange='binance',
                invert=False, is_direct_usd=False,
            ))
        elif binance_symbols and f'USDT/{ccy}' in binance_symbols:
            specs.append(FxPairSpec(
                quote_ccy=ccy, symbol=f'USDT/{ccy}', exchange='binance',
                invert=True, is_direct_usd=False,
            ))
        else:
            # Derive from BTC/{ccy} ÷ BTC/USDT on Binance
            specs.append(FxPairSpec(
                quote_ccy=ccy, symbol=f'BTC/{ccy}', exchange='binance',
                invert=True, is_direct_usd=False, denominator_symbol='BTC/USDT',
            ))

    return specs


class FxRateCache:
    def __init__(self, md_client: MarketDataClient, config: dict):
        self._md_client = md_client
        self._lock = threading.Lock()
        self._data: dict[str, pd.DataFrame] = {}
        self._binance_symbols = self._load_binance_symbols()
        self._specs: list[FxPairSpec] = derive_required_fx_pairs(config, self._binance_symbols)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._failed: set[str] = set()  # quote_ccy values that failed on startup — don't retry

    def _load_binance_symbols(self) -> set[str]:
        try:
            if 'binance' not in self._md_client.ccxt_connectors:
                self._md_client._add_connector('binance')
            ex = self._md_client.ccxt_connectors['binance']
            ex.load_markets()
            return set(ex.symbols)
        except Exception as e:
            print(f'[FX Cache] Failed to load Binance markets: {e}')
            return set()

    def refresh_pairs(self, config: dict):
        new_specs = derive_required_fx_pairs(config, self._binance_symbols)
        with self._lock:
            self._specs = new_specs
        # Fetch data for any new quote currencies
        for spec in new_specs:
            if spec.quote_ccy not in self._data:
                self._fetch_pair(spec, days=CACHE_DAYS)

    def refetch_all(self, status_callback=None):
        """Clear all cached data and re-fetch everything."""
        with self._lock:
            self._data.clear()
        self._failed.clear()
        self.startup_fetch(status_callback)

    def startup_fetch(self, status_callback=None):
        total = len(self._specs)
        for i, spec in enumerate(self._specs, 1):
            pair_label = f'{spec.symbol} / {spec.denominator_symbol}' if spec.denominator_symbol else spec.symbol
            label = f'Fetching FX rates: {pair_label} from {spec.exchange} ({i}/{total})...'
            if status_callback:
                status_callback(label)
            else:
                print(f'[FX Cache] {label}')
            self._fetch_pair(spec, days=CACHE_DAYS)
        if status_callback:
            status_callback(f'FX rate cache loaded ({total} pairs)')

    def _fetch_single(self, symbol: str, exchange: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame | None:
        """Fetch 1m klines for a single symbol, returning a timestamp-indexed DataFrame or None."""
        try:
            df = self._md_client.get_klines_df(
                symbol=symbol, exchange=exchange, timeframe='1m',
                start_dt=start_dt, end_dt=end_dt, request_limit=1000,
            )
            if df.empty:
                return None
            return df.set_index('timestamp').sort_index()
        except Exception as e:
            print(f'[FX Cache] Failed to fetch {symbol} on {exchange}: {e}')
            return None

    def _fetch_klines(self, spec: FxPairSpec, start_dt: datetime, end_dt: datetime) -> pd.DataFrame | None:
        """Fetch 1m klines for a spec. For derived pairs, fetches both symbols and divides."""
        if spec.denominator_symbol is None:
            return self._fetch_single(spec.symbol, spec.exchange, start_dt, end_dt)

        # Derived pair: rate = numerator / denominator
        num_df = self._fetch_single(spec.symbol, spec.exchange, start_dt, end_dt)
        den_df = self._fetch_single(spec.denominator_symbol, spec.exchange, start_dt, end_dt)
        if num_df is None or den_df is None:
            return None

        # Align on shared timestamps and divide OHLCV
        common = num_df.index.intersection(den_df.index)
        if common.empty:
            return None
        result = pd.DataFrame(index=common)
        for col in ('open', 'high', 'low', 'close'):
            result[col] = num_df.loc[common, col] / den_df.loc[common, col]
        result['volume'] = 0  # volume is meaningless for a derived rate
        return result

    def _fetch_pair(self, spec: FxPairSpec, days: int = CACHE_DAYS, since: datetime = None):
        if spec.quote_ccy in self._failed:
            return
        start_dt = since or (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days))
        df = self._fetch_klines(spec, start_dt, datetime.now(timezone.utc).replace(tzinfo=None))
        if df is None:
            self._failed.add(spec.quote_ccy)
            print(f'[FX Cache] No data for {spec.symbol} on {spec.exchange} — will not retry, defaulting to 1.0')
            return
        with self._lock:
            if since and spec.quote_ccy in self._data:
                existing = self._data[spec.quote_ccy]
                df = pd.concat([existing, df[~df.index.isin(existing.index)]]).sort_index()
            cutoff = pd.Timestamp.now('UTC').tz_localize(None) - pd.Timedelta(days=CACHE_DAYS)
            df = df[df.index >= cutoff]
            self._data[spec.quote_ccy] = df

    def start_background_refresh(self):
        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _refresh_loop(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(REFRESH_INTERVAL_SECONDS)
            if self._stop_event.is_set():
                break
            with self._lock:
                specs = list(self._specs)
            for spec in specs:
                if self._stop_event.is_set():
                    break
                # Fetch only new data since last timestamp
                last_ts = None
                with self._lock:
                    if spec.quote_ccy in self._data and not self._data[spec.quote_ccy].empty:
                        last_ts = self._data[spec.quote_ccy].index[-1].to_pydatetime()
                self._fetch_pair(spec, since=last_ts)

    def _get_spec(self, quote_ccy: str) -> FxPairSpec | None:
        return next((s for s in self._specs if s.quote_ccy == quote_ccy), None)

    def _to_rate_series(self, df: pd.DataFrame, timeframe: str, spec: FxPairSpec | None) -> pd.Series:
        pd_freq = _TF_TO_PANDAS.get(timeframe, timeframe)
        if timeframe != '1m':
            df = df.resample(pd_freq).agg(_RESAMPLE_AGG).dropna(subset=['open'])
        rate = (df['open'] + df['close']) / 2
        if spec and spec.invert:
            rate = 1 / rate
        return rate

    def _fallback_fetch(self, spec: FxPairSpec, start_dt: datetime, end_dt: datetime) -> pd.DataFrame | None:
        return self._fetch_klines(spec, start_dt, end_dt)

    def get_rate_series(self, quote_ccy: str, timeframe: str, start_dt: datetime, end_dt: datetime) -> pd.Series | None:
        spec = self._get_spec(quote_ccy)

        with self._lock:
            cached = self._data.get(quote_ccy)
            if cached is not None and not cached.empty:
                cached = cached.copy()
            else:
                cached = None

        req_start = pd.Timestamp(start_dt)
        req_end = pd.Timestamp(end_dt)

        if cached is not None:
            cache_start = cached.index.min()
            # Check if cache fully covers the requested range
            if req_start >= cache_start - pd.Timedelta(hours=1):
                # Cache covers the range — use it directly
                df = cached[(cached.index >= req_start - pd.Timedelta(hours=1)) & (cached.index <= req_end)]
                if not df.empty:
                    return self._to_rate_series(df, timeframe, spec)

            # Cache doesn't cover the start — fetch the missing portion and combine
            if spec:
                gap_df = self._fallback_fetch(spec, start_dt, cache_start.to_pydatetime())
                if gap_df is not None:
                    df = pd.concat([gap_df, cached]).sort_index()
                    df = df[~df.index.duplicated(keep='last')]
                    df = df[(df.index >= req_start - pd.Timedelta(hours=1)) & (df.index <= req_end)]
                    if not df.empty:
                        return self._to_rate_series(df, timeframe, spec)

        # No cache at all — full fallback
        if spec:
            df = self._fallback_fetch(spec, start_dt, end_dt)
            if df is not None and not df.empty:
                return self._to_rate_series(df, timeframe, spec)

        return None

    def get_usdc_usdt_series(self, timeframe: str, start_dt: datetime, end_dt: datetime) -> pd.Series | None:
        return self.get_rate_series('_USDC_USDT', timeframe, start_dt, end_dt)

    def get_status(self) -> list[dict]:
        """Return cache health info for each pair — used by the debug page."""
        with self._lock:
            specs = list(self._specs)
            data = {k: v.copy() for k, v in self._data.items()}
        rows = []
        for spec in specs:
            df = data.get(spec.quote_ccy)
            latest_rate = None
            if df is not None and not df.empty:
                last_row = df.iloc[-1]
                mid = (last_row['open'] + last_row['close']) / 2
                latest_rate = 1 / mid if spec.invert else mid
            rows.append({
                'quote_ccy': spec.quote_ccy,
                'symbol': f'{spec.symbol} / {spec.denominator_symbol}' if spec.denominator_symbol else spec.symbol,
                'exchange': spec.exchange,
                'invert': spec.invert,
                'latest_rate': latest_rate,
                'status': 'failed' if spec.quote_ccy in self._failed else ('ok' if df is not None and not df.empty else 'empty'),
                'rows': len(df) if df is not None else 0,
                'oldest': df.index.min() if df is not None and not df.empty else None,
                'newest': df.index.max() if df is not None and not df.empty else None,
                'memory_kb': round(df.memory_usage(deep=True).sum() / 1024, 1) if df is not None else 0,
            })
        return rows
