import calendar
import math
import threading
import ccxt
import pandas as pd
from datetime import datetime


def _norm_tick(mode, price_precision, mid):
    """ccxt price precision -> an actual tick size (quote units). Handles the three precision
    modes; SIGNIFICANT_DIGITS needs a price to resolve, so it falls back to None without one."""
    p = price_precision
    if p is None:
        return None
    if mode == ccxt.TICK_SIZE:
        return float(p)
    if mode == ccxt.DECIMAL_PLACES:
        return 10.0 ** -int(p)
    if mode == ccxt.SIGNIFICANT_DIGITS and mid:
        return 10.0 ** (math.floor(math.log10(mid)) - int(p) + 1)
    return float(p) if 0 < p < 1 else None


CCXT_EXCHANGE_NAMES = {
    'BINANCE': 'binance',
    'BYBIT': 'bybit',
    'GATEIO': 'gate',
    'MEXC': 'mexc',
    'KUCOIN': 'kucoin',
    'OKX': 'okx',
    'BITGET': 'bitget',
    'KRAKEN': 'kraken',
    'COINBASE': 'coinbaseexchange',
    'CRYPTO': 'cryptocom',
    'BITFINEX': 'bitfinex',
    'BITSTAMP': 'bitstamp',
    'HUOBI': 'htx',
    'BITVAVO': 'bitvavo',
    'BINGX': 'bingx',
    'BITMART': 'bitmart',
    'XT': 'xt',
    'BINANCEALPHA': 'binancealpha',
    'HASHKEY': 'hashkey',
    'LCX': 'lcx',
    'LMAX': 'lmax',
}


class MarketDataClient:
    def __init__(self):
        self.ccxt_connectors = {}
        # One reentrant lock per exchange so concurrent callers (e.g. a desk fetching many coins
        # at once) never race a shared ccxt connector — same-exchange calls serialise (which also
        # keeps within that exchange's rate limit), different exchanges run in parallel.
        self._ex_locks = {}
        self._ex_locks_guard = threading.Lock()

    _CUSTOM_CONNECTORS = {'binancealpha', 'lcx', 'lmax'}

    def _lock_for(self, exchange: str) -> threading.RLock:
        with self._ex_locks_guard:
            lock = self._ex_locks.get(exchange)
            if lock is None:
                lock = self._ex_locks[exchange] = threading.RLock()
            return lock

    # Per-exchange cap on candles/request. ccxt exchanges reject a limit above their max;
    # most majors allow 1000, but a few are lower. Callers may request a big limit (to page
    # less); we clamp to what the exchange accepts. Omitted exchanges use the caller's limit.
    _MAX_OHLCV_LIMIT = {'coinbaseexchange': 300, 'okx': 300, 'kraken': 720, 'bitstamp': 1000}

    def _add_connector(self, exchange: str, timeout: int = 60000): #1min timeout
        if exchange == 'binancealpha':
            from exchange_connectors.binance_alpha import BinanceAlpha
            self.ccxt_connectors[exchange] = BinanceAlpha()
        elif exchange == 'lcx':
            from exchange_connectors.lcx_exchange import LCXExchange
            self.ccxt_connectors[exchange] = LCXExchange()
        elif exchange == 'lmax':
            from exchange_connectors.lmax_exchange import LMAXExchange
            self.ccxt_connectors[exchange] = LMAXExchange()
        else:
            self.ccxt_connectors[exchange] = getattr(ccxt, exchange)({'enableRateLimit': True, 'timeout': timeout})

    def _reset_connector(self, exchange: str):
        if exchange in self.ccxt_connectors:
            del self.ccxt_connectors[exchange]
        self._add_connector(exchange)

    def ensure_connector(self, exchange: str):
        """Create the connector if it doesn't exist yet. Safe to call concurrently (per-exchange
        lock); still cheap to call single-threaded up front."""
        with self._lock_for(exchange):
            if exchange not in self.ccxt_connectors:
                self._add_connector(exchange)

    def get_ticks_and_mids(self, exchange: str, symbols) -> dict:
        """{symbol: (tick_size, mid)} for ccxt `symbols` on one exchange — one load_markets + one
        fetch_tickers under the per-exchange lock (tick is a static market property, mid is the last
        price for the bps conversion). Unknown/failed entries come back as (None, None)."""
        out = {s: (None, None) for s in symbols}
        with self._lock_for(exchange):
            if exchange not in self.ccxt_connectors:
                self._add_connector(exchange)
            ex = self.ccxt_connectors[exchange]
            try:
                ex.load_markets()
            except Exception:
                return out
            tickers = {}
            if (getattr(ex, 'has', None) or {}).get('fetchTickers'):
                try:
                    tickers = ex.fetch_tickers()
                except Exception:
                    tickers = {}
            mode = getattr(ex, 'precisionMode', None)
            markets = getattr(ex, 'markets', None) or {}
            has_fetch_ticker = (getattr(ex, 'has', None) or {}).get('fetchTicker')
            for s in symbols:
                m = markets.get(s)
                if not m:
                    continue
                t = tickers.get(s) or {}
                mid = t.get('last') or t.get('close')
                if mid is None and t.get('bid') and t.get('ask'):
                    mid = (t['bid'] + t['ask']) / 2
                # fetch_tickers() is spot-only on some venues (binance), so a swap/perp symbol comes
                # back with no mid — fall back to the single-market fetch_ticker for those.
                if mid is None and has_fetch_ticker:
                    try:
                        t2 = ex.fetch_ticker(s)
                        mid = t2.get('last') or t2.get('close')
                        if mid is None and t2.get('bid') and t2.get('ask'):
                            mid = (t2['bid'] + t2['ask']) / 2
                    except Exception:
                        pass
                tick = _norm_tick(mode, (m.get('precision') or {}).get('price'), mid)
                out[s] = (tick, mid)
        return out

    def get_klines(self, symbol: str, exchange: str, timeframe: str, start_dt: datetime, end_dt: datetime, request_limit=100) -> list:
        # serialise same-exchange access; different exchanges still run in parallel
        with self._lock_for(exchange):
            return self._get_klines_unlocked(symbol, exchange, timeframe, start_dt, end_dt, request_limit)

    def _get_klines_unlocked(self, symbol: str, exchange: str, timeframe: str, start_dt: datetime, end_dt: datetime, request_limit=100) -> list:
        request_limit = min(request_limit, self._MAX_OHLCV_LIMIT.get(exchange, request_limit))
        if exchange not in self.ccxt_connectors:
            self._add_connector(exchange)
        since = int(calendar.timegm(start_dt.timetuple())) * 1000
        end_ms = int(calendar.timegm(end_dt.timetuple())) * 1000
        # Conservative hop for empty windows: <= the page span any exchange scans
        # (page limits are >= 100), so we never step over the start of the data.
        empty_step_ms = int(pd.Timedelta(timeframe).total_seconds() * 1000) * min(request_limit, 100)
        all_klines = []
        while since < end_ms:
            klines = self.ccxt_connectors[exchange].fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=since, limit=request_limit)
            if not klines:
                # No candles in [since, since + span]. Some exchanges (e.g. OKX) return
                # empty for a window before the token was listed rather than the next
                # available candle, so advance past it instead of giving up.
                since += empty_step_ms
                continue
            all_klines += klines
            last_ts = datetime.utcfromtimestamp(klines[-1][0]/1000)
            if last_ts + pd.Timedelta(timeframe) > end_dt:
                break
            since = klines[-1][0] + 1
        while all_klines:
            last_ts = datetime.utcfromtimestamp(all_klines[-1][0]/1000)
            if last_ts >= end_dt:
                all_klines.pop()
            else:
                break
        return all_klines
    
    def get_klines_df(self, symbol: str, exchange: str, timeframe: str, start_dt: datetime, end_dt: datetime, request_limit: int = 100) -> pd.DataFrame:
        with self._lock_for(exchange):   # keep reset-then-retry atomic per exchange (RLock: get_klines re-enters)
            try:
                klines = self.get_klines(symbol, exchange, timeframe, start_dt, end_dt, request_limit)
            except ccxt.BadSymbol:
                # The long-lived connector caches its market list on first use, so a symbol
                # listed after that (e.g. a new token) looks unknown. Reload markets via a
                # fresh connector and retry once; a genuinely missing symbol will raise again.
                self._reset_connector(exchange)
                klines = self.get_klines(symbol, exchange, timeframe, start_dt, end_dt, request_limit)
            except ccxt.NetworkError:
                self._reset_connector(exchange)
                klines = self.get_klines(symbol, exchange, timeframe, start_dt, end_dt, request_limit)
        df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
