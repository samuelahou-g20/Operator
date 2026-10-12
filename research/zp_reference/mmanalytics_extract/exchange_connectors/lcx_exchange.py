"""CCXT-compatible connector for LCX Exchange."""
import requests
import ccxt


class LCXExchange:
    """Minimal CCXT-compatible connector for LCX Exchange market data.

    LCX pairs are mostly EUR-denominated (e.g. BTC/EUR, NEAR/EUR).
    API docs: https://docs.lcx.com/
    """

    KLINE_URL = 'https://api-kline.lcx.com/v1/market/kline'
    TICKERS_URL = 'https://exchange-api.lcx.com/api/tickers'

    # LCX uses minute-based resolutions, not CCXT-style strings
    _INTERVAL_MAP = {
        '1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30',
        '1h': '60', '2h': '120', '4h': '240', '6h': '360', '12h': '720',
        '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M',
    }
    # Interval durations in seconds for computing `to` parameter
    _INTERVAL_SECONDS = {
        '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
        '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600, '12h': 43200,
        '1d': 86400, '3d': 259200, '1w': 604800, '1M': 2592000,
    }

    def __init__(self, config=None):
        self.symbols = []
        self._session = requests.Session()
        self._markets_loaded = False

    def load_markets(self):
        """Fetch available trading pairs from LCX."""
        try:
            resp = self._session.get(self.TICKERS_URL, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get('status') != 'success' or not data.get('data'):
                print(f'[LCX] Tickers returned status={data.get("status")}')
                return
            self.symbols = sorted(data['data'].keys())
            self._markets_loaded = True
            print(f'[LCX] Loaded {len(self.symbols)} pairs')
        except Exception as e:
            raise ccxt.NetworkError(f'[LCX] Failed to load markets: {e}')

    def _ensure_markets(self):
        if not self._markets_loaded:
            self.load_markets()

    def fetch_ohlcv(self, symbol, timeframe='1h', since=None, limit=500):
        """Fetch OHLCV data from LCX.

        Args:
            symbol: Pair string (e.g. 'NEAR/EUR')
            timeframe: Candle interval (e.g. '1m', '1h', '1d')
            since: Start time in milliseconds (optional)
            limit: Max candles (not directly supported by LCX, uses time range)

        Returns:
            List of [timestamp_ms, open, high, low, close, volume]
        """
        self._ensure_markets()
        resolution = self._INTERVAL_MAP.get(timeframe)
        if not resolution:
            raise ccxt.BadRequest(f'[LCX] Unsupported timeframe: {timeframe}')

        # Build URL manually — LCX rejects URL-encoded slashes in pair names
        # LCX requires both `from` and `to` parameters
        interval_secs = self._INTERVAL_SECONDS.get(timeframe, 3600)
        if since is not None:
            from_ts = int(since / 1000)
        else:
            from_ts = int(__import__('time').time()) - (limit * interval_secs)
        to_ts = from_ts + (limit * interval_secs)
        url = f'{self.KLINE_URL}?pair={symbol}&resolution={resolution}&from={from_ts}&to={to_ts}'

        try:
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get('status') != 'success' or not data.get('data'):
                return []

            result = []
            seen_ts = set()
            for candle in data['data']:
                ts = int(candle['timestamp']) * 1000
                if ts in seen_ts:
                    continue
                seen_ts.add(ts)
                # Skip candles before our requested `since` (LCX may return earlier data)
                if since is not None and ts < since:
                    continue
                result.append([
                    ts,
                    float(candle['open']),
                    float(candle['high']),
                    float(candle['low']),
                    float(candle['close']),
                    float(candle['volume']),
                ])
            return sorted(result, key=lambda x: x[0])
        except requests.exceptions.RequestException as e:
            raise ccxt.NetworkError(f'[LCX] Klines request failed: {e}')
