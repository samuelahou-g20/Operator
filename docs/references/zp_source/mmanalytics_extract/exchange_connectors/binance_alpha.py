"""CCXT-compatible connector for Binance Alpha tokens."""
import requests
import ccxt


class BinanceAlpha:
    """Minimal CCXT-compatible connector for Binance Alpha market data.

    Implements fetch_ohlcv and load_markets to match the interface
    expected by MarketDataClient.
    """

    BASE_URL = 'https://www.binance.com'
    TOKEN_LIST_URL = f'{BASE_URL}/bapi/defi/v1/public/wallet-direct/buw/wallet/cex/alpha/all/token/list'
    KLINES_URL = f'{BASE_URL}/bapi/defi/v1/public/alpha-trade/klines'

    def __init__(self, config=None):
        self.symbols = []
        self._symbol_map = {}  # symbol_upper -> alpha_id (e.g. 'SAFE' -> '175')
        self._session = requests.Session()
        self._markets_loaded = False

    def load_markets(self):
        """Fetch token list and build symbol mapping."""
        try:
            resp = self._session.get(self.TOKEN_LIST_URL, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get('code') != '000000' or not data.get('data'):
                print(f'[BinanceAlpha] Token list returned code={data.get("code")}')
                return

            self.symbols = []
            self._symbol_map = {}
            for token in data['data']:
                alpha_id = str(token.get('alphaId', ''))  # e.g. 'ALPHA_137'
                symbol = str(token.get('symbol', '')).upper()
                if alpha_id and symbol:
                    self._symbol_map[symbol] = alpha_id
                    self.symbols.append(f'{symbol}/USDT')

            self._markets_loaded = True
            print(f'[BinanceAlpha] Loaded {len(self.symbols)} Alpha tokens')
        except Exception as e:
            raise ccxt.NetworkError(f'[BinanceAlpha] Failed to load markets: {e}')

    def _ensure_markets(self):
        if not self._markets_loaded:
            self.load_markets()

    def _to_alpha_symbol(self, symbol):
        """Convert CCXT-style symbol to Binance Alpha format.
        e.g. 'SAFE/USDT' -> 'ALPHA_175USDT'
        """
        self._ensure_markets()
        parts = symbol.split('/')
        if len(parts) != 2:
            raise ccxt.BadSymbol(f'[BinanceAlpha] Invalid symbol format: {symbol}')
        base, quote = parts
        alpha_id = self._symbol_map.get(base.upper())
        if not alpha_id:
            raise ccxt.BadSymbol(f'[BinanceAlpha] Unknown Alpha token: {base}')
        return f'{alpha_id}{quote}'

    def fetch_ohlcv(self, symbol, timeframe='1m', since=None, limit=500):
        """Fetch OHLCV data from Binance Alpha API.

        Args:
            symbol: CCXT-style symbol (e.g. 'SAFE/USDT')
            timeframe: Candle interval (e.g. '1m', '1h', '1d')
            since: Start time in milliseconds (optional)
            limit: Max candles to return (default 500, max 1500)

        Returns:
            List of [timestamp_ms, open, high, low, close, volume]
        """
        alpha_symbol = self._to_alpha_symbol(symbol)
        params = {
            'symbol': alpha_symbol,
            'interval': timeframe,
            'limit': min(limit, 1500),
        }
        if since is not None:
            params['startTime'] = int(since)

        try:
            resp = self._session.get(self.KLINES_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get('code') != '000000' or not data.get('data'):
                return []

            # Convert to CCXT format: [[ts, open, high, low, close, volume], ...]
            # API returns strings, convert to numeric
            result = []
            for candle in data['data']:
                result.append([
                    int(candle[0]),      # open time (ms)
                    float(candle[1]),    # open
                    float(candle[2]),    # high
                    float(candle[3]),    # low
                    float(candle[4]),    # close
                    float(candle[5]),    # volume
                ])
            return result
        except requests.exceptions.RequestException as e:
            raise ccxt.NetworkError(f'[BinanceAlpha] Klines request failed: {e}')
