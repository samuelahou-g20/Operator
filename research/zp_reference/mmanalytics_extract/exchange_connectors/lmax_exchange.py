"""Stub connector for LMAX Digital.

LMAX does not provide a standard REST klines/OHLCV API.
Their market data is delivered via CSV files and FIX protocol.
This stub allows LMAX to appear in the config without crashing,
but returns empty data for all kline requests.
"""
import ccxt


class LMAXExchange:
    """Stub CCXT-compatible connector for LMAX Digital."""

    def __init__(self, config=None):
        self.symbols = []
        self._warned = False

    def load_markets(self):
        self.symbols = []
        if not self._warned:
            print('[LMAX] LMAX does not support REST klines API. Market data will be unavailable.')
            self._warned = True

    def fetch_ohlcv(self, symbol, timeframe='1h', since=None, limit=500):
        if not self._warned:
            print(f'[LMAX] Cannot fetch klines for {symbol} — LMAX has no REST OHLCV endpoint.')
            self._warned = True
        return []
