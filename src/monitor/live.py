"""Track live price and build live bars per symbol off the Redis trade stream.

Runs a :class:`~src.data.streaming.consumer.Consumer` in a background daemon thread
and, per trade, (a) keeps the last price + event timestamp for the Price/Updated
columns and (b) folds the trade into a :class:`BarBuilder` per (symbol, timeframe)
so features can be computed on the lake tail extended to the live edge (no gap).

MUST NOT: talk to Redis except through ``Consumer``/``streaming`` helpers, or compute
signals — it only relays the last price and the freshly-built bars.
"""

from __future__ import annotations

import statistics
import threading
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.data.streaming import raw_channel
from src.data.streaming.aggregator import BarBuilder
from src.data.streaming.consumer import Consumer
from src.monitor.config import Instrument

_log = get_logger("monitor")
_MAX_CLOSED = 64  # recent closed live bars kept per (symbol, timeframe)
_SIGMA_MIN_SAMPLES = 30  # min window-moves before sigma is trusted (else abnormal score = None)


class LivePrices:
    """Subscribe to every instrument's trade channel; expose the latest tick + live bars."""

    def __init__(self, instruments: list[Instrument], *, tick_history_sec: float = 4200.0) -> None:
        # One channel per distinct (source, symbol); map it back for lookups.
        self._by_channel = {
            raw_channel(i.source, i.symbol, "trades"): (i.source, i.symbol)
            for i in instruments
        }
        # A bar builder + recent-closed buffer per (source, symbol, timeframe).
        self._builders = {
            (i.source, i.symbol, i.timeframe): BarBuilder(i.timeframe) for i in instruments
        }
        self._closed: dict[tuple[str, str, str], deque] = defaultdict(
            lambda: deque(maxlen=_MAX_CLOSED)
        )
        self._seed: dict[tuple[str, str, str], pd.DataFrame] = {}
        self._tfs_by_symbol: dict[tuple[str, str], list[str]] = defaultdict(list)
        for i in instruments:
            self._tfs_by_symbol[(i.source, i.symbol)].append(i.timeframe)
        self._latest: dict[tuple[str, str], tuple[float, pd.Timestamp]] = {}
        # Downsampled (~1/sec) price history per symbol for the abnormal-move score; sized to
        # hold the sigma lookback + the scored window (ticks arrive ~1/sec, so maxlen ~= seconds).
        self._tick_maxlen = int(tick_history_sec)
        self._ticks: dict[tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=self._tick_maxlen)
        )
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="live-prices", daemon=True)
        self._thread.start()

    def latest(self, source: str, symbol: str) -> tuple[float, pd.Timestamp] | None:
        with self._lock:
            return self._latest.get((source, symbol))

    def seed(self, source: str, symbol: str, timeframe: str, bars: pd.DataFrame) -> None:
        """Prime the recent-bars buffer with a ts-indexed OHLCV frame (startup catch-up)."""
        with self._lock:
            self._seed[(source, symbol, timeframe)] = bars

    def recent_bars(self, source: str, symbol: str, timeframe: str) -> pd.DataFrame:
        """Seeded catch-up + closed live bars + the forming bar, ts-indexed OHLCV.

        Live bars win over the seed on any shared timestamp (they are fresher)."""
        cols = ["open", "high", "low", "close", "volume"]
        key = (source, symbol, timeframe)
        with self._lock:
            seed = self._seed.get(key)
            rows = list(self._closed[key])
            forming = self._builders[key].current if key in self._builders else None
        if forming is not None:
            rows = [*rows, forming]
        frames = []
        if seed is not None and not seed.empty:
            frames.append(seed[cols])
        if rows:
            live = pd.DataFrame(rows)
            live["ts"] = pd.to_datetime(live["ts"], unit="ns")
            frames.append(live.set_index("ts")[cols])
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames)
        return df[~df.index.duplicated(keep="last")].sort_index()

    def _on_message(self, channel: str, msg: dict) -> None:
        key = self._by_channel.get(channel)
        if key is None or "price" not in msg or "ts" not in msg:
            return
        price = float(msg["price"])
        if price <= 0:  # synthetic zero tick (p="0"); never fold into a bar
            return
        ts_ns, size = int(msg["ts"]), float(msg.get("size", 0.0))
        ts = pd.Timestamp(ts_ns, unit="ns", tz="UTC").tz_localize(None)
        with self._lock:
            self._latest[key] = (price, ts)
            ticks = self._ticks[key]
            if not ticks or ts_ns - ticks[-1][0] >= 1_000_000_000:   # ~1/sec downsample
                ticks.append((ts_ns, price))
            for tf in self._tfs_by_symbol.get(key, ()):
                closed = self._builders[(*key, tf)].add(ts_ns, price, size)
                if closed is not None:
                    self._closed[(*key, tf)].append(closed)

    def recent_move(
        self, source: str, symbol: str, window_sec: float, smooth_sec: float = 20.0
    ) -> float | None:
        """Net directional move over the last ``window_sec`` (None until the window is covered).

        Both endpoints are the *median* of the ticks in a short ``smooth_sec`` bracket (now, and
        around now - window) rather than single prints, so one wick/outlier tick can't swing the
        move. Signed: positive = up over the window, negative = down.
        """
        with self._lock:
            ticks = list(self._ticks[(source, symbol)])
        if len(ticks) < 2:
            return None
        cur_ts = ticks[-1][0]
        cutoff = cur_ts - int(window_sec * 1_000_000_000)
        if ticks[0][0] > cutoff:            # not enough history to span the window yet
            return None
        b = int(smooth_sec * 1_000_000_000)
        cur = statistics.median([p for t, p in ticks if t >= cur_ts - b])
        bracket = [p for t, p in ticks if cutoff - b <= t <= cutoff + b]
        ref = statistics.median(bracket) if bracket else next(p for t, p in ticks if t >= cutoff)
        return None if ref <= 0 else cur / ref - 1.0

    def move_sigma(
        self, source: str, symbol: str, window_sec: float, lookback_sec: float
    ) -> float | None:
        """Stddev of the last ``lookback_sec`` of ``window_sec``-length moves (None if too few).

        This is the empirical scale of a "normal" window move, replacing a sqrt-time projection
        of hourly vol: `move / sigma` is then how unusual the current move is versus its own
        recent short-horizon distribution. References are matched the same way as `recent_move`
        (first tick at/after t - window), and only anchors with a full window of history count.
        """
        with self._lock:
            ticks = list(self._ticks[(source, symbol)])
        if len(ticks) < _SIGMA_MIN_SAMPLES + 1:
            return None
        ts = np.fromiter((t for t, _ in ticks), dtype=np.int64, count=len(ticks))
        px = np.fromiter((p for _, p in ticks), dtype=np.float64, count=len(ticks))
        win = int(window_sec * 1_000_000_000)
        look = int(lookback_sec * 1_000_000_000)
        anchors = np.nonzero(ts >= ts[-1] - look)[0]
        targets = ts[anchors] - win
        refs = np.searchsorted(ts, targets, side="left")     # first tick at/after t - window
        valid = (targets >= ts[0]) & (refs < anchors)         # full window covered, distinct ref
        anchors, refs = anchors[valid], refs[valid]
        if len(anchors) < _SIGMA_MIN_SAMPLES:
            return None
        moves = px[anchors] / px[refs] - 1.0
        moves = moves[np.isfinite(moves)]
        if len(moves) < _SIGMA_MIN_SAMPLES:
            return None
        sigma = float(np.std(moves))
        return sigma if sigma > 0 else None

    def _run(self) -> None:
        try:
            Consumer(list(self._by_channel)).run(self._on_message)
        except Exception as exc:  # noqa: BLE001 - report, don't kill the app
            self.error = str(exc)
            _log.warning("live price stream stopped: %s", exc)
