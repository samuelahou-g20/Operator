"""Move alert: posts the moment an MM3 token moves far more than it usually does, net of BTC.

Each 5m bar close, every token's move over 5m / 15m / 1h is scored in sigmas: the part BTC does not
explain (EWMA beta), over the token's own recent size of move (EWMA vol). Scaling by each token's
vol is what makes one rule fit a sleepy coin and a wild one; a fixed % either floods on the wild
ones or misses the sleepy ones. Thresholds and the 2h per-token quiet period are the ~6 alerts/day
budget from ``research/move_alert`` (87% of 4h big moves caught out of sample), so the features
here must stay computed exactly as there.

Prices are each venue's public 5m bars — the series the thresholds were tuned on — not the
CoinRoutes book. A start seeds 20 days per token so vol and beta are ready at once; that history
is never alerted on, only bars that close afterwards. This posts immediately; it does not wait
for ``watch.Hourly``, because a move reported at the next :00 is often over.

MUST NOT: place or change anything, read the lake (it fetches live bars), or post outside the
MM3 chats it was given.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.common.logging import get_logger

_log = get_logger("operator")

# Calibrated venue per token: the market with the deepest history for it (research/move_alert).
VENUES = {
    **dict.fromkeys("AIGENSYN DODO GAS HEMI MINA NEO PARTI QI SKL VTHO".split(), "binance"),
    **dict.fromkeys("AURORA AZTEC BILL HNT NYM".split(), "bybit"),
    **dict.fromkeys("BTW CHECK CROSS DNX GENIUS MAGMA NOCK PUBLIC RHEA XCH".split(), "mexc"),
}
BTC_VENUE = "bybit"
HORIZONS = {"5m": 1, "15m": 3, "1h": 12}           # in 5m bars
THRESHOLDS = {"5m": 12.0, "15m": 7.0, "1h": 5.0}   # sigmas; 5m is high: lone 5m jumps are thin prints
VOL_HL = 3 * 288                                   # EWMA half-lives in bars: 3 days of vol,
BETA_HL = 7 * 288                                  # 7 days of beta
BAR = pd.Timedelta(minutes=5)
COOLDOWN = pd.Timedelta(hours=2)
# A 7-day history put live sigmas within +-15% of the calibration's; 20 days, within +-1.5%.
SEED = pd.Timedelta(days=20)
KEEP = pd.Timedelta(days=21)                       # bound the in-memory history


def fetch_closes(venue: str, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """5m spot closes for ``symbol`` on ``venue`` (deferred imports: fetchers are heavy)."""
    from src.common.config import load_config

    config = load_config(f"data/{venue}.yaml")
    if venue == "binance":
        from src.data.fetching.binance.fetcher import fetch_recent_bars as binance_bars

        bars = binance_bars(config, symbol, "5m", start, end, market="spot")
    elif venue == "bybit":
        from src.data.fetching.bybit.fetcher import fetch_recent_bars as bybit_bars

        bars = bybit_bars(config, symbol, "spot", start, end, "5m")
    else:
        from src.data.fetching.mexc.fetcher import fetch_recent_bars as mexc_bars

        bars = mexc_bars(config, symbol, "spot", start, end, "5m")
    return bars["close"].astype(float)


def scores(px: pd.Series, btc: pd.Series) -> pd.DataFrame:
    """Per bar: the BTC-residual move over each horizon in sigmas (``z5m`` ...), plus ``px``.

    Beta and vol use only the past (shift 1): the bar being scored must not shape its own yardstick.
    """
    r = np.log(px).diff()
    rb = np.log(btc).diff().reindex(r.index)
    cov = (r * rb).ewm(halflife=BETA_HL, min_periods=288).mean().shift(1)
    var_b = (rb ** 2).ewm(halflife=BETA_HL, min_periods=288).mean().shift(1)
    e = r - (cov / var_b).clip(-3, 3) * rb
    sig = np.sqrt((e ** 2).ewm(halflife=VOL_HL, min_periods=288).mean().shift(1))
    f = pd.DataFrame({"px": px, "e": e, "sig": sig}).dropna()
    for k, h in HORIZONS.items():
        f[f"z{k}"] = f["e"].rolling(h).sum() / (f["sig"] * np.sqrt(h))
    return f


def verdict(f: pd.DataFrame) -> tuple[str, float, float] | None:
    """``(horizon, z, raw % move)`` when the last bar breaches a threshold, else ``None``.

    Several horizons can breach at once; the one furthest past its own threshold names the move."""
    last = f.iloc[-1]
    ratio = {k: abs(last[f"z{k}"]) / THRESHOLDS[k] for k in HORIZONS}
    k = max(ratio, key=ratio.get)
    if not ratio[k] >= 1:                  # also False on NaN: too little history to score
        return None
    h = HORIZONS[k]
    return k, float(last[f"z{k}"]), float(last["px"] / f["px"].iloc[-1 - h] - 1)


def render(token: str, horizon: str, z: float, pct: float, px: float) -> str:
    return f"⚡ **{token}** {pct:+.1%} in {horizon} ({z:+.1f}σ after BTC) · {px:.4g}"


class MoveWatcher:
    """Keeps each token's recent closes in memory and scores every newly closed bar.

    ``fetch(venue, symbol, start, end) -> closes`` and ``clock`` are injectable for tests."""

    def __init__(self, chat_ids: list[str], say, fetch=fetch_closes, clock=time.time) -> None:
        self.chat_ids = chat_ids
        self.say = say
        self.fetch = fetch
        self.clock = clock
        self.px: dict[str, pd.Series] = {}             # "BTC" and each token -> closes
        self.done: dict[str, pd.Timestamp] = {}        # token -> last bar scored (or seeded)
        self.quiet_until: dict[str, pd.Timestamp] = {}

    def _update(self, name: str, venue: str, now: pd.Timestamp) -> bool:
        """Append the bars closed since the last round; ``False`` when the fetch failed."""
        held = self.px.get(name)
        start = held.index[-1] if held is not None and len(held) else now - SEED
        try:
            new = self.fetch(venue, f"{name.lower()}usdt", start, now)
        except Exception:  # noqa: BLE001 - one venue hiccup must not stop the other tokens
            _log.exception("move watch could not fetch %s on %s", name, venue)
            return False
        new = new[new.index + BAR <= now]               # the forming bar is not a close yet
        merged = new if held is None else pd.concat([held, new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        self.px[name] = merged[merged.index > now - KEEP]
        return True

    def round_once(self) -> None:
        now = pd.Timestamp(self.clock(), unit="s")
        if not self._update("BTC", BTC_VENUE, now):
            return                                       # without BTC nothing can be scored
        btc = self.px["BTC"]
        lines = []
        for token, venue in VENUES.items():
            if not self._update(token, venue, now) or self.px[token].empty:
                continue
            newest = self.px[token].index[-1]
            if token not in self.done:                  # first sight: the seed, never alerted on
                self.done[token] = newest
                continue
            if newest <= self.done[token]:
                continue
            self.done[token] = newest
            if newest < self.quiet_until.get(token, newest):
                continue
            f = scores(self.px[token], btc)
            if f.empty or f.index[-1] != newest:        # BTC lacks this bar: the residual is unknown
                continue
            hit = verdict(f)
            if hit:
                self.quiet_until[token] = newest + COOLDOWN
                lines.append(render(token, *hit, float(f["px"].iloc[-1])))
        if not lines:
            return
        text = "\n".join(lines)
        _log.info("move post to %s: %s", self.chat_ids, text.replace("\n", " | "))
        for chat_id in self.chat_ids:
            try:
                self.say(chat_id, text)
            except Exception:  # noqa: BLE001 - one unreachable chat must not stop the others
                _log.exception("move watch could not post to %s", chat_id)
