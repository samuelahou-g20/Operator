"""The move alert: scores only bars that close after the seed, nets out BTC, and keeps quiet 2h."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.the_operator import moves

T0 = pd.Timestamp("2026-10-01")
N = 3 * 288                                    # 3 days of 5m bars: past the 1-day warm-up


def _market(seed: int = 0) -> dict[str, pd.Series]:
    """Closes per symbol: BTC, and every token = BTC (beta 1) + its own noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(T0, periods=N + 48, freq="5min")
    rb = rng.normal(0, 0.002, len(idx))
    out = {"btcusdt": pd.Series(60_000 * np.exp(np.cumsum(rb)), idx)}
    for t in moves.VENUES:
        e = rng.normal(0, 0.004, len(idx))
        out[f"{t.lower()}usdt"] = pd.Series(np.exp(np.cumsum(rb + e)), idx)
    return out


def _jump(px: pd.Series, at: pd.Timestamp, pct: float) -> None:
    px[px.index >= at] *= 1 + pct


class _Feed:
    """``fetch`` over a fixed market, as a venue would answer: up to ``end``, forming bar included."""

    def __init__(self, market: dict[str, pd.Series]) -> None:
        self.market = market
        self.fail: set[str] = set()

    def __call__(self, venue, symbol, start, end):
        if symbol in self.fail:
            raise RuntimeError("venue down")
        s = self.market[symbol]
        return s[(s.index >= start) & (s.index <= end)]


class _Clock:
    def __init__(self, bar: int) -> None:
        self.at(bar)

    def at(self, bar: int) -> None:
        """Just after bar ``bar`` closes (bar ``bar + 1`` is forming)."""
        self.t = (T0 + moves.BAR * (bar + 1) + pd.Timedelta(seconds=20)).timestamp()

    def __call__(self) -> float:
        return self.t


def _watcher(market, bar=N - 1):
    feed, clock, said = _Feed(market), _Clock(bar), []
    w = moves.MoveWatcher(["c1", "c2"], lambda chat, text: said.append((chat, text)),
                          fetch=feed, clock=clock)
    return w, feed, clock, said


def _bar(i: int) -> pd.Timestamp:
    return T0 + moves.BAR * i


def test_a_quiet_market_says_nothing():
    w, _feed, clock, said = _watcher(_market())
    w.round_once()
    for i in range(N, N + 12):
        clock.at(i)
        w.round_once()
    assert said == []


def test_a_big_token_move_posts_once_to_every_chat():
    market = _market()
    _jump(market["qiusdt"], _bar(N), 0.10)
    w, _feed, clock, said = _watcher(market)
    w.round_once()
    clock.at(N)
    w.round_once()
    assert [c for c, _ in said] == ["c1", "c2"]
    assert said[0][1].startswith("⚡ **QI** +10.") and "σ after BTC" in said[0][1]


def test_the_seed_is_never_alerted_on():
    market = _market()
    _jump(market["qiusdt"], _bar(N - 1), 0.10)     # the last seeded bar is a huge move
    w, _feed, clock, said = _watcher(market)
    w.round_once()
    assert said == []


def test_the_forming_bar_is_not_scored():
    market = _market()
    _jump(market["qiusdt"], _bar(N + 1), 0.10)     # still forming when the round runs
    w, _feed, clock, said = _watcher(market)
    w.round_once()
    clock.at(N)
    w.round_once()
    assert said == []
    clock.at(N + 1)
    w.round_once()
    assert len(said) == 2


def test_a_token_stays_quiet_for_two_hours_after_it_alerts():
    market = _market()
    _jump(market["qiusdt"], _bar(N), 0.10)
    _jump(market["qiusdt"], _bar(N + 6), 0.10)     # 30 min later, inside the cooldown
    w, _feed, clock, said = _watcher(market)
    w.round_once()
    for i in range(N, N + 12):
        clock.at(i)
        w.round_once()
    assert len(said) == 2                          # the first jump, to both chats; nothing after


def test_a_btc_move_the_token_follows_is_not_news():
    market = _market()
    for s in market.values():
        _jump(s, _bar(N), 0.10)                    # everything, BTC included, +10%
    w, _feed, clock, said = _watcher(market)
    w.round_once()
    clock.at(N)
    w.round_once()
    assert said == []


def test_one_venue_failing_does_not_stop_the_others():
    market = _market()
    _jump(market["qiusdt"], _bar(N), 0.10)
    w, feed, clock, said = _watcher(market)
    w.round_once()
    feed.fail = {"magmausdt"}
    clock.at(N)
    w.round_once()
    assert len(said) == 2


def test_without_btc_nothing_is_scored():
    market = _market()
    _jump(market["qiusdt"], _bar(N), 0.10)
    w, feed, clock, said = _watcher(market)
    w.round_once()
    feed.fail = {"btcusdt"}
    clock.at(N)
    w.round_once()
    assert said == []
