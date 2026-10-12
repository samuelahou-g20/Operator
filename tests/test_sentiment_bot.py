"""Offline tests for the sentiment bot's in-process tweet cache (no network)."""

from __future__ import annotations

import time

from scripts import sentiment_bot as sb


def _key(cap):
    return f"{cap}:{sb.SORT_ORDER}:q"


def test_pull_uses_cache_within_ttl(monkeypatch):
    sb._CACHE.clear()
    sb._CACHE[_key(5)] = (time.time(), [{"user": "a"}])

    def _boom(*a, **k):
        raise AssertionError("should not hit the X API on a cache hit")

    monkeypatch.setattr(sb, "_x_get", _boom)
    assert sb._pull("q", 5, "tok", "start") == [{"user": "a"}]   # served from cache


def test_pull_refetches_after_expiry(monkeypatch):
    sb._CACHE.clear()
    sb._CACHE[_key(5)] = (time.time() - (sb.CACHE_TTL_MIN * 60 + 1), [{"user": "old"}])
    monkeypatch.setattr(sb, "_x_get", lambda *a, **k: {"data": [], "includes": {}, "meta": {}})
    assert sb._pull("q", 5, "tok", "start") == []                # stale -> refetched
    assert sb._CACHE[_key(5)][1] == []                           # cache refreshed


def test_distinct_cap_keys_separately(monkeypatch):
    sb._CACHE.clear()
    sb._CACHE[_key(5)] = (time.time(), [{"user": "small"}])
    monkeypatch.setattr(sb, "_x_get", lambda *a, **k: {"data": [], "includes": {}, "meta": {}})
    assert sb._pull("q", 50, "tok", "start") == []               # different cap -> not the cap-5 entry
