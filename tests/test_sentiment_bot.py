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


_HANDLES = [f"h{i}" for i in range(29)]


def _record_pulls(monkeypatch):
    calls = []

    def fake_pull(query, cap, bearer, start, sort=sb.SORT_ORDER):
        calls.append((query, cap))
        return [{"user": "a"}] * cap

    monkeypatch.setattr(sb, "_pull", fake_pull)
    monkeypatch.setattr(sb, "require_env", lambda name: "tok")
    monkeypatch.setattr(sb, "FOLLOWING", _HANDLES)
    return calls


def test_following_reads_every_handle_within_cap(monkeypatch):
    calls = _record_pulls(monkeypatch)
    out = sb.fetch_following()
    assert {h for h in _HANDLES if any(f"from:{h} " in q or f"from:{h})" in q for q, _ in calls)} == set(_HANDLES)
    assert all(len(q) <= 512 for q, _ in calls)
    assert len(out) <= sb.FOLLOWING_CAP


def test_infl_topic_covers_every_handle_and_stops_at_cap(monkeypatch):
    calls = _record_pulls(monkeypatch)
    out = sb.fetch_tweets("btc", "infl")
    assert len(out) == sb.MAX_TWEETS and len(calls) == 1      # first chunk filled the cap -> stop
    calls.clear()
    monkeypatch.setattr(sb, "_pull", lambda q, cap, *a, **k: calls.append(q) or [])
    sb.fetch_tweets("btc", "infl")
    assert all(any(f"from:{h} " in q or f"from:{h})" in q for q in calls) for h in _HANDLES)
    assert all(len(q) <= 512 for q in calls)


def test_crowd_topic_is_one_keyword_query(monkeypatch):
    calls = _record_pulls(monkeypatch)
    sb.fetch_tweets("near", "crowd")
    assert len(calls) == 1 and "from:" not in calls[0][0]
