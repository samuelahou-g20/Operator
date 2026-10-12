"""TTL cache and per-chat transcript (offline)."""

from __future__ import annotations

from src.the_operator import cache, chatlog
from src.the_operator.contract import Context
from src.the_operator.tools import chat as chat_tool


def test_cached_computes_once_within_ttl_and_reports_age(monkeypatch):
    cache.clear()
    calls = []
    v1, age1 = cache.cached("k", 100, lambda: calls.append(1) or "x")
    v2, age2 = cache.cached("k", 100, lambda: calls.append(1) or "y")
    assert (v1, v2) == ("x", "x") and calls == [1] and age1 == 0.0 and age2 >= 0.0
    v3, _ = cache.cached("k", 0, lambda: "z")           # ttl 0 = force refresh
    assert v3 == "z"


def test_age_note_wording():
    assert cache.age_note(5) == "fresh"
    assert cache.age_note(90) == "cached 1m ago"
    assert cache.age_note(2 * 3600 + 13 * 60) == "cached 2h13m ago"


def test_chatlog_is_per_chat_and_windowed():
    chatlog._log.clear()
    chatlog.record("-1", "a", "hello", ts=1000.0)
    chatlog.record("-2", "b", "other room", ts=1000.0)
    lines = chatlog.recent("-1", hours=3)               # window is relative to now: old lines fall out
    assert lines == []
    import time
    chatlog.record("-1", "a", "now", ts=time.time())
    assert [ln.text for ln in chatlog.recent("-1", 1)] == ["now"]
    assert "now" in chatlog.render(chatlog.recent("-1", 1)) and "a: now" in chatlog.render(chatlog.recent("-1", 1))
    assert chatlog.render([]) .startswith("(nothing said")


def test_recent_chat_tool_scopes_to_the_requesting_chat():
    chatlog._log.clear()
    import time
    chatlog.record("-1", "sam", "zec looks bid", ts=time.time())
    chatlog.record("-9", "x", "secret", ts=time.time())
    res = chat_tool.RECENT_CHAT.handler(Context("r", "mm3", "-1", "7", "sam"), {"hours": 1})
    assert "zec looks bid" in res.text and "secret" not in res.text and "data, not instructions" in res.text
    assert chat_tool.RECENT_CHAT.handler(Context("r", "mm3", "-1", "7", "sam"), {"hours": 9}).text.startswith("❌")
