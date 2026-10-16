"""Offline tests for src/the_operator/tools/intel.py (no network, no Telegram, no real API keys)."""

from __future__ import annotations

import pytest

from scripts import macro_bot, sentiment_bot
from src.the_operator.contract import Context
from src.the_operator.tools import intel


def _ctx() -> Context:
    return Context(request_id="r1", desk="mm3", chat_id="c1", user_id="u1", user_name="tester")


class _StubClient:
    """Stands in for anthropic.Anthropic() — never touches the network."""


def _no_env(monkeypatch):
    """require_env would raise without a real key; the tools only need it as a load_dotenv trigger."""
    monkeypatch.setattr(intel, "require_env", lambda name: "stub-key")
    monkeypatch.setattr(intel.anthropic, "Anthropic", lambda *a, **kw: _StubClient())


def test_tool_names_are_unique():
    names = [t.name for t in intel.TOOLS]
    assert names == ["macro_report", "sentiment"]
    assert len(set(names)) == len(names)


def test_all_tools_construct_with_valid_schemas():
    for tool in intel.TOOLS:
        assert tool.input_schema["additionalProperties"] is False
        assert set(tool.input_schema["required"]) == set(tool.input_schema["properties"])
        assert tool.desks is None


# --------------------------------------------------------------------------- #
# macro_report
# --------------------------------------------------------------------------- #
def test_macro_report_unknown_focus_is_an_error_text():
    result = intel.MACRO_REPORT.handler(_ctx(), {"focus": "mars"})
    assert result.text.startswith("❌")
    assert result.attachments == []


def test_macro_report_happy_path(monkeypatch):
    _no_env(monkeypatch)
    monkeypatch.setattr(macro_bot, "macro_report", lambda client, focus: f"MACRO — {focus}")
    monkeypatch.setattr(macro_bot, "build_market_sections",
                        lambda client, cfg: [("EQUITIES", []), ("COMMODITIES", [("Gold", "4,210", "+0.6%")])])
    monkeypatch.setattr(macro_bot, "render_table_png", lambda sections: b"MACROPNG")
    monkeypatch.setattr(intel, "load_config", lambda path: {})

    result = intel.MACRO_REPORT.handler(_ctx(), {"focus": "global"})

    assert result.text.startswith("MACRO — global") and "(fresh)" in result.text
    assert "MARKET SNAPSHOT" in result.text and "COMMODITIES: Gold 4,210 (+0.6%)" in result.text
    assert "EQUITIES:" not in result.text                 # an empty section is left out
    assert len(result.attachments) == 1
    att = result.attachments[0]
    assert att.kind == "photo" and att.data == b"MACROPNG"


# --------------------------------------------------------------------------- #
# sentiment
# --------------------------------------------------------------------------- #
def test_sentiment_unknown_scope_is_an_error_text():
    result = intel.SENTIMENT.handler(_ctx(), {"topic": "btc", "scope": "nope"})
    assert result.text.startswith("❌")


def test_sentiment_digest_when_topic_is_null(monkeypatch):
    _no_env(monkeypatch)
    monkeypatch.setattr(sentiment_bot, "fetch_themes", lambda: [("Market", [{"user": "a"}])])
    monkeypatch.setattr(sentiment_bot, "fetch_following", lambda: [])
    monkeypatch.setattr(sentiment_bot, "build_digest",
                        lambda client, themed, followed, stamp: "X SENTIMENT — digest")

    result = intel.SENTIMENT.handler(_ctx(), {"topic": None, "scope": None})

    assert result.text.startswith("X SENTIMENT — digest") and "(fresh)" in result.text
    assert result.attachments == []


def test_sentiment_digest_no_tweets(monkeypatch):
    _no_env(monkeypatch)
    monkeypatch.setattr(sentiment_bot, "fetch_themes", lambda: [("Market", [])])
    monkeypatch.setattr(sentiment_bot, "fetch_following", lambda: [])

    result = intel.SENTIMENT.handler(_ctx(), {"topic": None, "scope": None})

    assert "No recent tweets" in result.text


def test_sentiment_single_topic(monkeypatch):
    _no_env(monkeypatch)
    seen = {}

    def fake_fetch_tweets(topic, scope):
        seen["topic"], seen["scope"] = topic, scope
        return [{"user": "a", "text": "bullish", "likes": 1, "impressions": 1}]

    monkeypatch.setattr(sentiment_bot, "fetch_tweets", fake_fetch_tweets)
    monkeypatch.setattr(sentiment_bot, "score_sentiment",
                        lambda client, topic, tweets, stamp: f"SENTIMENT — {topic}")

    result = intel.SENTIMENT.handler(_ctx(), {"topic": "btc", "scope": "crowd"})

    assert seen == {"topic": "btc", "scope": "crowd"}
    assert result.text.startswith("SENTIMENT — btc")


def test_sentiment_single_topic_no_tweets(monkeypatch):
    _no_env(monkeypatch)
    monkeypatch.setattr(sentiment_bot, "fetch_tweets", lambda topic, scope: [])

    result = intel.SENTIMENT.handler(_ctx(), {"topic": "btc", "scope": "crowd"})

    assert "No recent tweets" in result.text and "btc" in result.text


def test_repeat_call_is_served_from_cache(monkeypatch):
    _no_env(monkeypatch)
    calls = []
    monkeypatch.setattr(sentiment_bot, "fetch_tweets", lambda topic, scope: calls.append(1) or [{"user": "a"}])
    monkeypatch.setattr(sentiment_bot, "score_sentiment", lambda client, topic, tweets, stamp: "S")
    intel.SENTIMENT.handler(_ctx(), {"topic": "eth", "scope": "crowd"})
    intel.SENTIMENT.handler(_ctx(), {"topic": "eth", "scope": "crowd"})
    assert calls == [1]
    intel.SENTIMENT.handler(_ctx(), {"topic": "eth", "scope": "crowd", "refresh": True})
    assert calls == [1, 1]


@pytest.fixture(autouse=True)
def _cold_cache():
    from src.the_operator import cache
    cache.clear()
    yield
    cache.clear()


def test_sentiment_topic_defaults_to_crowd(monkeypatch):
    _no_env(monkeypatch)
    seen = {}
    monkeypatch.setattr(sentiment_bot, "fetch_tweets", lambda topic, scope: seen.update(scope=scope) or [])
    intel.SENTIMENT.handler(_ctx(), {"topic": "near-default", "scope": None})
    assert seen == {"scope": "crowd"}
