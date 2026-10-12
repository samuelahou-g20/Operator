"""Offline tests for src/the_operator/buzz.py and tools/discovery.py.

No network, no Telegram, no real API keys: _pull, requests.get, the anthropic client,
scripts.macro_bot._claude_web, load_config and the mentions state file are all monkeypatched.
"""

from __future__ import annotations

import json

import requests

from scripts import macro_bot
from src.the_operator import buzz, cache
from src.the_operator.contract import Context
from src.the_operator.tools import discovery


def _ctx() -> Context:
    return Context(request_id="r1", desk="mm3", chat_id="c1", user_id="u1", user_name="tester")


class _FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResp:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, texts):
        self._texts = list(texts)

    def create(self, **kwargs):
        return _FakeResp(self._texts.pop(0))


class _FakeClient:
    def __init__(self, texts):
        self.messages = _FakeMessages(texts)


# --------------------------------------------------------------------------- #
# accounts / fetch_x
# --------------------------------------------------------------------------- #
def test_accounts_from_config(monkeypatch):
    monkeypatch.setattr(buzz, "load_config", lambda path: {"x_accounts": ["alice", "bob"]})
    assert buzz.accounts() == ["alice", "bob"]


def test_accounts_empty_when_missing_key(monkeypatch):
    monkeypatch.setattr(buzz, "load_config", lambda path: {})
    assert buzz.accounts() == []


def test_fetch_x_empty_handles_skips_pull(monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("_pull should not be called with no handles")
    monkeypatch.setattr(buzz, "_pull", _boom)
    assert buzz.fetch_x([]) == []


def test_fetch_x_chunks_at_25_handles(monkeypatch):
    monkeypatch.setattr(buzz, "require_env", lambda name: "tok")
    calls = []

    def fake_pull(query, cap, bearer, start, sort="recency"):
        calls.append(query)
        return [{"user": "x", "text": "t", "likes": 0, "impressions": 0, "ts": ""}]

    monkeypatch.setattr(buzz, "_pull", fake_pull)
    handles = [f"h{i}" for i in range(30)]
    out = buzz.fetch_x(handles, hours=24)

    assert len(calls) == 2
    assert calls[0].count("from:") == 25
    assert calls[1].count("from:") == 5
    assert len(out) == 2  # one tweet returned per chunk


def test_fetch_topic_builds_crowd_query_capped_at_100(monkeypatch):
    monkeypatch.setattr(buzz, "require_env", lambda name: "tok")
    captured = {}

    def fake_pull(query, cap, bearer, start):
        captured["query"] = query
        captured["cap"] = cap
        return [{"user": "x", "text": "zcash privacy pump", "likes": 1, "impressions": 0, "ts": ""}]

    monkeypatch.setattr(buzz, "_pull", fake_pull)
    out = buzz.fetch_topic("privacy tokens ZEC", hours=24)

    assert captured["cap"] == 100
    assert "-is:retweet" in captured["query"]
    assert len(out) == 1


# --------------------------------------------------------------------------- #
# fetch_trending
# --------------------------------------------------------------------------- #
def test_fetch_trending_happy_path(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "coins": [{"item": {
                    "symbol": "btc", "name": "Bitcoin", "market_cap_rank": 1,
                    "data": {"price_change_percentage_24h": {"usd": 3.2}},
                }}],
                "categories": [{"name": "AI"}, {"name": "Meme"}],
            }

    monkeypatch.setattr(buzz.requests, "get", lambda *a, **kw: _Resp())
    result = buzz.fetch_trending()

    assert result["coins"] == [{"symbol": "btc", "name": "Bitcoin", "rank": 1, "price_change_24h": 3.2}]
    assert result["categories"] == ["AI", "Meme"]
    assert "error" not in result


def test_fetch_trending_error_never_raises(monkeypatch):
    def _boom(*a, **kw):
        raise requests.exceptions.ConnectionError("no network")

    monkeypatch.setattr(buzz.requests, "get", _boom)
    result = buzz.fetch_trending()

    assert result == {"coins": [], "categories": [], "error": "no network"}


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #
_TWEETS = [{"user": "a", "text": "BTC looking strong", "likes": 5, "impressions": 100, "ts": ""}]
_TRENDING = {"coins": [], "categories": []}
_GOOD_THEME = {
    "theme": "BTC strength", "tokens": ["BTC"], "stance": "bullish", "status": "building",
    "happened": "BTC broke 70k on spot volume", "thesis": "desk expects continuation into the "
    "ETF flow window as CTAs re-lever", "catalysts": [], "laggards": [], "risk": "",
    "evidence": [], "related": [],
}
_GOOD_JSON = json.dumps({"themes": [_GOOD_THEME], "notes": ""})


def test_extract_parses_plain_json():
    client = _FakeClient([_GOOD_JSON])
    result = buzz.extract(client, _TWEETS, _TRENDING)
    assert result["themes"][0]["theme"] == "BTC strength"
    assert result["themes"][0]["status"] == "building"


def test_extract_parses_fenced_json():
    client = _FakeClient([f"```json\n{_GOOD_JSON}\n```"])
    result = buzz.extract(client, _TWEETS, _TRENDING)
    assert result["themes"][0]["theme"] == "BTC strength"


def test_extract_retries_once_on_bad_json_then_succeeds():
    client = _FakeClient(["not json", _GOOD_JSON])
    result = buzz.extract(client, _TWEETS, _TRENDING)
    assert result["themes"][0]["theme"] == "BTC strength"


def test_extract_falls_back_after_second_failure():
    client = _FakeClient(["not json", "still not json"])
    result = buzz.extract(client, _TWEETS, _TRENDING)
    assert result["themes"] == []
    assert result["notes"].startswith("extraction failed:")


def test_extract_includes_topic_directive_in_user_content():
    captured = {}

    class _SpyMessages:
        def create(self, **kwargs):
            captured["content"] = kwargs["messages"][0]["content"]
            return _FakeResp(_GOOD_JSON)

    class _SpyClient:
        messages = _SpyMessages()

    buzz.extract(_SpyClient(), _TWEETS, _TRENDING, topic="privacy tokens ZEC")

    assert "The trader asked specifically about: privacy tokens ZEC" in captured["content"]
    assert "FIRST" in captured["content"]


def test_extract_omits_topic_directive_when_no_topic():
    captured = {}

    class _SpyMessages:
        def create(self, **kwargs):
            captured["content"] = kwargs["messages"][0]["content"]
            return _FakeResp(_GOOD_JSON)

    class _SpyClient:
        messages = _SpyMessages()

    buzz.extract(_SpyClient(), _TWEETS, _TRENDING)

    assert "asked specifically about" not in captured["content"]


# --------------------------------------------------------------------------- #
# rank_themes
# --------------------------------------------------------------------------- #
def test_rank_themes_already_moved_last_early_before_building():
    themes = [
        {"theme": "Moved", "status": "already_moved", "evidence": []},
        {"theme": "Build", "status": "building", "evidence": []},
        {"theme": "Early", "status": "early", "evidence": []},
    ]
    ranked = buzz.rank_themes(themes)
    assert [t["theme"] for t in ranked] == ["Early", "Build", "Moved"]


def test_rank_themes_ties_broken_by_evidence_count_desc():
    themes = [
        {"theme": "Few", "status": "early", "evidence": [{}]},
        {"theme": "Many", "status": "early", "evidence": [{}, {}, {}]},
    ]
    ranked = buzz.rank_themes(themes)
    assert [t["theme"] for t in ranked] == ["Many", "Few"]


def test_rank_themes_missing_status_treated_as_building():
    themes = [
        {"theme": "Unknown", "evidence": []},
        {"theme": "Early", "status": "early", "evidence": []},
        {"theme": "Moved", "status": "already_moved", "evidence": []},
    ]
    ranked = buzz.rank_themes(themes)
    assert [t["theme"] for t in ranked] == ["Early", "Unknown", "Moved"]


# --------------------------------------------------------------------------- #
# research
# --------------------------------------------------------------------------- #
def test_research_empty_themes_returns_empty_without_calling(monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("_claude_web should not be called with no themes")
    monkeypatch.setattr(macro_bot, "_claude_web", _boom)
    assert buzz.research(object(), [], top_n=3) == {}


def test_research_covers_only_top_n_ranked_themes(monkeypatch):
    calls = []

    def fake_claude_web(client, system, user, max_tokens):
        calls.append(user)
        return "catalyst on the 12th (example.com)"

    monkeypatch.setattr(macro_bot, "_claude_web", fake_claude_web)

    themes = [
        {"theme": "Moved", "tokens": ["M"], "status": "already_moved", "evidence": [{}, {}]},
        {"theme": "EarlyBig", "tokens": ["A"], "status": "early", "evidence": [{}, {}]},
        {"theme": "EarlySmall", "tokens": ["B"], "status": "early", "evidence": [{}]},
        {"theme": "Building", "tokens": ["C"], "status": "building", "evidence": [{}]},
    ]
    result = buzz.research(object(), themes, top_n=3)

    assert set(result) == {"EarlyBig", "EarlySmall", "Building"}
    assert "Moved" not in result
    assert len(calls) == 3
    assert all("catalyst on the 12th" in v for v in result.values())
    assert any("Theme: EarlyBig" in c and "Tokens: A" in c for c in calls)


def test_research_exception_becomes_message(monkeypatch):
    def fake_claude_web(client, system, user, max_tokens):
        raise RuntimeError("boom")

    monkeypatch.setattr(macro_bot, "_claude_web", fake_claude_web)
    themes = [{"theme": "A", "tokens": ["A"], "status": "early", "evidence": []}]
    result = buzz.research(object(), themes, top_n=1)
    assert result == {"A": "research unavailable: RuntimeError"}


def test_research_appends_focus_line_when_topic_set(monkeypatch):
    calls = []

    def fake_claude_web(client, system, user, max_tokens):
        calls.append(user)
        return "line (x.com)"

    monkeypatch.setattr(macro_bot, "_claude_web", fake_claude_web)
    themes = [{"theme": "A", "tokens": ["A"], "status": "early", "evidence": []}]
    buzz.research(object(), themes, top_n=1, topic="privacy tokens ZEC")

    assert calls[0].endswith("Focus: privacy tokens ZEC.")


def test_research_no_focus_line_when_no_topic(monkeypatch):
    calls = []

    def fake_claude_web(client, system, user, max_tokens):
        calls.append(user)
        return "line (x.com)"

    monkeypatch.setattr(macro_bot, "_claude_web", fake_claude_web)
    themes = [{"theme": "A", "tokens": ["A"], "status": "early", "evidence": []}]
    buzz.research(object(), themes, top_n=1)

    assert "Focus:" not in calls[0]


# --------------------------------------------------------------------------- #
# tradable
# --------------------------------------------------------------------------- #
def _fake_registries():
    binance = {
        "btcusdt_perp": {"base_asset": "btc", "asset_type": "perps", "status": "trading"},
        "btcusdt_spot": {"base_asset": "btc", "asset_type": "spot", "status": "trading"},
        "fooold_perp": {"base_asset": "foo", "asset_type": "perps", "status": "delisted"},
    }
    bybit = {"btcusdt_perp": {"base_asset": "btc", "asset_type": "perps", "status": "trading"}}
    okx = {}
    hyperliquid = {}
    desks = {
        "mm3": {"strategy": {"BTC": "some-strategy"}},
        "mm2": {"strategy": {}},
    }

    def fake_load_config(path):
        return {
            "universe/binance.yaml": binance,
            "universe/bybit.yaml": bybit,
            "universe/okx.yaml": okx,
            "universe/hyperliquid.yaml": hyperliquid,
            "desks.yaml": desks,
        }[path]

    return fake_load_config


def test_tradable_maps_venues_and_desk(monkeypatch):
    monkeypatch.setattr(buzz, "_registries", None)
    monkeypatch.setattr(buzz, "load_config", _fake_registries())

    result = buzz.tradable(["btc", "FOO", "ZZZ"])

    assert set(result["BTC"]["venues"]) == {"binance perp", "binance spot", "bybit perp"}
    assert result["BTC"]["desk"] == "mm3"
    assert result["FOO"]["venues"] == []  # delisted entry excluded
    assert result["FOO"]["desk"] is None
    assert result["ZZZ"]["venues"] == []
    assert result["ZZZ"]["desk"] is None


# --------------------------------------------------------------------------- #
# velocity
# --------------------------------------------------------------------------- #
def test_velocity_new_and_average(monkeypatch, tmp_path):
    mentions_file = tmp_path / "mentions.json"
    monkeypatch.setattr(buzz, "MENTIONS_FILE", mentions_file)

    themes1 = [{"tokens": ["BTC"], "evidence": [{"handle": "a"}, {"handle": "b"}]}]
    v1 = buzz.velocity(themes1)
    assert v1["BTC"] == {"now": 2, "avg_prev": 0.0, "new": True}

    themes2 = [{"tokens": ["BTC"], "evidence": [{"handle": "a"}]}]
    v2 = buzz.velocity(themes2)
    assert v2["BTC"]["now"] == 1
    assert v2["BTC"]["avg_prev"] == 2.0
    assert v2["BTC"]["new"] is False

    # ETH never appeared before -> new, even on the third run
    themes3 = [{"tokens": ["ETH"], "evidence": [{"handle": "c"}]}]
    v3 = buzz.velocity(themes3)
    assert v3["ETH"] == {"now": 1, "avg_prev": 0.0, "new": True}


def test_velocity_rotates_at_50_runs(monkeypatch, tmp_path):
    mentions_file = tmp_path / "mentions.json"
    seed = [{"ts": f"run{i}", "counts": {"BTC": 1}} for i in range(50)]
    mentions_file.write_text(json.dumps(seed))
    monkeypatch.setattr(buzz, "MENTIONS_FILE", mentions_file)

    buzz.velocity([{"tokens": ["BTC"], "evidence": [{"handle": "a"}]}])

    saved = json.loads(mentions_file.read_text())
    assert len(saved) == 50
    assert saved[0]["ts"] != "run0"  # oldest entry was dropped to make room


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
def test_scan_topic_merges_and_dedupes_with_followed_tweets(monkeypatch):
    monkeypatch.setattr(buzz, "require_env", lambda name: "tok")
    monkeypatch.setattr(buzz, "accounts", lambda: ["alice"])

    def fake_pull(query, cap, bearer, start, sort="recency"):
        if "from:alice" in query:
            return [{"user": "alice", "text": "dup text", "likes": 1, "impressions": 0, "ts": ""}]
        return [
            {"user": "bob", "text": "dup text", "likes": 2, "impressions": 0, "ts": ""},
            {"user": "carol", "text": "unique zec take", "likes": 3, "impressions": 0, "ts": ""},
        ]

    monkeypatch.setattr(buzz, "_pull", fake_pull)
    monkeypatch.setattr(buzz, "fetch_trending", lambda: {"coins": [], "categories": []})
    monkeypatch.setattr(buzz, "tradable", lambda tickers: {})
    monkeypatch.setattr(buzz, "velocity", lambda themes: {})
    monkeypatch.setattr(buzz, "research", lambda client, themes, topic=None: {})

    captured = {}

    def fake_extract(client, tweets, trending, topic=None):
        captured["tweets"] = tweets
        captured["topic"] = topic
        return {"themes": [], "notes": ""}

    monkeypatch.setattr(buzz, "extract", fake_extract)

    result = buzz.scan(object(), hours=24, topic="privacy tokens ZEC")

    texts = [t["text"] for t in captured["tweets"]]
    assert texts.count("dup text") == 1  # deduped on text
    assert "unique zec take" in texts
    assert captured["topic"] == "privacy tokens ZEC"
    assert result["topic"] == "privacy tokens ZEC"
    assert result["n_tweets"] == 2


def test_scan_without_topic_skips_keyword_pull(monkeypatch):
    monkeypatch.setattr(buzz, "require_env", lambda name: "tok")
    monkeypatch.setattr(buzz, "accounts", lambda: ["alice"])
    monkeypatch.setattr(buzz, "_pull", lambda *a, **kw: [
        {"user": "alice", "text": "hi", "likes": 1, "impressions": 0, "ts": ""}
    ])
    monkeypatch.setattr(buzz, "fetch_topic",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no topic call")))
    monkeypatch.setattr(buzz, "fetch_trending", lambda: {"coins": [], "categories": []})
    monkeypatch.setattr(buzz, "extract", lambda client, tweets, trending, topic=None: {"themes": []})
    monkeypatch.setattr(buzz, "tradable", lambda tickers: {})
    monkeypatch.setattr(buzz, "velocity", lambda themes: {})
    monkeypatch.setattr(buzz, "research", lambda client, themes, topic=None: {})

    result = buzz.scan(object(), hours=24)
    assert result["topic"] is None
    assert result["n_tweets"] == 1


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def _theme(**overrides):
    theme = {
        "theme": "BTC strength", "tokens": ["BTC"], "stance": "bullish", "status": "early",
        "happened": "", "thesis": "desk expects continuation as CTAs re-lever",
        "catalysts": [], "laggards": [], "risk": "", "evidence": [], "related": [],
    }
    theme.update(overrides)
    return theme


def _scan_result(themes, **overrides):
    result = {
        "themes": themes,
        "notes": "",
        "trending": {"coins": [{"symbol": "btc"}], "categories": []},
        "tradable": {},
        "velocity": {},
        "research": {},
        "n_tweets": 10,
        "n_accounts": 3,
        "ts": "2026-09-04T12:34:00+00:00",
    }
    result.update(overrides)
    return result


def test_render_header_and_rule():
    text = buzz.render(_scan_result([]))
    lines = text.split("\n")
    assert lines[0].startswith("**\U0001F4E1 BUZZ · 12:34 UTC** · 10 posts from 3 accounts")
    assert lines[1].startswith("**Trending (CoinGecko):**") and lines[2] == "━" * 22
    assert "Narratives, not trades." in text


def test_render_orders_early_before_building_before_already_moved():
    themes = [
        _theme(theme="Moved", status="already_moved", evidence=[{}, {}, {}]),
        _theme(theme="Building", status="building", evidence=[{}]),
        _theme(theme="Early", status="early", evidence=[{}]),
    ]
    text = buzz.render(_scan_result(themes))
    assert text.index("Early") < text.index("Building") < text.index("Moved")


def test_render_status_label_and_header_line_order():
    themes = [_theme(theme="Theme X", tokens=["X"], stance="bullish", status="early")]
    text = buzz.render(_scan_result(themes))
    assert "🟢 **Theme X** · EARLY · X" in text


def test_render_already_moved_label():
    themes = [_theme(theme="Old news", status="already_moved")]
    text = buzz.render(_scan_result(themes))
    assert "ALREADY MOVED" in text


def test_render_happened_line_present_and_omitted():
    with_happened = buzz.render(_scan_result([_theme(happened="ETF approved yesterday")]))
    assert "• **Happened** ETF approved yesterday" in with_happened

    without_happened = buzz.render(_scan_result([_theme(happened="")]))
    assert "happened:" not in without_happened


def test_render_next_line_always_present():
    themes = [_theme(thesis="no forward thesis in the posts")]
    text = buzz.render(_scan_result(themes))
    assert "• **Next** no forward thesis in the posts" in text


def test_render_catalysts_present_and_omitted():
    with_cat = buzz.render(_scan_result([_theme(catalysts=["token unlock Sep 12", "mainnet Q4"])]))
    assert "• **Catalysts** token unlock Sep 12; mainnet Q4" in with_cat

    without_cat = buzz.render(_scan_result([_theme(catalysts=[])]))
    assert "catalysts:" not in without_cat


def test_render_laggards_with_venue_annotation():
    themes = [_theme(laggards=["FOO", "BAR"])]
    trad = {
        "FOO": {"venues": ["binance perp"], "desk": None},
        "BAR": {"venues": [], "desk": None},
    }
    text = buzz.render(_scan_result(themes, tradable=trad))
    assert "• **Laggards** FOO (on binance perp), BAR (not on our venues)" in text


def test_render_laggards_omitted_when_none():
    text = buzz.render(_scan_result([_theme(laggards=[])]))
    assert "laggards:" not in text


def test_render_risk_present_and_omitted():
    with_risk = buzz.render(_scan_result([_theme(risk="regulatory pushback")]))
    assert "• **Risk** regulatory pushback" in with_risk

    without_risk = buzz.render(_scan_result([_theme(risk="")]))
    assert "risk:" not in without_risk


def test_render_web_research_indented_and_omitted():
    themes = [_theme(theme="BTC strength")]
    research = {"BTC strength": "line one (a.com)\nline two (b.com)"}
    text = buzz.render(_scan_result(themes, research=research))
    assert "• **Web** line one (a.com) line two (b.com)" in text

    without_research = buzz.render(_scan_result(themes, research={}))
    assert "web:" not in without_research


def test_render_evidence_capped_at_two():
    themes = [_theme(evidence=[
        {"handle": "a", "quote": "q1", "likes": 1},
        {"handle": "b", "quote": "q2", "likes": 2},
        {"handle": "c", "quote": "q3", "likes": 3},
    ])]
    text = buzz.render(_scan_result(themes))
    assert '“q1” — @a (1)' in text
    assert '“q2” — @b (2)' in text
    assert '“q3” — @c (3)' not in text


def test_render_caps_total_length():
    themes = [_theme(theme=f"Theme {i}", tokens=["X"],
                     thesis="t" * 100, catalysts=["c" * 50],
                     evidence=[{"handle": "h", "quote": "q" * 100, "likes": 1}])
              for i in range(50)]
    text = buzz.render(_scan_result(themes))
    assert len(text) <= buzz._CHAR_CAP


def test_render_includes_notes_and_trending_error():
    result = _scan_result([], notes="thin sample", trending={"coins": [], "categories": [], "error": "boom"})
    text = buzz.render(result)
    assert "**Notes** thin sample" in text
    assert "**Trending unavailable** boom" in text


def test_render_header_shows_focus_when_topic_set():
    text = buzz.render(_scan_result([], topic="privacy tokens ZEC"))
    assert "**Focus:** privacy tokens ZEC · **Trending (CoinGecko):**" in text


def test_render_header_omits_focus_when_no_topic():
    text = buzz.render(_scan_result([]))
    assert "**Focus:**" not in text


# --------------------------------------------------------------------------- #
# tools/discovery.py: buzz tool
# --------------------------------------------------------------------------- #
def test_buzz_tool_rejects_out_of_range_hours():
    result = discovery.BUZZ.handler(_ctx(), {"hours": 0, "refresh": False, "card": True, "topic": None})
    assert result.text.startswith("❌")

    result = discovery.BUZZ.handler(_ctx(), {"hours": 100, "refresh": False, "card": True, "topic": None})
    assert result.text.startswith("❌")


def test_buzz_tool_caches_then_refreshes(monkeypatch):
    cache.clear()
    monkeypatch.setattr(discovery, "require_env", lambda name: "stub-key")
    monkeypatch.setattr(discovery.anthropic, "Anthropic", lambda *a, **kw: object())

    calls = []

    def fake_scan(client, hours, topic):
        calls.append((hours, topic))
        return buzz.__dict__  # any sentinel object is fine

    monkeypatch.setattr(discovery.buzz, "scan", fake_scan)
    monkeypatch.setattr(discovery.buzz, "render", lambda result: "CARD")

    args = {"hours": 11, "refresh": False, "card": True, "topic": None}
    r1 = discovery.BUZZ.handler(_ctx(), args)
    r2 = discovery.BUZZ.handler(_ctx(), args)
    assert len(calls) == 1          # second call was a cache hit
    assert "CARD" in r1.text and "CARD" in r2.text
    assert "fresh" in r1.text

    discovery.BUZZ.handler(_ctx(), {**args, "refresh": True})
    assert len(calls) == 2          # refresh bypassed the cache


def test_buzz_tool_verbatim_reflects_card_flag(monkeypatch):
    cache.clear()
    monkeypatch.setattr(discovery, "require_env", lambda name: "stub-key")
    monkeypatch.setattr(discovery.anthropic, "Anthropic", lambda *a, **kw: object())
    monkeypatch.setattr(discovery.buzz, "scan", lambda client, hours, topic: {})
    monkeypatch.setattr(discovery.buzz, "render", lambda result: "CARD")

    r_card = discovery.BUZZ.handler(_ctx(), {"hours": 11, "refresh": False, "card": True, "topic": None})
    assert r_card.verbatim is True

    r_research = discovery.BUZZ.handler(
        _ctx(), {"hours": 11, "refresh": True, "card": False, "topic": None})
    assert r_research.verbatim is False


def test_buzz_tool_cache_key_includes_topic(monkeypatch):
    cache.clear()
    monkeypatch.setattr(discovery, "require_env", lambda name: "stub-key")
    monkeypatch.setattr(discovery.anthropic, "Anthropic", lambda *a, **kw: object())

    calls = []

    def fake_scan(client, hours, topic):
        calls.append((hours, topic))
        return {}

    monkeypatch.setattr(discovery.buzz, "scan", fake_scan)
    monkeypatch.setattr(discovery.buzz, "render", lambda result: "CARD")

    base = {"hours": 11, "refresh": False, "card": True}
    discovery.BUZZ.handler(_ctx(), {**base, "topic": "ZEC"})
    discovery.BUZZ.handler(_ctx(), {**base, "topic": "ZEC"})   # cache hit, same key
    discovery.BUZZ.handler(_ctx(), {**base, "topic": None})   # different key -> fresh scan

    assert calls == [(11, "ZEC"), (11, None)]
