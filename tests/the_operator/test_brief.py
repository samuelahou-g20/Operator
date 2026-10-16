"""Offline tests for the morning brief: slot timing, non-trader context, targets, /operator brief."""

from __future__ import annotations

from datetime import datetime, timezone

from src.the_operator import brief, gateway


def _utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


def test_next_run_is_the_next_slot_and_a_late_start_skips_to_tomorrow():
    assert brief.next_run(_utc(2026, 10, 17, 6, 59)) == _utc(2026, 10, 17, 7, 0)
    assert brief.next_run(_utc(2026, 10, 17, 7, 0)) == _utc(2026, 10, 18, 7, 0)
    assert brief.next_run(_utc(2026, 10, 17, 7, 10)) == _utc(2026, 10, 18, 7, 0)


def test_post_runs_as_non_trader_per_chat_and_survives_a_failure():
    asked = []

    def ask(ctx, text):
        asked.append((ctx.desk, ctx.chat_id, ctx.is_trader, ctx.user_id, text))
        if ctx.chat_id == "-1":
            raise RuntimeError("boom")

    brief.post(ask, [("mm3", "-1"), ("mm3", "-2")])
    assert [(d, c, t, u) for d, c, t, u, _ in asked] == [("mm3", "-1", False, "morning-brief"),
                                                          ("mm3", "-2", False, "morning-brief")]
    assert all(text == brief.PROMPT for *_, text in asked)


def test_start_only_targets_served_pinned_chats(monkeypatch):
    monkeypatch.setattr(brief, "chats", lambda: ["-1", "-9"])
    started = []
    monkeypatch.setattr(brief.threading, "Thread",
                        lambda target, daemon, name: type("T", (), {"start": lambda self: started.append(name)})())
    assert brief.start(lambda c, t: None, {"-1": "mm3"}, {"-1"}) is not None and started == ["operator-brief"]
    monkeypatch.setattr(brief, "chats", lambda: [])
    assert brief.start(lambda c, t: None, {"-1": "mm3"}, {"-1"}) is None


def test_prompt_is_information_only():
    assert "NO trade ideas" in brief.PROMPT and "do not call desk_view" in brief.PROMPT
    for section in ("Macro & rates", "Geopolitics", "Tech & AI", "Crypto", "Today (UTC)"):
        assert section in brief.PROMPT


def test_prebuild_runs_every_input_and_survives_a_failure(monkeypatch):
    from src.the_operator.tools import discovery, intel, market
    ran = []
    monkeypatch.setattr(intel, "macro_cached", lambda focus: ran.append(("macro", focus)))
    monkeypatch.setattr(intel, "sentiment_digest_cached", lambda: ran.append("sentiment"))
    monkeypatch.setattr(market, "positioning_cached", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(market, "prediction_markets_cached", lambda: ran.append("pm"))
    monkeypatch.setattr(discovery, "buzz_cached", lambda hours, topic: ran.append(("buzz", hours, topic)))
    brief.prebuild()
    assert sorted(map(str, ran)) == sorted(map(str, [("macro", "global"), "sentiment", "pm", ("buzz", 24, None)]))


def test_operator_brief_command_runs_the_prompt_in_that_chat_as_non_trader(tmp_path, monkeypatch):
    from tests.the_operator.test_gateway import _gw, _upd
    sent = []
    gw = _gw(tmp_path, sent)
    asked = []
    monkeypatch.setattr(gw, "ask", lambda ctx, text: asked.append((ctx, text)))
    monkeypatch.setattr(brief, "prebuild", lambda: asked.append("prebuilt"))
    monkeypatch.setattr(gateway.threading, "Thread",
                        lambda target, args, daemon, name: type("T", (), {"start": lambda self: target(*args)})())
    assert gateway.parse("/operator brief") == ("brief", "")
    gw.handle_update(_upd("-1", "/operator brief", user="7"))   # user 7 is a trader on mm3
    prebuilt, (ctx, text) = asked                               # prebuild first, then the model
    assert prebuilt == "prebuilt"
    assert (ctx.desk, ctx.chat_id, ctx.user_id, ctx.is_trader) == ("mm3", "-1", "7", False)
    assert text == brief.PROMPT
    assert any("Building the morning brief" in t for _, t in sent)
