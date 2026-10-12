"""Offline tests for src/common/telegram.py (no real network calls)."""

from __future__ import annotations

import requests

from src.common import telegram as tg


class _FakeResponse:
    def __init__(self, json_data=None, ok=True, status_code=200):
        self._json = json_data or {}
        self.ok = ok
        self.status_code = status_code

    def json(self):
        return self._json


# --------------------------------------------------------------------------- #
# drain_backlog
# --------------------------------------------------------------------------- #
def test_drain_backlog_returns_next_offset_from_batch(monkeypatch):
    updates = [{"update_id": 10}, {"update_id": 11}]
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse({"result": updates}))
    assert tg.drain_backlog("api") == 12


def test_drain_backlog_returns_none_on_empty_batch(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse({"result": []}))
    assert tg.drain_backlog("api") is None


def test_drain_backlog_returns_none_on_request_exception(monkeypatch):
    def _boom(*a, **k):
        raise requests.exceptions.RequestException("network down")

    monkeypatch.setattr(requests, "get", _boom)
    assert tg.drain_backlog("api") is None


# --------------------------------------------------------------------------- #
# poll_updates
# --------------------------------------------------------------------------- #
def test_poll_updates_returns_updates_and_advances_offset(monkeypatch):
    updates = [{"update_id": 5}, {"update_id": 6}]
    captured = {}

    def _fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return _FakeResponse({"ok": True, "result": updates})

    monkeypatch.setattr(requests, "get", _fake_get)
    got, new_offset = tg.poll_updates("https://api", 5, poll_timeout=25)
    assert got == updates
    assert new_offset == 7
    assert captured["url"] == "https://api/getUpdates"
    assert captured["params"] == {"offset": 5, "timeout": 25}
    assert captured["timeout"] == 35  # poll_timeout + 10


def test_poll_updates_keeps_offset_unchanged_when_no_updates(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse({"ok": True, "result": []}))
    got, new_offset = tg.poll_updates("api", 5)
    assert got == []
    assert new_offset == 5


def test_poll_updates_sleeps_and_returns_empty_on_request_exception(monkeypatch):
    def _boom(*a, **k):
        raise requests.exceptions.RequestException("timed out")

    slept = []
    monkeypatch.setattr(requests, "get", _boom)
    monkeypatch.setattr(tg.time, "sleep", lambda s: slept.append(s))

    got, new_offset = tg.poll_updates("api", 5)
    assert got == []
    assert new_offset == 5
    assert slept == [2]


# --------------------------------------------------------------------------- #
# send
# --------------------------------------------------------------------------- #
def test_send_plain_posts_sendmessage_without_parse_mode(monkeypatch):
    calls = []

    def _fake_post(url, data=None, timeout=None):
        calls.append((url, data, timeout))
        return _FakeResponse(ok=True)

    monkeypatch.setattr(requests, "post", _fake_post)
    tg.send("https://api", "123", "hello")
    assert len(calls) == 1
    url, data, timeout = calls[0]
    assert url == "https://api/sendMessage"
    assert data == {"chat_id": "123", "text": "hello"}
    assert "parse_mode" not in data
    assert timeout == 30


def test_send_markdown_success_only_posts_once(monkeypatch):
    calls = []

    def _fake_post(url, data=None, timeout=None):
        calls.append(data)
        return _FakeResponse(ok=True)

    monkeypatch.setattr(requests, "post", _fake_post)
    tg.send("api", "123", "*bold*", markdown=True)
    assert len(calls) == 1
    assert calls[0]["parse_mode"] == "Markdown"


def test_send_markdown_failure_resends_as_plain(monkeypatch):
    calls = []

    def _fake_post(url, data=None, timeout=None):
        calls.append(dict(data))
        return _FakeResponse(ok=(len(calls) > 1))  # first call fails, resend succeeds

    monkeypatch.setattr(requests, "post", _fake_post)
    tg.send("api", "123", "bad *markdown", markdown=True)
    assert len(calls) == 2
    assert calls[0]["parse_mode"] == "Markdown"
    assert "parse_mode" not in calls[1]
    assert calls[1] == {"chat_id": "123", "text": "bad *markdown"}


def test_send_swallows_request_exception(monkeypatch):
    def _boom(*a, **k):
        raise requests.exceptions.RequestException("down")

    monkeypatch.setattr(requests, "post", _boom)
    tg.send("api", "123", "hello")  # must not raise


# --------------------------------------------------------------------------- #
# send_photo
# --------------------------------------------------------------------------- #
def test_send_photo_posts_sendphoto_with_png_and_caption(monkeypatch):
    calls = []

    def _fake_post(url, data=None, files=None, timeout=None):
        calls.append((url, data, files, timeout))
        return _FakeResponse(ok=True)

    monkeypatch.setattr(requests, "post", _fake_post)
    tg.send_photo("https://api", "123", b"pngbytes", caption="a caption")
    assert len(calls) == 1
    url, data, files, timeout = calls[0]
    assert url == "https://api/sendPhoto"
    assert data == {"chat_id": "123", "caption": "a caption"}
    assert files["photo"][1] == b"pngbytes"
    assert files["photo"][2] == "image/png"
    assert timeout == 60


def test_send_photo_omits_caption_when_none(monkeypatch):
    calls = []

    def _fake_post(url, data=None, files=None, timeout=None):
        calls.append(data)
        return _FakeResponse(ok=True)

    monkeypatch.setattr(requests, "post", _fake_post)
    tg.send_photo("api", "123", b"pngbytes")
    assert calls[0] == {"chat_id": "123"}


def test_send_photo_swallows_request_exception(monkeypatch):
    def _boom(*a, **k):
        raise requests.exceptions.RequestException("down")

    monkeypatch.setattr(requests, "post", _boom)
    tg.send_photo("api", "123", b"pngbytes")  # must not raise


# --------------------------------------------------------------------------- #
# inline keyboard helpers (the Operator's confirm step)
# --------------------------------------------------------------------------- #
def test_send_buttons_posts_inline_keyboard_and_returns_message_id(monkeypatch):
    seen = {}

    def post(url, **kw):
        seen["url"], seen["json"] = url, kw["json"]
        return _FakeResponse({"ok": True, "result": {"message_id": 42}})
    monkeypatch.setattr(requests, "post", post)
    mid = tg.send_buttons("api", "-1", "Release?", [[("✅ Confirm", "confirm:ab"), ("❌ Abort", "abort:ab")]])
    assert mid == 42 and seen["url"].endswith("/sendMessage")
    assert seen["json"]["reply_markup"] == {"inline_keyboard": [[
        {"text": "✅ Confirm", "callback_data": "confirm:ab"}, {"text": "❌ Abort", "callback_data": "abort:ab"}]]}


def test_send_buttons_returns_none_on_request_exception(monkeypatch):
    def post(*a, **k):
        raise requests.exceptions.ConnectionError("down")
    monkeypatch.setattr(requests, "post", post)
    assert tg.send_buttons("api", "-1", "x", [[("a", "b")]]) is None


def test_clear_buttons_sends_empty_keyboard(monkeypatch):
    seen = {}
    monkeypatch.setattr(requests, "post", lambda url, **kw: seen.update(url=url, json=kw["json"]) or _FakeResponse())
    tg.clear_buttons("api", "-1", 42)
    assert seen["url"].endswith("/editMessageReplyMarkup")
    assert seen["json"] == {"chat_id": "-1", "message_id": 42, "reply_markup": {"inline_keyboard": []}}


def test_send_html_falls_back_to_stripped_plain_text(monkeypatch):
    posts = []

    def post(url, **kw):
        posts.append(kw["data"])
        return _FakeResponse(ok=len(posts) > 1, status_code=400 if len(posts) == 1 else 200)
    monkeypatch.setattr(requests, "post", post)
    tg.send("api", "-1", "<b>hi</b> <code>x</code>", html=True)
    assert posts[0]["parse_mode"] == "HTML" and posts[1] == {"chat_id": "-1", "text": "hi x"}
