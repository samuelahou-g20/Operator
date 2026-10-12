"""TelegramNotifier: disabled (no-op) without creds, enabled with them."""

from src.monitor.notify import TelegramNotifier


def test_disabled_without_creds(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    n = TelegramNotifier()
    assert n.enabled is False
    n.send("must not raise")            # no-op


def test_enabled_with_creds(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    assert TelegramNotifier().enabled is True
