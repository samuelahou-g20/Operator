"""Telegram alerting for the dashboard (abnormal-move notifications).

Reads ``TELEGRAM_BOT_TOKEN`` + ``TELEGRAM_CHAT_ID`` from the environment (loading repo-root
``.env`` first, per the credential convention). If either is unset the notifier is disabled
(logged once) so the dashboard still runs. Sends are fire-and-forget on a daemon thread so
the UI event loop never blocks on the network.

MUST NOT: hold trading/strategy logic, block the event loop, or raise into the caller.
"""

from __future__ import annotations

import os
import threading
import urllib.parse
import urllib.request

from src.common import config as cfg
from src.common.logging import get_logger

_log = get_logger("monitor")


class TelegramNotifier:
    def __init__(self) -> None:
        cfg.load_dotenv()
        self._token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self._chat = os.environ.get("TELEGRAM_CHAT_ID")
        self.enabled = bool(self._token and self._chat)
        if not self.enabled:
            _log.warning("telegram alerts disabled: set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID")

    def send(self, text: str) -> None:
        """Fire-and-forget a message (no-op if disabled)."""
        if not self.enabled:
            return
        threading.Thread(target=self._post, args=(text,), name="telegram", daemon=True).start()

    def _post(self, text: str) -> None:
        try:
            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            data = urllib.parse.urlencode({"chat_id": self._chat, "text": text}).encode()
            with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10):  # noqa: S310
                pass
        except Exception as exc:  # noqa: BLE001 - alerting must never break the app
            _log.warning("telegram send failed: %s", exc)
