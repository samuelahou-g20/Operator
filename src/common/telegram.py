"""Shared Telegram long-poll plumbing for the ``scripts/*_bot.py`` family.

Centralizes the HTTP calls every bot currently copy-pastes: ``sendMessage``/``sendPhoto``, the
startup backlog drain, one iteration of the ``getUpdates`` offset loop (including the
``RequestException`` -> ``time.sleep(2)`` backoff), and the inline-keyboard trio
(``send_buttons`` / ``answer_callback`` / ``clear_buttons``) the Operator's confirm step uses. Behavior matches the pattern already deployed
across ``scripts/jason_bot.py``, ``scripts/hedge_bot.py``, ``scripts/vol_bot.py``, etc. byte-for-byte.

MUST NOT: own command dispatch, chat/trader allowlists, or desk (mm2/mm3) routing — those stay
inline in each bot's own loop so nothing safety-relevant is hidden in shared code. Plain functions
only, no classes/registries/DI — callers own their own loop and state.
"""

from __future__ import annotations

import fcntl
import os
import re
import time
from typing import IO

import requests

from src.common.logging import get_logger

_log = get_logger("common.telegram")


def single_instance_lock(name: str) -> IO | None:
    """Acquire an exclusive ``/tmp/mm3_{name}.lock`` so a second copy of the bot can't run.
    A duplicate long-poller triggers Telegram 409 conflicts and the bot goes silently deaf to
    commands while its timers keep running. Returns the held handle (keep it referenced for the
    process lifetime; the OS frees it on exit) or ``None`` if another instance holds it."""
    fh = open(f"/tmp/mm3_{name}.lock", "w")  # noqa: SIM115 - must stay open for the process lifetime
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    fh.write(str(os.getpid()))
    fh.flush()
    return fh


def send(api: str, chat_id: str, text: str, *, markdown: bool = False, html: bool = False) -> None:
    """POST sendMessage. With ``markdown`` (legacy Markdown) or ``html`` (Telegram HTML), a
    rejected formatting resends as plain text so nothing is lost (mirrors scripts/trade_bot.py::_send)."""
    data = {"chat_id": chat_id, "text": text}
    if html:
        data["parse_mode"] = "HTML"
    elif markdown:
        data["parse_mode"] = "Markdown"
    try:
        r = requests.post(f"{api}/sendMessage", data=data, timeout=30)
        if (markdown or html) and not r.ok:   # bad formatting -> resend as plain so nothing is lost
            plain = re.sub(r"</?(b|i|code|pre|a)[^>]*>", "", text) if html else text
            requests.post(f"{api}/sendMessage", data={"chat_id": chat_id, "text": plain}, timeout=30)
    except requests.exceptions.RequestException as e:
        _log.warning("telegram send failed: %s", e)


def send_photo(api: str, chat_id: str, png: bytes, *, caption: str | None = None) -> None:
    """POST sendPhoto with a PNG payload (mirrors the ``_send_photo`` helper in every render bot)."""
    data = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption
    try:
        requests.post(f"{api}/sendPhoto", data=data,
                      files={"photo": ("chart.png", png, "image/png")}, timeout=60)
    except requests.exceptions.RequestException as e:
        _log.warning("telegram photo send failed: %s", e)


def send_document(api: str, chat_id: str, data: bytes, filename: str, *,
                  caption: str | None = None, mime: str = "text/csv") -> None:
    """POST sendDocument with an in-memory payload. Sends bytes, never a path, so a caller can
    attach something it just computed without writing a temp file first."""
    form: dict = {"chat_id": chat_id}
    if caption:
        form["caption"] = caption
    try:
        requests.post(f"{api}/sendDocument", data=form,
                      files={"document": (filename, data, mime)}, timeout=60)
    except requests.exceptions.RequestException as e:
        _log.warning("telegram document send failed: %s", e)


def send_buttons(api: str, chat_id: str, text: str,
                 buttons: list[list[tuple[str, str]]]) -> int | None:
    """POST sendMessage with an inline keyboard: ``buttons`` is rows of ``(label, callback_data)``.
    A tap arrives in ``getUpdates`` as a ``callback_query`` carrying the tapper's ``from`` and the
    data. Returns the message_id (so the caller can strip the buttons later) or None."""
    markup = {"inline_keyboard": [[{"text": t, "callback_data": d} for t, d in row] for row in buttons]}
    try:
        r = requests.post(f"{api}/sendMessage", json={"chat_id": chat_id, "text": text,
                                                     "reply_markup": markup}, timeout=30).json()
    except (requests.exceptions.RequestException, ValueError) as e:
        _log.warning("telegram buttons send failed: %s", e)
        return None
    return (r.get("result") or {}).get("message_id") if isinstance(r, dict) else None


def answer_callback(api: str, callback_id: str, text: str | None = None) -> None:
    """POST answerCallbackQuery: stops the client's spinner; ``text`` shows as a short toast."""
    data = {"callback_query_id": callback_id}
    if text:
        data["text"] = text
    try:
        requests.post(f"{api}/answerCallbackQuery", json=data, timeout=30)
    except requests.exceptions.RequestException as e:
        _log.warning("telegram answerCallbackQuery failed: %s", e)


def clear_buttons(api: str, chat_id: str, message_id: int) -> None:
    """POST editMessageReplyMarkup with an empty keyboard: the message stays, the buttons go."""
    try:
        requests.post(f"{api}/editMessageReplyMarkup",
                      json={"chat_id": chat_id, "message_id": message_id,
                            "reply_markup": {"inline_keyboard": []}}, timeout=30)
    except requests.exceptions.RequestException as e:
        _log.warning("telegram editMessageReplyMarkup failed: %s", e)


def drain_backlog(api: str) -> int | None:
    """Startup drain: consume any backlog so a restart doesn't replay old commands. Returns the
    offset to resume from, or ``None`` if there was no backlog (or the drain itself failed)."""
    try:
        drained = requests.get(f"{api}/getUpdates", timeout=30).json().get("result", [])
    except requests.exceptions.RequestException as e:
        _log.warning("startup drain failed: %s", e)
        return None
    if drained:
        return drained[-1]["update_id"] + 1
    return None


def poll_updates(
    api: str, offset: int | None, *, poll_timeout: int = 25
) -> tuple[list[dict], int | None]:
    """One iteration of the getUpdates offset loop. On a ``RequestException`` this sleeps 2s
    (the backoff every bot now has) and returns ``([], offset)`` unchanged; otherwise returns
    ``(updates, new_offset)`` with ``new_offset`` advanced past the last update in the batch."""
    try:
        resp = requests.get(
            f"{api}/getUpdates",
            params={"offset": offset, "timeout": poll_timeout},
            timeout=poll_timeout + 10,
        ).json()
    except requests.exceptions.RequestException as e:
        _log.warning("getUpdates failed: %s", e)
        time.sleep(2)
        return [], offset
    except ValueError as e:                      # non-JSON body (Telegram 5xx/HTML) — don't crash
        _log.warning("getUpdates returned a non-JSON body: %s", e)
        time.sleep(2)
        return [], offset
    if not isinstance(resp, dict) or not resp.get("ok", False):
        # ok:false includes the 409 "terminated by other getUpdates" duplicate-poller conflict —
        # log it so a silently-deaf bot is diagnosable (a running duplicate steals the updates).
        desc = resp.get("description") if isinstance(resp, dict) else type(resp).__name__
        _log.warning("getUpdates not ok: %s", desc)
        time.sleep(2)
        return [], offset
    updates = resp.get("result") or []
    new_offset = offset
    for upd in updates:
        new_offset = upd["update_id"] + 1
    return updates, new_offset
