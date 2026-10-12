"""Rolling per-chat transcript of what traders say to each other (bot privacy mode is off).

The gateway ``record``s every non-command message; the ``recent_chat`` tool reads ``recent`` for
the requesting chat so the model can take the room's context into account WHEN IT NEEDS IT, rather
than having every request carry the transcript (which would also break the cached prompt prefix).
In memory only, ``KEEP_S`` seconds / ``KEEP_N`` messages per chat.

MUST NOT: be written to the journal or disk (it is other people's conversation), or be readable
across chats.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

# Matches the gateway's own history window: the model should not be able to recall
# its own replies from this morning but not what the room said alongside them.
KEEP_S = 8 * 3600
KEEP_N = 400


@dataclass(frozen=True)
class Line:
    ts: float
    user: str
    text: str


_log: dict[str, deque[Line]] = defaultdict(lambda: deque(maxlen=KEEP_N))
_lock = threading.Lock()


def record(chat_id: str, user: str, text: str, ts: float | None = None) -> None:
    with _lock:
        _log[chat_id].append(Line(ts if ts is not None else time.time(), user, text))


def recent(chat_id: str, hours: float = 3.0) -> list[Line]:
    cutoff = time.time() - min(hours * 3600, KEEP_S)
    with _lock:
        return [ln for ln in _log.get(chat_id, ()) if ln.ts >= cutoff]


def render(lines: list[Line]) -> str:
    if not lines:
        return "(nothing said in this chat in the window)"
    return "\n".join(f"{time.strftime('%H:%M', time.gmtime(ln.ts))} {ln.user}: {ln.text}" for ln in lines)
