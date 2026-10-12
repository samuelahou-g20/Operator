"""Desk notes: what traders tell the Operator about how their desk works.

They go in the desk's own playbook, ``config/operator/{desk}.md``, which ``desk.desk_facts`` already
reads into every prompt for that desk. That is what the file is for — an earlier version of this
module kept a parallel store instead, and the result was a playbook nobody ever edited.

Notes are appended under a marker line, so the hand-written part above stays the human's and the
appended part is obviously the Operator's. Both halves reach the model as one text; a trader who
disagrees with a note edits or deletes the line, in git, where the change is reviewable. The file
is versioned, so a note written by the bot shows up in the working tree as an uncommitted change —
deliberate: the owner sees what the desk taught it.

MUST NOT: be written from a tool handler (a note is confirmed first, like any write), rewrite or
delete what a human wrote above the marker, or be treated as instructions — it is what somebody
said, and ``desk_facts`` says so.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.common.logging import get_logger
from src.common.paths import CONFIG_DIR

_log = get_logger("operator")

DIR = CONFIG_DIR / "operator"
MARKER = "<!-- operator-notes: appended by /operator remember; edit or delete these freely -->"
PREAMBLE = ("What traders have told me on this desk (their words, not desk policy — the notes "
            "above win if they disagree):")
MAX_CHARS = 400


@dataclass(frozen=True)
class Note:
    ts: float
    author: str
    text: str

    def line(self) -> str:
        when = datetime.fromtimestamp(self.ts, tz=timezone.utc).strftime("%Y-%m-%d")
        return f"- ({when}, {self.author}) {self.text}"


def path(desk: str) -> Path:
    return DIR / f"{desk}.md"


def add(desk: str, author: str, text: str, ts: float | None = None) -> Note:
    """Append one note under the marker, creating the marker (and the file) if absent.

    Raises OSError if it cannot be written — the caller reports that to the chat."""
    note = Note(ts if ts is not None else time.time(), author, " ".join(text.split())[:MAX_CHARS])
    p = path(desk)
    DIR.mkdir(parents=True, exist_ok=True)
    body = p.read_text(encoding="utf-8").rstrip("\n") if p.is_file() else ""
    if MARKER not in body:
        body = f"{body}\n\n{MARKER}\n{PREAMBLE}" if body else f"{MARKER}\n{PREAMBLE}"
    p.write_text(f"{body}\n{note.line()}\n", encoding="utf-8")
    _log.info("desk note recorded on %s by %s: %s", desk, author, note.text[:120])
    return note


def read(desk: str) -> list[Note]:
    """The appended notes, oldest first. Anything above the marker is the human's, not ours."""
    p = path(desk)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return []
    if MARKER not in text:
        return []
    out: list[Note] = []
    for line in text.split(MARKER, 1)[1].splitlines():
        line = line.strip()
        if not line.startswith("- (") or ")" not in line:
            continue
        head, _, body = line[3:].partition(")")
        when, _, author = head.partition(",")
        try:
            stamp = datetime.strptime(when.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        out.append(Note(stamp.timestamp(), author.strip(), body.strip()))
    return out
