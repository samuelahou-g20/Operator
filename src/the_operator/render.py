"""Telegram HTML rendering for everything the Operator sends.

Tools and the model write a tiny markdown subset — ``**bold**``, `` `code` `` and fenced
```` ``` ```` blocks — and ``to_html`` turns it into the HTML Telegram renders (bold, tap-to-copy
code, monospace blocks) after escaping everything else, so a stray ``<`` or ``&`` in a tool
result can never break a message. Plain text is a valid input and comes back merely escaped.

MUST NOT: accept raw HTML from tools or the model (it is escaped, by design), or be bypassed by a
send path in the gateway.
"""

from __future__ import annotations

import html
import re

_FENCE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)```", re.S)
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_CODE = re.compile(r"`([^`\n]+)`")


def to_html(text: str) -> str:
    """Escape, then bold / inline code / fenced blocks → Telegram HTML."""
    parts: list[str] = []
    pos = 0
    for m in _FENCE.finditer(text):            # fenced blocks are escaped but not further parsed
        parts.append(_inline(text[pos:m.start()]))
        parts.append(f"<pre>{html.escape(m.group(1).rstrip())}</pre>")
        pos = m.end()
    parts.append(_inline(text[pos:]))
    return "".join(parts)


def _inline(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = _BOLD.sub(r"<b>\1</b>", escaped)
    return _CODE.sub(r"<code>\1</code>", escaped)
