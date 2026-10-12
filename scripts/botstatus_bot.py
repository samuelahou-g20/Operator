"""Telegram bot: ``/botstatus`` reports the calling desk's Pluto trading bots — their run status,
and (per bot) the parameters they are running with — as house-style PNG tables.

- ``/botstatus`` — the desk roster: Bot / Type / Status / Operational / Conn, with each bot's
  ``lastError`` listed underneath in the caption.
- ``/botstatus <bot>`` — that bot's ``Strategy<N>`` tabs, one row each (what it is actually doing).
- ``/botstatus <bot> tabs`` — every config tab and its field count.
- ``/botstatus <bot> <tab>`` — one tab's fields (Field / Value / Type).

Reads two Pluto **Read** endpoints — ``GET /api/BotManager/status`` and
``GET /api/BotManager/{botName}/config`` (auth ``PLUTO_API_KEY`` / ``PLUTO_API_SECRET``; internal
self-signed host, so ``verify=False``). The roster comes from ``config/desks.yaml``
(``<desk>.bots``), never from a name glob or a list hardcoded here: only listed bots are shown, a
listed bot the API does not return is rendered as ``MISSING`` instead of silently disappearing, and
``<bot>`` only ever resolves inside the calling desk's own roster.

Its own BotFather token ``TELEGRAM_BOTSTATUS_BOT_TOKEN`` (one long-poller per token). Desk routing
is explicit by env: ``TELEGRAM_MM3_CHAT_ID`` -> mm3, ``TELEGRAM_MM2_CHAT_ID`` -> mm2,
``TELEGRAM_TREASURY_CHAT_ID`` -> treasury (all disjoint; an unpinned chat is refused, never
defaulted to a desk).

MUST NOT: start, stop, restart or reconfigure a bot. Only the BotManager **Read** endpoints may be
wired here — reading ``/config`` is fine; no ``start``/``stop``/``operations`` or config-**write**
path is implemented, called or imported, and none may be added without the owner. (A future version
may add control behind a trader allowlist like ``trade_bot``'s; until then this module is strictly
read-and-render.) It also does not place orders or touch the data lake.

Run: ``uv run python -m scripts.botstatus_bot``
"""

from __future__ import annotations

import difflib
import io
import os
import re
import threading
import time
from datetime import datetime, timezone
from itertools import combinations
from urllib.parse import quote

import matplotlib
import pandas as pd
import urllib3

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Liberation Sans"
import matplotlib.pyplot as plt  # noqa: E402 - must follow backend selection

from src.common.config import load_config, require_env  # noqa: E402
from src.common.deadline import run_with_deadline  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.common.pluto import BOTMANAGER_URL, get as pluto_get  # noqa: E402
from src.common.telegram import (  # noqa: E402
    drain_backlog,
    poll_updates,
    send,
    send_photo,
    single_instance_lock,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # Pluto internal self-signed

_log = get_logger("botstatus_bot")

# The ONLY two BotManager URLs this module can build, and BOTH are Read endpoints. See the
# docstring: the write side (start/stop/operations, config writes) is deliberately absent, and
# tests/test_botstatus_bot.py asserts no third path template can appear here.
_BOTMANAGER = BOTMANAGER_URL
_STATUS_URL = f"{_BOTMANAGER}/status"
_CONFIG_URL = f"{_BOTMANAGER}/{{}}/config"     # .format(quote(botName, safe=""))
_HTTP_TIMEOUT = 30       # per-request; requests' timeout bounds byte gaps, not total duration
_FETCH_DEADLINE = 60     # hard wall-clock ceiling so a stalled Pluto call still yields a reply
_HEARTBEAT_SECONDS = 300

_GRAY, _DARK = "#6b7280", "#111827"
_GREEN, _RED = "#047857", "#b91c1c"

_COLS = ["Bot", "Type", "Status", "Operational", "Conn"]
_MISSING = "MISSING"     # in the desk roster but not returned by the API
_ERR_CHARS = 200         # per-bot cap: lastError is free text and often a multi-line stack trace
_CAPTION_LIMIT = 1024    # Telegram's sendPhoto caption limit — an over-long caption is rejected

# A config is far too big to render whole (*TREASURY_PROP_JASONDH: 58 tabs / 862 fields), so every
# view is a slice and every cell is truncated — a value or an enum `type` is arbitrarily long
# (longest type seen: 156 chars of comma-separated allowed values).
_MAX_ROWS = 60           # per-table row cap; truncation is ALWAYS announced in the caption
_TWO_COL_ROWS = 30       # above this, fold into two side-by-side column pairs (house rule)
_VAL_CHARS = 44
_TYPE_CHARS = 28
_STRATEGY_RE = re.compile(r"^Strategy(\d+)$")
# The Strategy<N> fields worth a summary row: (field name in the config, column header).
_STRATEGY_COLS = [("Biprod", "Biprod"), ("Active", "Active"),
                  ("Hedge Instrument", "Hedge Instr"), ("Liquidity Provider", "LP"),
                  ("Max Execution", "Max Exec"), ("GateOpen-RestingStateDelta", "Resting Δ"),
                  ("GateOpen-ActivationDeadBand", "Deadband")]

_DESK_LABEL = {"mm3": "MM3 Bots", "mm2": "MM2 Bots", "treasury": "Treasury Bots"}

_render_lock = threading.Lock()  # matplotlib pyplot keeps global figure state; handlers are threads


# --------------------------------------------------------------------------- #
# Roster + fetch
# --------------------------------------------------------------------------- #
def desk_roster(desk: str) -> list[str]:
    """The desk's bot names from ``config/desks.yaml`` (``<desk>.bots``), in file order."""
    bots = load_config("desks.yaml").get(desk, {}).get("bots")
    if not bots:
        raise RuntimeError(f"no `bots` roster for desk {desk} in config/desks.yaml")
    return list(bots)


def roster_glob(roster: list[str]) -> str:
    """A ``name=`` glob covering the whole roster in ONE request: the names' common prefix plus
    ``*``. The API honours only the first ``name`` value, so repeating the param does not work, and
    one request per bot would be N calls for a table that changes together. A roster with no shared
    prefix degrades to ``*`` (fetch all, ~172 bots) — wider than needed but never wrong, since the
    reply is filtered against the exact roster anyway."""
    return os.path.commonprefix(roster) + "*"


def fetch_status(glob: str) -> dict[str, dict]:
    """``GET /api/BotManager/status?name=<glob>`` -> ``{name: item}``.

    ``include_inactive`` is sent explicitly even though it defaults true: a server-side default flip
    would hide exactly the dead bots this table is for."""
    r = pluto_get(_STATUS_URL, headers=_pluto_headers(), timeout=_HTTP_TIMEOUT,
                     params={"name": glob, "include_inactive": "true"})
    r.raise_for_status()
    d = r.json()
    if not isinstance(d, dict) or "items" not in d:
        got = list(d)[:6] if isinstance(d, dict) else type(d).__name__
        raise RuntimeError(f"unexpected BotManager status payload: {got}")
    return {str(it.get("name")): it for it in (d.get("items") or [])}


def _pluto_headers() -> dict[str, str]:
    return {"X-Api-Key": require_env("PLUTO_API_KEY"),
            "X-Api-Secret": require_env("PLUTO_API_SECRET")}


def fetch_config(bot: str) -> tuple[list[dict], bool]:
    """``GET /api/BotManager/{botName}/config`` -> ``(tabs, secrets_redacted)``.

    Read-only. The name is URL-encoded with ``safe=""`` because every bot name starts with a literal
    ``*`` and some contain spaces (``*TREASURYPROP Complex futures``).

    ⚠️ The tabs are nested: the payload is ``{success, botName, secretsRedacted, config}`` and
    ``config`` is a MAPPING (``type_of_bot``/``name``/``class_name``/``tabs``) — the tab list lives
    under ``config["tabs"]``, not at ``config`` itself."""
    r = pluto_get(_CONFIG_URL.format(quote(bot, safe="")), headers=_pluto_headers(), timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    d = r.json()
    cfg = d.get("config") if isinstance(d, dict) else None
    tabs = cfg.get("tabs") if isinstance(cfg, dict) else None
    if not isinstance(tabs, list):
        got = list(cfg)[:6] if isinstance(cfg, dict) else type(cfg).__name__
        raise RuntimeError(f"unexpected BotManager config payload: {got}")
    return tabs, bool(d.get("secretsRedacted"))


# --------------------------------------------------------------------------- #
# Bot / tab resolution
# --------------------------------------------------------------------------- #
def resolve_bot(roster: list[str], query: str) -> tuple[str | None, list[str]]:
    """Resolve a typed ``<bot>`` against THIS DESK'S roster only -> ``(name, candidates)``.

    Case-insensitive substring match on the name with the leading ``*`` stripped, so ``jasondh``
    finds ``*TREASURY_PROP_JASONDH``. Never looks outside the roster: the firm runs ~172 bots and a
    desk chat has no business reporting another desk's.

    An exact (case-folded) match wins outright — otherwise ``*MM2_FAKE_GAMMA_HEDGER`` would be
    permanently unaddressable, since its full name is also a substring of
    ``*MM2_FAKE_GAMMA_HEDGER_OKX``. Anything else ambiguous returns ``(None, candidates)`` for the
    caller to list; guessing between two live trading bots is not this bot's call."""
    q = query.strip().lower()
    if not q:
        return None, []
    matches = [n for n in roster if q in n.lstrip("*").lower()]
    exact = [n for n in matches if n.lstrip("*").lower() == q]
    if exact:
        return exact[0], exact
    if len(matches) == 1:
        return matches[0], matches
    return None, matches


def find_tab(tabs: list[dict], query: str) -> dict | None:
    """A tab by case-insensitive exact name, else a unique case-insensitive substring match."""
    q = query.strip().lower()
    for t in tabs:
        if str(t.get("name", "")).lower() == q:
            return t
    hits = [t for t in tabs if q in str(t.get("name", "")).lower()]
    return hits[0] if len(hits) == 1 else None


def close_tab_names(tabs: list[dict], query: str, n: int = 6) -> list[str]:
    """The few tab names closest to a miss — a bot has 35-58 tabs, so dumping them all as an error
    is unreadable. Substring hits first (an ambiguous match lands here), then fuzzy neighbours."""
    q = query.strip().lower()
    names = [str(t.get("name", "")) for t in tabs]
    hits = [nm for nm in names if q in nm.lower()]
    for nm in difflib.get_close_matches(query, names, n=n, cutoff=0.4):
        if nm not in hits:
            hits.append(nm)
    return hits[:n]


# --------------------------------------------------------------------------- #
# Table + caption
# --------------------------------------------------------------------------- #
def _trunc(text: object, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[:limit - 1] + "…"


def _fields(tab: dict) -> dict[str, str]:
    return {str(f.get("name", "")): str(f.get("value") or "") for f in (tab.get("fields") or [])}


_MONEY_RE = re.compile(r"^(-?[\d.,]+)\$$")


def money(value: object) -> str:
    """A BotManager money value rendered with its unit made explicit: ``25$`` -> ``$25k``.

    Pluto stores these in THOUSANDS — a ``Max Execution`` of ``25$`` is $25,000, not $25 (owner-
    confirmed; the CircuitBreaker tab spells the same convention out as ``Change in PV(in K)``).
    Printing the raw ``25$`` in a money column reads as twenty-five dollars, which is wrong by
    1000x on a sizing parameter. The DIGITS are left exactly as Pluto shows them so a row still
    ties to the Pluto UI line by line; only the unit is spelled out. Anything not matching the
    ``<number>$`` shape is passed through untouched."""
    s = " ".join(str(value or "").split())
    m = _MONEY_RE.match(s)
    if not m:
        return s
    return f"${m.group(1)}k"


def strategy_tabs(tabs: list[dict]) -> list[dict]:
    """The ``Strategy<N>`` tabs sorted NUMERICALLY. The API returns tabs lexicographically, which
    puts Strategy10 between Strategy1 and Strategy2 — verified live on the treasury JasonDH."""
    keyed = [(int(m.group(1)), t) for t in tabs
             if (m := _STRATEGY_RE.match(str(t.get("name", ""))))]
    return [t for _n, t in sorted(keyed, key=lambda kt: kt[0])]


def build_strategy_rows(tabs: list[dict]) -> pd.DataFrame:
    """One row per ``Strategy<N>`` tab: what that leg trades, whether it is on, and its hedge and
    gate-open sizing. Fields the bot type does not define come back blank, never as an error."""
    rows = []
    for t in strategy_tabs(tabs):
        f = _fields(t)
        rows.append([str(t.get("name", ""))]
                    + [_trunc(money(f.get(key, "")), _VAL_CHARS)
                       for key, _hdr in _STRATEGY_COLS])
    return pd.DataFrame(rows, columns=["Strategy", *(hdr for _k, hdr in _STRATEGY_COLS)])


def build_tab_rows(tabs: list[dict]) -> pd.DataFrame:
    """Every tab and its field count — the index for ``/botstatus <bot> <tab>``."""
    return pd.DataFrame([[str(t.get("name", "")), str(len(t.get("fields") or []))] for t in tabs],
                        columns=["Tab", "Fields"])


def build_field_rows(tab: dict) -> pd.DataFrame:
    """One tab's fields. ``type`` is either a scalar type (``Double``, ``Boolean``) or the
    comma-separated list of ALLOWED VALUES for an enum, which runs to 156 chars — truncated, since
    the column is context, not the answer."""
    rows = [[_trunc(f.get("name"), _VAL_CHARS), _trunc(money(f.get("value")), _VAL_CHARS),
             _trunc(f.get("type"), _TYPE_CHARS)] for f in (tab.get("fields") or [])]
    return pd.DataFrame(rows, columns=["Field", "Value", "Type"])


def cap_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Trim to ``_MAX_ROWS`` -> ``(df, n_dropped)``. The caller MUST report a non-zero drop in the
    caption: a silently short table reads as a complete one."""
    if len(df) <= _MAX_ROWS:
        return df, 0
    return df.iloc[:_MAX_ROWS], len(df) - _MAX_ROWS


def two_column(df: pd.DataFrame) -> pd.DataFrame:
    """Fold a tall table into side-by-side column pairs. Telegram downscales the long side of a
    tall image, shrinking the text to unreadable — the house rule in docs/bots.md. The right half's
    headers carry a trailing space so pandas keeps them distinct; it is invisible when rendered."""
    half = (len(df) + 1) // 2
    left = df.iloc[:half].reset_index(drop=True)
    right = df.iloc[half:].reset_index(drop=True).reindex(range(half)).fillna("")
    out = pd.concat([left, right], axis=1)
    out.columns = [*df.columns, *(f"{c} " for c in df.columns)]
    return out


def build_rows(roster: list[str], items: dict[str, dict]) -> pd.DataFrame:
    """One row per rostered bot, roster order. A bot the API did not return still gets a row with
    status ``MISSING`` — dropping it would hide the loudest failure there is."""
    rows = []
    for name in roster:
        it = items.get(name)
        disp = name.lstrip("*")           # leading '*' marks a Firm-owned bot; noise in a table
        if it is None:
            rows.append([disp, "", _MISSING, "not returned by BotManager", ""])
            continue
        rows.append([disp, str(it.get("type") or ""), str(it.get("status") or ""),
                     str(it.get("operationalStatus") or ""), str(it.get("connectionStatus") or "")])
    return pd.DataFrame(rows, columns=_COLS)


def error_lines(roster: list[str], items: dict[str, dict]) -> list[str]:
    """One collapsed, truncated line per bot carrying a non-empty ``lastError``. Errors are free
    text and routinely multi-line, so they are whitespace-collapsed and capped — a raw stack trace
    would blow the caption limit and the photo would arrive with no caption at all."""
    out = []
    for name in roster:
        err = " ".join(str((items.get(name) or {}).get("lastError") or "").split())
        if not err:
            continue
        if len(err) > _ERR_CHARS:
            err = err[:_ERR_CHARS - 1] + "…"
        out.append(f"{name.lstrip('*')}: {err}")
    return out


def build_caption(label: str, df: pd.DataFrame, errors: list[str]) -> str:
    """Summary line + the error lines, hard-capped at Telegram's caption limit."""
    n_active = int((df["Status"] == "ACTIVE").sum())
    n_missing = int((df["Status"] == _MISSING).sum())
    head = f"{label} · {len(df)} bots · {n_active} ACTIVE"
    if n_missing:
        head += f" · {n_missing} MISSING"
    body = "\n".join(errors) if errors else "No errors reported."
    cap = f"{head}\n{body}"
    return cap if len(cap) <= _CAPTION_LIMIT else cap[:_CAPTION_LIMIT - 1] + "…"


def render_table_image(label: str, df: pd.DataFrame, *,
                       flag_col: str | None = "Status", good: str = "ACTIVE") -> bytes:
    """House format-B render: ruled (bottom edges only), gray body, dark bold key column, and
    ``flag_col`` coloured green when it equals ``good`` and red for any other non-empty value
    (``Status`` ACTIVE/NOT_LOADED on the roster table, ``Active`` true/false on the strategy
    table). ``flag_col=None`` renders without a flag column."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cols = list(df.columns)
    nrows = len(df)
    fig, ax = plt.subplots(figsize=(1.9 * len(cols) + 1.0, 0.55 + 0.32 * (nrows + 1)))
    ax.axis("off")
    ax.set_title(f"{label} · {stamp}", fontsize=12, fontweight="bold", loc="left", pad=12)
    tbl = ax.table(cellText=df.values, colLabels=cols, cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(range(len(cols)))
    for cell in tbl.get_celld().values():
        cell.PAD = 0.3
        cell.set_width(cell.get_width() * 1.4)
    tbl.scale(1, 1.34)
    for (r, c), cell in tbl.get_celld().items():
        col = cols[c]
        cell.visible_edges = "B"
        cell.set_edgecolor("#e5e7eb")
        if r == 0:
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
            cell.set_text_props(ha="left")
            cell.set_edgecolor("#111827")
            cell.set_linewidth(1.3)
            continue
        cell.get_text().set_color(_GRAY)
        if c == 0:
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
        elif flag_col is not None and col == flag_col:
            flag = str(df.iloc[r - 1][col]).strip()
            if flag:                    # anything other than `good` is a problem, incl. MISSING
                cell.get_text().set_color(_GREEN if flag.lower() == good.lower() else _RED)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Telegram
# --------------------------------------------------------------------------- #
def botstatus_report(desk: str) -> tuple[pd.DataFrame, bytes, str]:
    """Fetch + render one ``/botstatus`` -> ``(rows df, PNG, caption)``. Raises on failure — the
    Telegram caller (``handle_botstatus``) reports errors over chat; a non-Telegram caller (the
    ``bot_status`` Operator tool) gets the df too, so it can show the roster as text without a
    second Pluto fetch."""
    roster = desk_roster(desk)
    items = run_with_deadline(fetch_status, roster_glob(roster), deadline=_FETCH_DEADLINE)
    label = _DESK_LABEL.get(desk, desk)
    df = build_rows(roster, items)
    with _render_lock:
        png = render_table_image(label, df)
    caption = build_caption(label, df, error_lines(roster, items))
    return df, png, caption


def handle_botstatus(api: str, chat_id: str, desk: str) -> None:
    """Fetch + render + reply for one ``/botstatus``. Runs on its own thread; never raises."""
    try:
        _df, png, caption = botstatus_report(desk)
        send_photo(api, chat_id, png, caption=caption)
    except Exception as exc:  # noqa: BLE001 - report over chat, never crash the bot
        _log.exception("/botstatus failed")
        send(api, chat_id, f"/botstatus failed: {exc}")


def _config_caption(bot: str, head: str, dropped: int, redacted: bool) -> str:
    """Caption for a config view. Truncation is stated outright, and a payload that did NOT redact
    its secrets is called out — never assume redaction happened just because it usually does."""
    parts = [f"{bot.lstrip('*')} · {head}"]
    if dropped:
        parts.append(f"⚠️ {dropped} more row(s) not shown (capped at {_MAX_ROWS}).")
    if not redacted:
        parts.append("⚠️ secretsRedacted=false — this config may contain unredacted secrets.")
    cap = "\n".join(parts)
    return cap if len(cap) <= _CAPTION_LIMIT else cap[:_CAPTION_LIMIT - 1] + "…"


def bot_config_report(desk: str, bot_query: str, tab_query: str) -> tuple[bytes | None, str]:
    """``/botstatus <bot> [tabs|<tab>]`` -> ``(PNG, caption)``, or ``(None, message)`` when the bot
    or tab can't be resolved (unknown/ambiguous name, no Strategy tabs) — the caller sends
    ``message`` as plain text in that case, a photo+caption otherwise.

    Read-only: it fetches ``/config`` and renders a slice of it. Resolution is confined to the
    calling desk's roster, so this can never report a bot the chat isn't entitled to. May raise on
    a real fetch failure; the Telegram caller (``handle_bot_config``) reports that over chat."""
    roster = desk_roster(desk)
    bot, candidates = resolve_bot(roster, bot_query)
    if bot is None:
        if candidates:
            names = ", ".join(n.lstrip("*") for n in candidates)
            return None, f"'{bot_query}' matches {len(candidates)} bots: {names}"
        names = ", ".join(n.lstrip("*") for n in roster)
        return None, f"No bot matching '{bot_query}' on {desk}. Roster: {names}"
    tabs, redacted = run_with_deadline(fetch_config, bot, deadline=_FETCH_DEADLINE)

    if tab_query.lower() == "tabs":
        df, dropped = cap_rows(build_tab_rows(tabs))
        head = f"{len(tabs)} tabs · {sum(len(t.get('fields') or []) for t in tabs)} fields"
        shown = two_column(df) if len(df) > _TWO_COL_ROWS else df
        label, flag = f"{bot.lstrip('*')} · tabs", None
    elif tab_query:
        tab = find_tab(tabs, tab_query)
        if tab is None:
            near = close_tab_names(tabs, tab_query)
            hint = (f" Closest: {', '.join(near)}" if near
                    else f" Try '/botstatus {bot_query} tabs' for the full list.")
            return None, (f"No single tab matching '{tab_query}' on "
                          f"{bot.lstrip('*')} ({len(tabs)} tabs).{hint}")
        df, dropped = cap_rows(build_field_rows(tab))
        head = f"{tab['name']} · {len(tab.get('fields') or [])} fields"
        shown = two_column(df) if len(df) > _TWO_COL_ROWS else df
        label, flag = f"{bot.lstrip('*')} · {tab['name']}", None
    else:
        df, dropped = cap_rows(build_strategy_rows(tabs))
        if df.empty:
            return None, (f"{bot.lstrip('*')} has no Strategy tabs — try "
                          f"'/botstatus {bot_query} tabs' for its {len(tabs)} config tabs.")
        n_on = int(df["Active"].str.lower().eq("true").sum())
        head = f"{len(df)} strategies · {n_on} active"
        shown, label, flag = df, f"{bot.lstrip('*')} · strategies", "Active"

    with _render_lock:
        png = render_table_image(label, shown, flag_col=flag, good="true")
    return png, _config_caption(bot, head, dropped, redacted)


def handle_bot_config(api: str, chat_id: str, desk: str, bot_query: str, tab_query: str) -> None:
    """``/botstatus <bot> [tabs|<tab>]``. Runs on its own thread; never raises."""
    try:
        png, text = bot_config_report(desk, bot_query, tab_query)
        if png is None:
            send(api, chat_id, text)
        else:
            send_photo(api, chat_id, png, caption=text)
    except Exception as exc:  # noqa: BLE001 - report over chat, never crash the bot
        _log.exception("/botstatus config view failed")
        send(api, chat_id, f"/botstatus failed: {exc}")


def main() -> None:
    lock = single_instance_lock("botstatus_bot")
    if lock is None:
        _log.error("another botstatus_bot instance is already running — refusing to start "
                   "(a duplicate poller causes Telegram 409s and silent deafness)")
        return
    token = require_env("TELEGRAM_BOTSTATUS_BOT_TOKEN")

    # Explicit chat -> desk (no defaulting): a chat is a desk ONLY if pinned in its own var.
    def pinned(var: str, required: bool = False) -> set[str]:
        raw = require_env(var) if required else (os.environ.get(var) or "")
        return {c.strip() for c in raw.split(",") if c.strip()}

    pins = [("mm3", pinned("TELEGRAM_MM3_CHAT_ID", required=True)),
            ("mm2", pinned("TELEGRAM_MM2_CHAT_ID")),
            ("treasury", pinned("TELEGRAM_TREASURY_CHAT_ID"))]
    for (na, a), (nb, b) in combinations(pins, 2):        # all pins must be disjoint
        if a & b:
            raise RuntimeError(f"chat(s) {sorted(a & b)} pinned to both {na} and {nb} — fix .env")
    desk_by_chat = {c: name for name, chats in pins for c in chats}
    api = f"https://api.telegram.org/bot{token}"

    offset = drain_backlog(api)  # so a restart doesn't replay old commands
    _log.info("botstatus_bot started; serving /botstatus")
    last_beat = 0.0
    while True:
        updates, offset = poll_updates(api, offset, poll_timeout=25)
        if time.time() - last_beat >= _HEARTBEAT_SECONDS:
            last_beat = time.time()
            _log.info("poll loop alive (offset=%s)", offset)
        try:
            for upd in updates:
                msg = upd.get("message") or {}
                text = (msg.get("text") or "").strip()
                chat_id = str((msg.get("chat") or {}).get("id"))
                parts = text.split()
                if not parts or parts[0].split("@")[0].lower() != "/botstatus":
                    continue
                desk = desk_by_chat.get(chat_id)
                if desk is None:
                    _log.warning("refused /botstatus from chat %s", chat_id)
                    send(api, chat_id,
                         f"Not authorized. This chat id is {chat_id} — pin it to a desk via "
                         "TELEGRAM_MM3_CHAT_ID (MM3), TELEGRAM_MM2_CHAT_ID (MM2) "
                         "or TELEGRAM_TREASURY_CHAT_ID (Treasury).")
                    continue
                _log.info("/botstatus %s (%s) requested by chat %s", parts[1:], desk, chat_id)
                # Background thread: a slow/hung Pluto call must never stall the next
                # poll_updates(), which is how a bot ends up "running" in systemd but deaf.
                if len(parts) == 1:
                    target, args = handle_botstatus, (api, chat_id, desk)
                else:
                    # parts[1] = the bot; everything after it is the tab (joined, since a tab name
                    # could carry a space even though none observed does).
                    target = handle_bot_config
                    args = (api, chat_id, desk, parts[1], " ".join(parts[2:]))
                threading.Thread(target=target, args=args, daemon=True).start()
        except Exception:  # noqa: BLE001 - an uncaught error here would exit main() outright
            _log.exception("botstatus_bot update dispatch failed")


if __name__ == "__main__":
    main()
