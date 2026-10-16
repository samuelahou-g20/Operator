"""Telegram bot that serves a macro market view on demand (any time of day, CET).

Long-polls Telegram; on ``/macro`` it posts TWO messages, both built before either is sent
(so they arrive back-to-back with no wait):

  MESSAGE 1 — MACRO REPORT (text): overnight/current market-moving news + today's economic
  calendar, synthesized by Claude (Sonnet 5) with the server-side web_search tool.

  MESSAGE 2 — MARKET TABLE (PNG): a styled table image (green/red change). Hybrid data —
  crypto + metals + WTI pulled live and exact from Binance USDⓈ-M REST; equity-index futures,
  FX, US 2Y/10Y yields, VIX, BVIV, and KOSPI fetched via Claude web search (they aren't in the
  lake). Python assembles + renders the table deterministically.

Env (loaded from repo-root .env):
  TELEGRAM_MACRO_BOT_TOKEN  its OWN BotFather bot (one long-poller per token).
  TELEGRAM_CHAT_ID          authorized chat/group id(s); comma-separated to serve several.
  ANTHROPIC_API_KEY         Claude API key (Sonnet 5 + web_search tool).

Run: ``uv run python -m scripts.macro_bot``

MUST NOT: place/modify orders or touch the data lake — read-only reporting only.
"""

from __future__ import annotations

import html
import io
import json
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import anthropic
import matplotlib
import requests

from src.common.config import load_config, require_env
from src.common.logging import get_logger
from src.common.telegram import drain_backlog, poll_updates
from src.data.fetching.binance.backfill import _get_json

matplotlib.use("Agg")  # headless: render the table to PNG, no display
matplotlib.rcParams["font.family"] = "Liberation Sans"  # clean, professional sans
import matplotlib.pyplot as plt  # noqa: E402 - must follow backend selection

_log = get_logger("macro_bot")

MODEL = "claude-sonnet-5"
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}  # Sonnet 5; no beta header
CET = ZoneInfo("Europe/Zurich")
TG_MSG_LIMIT = 4096

# Table PNG palette
_DARK = (0.10, 0.10, 0.12)
_GREEN = (21 / 255, 128 / 255, 61 / 255)
_RED = (185 / 255, 28 / 255, 28 / 255)
_LIGHT = "#f3f4f6"

# Message 2 layout. Each row is (label, source): source is either ("bin", TICKER) for a
# live Binance quote or ("web", KEY) for a value Claude returns via web search.
SECTIONS = [
    ("EQUITIES", [("S&P 500 (f)", ("web", "SP500")), ("Nasdaq (f)", ("web", "NDX")),
                  ("KOSPI", ("web", "KOSPI"))]),
    ("CRYPTO", [("BTC", ("bin", "BTCUSDT")), ("ETH", ("bin", "ETHUSDT")),
                ("HYPE", ("bin", "HYPEUSDT"))]),
    ("FX", [("DXY", ("web", "DXY")), ("EUR/USD", ("web", "EURUSD")), ("USD/JPY", ("web", "USDJPY")),
            ("AUD/USD", ("web", "AUDUSD"))]),
    ("RATES", [("US 2Y", ("web", "US2Y")), ("US 10Y", ("web", "US10Y"))]),
    ("COMMODITIES", [("Gold", ("bin", "XAUUSDT")), ("Silver", ("bin", "XAGUSDT")),
                     ("WTI", ("bin", "CLUSDT"))]),
    ("VOL", [("VIX", ("web", "VIX")), ("BVIV", ("web", "BVIV"))]),
]
WEB_KEYS = ["SP500", "NDX", "KOSPI", "DXY", "EURUSD", "USDJPY", "AUDUSD", "US2Y", "US10Y", "VIX", "BVIV"]

MACRO_SYSTEM = """You are a markets desk analyst writing a MACRO REPORT for a trading \
desk group chat. It can be produced at any time of day; all times in CET.

Search the web for overnight and current market-moving news and today's economic calendar. \
HARD RECENCY RULE: the user message gives today's reference date. Use ONLY items published within \
the last 48 hours of that date. Reject anything older than 48h — not only prior-YEAR articles but \
also SAME-YEAR articles that are merely days old (last week's news is stale even though the year \
matches). Web search returns a mix of dates and often ranks older pieces highly: check each \
source's publication date/timestamp and discard any beyond 48h; when several conflict, trust the \
most recent. Every price/level you cite must match a *today* (or latest-close) quote — if you \
cannot confirm an item is within the last 48h, drop it. Prefer primary / reputable sources (wires, \
official releases, exchange status pages); label unconfirmed reports as rumors and down-weight them.

Output format (Telegram — bold via **double asterisks**):
- Title line: **MACRO — {day} {d} {mon}**
- Then 3-4 bold-led bullets. Each: a short **bold lead phrase**, then one or two SHORT sentences \
(≤22 words total after the lead phrase) — what happened + why it matters, nothing else. \
**Bold key numbers.** No clause-stacking or subordinate pile-ups; never repeat a figure.
- Then **→ Net:** one-line takeaway.
- Then **Calendar (CET):** today's events, plus an "Ahead:" note for major upcoming events \
(CPI, FOMC, earnings).

Style: compact — clarity from plain wording, not length. No desk shorthand ("futures markets \
now expect", not "priced/modal outcome"; "automatic trading halt", not "sidecar"). Gloss \
unavoidable jargon in 2-3 words inline. Spell out units on first use (barrels/day, not b/d). \
All times CET.

Content priorities: rates/Fed, geopolitics affecting oil, equity drivers (especially chips / \
Korea), crypto-specific flows, upcoming binary events.

Be concise — keep the ENTIRE report under ~1100 characters; a desk reader should absorb it in \
about 20 seconds. Fewer words beats more. Output ONLY the final report: no narration of your \
research or tool use, no "let me..." / "now I'll..." lines, no preamble, no postamble."""

MARKET_SYSTEM = """You fetch current market data via web search and return STRICT JSON only.

Return a JSON object whose keys are exactly: SP500, NDX, KOSPI, DXY, EURUSD, USDJPY, AUDUSD, \
US2Y, US10Y, VIX, BVIV. Each maps to an object {"last": "<string>", "chg": "<string>"}.
- last: current level as a short string — SP500 = S&P 500 front-month future in index points \
(e.g. 6,140); NDX = Nasdaq-100 front-month future in index points (e.g. 22,150); KOSPI index \
points; DXY = US Dollar Index level (e.g. 104.25); EURUSD/AUDUSD as a rate (e.g. 1.0842); USDJPY as a rate (e.g. 156.20); US2Y/US10Y as a \
yield percent (e.g. "4.28%"); VIX as a level; BVIV (Bitcoin implied-vol index) as a level.
- chg: daily change WITH sign — percent for SP500/NDX/KOSPI/DXY/EURUSD/USDJPY/AUDUSD/VIX/BVIV (e.g. \
"+0.4%"); basis points for US2Y/US10Y (e.g. "+3bp").
- If a market is CLOSED right now (e.g. Seoul overnight/holiday for KOSPI, or the weekend), \
report the change over its MOST RECENT completed trading session (prior close -> last close), \
and set last to that last close — NEVER report "0.0%" for a closed market. If you genuinely \
cannot determine a value, return "n/a" (not "0.0%").
Use the current date. Output ONLY the JSON object, nothing else."""


# --------------------------------------------------------------------------- #
# Claude (web search)
# --------------------------------------------------------------------------- #
def _final_text(content: list) -> str:
    """Only the trailing text — the final answer after the last tool call, dropping the
    inter-search narration blocks that precede it."""
    last_tool = -1
    for i, b in enumerate(content):
        if b.type != "text":
            last_tool = i
    return "".join(b.text for b in content[last_tool + 1:] if b.type == "text").strip()


def _claude_web(client: anthropic.Anthropic, system: str, user: str, max_tokens: int) -> str:
    """Run one web-search-enabled Claude turn; resume across pause_turn; return the final text."""
    messages: list[dict] = [{"role": "user", "content": user}]
    for _ in range(6):  # bound server-tool pause_turn resumes
        resp = client.messages.create(
            model=MODEL, max_tokens=max_tokens, system=system,
            tools=[WEB_SEARCH_TOOL], messages=messages,
        )
        if resp.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": resp.content})
            continue
        break
    return _final_text(resp.content)


# Report focus variants for the /macro command: "/macro" (global) or "/macro asia".
FOCUS = {
    "global": ("MACRO", ""),
    "asia": ("MACRO ASIA", " FOCUS — Asia: center the report on the Asian session and Asia-Pacific "
             "markets (Japan, China, Hong Kong, South Korea, India, Taiwan, Australia): their equity "
             "indices (Nikkei, Hang Seng, CSI 300, KOSPI, Sensex, ASX), central banks (BoJ, PBoC, "
             "BoK, RBI, RBA) and FX (USD/JPY, USD/CNH, ...). Lead with what moved the Asian session; "
             "treat US/EU only as context that drives Asia, and keep the calendar Asia-centric. "
             "Include exactly ONE bullet on crypto (BTC/ETH moves + any Asia-relevant crypto flow or "
             "regulation); up to 5 bullets total is fine here."),
}


def macro_report(client: anthropic.Anthropic, focus: str = "global") -> str:
    now = datetime.now(CET)
    cutoff = now - timedelta(hours=48)
    title, focus_note = FOCUS[focus]
    user = (f"Today is {now:%A %-d %B %Y}, {now:%H:%M} CET. Use ONLY news published on or after "
            f"{cutoff:%A %-d %B %Y} (the last 48 hours). Reject anything older than that — including "
            f"same-year articles that are days old (e.g. last week's news), not just prior-year "
            f"ones. Verify each source's date.{focus_note} Title the report exactly "
            f"'{title} — {now:%A %-d %b}'. Write it now.")
    text = _claude_web(client, MACRO_SYSTEM, user, 8000)
    m = re.search(r"\*{0,2}MACRO\b", text)        # drop any preamble before the report title
    return text[m.start():] if m else text


def _web_quotes(client: anthropic.Anthropic) -> dict[str, dict[str, str]]:
    stamp = datetime.now(CET).strftime("%A %-d %b %Y")
    text = _claude_web(client, MARKET_SYSTEM, f"Today is {stamp} (CET). Return the JSON now.", 3000)
    match = re.search(r"\{.*\}", text, re.S)
    data = json.loads(match.group(0)) if match else {}
    out = {}
    for k in WEB_KEYS:
        v = data.get(k) or {}
        out[k] = {"last": str(v.get("last", "n/a")), "chg": str(v.get("chg", "n/a"))}
    return out


# --------------------------------------------------------------------------- #
# Binance quotes (exact, live)
# --------------------------------------------------------------------------- #
def _fmt_price(v: float) -> str:
    """Comma-separated, no scientific notation; decimals scale with magnitude."""
    a = abs(v)
    dec = 0 if a >= 1000 else 2 if a >= 1 else 4 if a >= 0.01 else 6
    return f"{v:,.{dec}f}"


def _binance_quotes(cfg: dict, tickers: list[str]) -> dict[str, dict[str, str]]:
    rest = cfg["markets"]["perps"]["rest_url"]
    out = {}
    for t in tickers:
        try:
            d = _get_json(rest, "/fapi/v1/ticker/24hr", {"symbol": t})
            last, pct = float(d["lastPrice"]), float(d["priceChangePercent"])
            out[t] = {"last": _fmt_price(last), "chg": f"{pct:+.2f}%"}
        except Exception as exc:  # noqa: BLE001 - one bad symbol shouldn't sink the table
            _log.warning("binance quote failed for %s: %s", t, exc)
            out[t] = {"last": "n/a", "chg": "n/a"}
    return out


# --------------------------------------------------------------------------- #
# Box table
# --------------------------------------------------------------------------- #
def build_market_sections(
    client: anthropic.Anthropic, cfg: dict
) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """Merge live Binance quotes + web quotes into [(section, [(label, last, chg), ...]), ...]."""
    bin_tickers = [src[1] for _, rows in SECTIONS for _, src in rows if src[0] == "bin"]
    bq = _binance_quotes(cfg, bin_tickers)
    wq = _web_quotes(client)
    sections = []
    for name, rows in SECTIONS:
        built = []
        for label, (kind, key) in rows:
            q = (bq if kind == "bin" else wq).get(key, {"last": "n/a", "chg": "n/a"})
            built.append((label, q["last"], q["chg"]))
        sections.append((name, built))
    return sections


def _chg_color(chg: str):
    if chg.startswith("-"):
        return _RED
    if chg.startswith("+"):
        return _GREEN
    return _DARK


def render_table_png(sections: list[tuple[str, list[tuple[str, str, str]]]]) -> bytes:
    """Render the market table as a styled PNG (ruled rows, shaded sections, green/red chg)."""
    stamp = datetime.now(CET).strftime("%a %d %b")
    flat = []  # (kind, label, last, chg)
    for name, rows in sections:
        flat.append(("section", name, "", ""))
        flat.extend(("data", lbl, last, chg) for lbl, last, chg in rows)
    n = len(flat)
    fig, ax = plt.subplots(figsize=(4.4, 0.34 * (n + 1) + 0.4))
    ax.axis("off")
    ax.set_title(f"Market  ·  {stamp}", fontsize=13, fontweight="bold", loc="left",
                 pad=10, color=_DARK)
    tbl = ax.table(cellText=[[lbl, last, chg] for _, lbl, last, chg in flat],
                   colLabels=["", "Last", "Chg%"], cellLoc="right", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.auto_set_column_width(range(3))
    for cell in tbl.get_celld().values():
        cell.PAD = 0.25
        cell.set_width(cell.get_width() * 1.5)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.visible_edges = "B"
        cell.set_edgecolor("#e5e7eb")
        if r == 0:
            cell.set_facecolor("white")
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
            cell.set_text_props(ha="center" if c else "left")
            cell.set_edgecolor("#111827")
            cell.set_linewidth(1.2)
            continue
        kind, lbl, last, chg = flat[r - 1]
        if kind == "section":
            cell.set_facecolor(_LIGHT)
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
            if c == 0:
                cell.set_text_props(ha="left")
            continue
        if c == 0:
            cell.set_text_props(ha="left")
            cell.get_text().set_color(_DARK)
        elif c == 1:
            cell.get_text().set_color(_DARK)
        else:
            cell.get_text().set_color(_chg_color(chg))
            cell.get_text().set_fontweight("bold")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Telegram
# --------------------------------------------------------------------------- #
def _md_to_html(text: str) -> str:
    """Telegram-HTML: escape, then **bold** -> <b>bold</b>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(text), flags=re.S)


def _send(api: str, chat_id: str, text: str, parse_mode: str | None = None) -> None:
    payload = {"chat_id": chat_id, "text": text[:TG_MSG_LIMIT]}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        r = requests.post(f"{api}/sendMessage", data=payload, timeout=30)
        if r.status_code != 200 and parse_mode:  # formatting rejected -> resend as plain text
            requests.post(f"{api}/sendMessage",
                          data={"chat_id": chat_id, "text": text[:TG_MSG_LIMIT]}, timeout=30)
    except requests.exceptions.RequestException as e:
        _log.warning("telegram send failed: %s", e)


def _send_photo(api: str, chat_id: str, png: bytes) -> None:
    try:
        requests.post(f"{api}/sendPhoto", data={"chat_id": chat_id},
                      files={"photo": ("macro.png", png, "image/png")}, timeout=30)
    except requests.exceptions.RequestException as e:
        _log.warning("telegram sendPhoto failed: %s", e)


def main() -> None:
    token = require_env("TELEGRAM_MACRO_BOT_TOKEN")
    authorized = {c.strip() for c in require_env("TELEGRAM_CHAT_ID").split(",") if c.strip()}
    client = anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))
    cfg = load_config("data/binance.yaml")
    api = f"https://api.telegram.org/bot{token}"

    offset = drain_backlog(api)  # so a restart doesn't replay old commands
    _log.info("macro_bot started; serving /macro to chats %s", sorted(authorized))
    while True:
        updates, offset = poll_updates(api, offset)
        for upd in updates:
            msg = upd.get("message") or {}
            text = (msg.get("text") or "").strip()
            chat_id = str((msg.get("chat") or {}).get("id"))
            parts = text.split()
            cmd = parts[0].split("@")[0].lower() if parts else ""
            if cmd != "/macro":
                continue
            if chat_id not in authorized:
                _send(api, chat_id, "Not authorized.")
                _log.warning("rejected /macro from unauthorized chat %s", chat_id)
                continue
            focus = parts[1].lower() if len(parts) > 1 else "global"  # "/macro asia" etc.
            if focus not in FOCUS:
                _send(api, chat_id, f"Unknown focus '{focus}'. Options: {', '.join(FOCUS)}")
                continue
            _send(api, chat_id, f"Building macro view ({focus})…")
            try:
                report = macro_report(client, focus)          # build BOTH first, then send
                png = render_table_png(build_market_sections(client, cfg))  # back-to-back so
                _send(api, chat_id, _md_to_html(report), "HTML")           # they arrive together
                _send_photo(api, chat_id, png)
            except Exception as exc:  # noqa: BLE001 - report over a chat, never crash the bot
                _log.exception("macro report failed")
                _send(api, chat_id, f"Macro report failed: {exc}")


if __name__ == "__main__":
    main()
