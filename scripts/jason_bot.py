"""Telegram bot that serves the JASON levels table on demand.

Long-polls Telegram; on ``/jason`` it fetches the JASON feature table from the Pluto
``jason_levels`` API and replies with a styled PNG. An optional universe keyword picks
which ccypair basket to fetch: ``/jason`` (default), ``/jason majors``, ``/jason mm3``,
``/jason rwa``. Mirrors ``scripts/hedge_bot.py`` / ``scripts/ls_bot.py``.

Env (loaded from repo-root .env):
  TELEGRAM_JASON_BOT_TOKEN  its OWN BotFather bot — Telegram allows only one long-poller
                            per token, so it must differ from the pnl/ls bot tokens.
  TELEGRAM_CHAT_ID          authorized chat/group id(s); comma-separated to serve several.
  PLUTO_API_KEY / PLUTO_API_SECRET   Pluto jason_levels API credentials.

Run: ``uv run python -m scripts.jason_bot``

MUST NOT: place/modify orders or touch the data lake — read-only reporting only.
"""

from __future__ import annotations

import io
import re
from datetime import datetime, timedelta, timezone

import matplotlib
import pandas as pd
import urllib3

matplotlib.use("Agg")  # headless: render to PNG, no display
matplotlib.rcParams["font.family"] = "Liberation Sans"  # clean, professional sans
import matplotlib.pyplot as plt  # noqa: E402 - must follow backend selection

from src.common.config import require_env  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.common.pluto import JASON_LEVELS_URL, get as pluto_get  # noqa: E402
from src.common.telegram import (  # noqa: E402
    drain_backlog, poll_updates, send, send_photo, single_instance_lock)
from src.data.fetching.binance.backfill import _get_json  # noqa: E402

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # internal host, self-signed

_log = get_logger("jason_bot")

PLUTO_URL = JASON_LEVELS_URL
# v1, deliberately NOT jason_levels_v2: v2 publishes only on COMPLETED 4h candles and then
# lags further, running 1-2 candles (4-12h) behind — measured 2026-08-27 15:59 UTC, v2 read
# BTC 79,440 against a live 80,329 (SOL was 5% out). No parameter reaches past it: dropping
# end_date, dropping both dates, a future end_date and every lookback from 8h to 240h all
# returned the same stale candle. v1 uses the in-progress candle and ties to the live tick.
# v1 also takes BARE ccypairs — no BINANCEF. prefix. Re-migrate once Pluto fixes v2's cadence.

# Named universes -> the ccypairs to request. "default" sends none (the API's own default
# set). Coverage was verified live against the jason API; mm3/mm2 keep only the desk names
# from config/desks.yaml that the (Binance-sourced) API actually carries.
UNIVERSES: dict[str, list[str] | None] = {
    "default": ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT", "BILLUSDT",
                "BNBUSDT", "BSBUSDT", "CCUSDT", "CELOUSDT", "CGPTUSDT", "FETUSDT", "FILUSDT",
                "GRASSUSDT", "HYPEUSDT", "ICPUSDT", "LINEAUSDT", "LINKUSDT", "MINAUSDT", "NEARUSDT",
                "PAXGUSDT", "POLUSDT", "RAREUSDT", "TNSRUSDT", "VETUSDT", "VVVUSDT", "ZECUSDT",
                "ZENUSDT", "ZKCUSDT", "BTWUSDT", "HNTUSDT", "GENIUSUSDT"],
    "majors": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "HYPEUSDT", "PAXGUSDT"],
    "mm3": ["AZTECUSDT", "BILLUSDT", "BSVUSDT", "BTWUSDT", "CROSSUSDT", "DODOUSDT", "GASUSDT",
            "GENIUSUSDT", "MAGMAUSDT", "MINAUSDT", "NEOUSDT", "PARTIUSDT", "VTHOUSDT"],
    "mm2": ["OXTUSDT", "FETUSDT", "ICPUSDT", "ZENUSDT", "SAFEUSDT", "NEARUSDT", "CELOUSDT",
            "CFXUSDT", "VETUSDT", "CCUSDT", "POLUSDT", "RAREUSDT",
            "TNSRUSDT", "LINEAUSDT", "ZKCUSDT", "COMPUSDT", "YBUSDT"],  # CSPR: not in this API
    "rwa": None,  # resolved live to the ~30 most-liquid tradfi perps — see _rwa_pairs()
}
DEFAULT_UNIVERSE = "default"
_FAPI = "https://fapi.binance.com"
_RWA_CAP = 30
_RWA_EXCLUDE = {  # leveraged/inverse ETFs + pre-IPO synthetics — not "real" tradfi underlyings
    "TQQQUSDT", "SQQQUSDT", "SOXLUSDT", "SOXSUSDT", "UVXYUSDT", "TZAUSDT",
    "OPENAIUSDT", "ANTHROPICUSDT", "SPCXUSDT",
}


def _rwa_pairs() -> list[str]:
    """The ~30 most-liquid Binance tradfi/equity perps (TRADIFI_PERPETUAL) by 24h quote
    volume, excluding leveraged/inverse ETFs and pre-IPO synthetics."""
    info = _get_json(_FAPI, "/fapi/v1/exchangeInfo", {})
    tradfi = {s["symbol"] for s in info["symbols"]
              if s["status"] == "TRADING" and s["contractType"] == "TRADIFI_PERPETUAL"}
    tradfi -= _RWA_EXCLUDE
    vol = {t["symbol"]: float(t["quoteVolume"]) for t in _get_json(_FAPI, "/fapi/v1/ticker/24hr", {})}
    return sorted(tradfi, key=lambda s: vol.get(s, 0.0), reverse=True)[:_RWA_CAP]

# Displayed columns: (header, alt_format API field). Symmetric around Last: up/support side
# on the left, down/resistance side on the right. "up" = move up from the low pivot, "dn" =
# move down from the high pivot (matches the alt field names zScoreOf{Up,Down}Move...).
COLMAP = [
    ("Pair", "assetOrCurrencyPair"),
    ("Vol dn", "volatilitySinceLastHighPivotAnnualized"),
    ("Days dn", "timeSinceHighPivotDays"),
    ("Z dn", "zScoreOfDownMoveFromHighPivot"),
    ("Hi piv", "highPivotPrice"),
    ("Lo Target", "targetPriceDown2StdevsFromHighPivot"),
    ("Last", "lastPrice"),
    ("Hi target", "targetPricUp2StdevsFromLowPivot"),
    ("Lo piv", "lowPivotPrice"),
    ("Z up", "zScoreOfUpMoveFromLowPivot"),
    ("Days up", "timeSinceLowPivotDays"),
    ("Vol up", "volatilitySinceLastLowPivotAnnualized"),
    ("Period Vol", "volatilityForWholePeriodAnnualized"),
]


# --------------------------------------------------------------------------- #
# Pluto jason_levels API
# --------------------------------------------------------------------------- #
_CHUNK = 45   # the API stalls on very large ccypairs batches; request in chunks


def _jason_request(headers: dict, params: dict) -> list[dict]:
    resp = pluto_get(PLUTO_URL, headers=headers, params=params, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"unexpected jason payload: {type(data).__name__}")
    return data


def _jason_params() -> dict:
    """Base jason_levels params: 10-day lookback running to NOW, composite pivot, OCHL vol, 4h
    candles. ``start_date`` is unix epoch seconds (API: ``long`` via ``fromUnix()``); ``end_date``
    is deliberately omitted — see below."""
    now = datetime.now(timezone.utc)
    # NO end_date ON PURPOSE. Sending it AT ALL truncates the study window to that day's 00:00,
    # whatever the format: epoch seconds, epoch millis and ISO all returned the midnight candle
    # (2026-08-31 07:19 UTC -> end 00:00, lastPrice 77,757 against a live 78,216), and %d%b%y only
    # reached 04:00. Omitted, the API runs to the CURRENT minute and lastPrice ties to the live
    # tick. start_date still sets the lookback exactly (2/7/10/20d all honoured).
    return {"start_date": int((now - timedelta(days=10)).timestamp()),
            "exp_weight": "NONE",
            "hist_vol_calc": "OCHL", "pivot": "COMPOSITE", "candle_size": "4h",
            "alt_format": "true"}


def fetch_jason(universe: str) -> list[dict]:
    """Fetch the JASON table for a named universe. Large universes (rwa = all tradfi) are
    chunked because the API stalls on very large single ccypairs batches."""
    headers = {"X-Api-Key": require_env("PLUTO_API_KEY"),
               "X-Api-Secret": require_env("PLUTO_API_SECRET")}
    pairs = _rwa_pairs() if universe == "rwa" else UNIVERSES[universe]
    if pairs is None:                                    # API default set
        return _jason_request(headers, _jason_params())
    out: list[dict] = []
    for i in range(0, len(pairs), _CHUNK):
        chunk = ",".join(pairs[i:i + _CHUNK])
        out.extend(_jason_request(headers, {**_jason_params(), "ccypairs": chunk}))
    return out


def _fmt_val(header: str, v: object) -> str:
    """Values arrive pre-formatted; vols shown as whole percent (Period Vol is a raw fraction)."""
    if header == "Period Vol":
        try:
            return f"{float(v) * 100:.0f} %"
        except (TypeError, ValueError):
            return "" if v in (None, "") else str(v)
    if header in ("Vol up", "Vol dn"):
        try:
            return f"{float(str(v).replace('%', '').replace(',', '').strip()):.0f} %"
        except (TypeError, ValueError):
            return "" if v in (None, "") else str(v)
    return "" if v is None else str(v)


def build_table(records: list[dict]) -> pd.DataFrame:
    """Project the API records onto the display columns (base symbol; values as formatted)."""
    rows = []
    for r in records:
        row = {header: _fmt_val(header, r.get(field, "")) for header, field in COLMAP}
        row["Pair"] = re.sub(r"USDT?$", "", str(row["Pair"]))  # BTCUSD/BTCUSDT -> BTC
        rows.append(row)
    return pd.DataFrame(rows, columns=[h for h, _ in COLMAP])


# --------------------------------------------------------------------------- #
# PNG rendering
# --------------------------------------------------------------------------- #
# Finalized style: ruled table (horizontal rules only), gray body so the emphasized
# cells carry the weight — Pair + targets are dark bold; Z's turn bold green (up) /
# amber→red (dn) only when |z| >= 1.5.
TARGET_COLS = {"Hi target", "Lo Target"}
Z_COLS = {"Z up", "Z dn"}
_GRAY, _DARK = "#6b7280", "#111827"
_FONT_SIZE = 9
_PADX = 1.55        # column-width padding factor — room between numbers
_CELL_PAD = 0.22    # text inset within each cell


def _z_color(col: str, text_val: object) -> tuple | None:
    """Text color for a Z cell: green (up) / amber→red (dn), only when |z| >= 1.5."""
    try:
        z = abs(float(str(text_val).replace(",", "")))
    except (TypeError, ValueError):
        return None
    if z < 1.5:
        return None
    t = min((z - 1.5) / 1.5, 1.0)   # 1.5 -> start, 3.0+ -> full
    s, e = ((21, 128, 61), (6, 78, 45)) if col == "Z up" else ((202, 138, 4), (185, 28, 28))
    return tuple((a + (b - a) * t) / 255 for a, b in zip(s, e))


def render_table_image(df: pd.DataFrame, universe: str) -> bytes:
    """Render the JASON table as a styled PNG (ruled, gray body, bold targets, Z heat)."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cols = list(df.columns)
    nrows, ncols = df.shape
    fig, ax = plt.subplots(figsize=(1.3 * ncols, 0.55 + 0.3 * (nrows + 1)))
    ax.axis("off")
    ax.set_title(f"JASON levels · {universe} · {stamp}", fontsize=12, fontweight="bold",
                 loc="left", pad=12)
    tbl = ax.table(cellText=df.values, colLabels=cols, cellLoc="right", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(_FONT_SIZE)
    tbl.auto_set_column_width(range(ncols))
    for cell in tbl.get_celld().values():          # horizontal breathing room
        cell.PAD = _CELL_PAD
        cell.set_width(cell.get_width() * _PADX)
    tbl.scale(1, 1.34)

    for (r, c), cell in tbl.get_celld().items():
        col = cols[c]
        cell.visible_edges = "B"                   # horizontal rules only
        cell.set_edgecolor("#e5e7eb")
        if r == 0:                                 # header: bold, dark bottom rule, no fill
            cell.set_facecolor("white")
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
            cell.set_text_props(ha="center")
            cell.set_edgecolor("#111827")
            cell.set_linewidth(1.3)
            continue
        cell.get_text().set_color(_GRAY)           # body default: soft gray
        if c == 0:                                 # Pair label: dark + bold, left-aligned
            cell.set_text_props(ha="left")
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
        if col in TARGET_COLS:                     # targets: dark + bold
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
        if col in Z_COLS:                          # Z: green / amber-red + bold, only >= 1.5
            zc = _z_color(col, df.iloc[r - 1][col])
            if zc is not None:
                cell.get_text().set_color(zc)
                cell.get_text().set_fontweight("bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _caption(universe: str, df: pd.DataFrame) -> str:
    parts = [f"JASON levels · {universe} · {len(df)} pairs"]
    if universe != DEFAULT_UNIVERSE:
        requested = UNIVERSES[universe] or []
        got = {re.sub(r'USDT?$', '', p) for p in df["Pair"]}
        missing = [re.sub(r"USDT?$", "", p) for p in requested
                   if re.sub(r"USDT?$", "", p) not in got]
        if missing:
            parts.append("no jason data for: " + ", ".join(missing))
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Telegram bot  (plumbing shared via src.common.telegram)
# --------------------------------------------------------------------------- #
def _parse_universe(text: str) -> tuple[str | None, str | None]:
    """Return (universe, error). text is the '/jason [word]' command."""
    parts = text.split()
    if len(parts) == 1:
        return DEFAULT_UNIVERSE, None
    word = parts[1].lower()
    if word not in UNIVERSES:
        return None, f"Unknown universe '{word}'. Options: {', '.join(UNIVERSES)}"
    return word, None


def main() -> None:
    lock = single_instance_lock("jason_bot")
    if lock is None:
        _log.error("another jason_bot instance is already running — refusing to start "
                   "(a duplicate poller causes Telegram 409s and silent deafness)")
        return
    token = require_env("TELEGRAM_JASON_BOT_TOKEN")
    authorized = {c.strip() for c in require_env("TELEGRAM_CHAT_ID").split(",") if c.strip()}
    api = f"https://api.telegram.org/bot{token}"

    offset = drain_backlog(api)  # so a restart doesn't replay old commands
    _log.info("jason_bot started; serving /jason to chats %s", sorted(authorized))
    while True:
        updates, offset = poll_updates(api, offset)
        for upd in updates:
            msg = upd.get("message") or {}
            text = (msg.get("text") or "").strip()
            chat_id = str((msg.get("chat") or {}).get("id"))
            cmd = text.split()[0].split("@")[0].lower() if text else ""  # case-insensitive, /jason@bot ok
            if cmd != "/jason":
                continue
            if chat_id not in authorized:
                send(api, chat_id, "Not authorized.")
                _log.warning("rejected /jason from unauthorized chat %s", chat_id)
                continue
            universe, err = _parse_universe(text)
            if err:
                send(api, chat_id, err)
                continue
            send(api, chat_id, f"Fetching JASON levels ({universe})…")
            try:
                df = build_table(fetch_jason(universe))
                send_photo(api, chat_id, render_table_image(df, universe), caption=_caption(universe, df))
            except Exception as exc:  # noqa: BLE001 - report over a chat, never crash the bot
                _log.exception("jason report failed")
                send(api, chat_id, f"JASON report failed: {exc}")


if __name__ == "__main__":
    main()
