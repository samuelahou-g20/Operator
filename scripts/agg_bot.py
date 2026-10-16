"""Telegram bot: on ``/agg`` it renders the calling desk's Aggregator Dashboard as a house-style
PNG table; on ``/mm2gamma``, the MM2 gamma book's balances and deltas in the same format.

Reads (read-only) the desk's Google Sheet Dashboard tab via a service account (shared as Viewer),
takes the requested columns for every coin row down to ALL, and replies with the format-B PNG
(Liberation Sans, ruled, gray body; P&L/delta columns colored by sign, ALL row bold). Headers are
matched against the sheet VERBATIM, line breaks included ("Time-Decay\nD+1"), and the header row
is per-table (`header_row`) because the gamma book puts its headers on row 5, not row 3.

``/analysis <symbol> <window>`` charts and explains the treasury desk's trades on one symbol over a
past window (perps AND options, from Pluto's historic cube + the trade service) — served in the
treasury chat only; the logic lives in ``scripts.treasury_analysis``.

``/mm2gamma`` is not desk-routed the way ``/agg`` is — it reads one fixed workbook (MM2GAMMA-MM),
so it is served only in the MM2 and MM3 chats rather than wherever the bot is pinned.

Auth: service-account JSON key (``GOOGLE_AGG_SA_KEY`` or repo-root ``.gsa_key.json``), scope
``spreadsheets.readonly``. Its own BotFather token ``TELEGRAM_AGG_BOT_TOKEN`` (one long-poller per
token). Desk routing is explicit by env: ``TELEGRAM_MM3_CHAT_ID`` -> MM3, ``TELEGRAM_MM2_CHAT_ID``
-> MM2 (disjoint; a chat not pinned to a desk is refused, never defaulted).

Desks: MM3/MM2/MM4 read their Google Sheet; treasury, the options book and MM5 read the Pluto
portfolio endpoint instead (``source="pluto"``) — greeks per asset, no sheet. Each desk is pinned
to its own chat var: TELEGRAM_MM3_CHAT_ID / _MM2_ / _MM4_ / _TREASURY_ / _OPTIONS_ / _MM5_CHAT_ID
(all disjoint, no default).

MUST NOT: write the sheet, place orders, or touch the data lake. Read-and-render only.

Run: ``uv run python -m scripts.agg_bot``
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from itertools import combinations

import gspread
import matplotlib
import pandas as pd
import urllib3

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Liberation Sans"
import matplotlib.pyplot as plt  # noqa: E402 - must follow backend selection

from src.common.config import load_config, require_env  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.common.paths import REPO_ROOT  # noqa: E402
from src.common.telegram import drain_backlog, poll_updates, send, send_photo  # noqa: E402
from src.common.pluto import PORTFOLIO_URL, get as pluto_get  # noqa: E402
from scripts.treasury_analysis import handle as handle_analysis  # noqa: E402

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # Pluto internal self-signed

_log = get_logger("agg_bot")

_SCOPE = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
_KEY = os.environ.get("GOOGLE_AGG_SA_KEY") or str(REPO_ROOT / ".gsa_key.json")
_HEADER_ROW = 2          # 0-indexed row holding the real column headers
_MKT_SHARE_LO, _MKT_SHARE_HI = 10.0, 40.0   # Mkt Share text: red <10%, purple >40%
_GRAY, _DARK = "#6b7280", "#111827"
_GREEN, _RED, _PURPLE = "#047857", "#b91c1c", "#7c3aed"

# Per-desk config: `columns` = (row-2 sheet header, display); `sign` = display cols colored by
# sign; `share` = the market-share display col (red <10% / purple >40%) or None. Chat routing is
# by env: TELEGRAM_MM3_CHAT_ID -> MM3, TELEGRAM_MM2_CHAT_ID -> MM2 (explicit, disjoint, no default).
# `label`/`sheet`/`gid` (the desk's
# Google Sheet wiring) come from config/desks.yaml, not here — a new owner edits config.
_AGG_CFG = load_config("desks.yaml")
_MM3 = {
    **_AGG_CFG["mm3"]["agg"],
    "columns": [("Coin", "Coin"), ("Price", "Price"), ("prev EOD %", "24h %"),
                ("Daily P/L", "Daily P/L"),
                ("Net Delta ($)", "Net Δ $"), ("Daily Net Delta ($)", "Daily Net Δ $"),
                ("Daily Trading PnL", "Daily Trd PnL"), ("G20 Total Volume ($)", "G20 Vol $"),
                ("Market Volume ($)", "Mkt Vol $"), ("Market Share", "Mkt Share")],
    "sign": {"24h %", "Daily P/L", "Net Δ $", "Daily Net Δ $", "Daily Trd PnL"},
    "share": "Mkt Share",
}
_MM2 = {
    **_AGG_CFG["mm2"]["agg"],
    "columns": [("Coin", "Coin"), ("Price", "Price"), ("prev EOD %", "24h %"),
                ("Daily P/L", "Daily P/L"),
                ("Net Delta ($)", "Net Δ $"), ("$Daily Net Delta (USD)", "Daily Net Δ $"),
                # Sits with the other risk columns, not the P&L block. The sheet header carries a
                # hard line break ("Time-Decay\nD+1") — matched verbatim, since read_dashboard
                # looks the name up exactly as row 2 spells it.
                ("Time-Decay\nD+1", "Theta D+1"),
                ("Daily PnL", "Daily PnL"), ("Total ($)", "Total Vol $"), ("MM ($)", "MM Vol $"),
                ("Market Share", "Mkt Share")],
    "sign": {"24h %", "Daily P/L", "Net Δ $", "Daily Net Δ $", "Theta D+1", "Daily PnL"},
    "share": "Mkt Share",
}
# MM4: a sheet desk like MM3/MM2. Its Dashboard carries every column MM3's view renders, under
# the same header names and on the same header row, so it reuses MM3's column spec verbatim.
# It has no ALL totals row — the reader stops at the first blank Coin, giving the 15 coin rows.
_MM4 = {
    **_AGG_CFG["mm4"]["agg"],
    "columns": _MM3["columns"], "sign": _MM3["sign"], "share": _MM3["share"],
}

# MM2 gamma book (`/mm2gamma`): a DIFFERENT spreadsheet from the MM2 aggregator — the MM2GAMMA-MM
# workbook's Dashboard tab, whose headers sit on row 5, not row 3, hence `header_row`. Balances and
# deltas only; no market-share column. Several headers carry hard line breaks, matched verbatim.
_MM2GAMMA = {
    "label": "MM2 Gamma Book", "sheet": "11Ne3nFRAg6I90qHpyWbxcKwgu7uo_8NLB5ragX7f9gU",
    "gid": 1242308581, "header_row": 4,
    "columns": [("Coin", "Coin"), ("Price", "Price"),
                ("Starting\nBalance", "Start Bal"), ("Current\nBalance", "Curr Bal"),
                ("Live Options MTM", "Opt MTM"), ("Pluto Delta (Coin)", "Pluto Δ"),
                ("Net Delta\n(Coin)", "Net Δ coins"), ("Net Delta ($)", "Net Δ $")],
    "sign": {"Opt MTM", "Pluto Δ", "Net Δ coins", "Net Δ $"},
    "share": None,
}

# Treasury desk: not a Google Sheet — its aggregator is the Pluto portfolio endpoint with
# `detailed=greeks`, giving per-asset greeks (delta $, gamma $, vega, theta, spot) + total PV.
# `source="pluto"` routes main() to the portfolio path instead of the sheet reader.
_PLUTO_PORTFOLIO_URL = PORTFOLIO_URL
# `risk_report` names a saved RiskReport object in the G20 Pluto account — the same selector as
# the UI (Pluto resolves it via GetUserObject("G20", "RiskReport", <name>), so the value must
# match the UI label EXACTLY, spaces included; underscored/concatenated forms 400). Omitting
# it lets Pluto fall back to its default model, which is why the numbers did not tie to the UI.
_TREASURY = {"label": "Treasury Portfolio", "source": "pluto", "portfolio": "TREASURYPROPTOTAL",
             "risk_report": "STICKY STRIKE FITTED"}
# Options book: same Pluto greeks path as treasury, different portfolio. Its greek columns carry
# real optionality (treasury is mostly delta-1), which is exactly why the RiskReport matters here.
_OPTIONS = {"label": "Options Portfolio", "source": "pluto", "portfolio": "TOTAL_OPTIONS",
            "risk_report": "STICKY STRIKE FITTED"}
# MM5: same Pluto greeks path again, portfolio QUANTMM_CLEAN (no Google Sheet for this desk).
_MM5 = {"label": "MM5 Portfolio", "source": "pluto", "portfolio": "QUANTMM_CLEAN",
        "risk_report": "STICKY STRIKE FITTED"}


# --------------------------------------------------------------------------- #
# Sheet read
# --------------------------------------------------------------------------- #
def read_dashboard(cfg: dict) -> pd.DataFrame:
    """Selected columns for every coin row from the first coin to ALL (inclusive), as displayed."""
    gc = gspread.service_account(filename=_KEY, scopes=_SCOPE)
    vals = gc.open_by_key(cfg["sheet"]).get_worksheet_by_id(cfg["gid"]).get_all_values()
    header_row = cfg.get("header_row", _HEADER_ROW)
    hdr = vals[header_row]
    missing = [name for name, _ in cfg["columns"] if name not in hdr]
    if missing:
        raise RuntimeError(f"aggregator columns not found: {missing}")
    idx = [(disp, hdr.index(name)) for name, disp in cfg["columns"]]
    coin_i = hdr.index("Coin")
    # MM3's table has a blank spacer row mid-table, so when an ALL row exists read through to it;
    # sheets without one (MM4, MM2 gamma) end at the first blank Coin.
    has_all = any(v[coin_i].strip() == "ALL" for v in vals[header_row + 1:])
    rows = []
    for r in range(header_row + 1, len(vals)):
        coin = vals[r][coin_i].strip()
        if not coin:
            if has_all:
                continue
            break
        rows.append([vals[r][i] for _disp, i in idx])
        if coin == "ALL":                  # totals row = last row of the main table
            break
    return pd.DataFrame(rows, columns=[disp for disp, _ in idx])


_PORT_COLS = ["Asset", "Spot", "Δ $", "Γ $", "Vega", "Theta"]
_PORT_SIGN = {"Δ $", "Γ $", "Vega", "Theta"}   # greek columns coloured green/red by sign


def _net_key(a: str) -> str:
    """The row a Pluto asset key nets into — its underlying, quote leg stripped.

    ``BTCUSD`` (spot) and ``BTCUSDT`` (perp) are one exposure quoted twice, and on a hedged
    market-making book they very nearly cancel (+121,945 against -122,809 = a -864 residual), so
    listing them apart doubled the table and buried the real delta.

    A MULTIPLIER contract is deliberately not merged: ``1000PEPEUSD`` prices at exactly 1000x
    ``PEPEUSD``, so folding them together would leave the Spot column meaningless. It keeps its own
    row (its delta $ is already in dollars, so the TOTAL is unaffected either way)."""
    s = a.upper()
    for q in ("USDT", "USD"):                  # USDT first: a bare USD test would mis-split it
        if s.endswith(q) and len(s) > len(q):
            return s[:-len(q)]
    return s


_STABLES = {"USD", "USDT", "USDC", "USDF", "USDD", "USDE", "DAI", "TUSD", "FDUSD", "PYUSD"}


def _is_stable_asset(a: str) -> bool:
    """True when BOTH legs of the asset key are stablecoins (``USDCUSD``, ``USDTUSD``,
    ``USDCUSDT``) — that is cash, not a risk position, so it is dropped from the greeks table.

    Both legs must match, or the quote leg alone would drop every row: ``BTCUSD`` and ``PAXGUSDT``
    are quoted in a stable but are not cash."""
    s = a.upper()
    if s in _STABLES:
        return True
    return any(s.endswith(q) and s[:-len(q)] in _STABLES for q in _STABLES)


def read_portfolio(cfg: dict) -> tuple[pd.DataFrame, float]:
    """Per-asset greeks for the treasury book from the Pluto portfolio endpoint (``detailed=greeks``):
    spot, delta $, gamma $, vega, theta per asset, sorted by |delta $|, with a TOTAL row. Returns
    ``(df, portfolio_pv)``.

    Sends ``RiskReport`` so the risk model matches the Pluto UI. Without it Pluto applies its own
    default and the greeks do not tie out. Note the book is revalued live — two identical calls
    seconds apart differ slightly, so never A/B a model change by comparing single calls."""
    headers = {"X-Api-Key": require_env("PLUTO_API_KEY"), "X-Api-Secret": require_env("PLUTO_API_SECRET")}
    params = {"portfolio": cfg["portfolio"], "detailed": "greeks"}
    if cfg.get("risk_report"):                      # must match the UI's RiskReport name exactly
        params["RiskReport"] = cfg["risk_report"]
    r = pluto_get(_PLUTO_PORTFOLIO_URL, headers=headers, timeout=60, params=params)
    r.raise_for_status()
    d = r.json()
    if not isinstance(d, dict) or "delta_usds" not in d:
        got = list(d)[:6] if isinstance(d, dict) else type(d).__name__
        raise RuntimeError(f"unexpected greeks payload: {got}")
    du, gu = d.get("delta_usds") or {}, d.get("gamma_usds") or {}
    vg, th, sp = d.get("vegas") or {}, d.get("thetas") or {}, d.get("spotrefs") or {}
    # Stablecoin legs are cash, not risk: they carry no gamma/vega/theta and their delta is just the
    # balance (USDCUSD alone read $298,539). Dropped BEFORE the TOTAL so the table sums to what it
    # shows. `pv` in the caption still comes from Pluto and does include the cash.
    groups: dict[str, list[str]] = {}
    for a in du:
        if _is_stable_asset(a):
            continue
        groups.setdefault(_net_key(a), []).append(a)

    def money(x: object) -> str:
        return "" if x is None else f"{float(x):,.0f}"

    def gsum(m: dict, legs: list[str]) -> float:
        """One greek summed across a row's legs. Delta $ and gamma $ are already in DOLLARS, so
        the spot and perp legs add directly — no spot-ratio scaling."""
        return sum(float(m.get(leg) or 0.0) for leg in legs)

    def gspot(legs: list[str]) -> str:
        """The row's spot reference: the cash (``...USD``) leg's where there is one. Legs of a
        netted row quote the same scale, so either is representative — BTCUSD 79954.07 against
        BTCUSDT 79972.06 is a 2bp basis, not a different price."""
        for leg in sorted(legs, key=lambda k: (not k.upper().endswith("USD"), k)):
            if sp.get(leg) is not None:
                return f"{float(sp[leg]):g}"
        return ""

    order = sorted(groups, key=lambda g: -abs(gsum(du, groups[g])))
    rows = [[g, gspot(groups[g]), money(gsum(du, groups[g])), money(gsum(gu, groups[g])),
             money(gsum(vg, groups[g])), money(gsum(th, groups[g]))] for g in order]
    legs_all = [leg for legs in groups.values() for leg in legs]
    rows.append(["TOTAL", "", money(gsum(du, legs_all)), money(gsum(gu, legs_all)),
                 money(gsum(vg, legs_all)), money(gsum(th, legs_all))])
    return pd.DataFrame(rows, columns=_PORT_COLS), float(d.get("pv") or 0.0)


def render_portfolio_image(cfg: dict, df: pd.DataFrame) -> bytes:
    """House format-B render of the treasury per-asset greeks (greek columns green/red by sign,
    TOTAL row bold + ruled)."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cols, nrows = list(df.columns), len(df)
    fig, ax = plt.subplots(figsize=(1.15 * len(cols) + 1.0, 0.55 + 0.32 * (nrows + 1)))
    ax.axis("off")
    ax.set_title(f"{cfg['label']} · {stamp}", fontsize=12, fontweight="bold", loc="left", pad=12)
    tbl = ax.table(cellText=df.values, colLabels=cols, cellLoc="right", loc="center")
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
            cell.set_text_props(ha="left" if c == 0 else "center")
            cell.set_edgecolor("#111827")
            cell.set_linewidth(1.3)
            continue
        is_total = str(df.iloc[r - 1]["Asset"]).strip() == "TOTAL"
        cell.get_text().set_color(_GRAY)
        if c == 0:
            cell.set_text_props(ha="left")
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
        elif col == "Spot":
            cell.get_text().set_color(_DARK)
        elif col in _PORT_SIGN:
            sc = _sign_color(df.iloc[r - 1][col])
            if sc:
                cell.get_text().set_color(sc)
        if is_total:
            cell.get_text().set_fontweight("bold")
            cell.visible_edges = "TB"
            cell.set_edgecolor("#111827")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# PNG rendering (house format-B)
# --------------------------------------------------------------------------- #
def _sign_color(text: object) -> str | None:
    """Green/red for a signed numeric string; None (gray) for zero/blank/non-numeric."""
    try:
        v = float(str(text).replace(",", "").replace("$", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None
    return _GREEN if v > 0 else _RED if v < 0 else None


def _mkt_share_color(text: object) -> str | None:
    """Red when market share < 10%, purple when > 40%; None (gray) inside the band or blank."""
    try:
        v = float(str(text).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None
    if v < _MKT_SHARE_LO:
        return _RED
    if v > _MKT_SHARE_HI:
        return _PURPLE
    return None


def render_table_image(cfg: dict, df: pd.DataFrame) -> bytes:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sign_cols, share_col = cfg["sign"], cfg["share"]
    cols = list(df.columns)
    nrows, ncols = df.shape
    fig, ax = plt.subplots(figsize=(1.35 * ncols, 0.55 + 0.3 * (nrows + 1)))
    ax.axis("off")
    ax.set_title(f"{cfg['label']} · {stamp}", fontsize=12, fontweight="bold", loc="left", pad=12)
    tbl = ax.table(cellText=df.values, colLabels=cols, cellLoc="right", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(range(ncols))
    for cell in tbl.get_celld().values():
        cell.PAD = 0.22
        cell.set_width(cell.get_width() * 1.55)
    tbl.scale(1, 1.34)

    for (r, c), cell in tbl.get_celld().items():
        col = cols[c]
        cell.visible_edges = "B"
        cell.set_edgecolor("#e5e7eb")
        if r == 0:                                    # header row
            cell.set_facecolor("white")
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
            cell.set_text_props(ha="center")
            cell.set_edgecolor("#111827")
            cell.set_linewidth(1.3)
            continue
        is_total = str(df.iloc[r - 1]["Coin"]).strip() == "ALL"
        cell.get_text().set_color(_GRAY)
        if c == 0:                                    # Coin: dark bold, left-aligned
            cell.set_text_props(ha="left")
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
        elif col in sign_cols:                        # P&L/delta: green/red by sign
            sc = _sign_color(df.iloc[r - 1][col])
            if sc:
                cell.get_text().set_color(sc)
        elif col == share_col:                        # market share: red <10% / purple >40%
            mc = _mkt_share_color(df.iloc[r - 1][col])
            if mc:
                cell.get_text().set_color(mc)
        if is_total:                                  # ALL totals row: bold + a top rule
            cell.get_text().set_fontweight("bold")
            cell.visible_edges = "TB"
            cell.set_edgecolor("#111827")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Telegram
# --------------------------------------------------------------------------- #
def main() -> None:
    token = require_env("TELEGRAM_AGG_BOT_TOKEN")
    # Explicit chat -> desk config (no defaulting): a chat is a desk ONLY if pinned in its own var.
    def pinned(var: str, required: bool = False) -> set[str]:
        raw = require_env(var) if required else (os.environ.get(var) or "")
        return {c.strip() for c in raw.split(",") if c.strip()}

    pins = [("mm3", pinned("TELEGRAM_MM3_CHAT_ID", required=True), _MM3),
            ("mm2", pinned("TELEGRAM_MM2_CHAT_ID"), _MM2),
            ("treasury", pinned("TELEGRAM_TREASURY_CHAT_ID"), _TREASURY),
            ("options", pinned("TELEGRAM_OPTIONS_CHAT_ID"), _OPTIONS),
            ("mm5", pinned("TELEGRAM_MM5_CHAT_ID"), _MM5),
            ("mm4", pinned("TELEGRAM_MM4_CHAT_ID"), _MM4)]
    for (na, a, _ca), (nb, b, _cb) in combinations(pins, 2):     # all pins must be disjoint
        if a & b:
            raise RuntimeError(f"chat(s) {sorted(a & b)} pinned to both {na} and {nb} — fix .env")
    desk = {c: cfg for _n, chats, cfg in pins for c in chats}
    api = f"https://api.telegram.org/bot{token}"

    # /mm2gamma reads ONE fixed book regardless of which chat asked, so it is not desk-routed the
    # way /agg is — it is served only where that book belongs (the MM2 and MM3 chats), never in the
    # treasury/options/mm5 chats, which would put another desk's positions in front of them.
    gamma_chats = {c for name, chats, _c in pins if name in ("mm2", "mm3") for c in chats}

    treasury_chats = {c for name, chats, _c in pins if name == "treasury" for c in chats}

    offset = drain_backlog(api)  # so a restart doesn't replay old commands
    _log.info("agg_bot started; serving /agg, /mm2gamma, /analysis")
    while True:
        updates, offset = poll_updates(api, offset, poll_timeout=25)
        for upd in updates:
            msg = upd.get("message") or {}
            text = (msg.get("text") or "").strip()
            chat_id = str((msg.get("chat") or {}).get("id"))
            cmd = text.split()[0].split("@")[0].lower() if text else ""
            if cmd not in ("/agg", "/mm2gamma", "/analysis"):
                continue
            cfg = desk.get(chat_id)
            if cfg is None:
                _log.warning("refused %s from chat %s", cmd, chat_id)
                send(api, chat_id, f"Not authorized. This chat id is {chat_id} — pin it to a desk via "
                                   "TELEGRAM_MM3_CHAT_ID (MM3), TELEGRAM_MM2_CHAT_ID (MM2), "
                                   "TELEGRAM_MM4_CHAT_ID (MM4), TELEGRAM_MM5_CHAT_ID (MM5), "
                                   "TELEGRAM_TREASURY_CHAT_ID or TELEGRAM_OPTIONS_CHAT_ID.")
                continue
            if cmd == "/mm2gamma":
                if chat_id not in gamma_chats:
                    send(api, chat_id, "/mm2gamma is served in the MM2 and MM3 chats only.")
                    continue
                cfg = _MM2GAMMA
            if cmd == "/analysis":
                if chat_id not in treasury_chats:
                    send(api, chat_id, "/analysis is served in the treasury chat only.")
                    continue
                _log.info("/analysis %s requested by chat %s", text.split()[1:], chat_id)
                # Blocks the poll loop for the whole run (~1 min for 7d), so say so up front
                # rather than look hung.
                send(api, chat_id, "⏳ Pulling Pluto snapshots and treasury fills…")
                try:
                    summary, pngs = handle_analysis(text.split()[1:])
                    send(api, chat_id, summary)
                    for png in pngs:
                        send_photo(api, chat_id, png)
                except Exception as exc:  # noqa: BLE001 - report over chat, never crash the bot
                    _log.exception("/analysis failed")
                    send(api, chat_id, f"/analysis failed: {exc}")
                continue
            _log.info("%s (%s) requested by chat %s", cmd, cfg["label"], chat_id)
            try:
                if cfg.get("source") == "pluto":                # treasury: Pluto greeks, not a sheet
                    df, pv = read_portfolio(cfg)
                    send_photo(api, chat_id, render_portfolio_image(cfg, df),
                               caption=f"{cfg['label']} · PV ${pv:,.0f} · {len(df) - 1} assets")
                else:
                    df = read_dashboard(cfg)
                    send_photo(api, chat_id, render_table_image(cfg, df),
                               caption=f"{cfg['label']} · {len(df)} rows")
            except Exception as exc:  # noqa: BLE001 - report over chat, never crash the bot
                _log.exception("%s failed", cmd)
                send(api, chat_id, f"{cmd} failed: {exc}")


if __name__ == "__main__":
    main()
