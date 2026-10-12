"""Telegram bot that serves the futures-hedge intraday PnL report on demand.

Long-polls Telegram; on ``/hedge`` it computes today's per-symbol PnL and replies with a
monospace table. The desk (mm2 vs mm3) is chosen by which chat the command came from, mirroring
scripts/agg_bot.py (CoinRoutes positions + internal trade service + ccxt marks, ported verbatim
from research/futures_pnl_report_sam.ipynb).

Env (loaded from repo-root .env): TELEGRAM_BOT_TOKEN, TELEGRAM_MM3_CHAT_ID (MM3 chat id(s)),
TELEGRAM_MM2_CHAT_ID (MM2 chat id(s)) — explicit, disjoint, no default; per-desk CoinRoutes keys
CR_API_TOKEN_MM3 / CR_API_TOKEN_MM2.

Run: ``uv run python -m scripts.hedge_bot``

MUST NOT: place/modify orders or touch the data lake — read-only reporting only.
"""

from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import ccxt
import matplotlib
import numpy as np
import pandas as pd
import requests
import urllib3

matplotlib.use("Agg")  # headless: render to PNG, no display
matplotlib.rcParams["font.family"] = "Liberation Sans"  # clean, professional sans
import matplotlib.pyplot as plt  # noqa: E402 - must follow backend selection

from src.adapters.trade_service.client import TradeService  # noqa: E402
from src.common.config import load_config, require_env  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.common.telegram import drain_backlog, poll_updates, send, send_photo  # noqa: E402
from src.common.pluto import JASON_LEVELS_URL, get as pluto_get  # noqa: E402
from src.data.fetching.binance.fetcher import fetch_recent_bars  # noqa: E402
from src.signals.jason_trend_score import trend_scores  # noqa: E402

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # Pluto internal self-signed

_log = get_logger("hedge_bot")

CR_STRATEGIES = {
    "mm2": "futurestrading_ha_j",
    "mm3": "mm3-futurestrading-ha-j"
}
_CR_TOKEN_ENV = {"mm2": "CR_API_TOKEN_MM2", "mm3": "CR_API_TOKEN_MM3"}   # each desk on its own CR key
LOCATION_IDS = {
    "mm2": {
        11262: "binance"
    },
    "mm3": {
        12097: "binance",
        11639: "bybit"
    }
}
EXCLUDE_EXCHANGES = {
    "mm2": ["deribit"],
    "mm3": []
}


# --------------------------------------------------------------------------- #
# Data sources (ported from the notebook)
# --------------------------------------------------------------------------- #
class CoinRoutes:
    def __init__(self, api_token: str):
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Token {api_token}"})
        self.base_url = "https://g20jpy.coinroutes.io"

    def _request(self, method, url, **kwargs):
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return None

    def get_positions(self, strategy=None):
        params = {"strategy": strategy} if strategy else {}
        result = self._request("get", self.base_url + "/api/positions/", params=params)
        return pd.DataFrame(result)


class MarketDataClient:
    def __init__(self):
        self.ccxt_connectors = {}

    def _add_connector(self, exchange: str, timeout: int = 60000):
        self.ccxt_connectors[exchange] = getattr(ccxt, exchange)(
            {"enableRateLimit": True, "timeout": timeout}
        )

    def get_klines(self, symbol, exchange, timeframe, start_dt, end_dt, request_limit=100):
        if exchange not in self.ccxt_connectors:
            self._add_connector(exchange)
        step = pd.Timedelta(timeframe.replace("d", "D"))  # ccxt '1d' -> pandas 'D'
        since = int(start_dt.timestamp()) * 1000
        all_klines = []
        while True:
            klines = self.ccxt_connectors[exchange].fetch_ohlcv(
                symbol=symbol, timeframe=timeframe, since=since, limit=request_limit
            )
            if not klines:
                break
            all_klines += klines
            last_ts = datetime.fromtimestamp(klines[-1][0] / 1000)
            if last_ts + step > end_dt:
                break
            since = klines[-1][0] + 1
        while all_klines:
            last_ts = datetime.fromtimestamp(all_klines[-1][0] / 1000)
            if last_ts >= end_dt:
                all_klines.pop()
            else:
                break
        return all_klines

    def get_klines_df(self, symbol, exchange, timeframe, start_dt, end_dt, request_limit=100):
        klines = self.get_klines(symbol, exchange, timeframe, start_dt, end_dt, request_limit)
        df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def build_report(cr_token: str, desk: str) -> pd.DataFrame:
    """Compute today's per-symbol positional + trading PnL (with TOTAL row)."""
    cr = CoinRoutes(api_token=cr_token)
    trade_service = TradeService()
    md_client = MarketDataClient()

    start_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = start_dt + timedelta(days=1)

    perp_positions = cr.get_positions(strategy=CR_STRATEGIES[desk])

    perp_positions = perp_positions[~perp_positions['exchange'].isin(EXCLUDE_EXCHANGES[desk])]

    binance_symbols = perp_positions[perp_positions["exchange"] == "binancefutures"][
        "quantity_currency"
    ].unique()
    bybit_symbols = perp_positions[perp_positions["exchange"] == "bybit"][
        "quantity_currency"
    ].unique()

    perp_positions["quantity"] = perp_positions["quantity"].astype(float)
    perp_positions = perp_positions.rename(
        columns={"quantity_currency": "symbol", "quantity": "position"}
    )
    perp_positions = perp_positions.groupby("symbol", as_index=False)["position"].sum()

    trades = trade_service.get_trades(
        location_ids=LOCATION_IDS[desk].keys(), start_dt=start_dt, end_dt=end_dt
    )
    trades["symbol"] = trades["native_inst"].str.extract(r"^P\.([^-]+)-")

    binance_loc_ids = [i for i in LOCATION_IDS[desk] if LOCATION_IDS[desk][i] == "binance"]
    binance_symbols = np.union1d(
        binance_symbols, trades[trades["location_id"].isin(binance_loc_ids)]["symbol"].unique()
    )
    bybit_loc_ids = [i for i in LOCATION_IDS[desk] if LOCATION_IDS[desk][i] == "bybit"]
    bybit_symbols = np.union1d(
        bybit_symbols, trades[trades["location_id"].isin(bybit_loc_ids)]["symbol"].unique()
    )

    trades["buy_qty"] = np.where(trades["side"] == "B", trades["qty"], 0)
    trades["sell_qty"] = np.where(trades["side"] == "S", trades["qty"], 0)
    trades["buy_notional"] = np.where(trades["side"] == "B", trades["notional"], 0)
    trades["sell_notional"] = np.where(trades["side"] == "S", trades["notional"], 0)
    trades = trades.groupby("symbol", as_index=False)[
        ["buy_qty", "sell_qty", "buy_notional", "sell_notional"]
    ].sum()
    trades["avg_buy_price"] = np.where(
        trades["buy_qty"] == 0, 0, trades["buy_notional"] / trades["buy_qty"]
    )
    trades["avg_sell_price"] = np.where(
        trades["sell_qty"] == 0, 0, trades["sell_notional"] / trades["sell_qty"]
    )

    perp_positions = perp_positions.merge(trades, on="symbol", how="outer").fillna(0)
    perp_positions["position(t-1)"] = (
        perp_positions["position"] - perp_positions["buy_qty"] + perp_positions["sell_qty"]
    )

    perp_positions["price(t-1)"] = np.nan
    perp_positions["price"] = np.nan
    for idx, row in perp_positions.iterrows():
        if row["symbol"] in binance_symbols:
            md_exch = "binance"
        elif row["symbol"] in bybit_symbols:
            md_exch = "bybit"
        else:
            _log.warning("exchange unknown for symbol %s, skipping", row["symbol"])
            continue
        data = md_client.get_klines_df(
            f"{row['symbol']}/USDT:USDT",
            md_exch,
            "1d",
            start_dt.replace(tzinfo=None),
            end_dt.replace(tzinfo=None),
        ).set_index("timestamp")
        bar_ts = pd.Timestamp(start_dt.replace(tzinfo=None))
        perp_positions.loc[idx, "price(t-1)"] = data.loc[bar_ts, "open"]
        perp_positions.loc[idx, "price"] = data.loc[bar_ts, "close"]

    perp_positions["positional_pnl"] = (
        perp_positions["price"] - perp_positions["price(t-1)"]
    ) * perp_positions["position(t-1)"]
    perp_positions["trading_pnl"] = (
        perp_positions["price"] - perp_positions["avg_buy_price"]
    ) * perp_positions["buy_qty"] + (
        perp_positions["avg_sell_price"] - perp_positions["price"]
    ) * perp_positions["sell_qty"]
    perp_positions["total_pnl"] = (
        perp_positions["positional_pnl"] + perp_positions["trading_pnl"]
    )
    perp_positions["position_usd"] = perp_positions["position"] * perp_positions["price"]
    perp_positions["pct_chg"] = (
        (perp_positions["price"] - perp_positions["price(t-1)"])
        / perp_positions["price(t-1)"] * 100
    )

    cols = [
        "symbol",
        "position(t-1)",
        "price(t-1)",
        "position",
        "position_usd",
        "price",
        "pct_chg",
        "buy_qty",
        "avg_buy_price",
        "sell_qty",
        "avg_sell_price",
        "positional_pnl",
        "trading_pnl",
        "total_pnl",
    ]
    total_row = {c: "" for c in cols}
    total_row["symbol"] = "TOTAL"
    for c in ("position_usd", "positional_pnl", "trading_pnl", "total_pnl"):
        total_row[c] = perp_positions[c].sum()
    return pd.concat([perp_positions[cols], pd.DataFrame([total_row])], ignore_index=True)


DISPLAY_COLS = [
    "symbol",
    "position",
    "position_usd",
    "price(t-1)",
    "price",
    "pct_chg",
    "buy_qty",
    "avg_buy_price",
    "sell_qty",
    "avg_sell_price",
    "positional_pnl",
    "trading_pnl",
    "total_pnl",
]
COL_HEADERS = {
    "symbol": "Sym",
    "position": "Pos",
    "position_usd": "Pos $",
    "price(t-1)": "Px t-1",
    "price": "Px t",
    "pct_chg": "%Chg",
    "buy_qty": "Buy Q",
    "avg_buy_price": "Avg B",
    "sell_qty": "Sell Q",
    "avg_sell_price": "Avg S",
    "positional_pnl": "Pos PnL",
    "trading_pnl": "Trd PnL",
    "total_pnl": "Tot PnL",
}
PNL_COLS = ("positional_pnl", "trading_pnl", "total_pnl")
SIGN_COLS = (*PNL_COLS, "pct_chg")          # green/red text by sign
_GRAY, _DARK = "#6b7280", "#111827"         # house-style body / label colors


def _fmt_num(v: float) -> str:
    """Comma-separated, no scientific notation; decimals scale with magnitude."""
    a = abs(float(v))
    if a == 0:
        return "0"
    if a >= 1000:
        dec = 0
    elif a >= 1:
        dec = 2
    elif a >= 0.01:
        dec = 4
    else:
        dec = 6
    return f"{v:,.{dec}f}"


def _fmt_cell(c: str, v: object) -> str:
    if v == "" or pd.isna(v):
        return ""
    if c == "pct_chg":
        return f"{v:+.2f}%"
    if c in PNL_COLS or c == "position_usd":
        return f"{v:,.0f}"
    return _fmt_num(v)


def _format_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return the display subset with values formatted as strings (PnL/notional with separators)."""
    disp = df[DISPLAY_COLS].copy()
    for c in disp.columns:
        if c == "symbol":
            continue
        disp[c] = disp[c].map(lambda v, c=c: _fmt_cell(c, v))
    return disp


def render_table_image(df: pd.DataFrame, desk: str) -> bytes:
    """Render the report as a styled PNG (house style: Liberation Sans, horizontal rules, soft-gray
    body, dark-bold symbols, green/red text on %Chg and the PnL columns, bold ruled TOTAL row)."""
    disp = _format_display(df)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    nrows, ncols = disp.shape
    total_row = nrows - 1  # build_report appends TOTAL last

    fig, ax = plt.subplots(figsize=(1.3 * ncols, 0.5 + 0.34 * (nrows + 1)))
    ax.axis("off")
    ax.set_title(f"{desk.upper()} Hedge PnL — {stamp}", fontsize=13, fontweight="bold",
                 loc="left", pad=12, color=_DARK)
    tbl = ax.table(cellText=disp.values, colLabels=[COL_HEADERS[c] for c in disp.columns],
                   cellLoc="right", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(range(ncols))
    for cell in tbl.get_celld().values():          # horizontal breathing room
        cell.PAD = 0.22
        cell.set_width(cell.get_width() * 1.55)
    tbl.scale(1, 1.34)

    sign_idx = {disp.columns.get_loc(c) for c in SIGN_COLS if c in disp.columns}
    for (r, c), cell in tbl.get_celld().items():
        cell.visible_edges = "B"                   # horizontal rules only
        cell.set_edgecolor("#e5e7eb")
        if r == 0:                                 # header: bold, dark bottom rule, no fill
            cell.set_facecolor("white")
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
            cell.set_text_props(ha="left" if c == 0 else "center")
            cell.set_edgecolor("#111827")
            cell.set_linewidth(1.3)
            continue
        data_r = r - 1
        cell.get_text().set_color(_GRAY)           # body default: soft gray
        if c == 0:                                 # symbol: dark, bold, left
            cell.set_text_props(ha="left")
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
        if data_r == total_row:                    # TOTAL: dark bold + dark top rule
            cell.visible_edges = "TB"
            cell.set_edgecolor("#111827")
            cell.set_linewidth(1.1)
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
        if c in sign_idx:                          # PnL / %Chg: green/red by sign, bold
            raw = df.iloc[data_r][disp.columns[c]]
            try:
                num = float(raw)
            except (TypeError, ValueError):
                num = 0.0
            if num != 0:
                cell.get_text().set_color("#047857" if num > 0 else "#b91c1c")
                cell.get_text().set_fontweight("bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Risk panel (v1): exposure + risk-to-level (entry -> nearer of the adverse JASON pivot or
# 2-stdev target) + wrong-way flags
# --------------------------------------------------------------------------- #
PLUTO_URL = JASON_LEVELS_URL   # pivots + z-scores
# v1, deliberately NOT jason_levels_v2: v2 publishes only on COMPLETED 4h candles and then
# lags further, running 1-2 candles (4-12h) behind — measured 2026-08-27 15:59 UTC, v2 read
# BTC 79,440 against a live 80,329 (SOL was 5% out). No parameter reaches past it: dropping
# end_date, dropping both dates, a future end_date and every lookback from 8h to 240h all
# returned the same stale candle. v1 uses the in-progress candle and ties to the live tick.
# v1 also takes BARE ccypairs — no BINANCEF. prefix. Re-migrate once Pluto fixes v2's cadence.
_Z_FLAG = 1.5                          # |z| beyond this = JASON says extended against the position
_GREEN, _RED, _PURPLE = "#047857", "#b91c1c", "#7c3aed"
RISK_COLS = ["Sym", "Net $", "Entry", "Mark", "uPnL", "Level", "Risk→Lvl", "VaR95", "Flag"]
RISK_EXCLUDE_SYMS = {"SOL"}   # hedges an unrelated book — not this desk's directional risk

# 1-day historical VaR: empirical quantile of the book revalued on each of the last VAR_LOOKBACK_D
# days' returns. Historical (not parametric) because crypto tails are far from normal. 95% not 99%:
# on a ~1y sample the 99th percentile rests on 2-3 observations and is noise.
VAR_CONF = 0.05                # 5th percentile -> 95% VaR
VAR_LOOKBACK_D = 365           # calendar days of daily bars to request
VAR_MIN_DAYS = 180             # a symbol with less usable history is dropped from VaR (renders —)
VAR_MAX_WORKERS = 8            # parallel bar fetches (serial would add ~N round-trips to /hedge)


def _num(x: object) -> float | None:
    try:
        return float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _adverse_level(long: bool, mark: float, hp: float | None, lp: float | None,
                    up_target: float | None, dn_target: float | None) -> tuple[float | None, str | None]:
    """Adverse level = whichever candidate (JASON pivot or 2-stdev target) on the position's
    losing side sits CLOSER to mark, i.e. the level price would reach first.
    long -> downside candidates (low pivot, down-2σ target); short -> upside candidates
    (high pivot, up-2σ target). Missing candidates are skipped; if none remain -> (None, None).
    A candidate already on the wrong side of mark is still eligible if it is genuinely nearest."""
    cands = [(lp, "P"), (dn_target, "2σ")] if long else [(hp, "P"), (up_target, "2σ")]
    cands = [(v, tag) for v, tag in cands if v is not None and not pd.isna(v)]
    if not cands:
        return None, None
    return min(cands, key=lambda vt: abs(vt[0] - mark))


def jason_pivots(symbols: list[str]) -> dict[str, tuple]:
    """{sym: (high_pivot, low_pivot, z_up, z_dn, up_2std_target, dn_2std_target)} from the Pluto
    jason_levels API (chunked). Missing/uncovered names are simply absent."""
    if not symbols:
        return {}
    headers = {"X-Api-Key": require_env("PLUTO_API_KEY"), "X-Api-Secret": require_env("PLUTO_API_SECRET")}
    pairs = [f"{s}USDT" for s in symbols]
    now = datetime.now(timezone.utc)
    # NO end_date ON PURPOSE. Sending it AT ALL truncates the study window to that day's 00:00,
    # whatever the format: epoch seconds, epoch millis and ISO all returned the midnight candle
    # (2026-08-31 07:19 UTC -> end 00:00, lastPrice 77,757 against a live 78,216), and %d%b%y only
    # reached 04:00. Omitted, the API runs to the CURRENT minute and lastPrice ties to the live
    # tick. start_date still sets the lookback exactly (2/7/10/20d all honoured).
    base_params = {"start_date": int((now - timedelta(days=10)).timestamp()),
                   "exp_weight": "NONE",
                   "hist_vol_calc": "OCHL", "pivot": "COMPOSITE", "candle_size": "4h",
                   "alt_format": "true"}
    out: dict[str, tuple] = {}
    for i in range(0, len(pairs), 45):                    # API stalls on very large ccypairs batches
        chunk = ",".join(pairs[i:i + 45])
        try:
            r = pluto_get(PLUTO_URL, headers=headers, timeout=90,
                             params={**base_params, "ccypairs": chunk})
            recs = r.json()
        except (requests.exceptions.RequestException, ValueError):
            continue
        for x in recs if isinstance(recs, list) else []:
            p = str(x.get("assetOrCurrencyPair", ""))
            tok = p[:-4] if p.endswith("USDT") else p
            out[tok] = (_num(x.get("highPivotPrice")), _num(x.get("lowPivotPrice")),
                        _num(x.get("zScoreOfUpMoveFromLowPivot")),
                        _num(x.get("zScoreOfDownMoveFromHighPivot")),
                        _num(x.get("targetPricUp2StdevsFromLowPivot")),
                        _num(x.get("targetPriceDown2StdevsFromHighPivot")))
    return out


def hedge_trend(cfg: dict, symbols: list[str]) -> dict[str, float | None]:
    """{sym: long-horizon ls trend score in [-100, 100]} on 4h bars (cross-sectional over the
    hedge book); None where history is too short. Degrades per-symbol, never raises."""
    now = pd.Timestamp.now("UTC").tz_localize(None)
    bars4: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            h4 = fetch_recent_bars(cfg, f"{s.lower()}usdt", "4h", now - pd.Timedelta(days=130), now)
            h4 = h4[h4.index < now.floor("4h")]           # drop the in-progress candle
            if not h4.empty:
                bars4[s] = h4
        except Exception as exc:  # noqa: BLE001 - drop a bad symbol, never crash the report
            _log.warning("trend bars failed for %s: %s", s, exc)
    if not bars4:
        return {}
    sc = trend_scores(bars4, freq="4h", window_days=90)
    return {s: (None if s not in sc.index or pd.isna(sc.loc[s, "long"])
                else float(sc.loc[s, "long"]) * 100.0) for s in symbols}


def _agg_positions(cr_token: str, desk: str) -> pd.DataFrame:
    """Per-symbol net futures-book position: signed qty + $, |qty|-weighted entry/mark, summed uPnL."""
    raw = CoinRoutes(cr_token).get_positions(CR_STRATEGIES[desk])
    if raw is None or raw.empty:
        return pd.DataFrame()
    raw = raw[~raw["exchange"].isin(EXCLUDE_EXCHANGES[desk])].copy()
    for c in ("quantity", "entry_price", "position_mark_price", "usd_value", "unrealized_pnl"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["aw"] = raw["quantity"].abs()
    rows = []
    for sym, g in raw.groupby("quantity_currency"):
        qty, w = g["quantity"].sum(), g["aw"].sum()
        if qty == 0 or w == 0:
            continue
        rows.append({"sym": sym, "qty": qty, "notional": g["usd_value"].sum(),
                     "entry": (g["entry_price"] * g["aw"]).sum() / w,
                     "mark": (g["position_mark_price"] * g["aw"]).sum() / w,
                     "upnl": g["unrealized_pnl"].sum()})
    return pd.DataFrame(rows)


def daily_returns(cfg: dict, symbols: list[str]) -> pd.DataFrame:
    """{date x symbol} daily close-to-close returns over VAR_LOOKBACK_D. Fetched in parallel
    (serial would add one round-trip per symbol to every /hedge). Symbols whose bars fail or
    are shorter than VAR_MIN_DAYS are omitted — the caller renders them as `—`."""
    now = pd.Timestamp.now("UTC").tz_localize(None)
    start = now - pd.Timedelta(days=VAR_LOOKBACK_D + 15)     # slack for holidays//gaps

    def one(sym: str) -> tuple[str, pd.Series | None]:
        try:
            bars = fetch_recent_bars(cfg, f"{sym.lower()}usdt", "1d", start, now)
            bars = bars[bars.index < now.floor("1d")]        # drop the in-progress candle
            rets = bars["close"].pct_change().dropna()
            return sym, (rets if len(rets) >= VAR_MIN_DAYS else None)
        except Exception as exc:  # noqa: BLE001 - drop a bad symbol, never crash the report
            _log.warning("VaR bars failed for %s: %s", sym, exc)
            return sym, None

    with ThreadPoolExecutor(max_workers=VAR_MAX_WORKERS) as ex:
        series = {s: r for s, r in ex.map(one, symbols) if r is not None}
    return pd.DataFrame(series).dropna() if series else pd.DataFrame()


def book_var(pos: pd.DataFrame, rets: pd.DataFrame) -> tuple[dict[str, float], float | None,
                                                             float | None, int]:
    """1-day historical VaR. Returns (per-symbol VaR, portfolio VaR, portfolio ES, n days).

    Portfolio VaR revalues TODAY's book on each historical day's returns —
    ``pnl[t] = sum_i signed_notional_i * r[t, i]`` — so cross-name correlation is captured by
    using the same day's contemporaneous returns. Per-symbol VaR is that name standalone, so the
    two deliberately do NOT sum; the gap is the book's diversification/hedge benefit.
    Sign convention matches uPnL/Risk→Lvl: negative = loss."""
    if rets.empty:
        return {}, None, None, 0
    notional = {r["sym"]: r["qty"] * r["mark"] for _, r in pos.iterrows() if r["sym"] in rets.columns}
    if not notional:
        return {}, None, None, 0
    per_sym = {s: float(np.quantile(rets[s] * n, VAR_CONF)) for s, n in notional.items()}
    pnl = sum(rets[s] * n for s, n in notional.items())
    var = float(np.quantile(pnl, VAR_CONF))
    tail = pnl[pnl <= var]
    return per_sym, var, (float(tail.mean()) if not tail.empty else None), len(pnl)


def build_risk(cr_token: str, desk: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Risk rows for the futures book. `risk` = P&L if price travels from entry to the adverse
    level, chosen as whichever of the JASON pivot or 2-stdev target sits CLOSER to mark on the
    position's losing side (long -> nearer of low pivot / down-2σ target; short -> nearer of high
    pivot / up-2σ target; see `_adverse_level`). `flag` = position opposes BOTH JASON
    mean-reversion (|z| >= _Z_FLAG against us) AND the ls trend sign. Excludes
    `RISK_EXCLUDE_SYMS` before totals are computed."""
    pos = _agg_positions(cr_token, desk)
    pos = pos[~pos["sym"].isin(RISK_EXCLUDE_SYMS)] if not pos.empty else pos
    if pos.empty:
        return pd.DataFrame(), {}
    symbols = pos["sym"].tolist()
    piv, trend = jason_pivots(symbols), hedge_trend(cfg, symbols)
    rows = []
    for _, p in pos.iterrows():
        long = p["qty"] > 0
        hp, lp, z_up, z_dn, up_t, dn_t = piv.get(p["sym"], (None, None, None, None, None, None))
        adverse, src = _adverse_level(long, p["mark"], hp, lp, up_t, dn_t)
        risk = p["qty"] * (adverse - p["entry"]) if adverse is not None else None
        tr, z = trend.get(p["sym"]), (z_up if long else z_dn)
        jason_wrong = z is not None and z >= _Z_FLAG
        trend_wrong = tr is not None and ((long and tr < 0) or (not long and tr > 0))
        rows.append({**p, "pivot": adverse, "pivot_src": src, "risk": risk, "trend": tr, "z": z,
                     "flag": bool(jason_wrong and trend_wrong)})
    df = pd.DataFrame(rows)
    per_var, pf_var, pf_es, n_days = book_var(pos, daily_returns(cfg, symbols))
    df["var"] = df["sym"].map(per_var)
    df = (df.assign(_r=df["risk"].fillna(0.0))
            .sort_values(["flag", "_r"], ascending=[False, True])
            .drop(columns="_r").reset_index(drop=True))
    totals = {"gross": pos["notional"].abs().sum(), "net": pos["notional"].sum(),
              "risk": df["risk"].dropna().sum(), "upnl": df["upnl"].sum(),
              "flagged": df.loc[df["flag"], "sym"].tolist(),
              "var": pf_var, "es": pf_es, "var_days": n_days}
    return df, totals


def render_risk_image(df: pd.DataFrame, totals: dict, desk: str) -> bytes:
    """Format-B risk table: exposure, entry/mark/uPnL, adverse level (pivot or 2σ target),
    risk-to-level, wrong-way flag."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def money(v: object) -> str:
        return "" if v is None or pd.isna(v) else f"{v:,.0f}"

    def price(v: object) -> str:
        return "—" if v is None or pd.isna(v) else _fmt_num(v)

    def level(v: object, src: object) -> str:
        return "—" if v is None or pd.isna(v) else f"{_fmt_num(v)} {src}"

    def var_cell(v: object) -> str:
        return "—" if v is None or pd.isna(v) else f"{v:,.0f}"

    body = [[r["sym"], money(r["notional"]), price(r["entry"]), price(r["mark"]), money(r["upnl"]),
             level(r["pivot"], r["pivot_src"]), money(r["risk"]) if r["risk"] is not None else "—",
             var_cell(r.get("var")), "⚠" if r["flag"] else ""] for _, r in df.iterrows()]
    body.append(["TOTAL", money(totals["net"]), "", "", money(totals["upnl"]), "",
                 money(totals["risk"]), var_cell(totals.get("var")), ""])
    signed = {"Net $": [r["notional"] for _, r in df.iterrows()] + [totals["net"]],
              "uPnL": [r["upnl"] for _, r in df.iterrows()] + [totals["upnl"]],
              "Risk→Lvl": [r["risk"] for _, r in df.iterrows()] + [totals["risk"]],
              "VaR95": [r.get("var") for _, r in df.iterrows()] + [totals.get("var")]}
    cols, nrows = RISK_COLS, len(body)
    total_row = nrows - 1

    fig, ax = plt.subplots(figsize=(1.3 * len(cols), 0.5 + 0.34 * (nrows + 1)))
    ax.axis("off")
    ax.set_title(f"{desk.upper()} Hedge Risk — {stamp}", fontsize=13, fontweight="bold",
                 loc="left", pad=12, color=_DARK)
    tbl = ax.table(cellText=body, colLabels=cols, cellLoc="right", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(range(len(cols)))
    for cell in tbl.get_celld().values():
        cell.PAD = 0.22
        cell.set_width(cell.get_width() * 1.55)
    tbl.scale(1, 1.34)

    for (r, c), cell in tbl.get_celld().items():
        col = cols[c]
        cell.visible_edges = "B"
        cell.set_edgecolor("#e5e7eb")
        if r == 0:
            cell.set_facecolor("white")
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
            cell.set_text_props(ha="left" if c == 0 else "center")
            cell.set_edgecolor("#111827")
            cell.set_linewidth(1.3)
            continue
        dr = r - 1
        cell.get_text().set_color(_GRAY)
        if c == 0:
            cell.set_text_props(ha="left")
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
        if dr == total_row:
            cell.visible_edges = "TB"
            cell.set_edgecolor("#111827")
            cell.set_linewidth(1.1)
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
        if col == "Flag" and body[dr][c]:
            cell.get_text().set_color(_PURPLE)
            cell.get_text().set_fontweight("bold")
        elif col in signed:
            v = signed[col][dr]
            if v is not None and not pd.isna(v) and v != 0:
                cell.get_text().set_color(_GREEN if v > 0 else _RED)
                if dr != total_row:
                    cell.get_text().set_fontweight("bold")

    if totals.get("var") is not None:
        # Per-symbol VaR is standalone, so the column does NOT sum to the TOTAL — without this
        # note the difference reads as an arithmetic bug rather than the hedge benefit it is.
        es = totals.get("es")
        foot = (f"VaR95 = 1d 95% historical, {totals.get('var_days', 0)}d sample"
                + (f" · book ES {es:,.0f}" if es is not None else "")
                + " · per-symbol VaR is standalone and does not sum to the book total")
        ax.text(0, -0.04, foot, transform=ax.transAxes, fontsize=7, color=_GRAY, va="top")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def risk_caption(desk: str, tot: dict) -> str:
    """The ``/hedge`` risk-panel caption: gross/net/risk-to-level, VaR/ES when available, and
    wrong-way flags. Lifted out of ``main()`` so a caller other than Telegram (the_operator's
    ``hedge_risk`` tool) gets byte-identical text without going through chat."""
    cap = (f"{desk.upper()} risk · gross ${tot['gross']:,.0f} · net ${tot['net']:,.0f}"
          f" · risk-to-level ${tot['risk']:,.0f}")
    if tot.get("var") is not None:
        cap += f" · VaR 95% ${tot['var']:,.0f}"
        if tot.get("es") is not None:
            cap += f" · ES ${tot['es']:,.0f}"
    if tot["flagged"]:
        cap += "\n⚠ wrong-way: " + ", ".join(tot["flagged"])
    return cap


# --------------------------------------------------------------------------- #
# Telegram bot
# --------------------------------------------------------------------------- #
def main() -> None:
    token = require_env("TELEGRAM_BOT_TOKEN")
    # Explicit chat -> desk (no defaulting): a chat is a desk ONLY if pinned in its own var.
    mm3 = {c.strip() for c in require_env("TELEGRAM_MM3_CHAT_ID").split(",") if c.strip()}
    mm2 = {c.strip() for c in (os.environ.get("TELEGRAM_MM2_CHAT_ID") or "").split(",") if c.strip()}
    if mm3 & mm2:
        raise RuntimeError(f"chat(s) {sorted(mm3 & mm2)} pinned to multiple desks — fix .env")
    desk_by_chat = {**{c: "mm3" for c in mm3}, **{c: "mm2" for c in mm2}}
    cfg = load_config("data/binance.yaml")       # for the risk panel's 4h trend bars
    api = f"https://api.telegram.org/bot{token}"

    offset = drain_backlog(api)  # so a restart doesn't replay old commands
    _log.info("hedge_bot started; serving /hedge to chats %s", sorted(desk_by_chat))
    while True:
        updates, offset = poll_updates(api, offset)
        for upd in updates:
            msg = upd.get("message") or {}
            text = (msg.get("text") or "").strip()
            chat_id = str((msg.get("chat") or {}).get("id"))
            if not text or text.split()[0].split("@")[0].lower() != "/hedge":
                continue
            desk = desk_by_chat.get(chat_id)
            if desk is None:
                send(api, chat_id, "Not authorized.")
                _log.warning("rejected /hedge from unauthorized chat %s", chat_id)
                continue
            cr_token = require_env(_CR_TOKEN_ENV[desk])      # this desk's CoinRoutes key
            send(api, chat_id, f"Generating {desk.upper()} Hedge PnL report…")
            try:
                report = build_report(cr_token, desk)
                caption = f"Total PnL: {report.iloc[-1]['total_pnl']:,.0f}"
                send_photo(api, chat_id, render_table_image(report, desk), caption=caption)
            except Exception as exc:  # noqa: BLE001 - report over a chat, never crash the bot
                _log.exception("report failed")
                send(api, chat_id, f"Report failed: {exc}")
            try:
                rdf, tot = build_risk(cr_token, desk, cfg)
                if rdf.empty:
                    send(api, chat_id, "No futures-book positions.")
                else:
                    cap = risk_caption(desk, tot)
                    send_photo(api, chat_id, render_risk_image(rdf, tot, desk), caption=cap)
            except Exception as exc:  # noqa: BLE001 - report over a chat, never crash the bot
                _log.exception("risk panel failed")
                send(api, chat_id, f"Risk panel failed: {exc}")


if __name__ == "__main__":
    main()
