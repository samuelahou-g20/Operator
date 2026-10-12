"""Telegram bot that serves the LS cross-universe trend ranking on demand.

Long-polls Telegram; on ``/ls`` it computes each ticker's short- and long-horizon trend
score (JasonAnalytics Trend Table methodology: 7d / 28d half-life, on 4h bars, cross-sectionally
normalized over the universe) plus the JASON support/resistance z-scores (z_up, z_dn on 1h bars)
from Binance USDⓈ-M fapi REST klines, and replies with a styled PNG table sorted most-long ->
most-short. Mirrors ``scripts/hedge_bot.py`` (same long-poll / authorized-chat / PNG plumbing).

The trend score is the exact dashboard Trend Table score (src.signals.jason_trend_score): six
JASON/OHLC factors, each a half-life-weighted EWMA, cross-sectionally normalized over the universe
and averaged into an Average Score in [-1, 1] (displayed x100). z_up/z_dn are the JASON metrics
from src.signals.jason, exactly as src.monitor.features.compute_features surfaces them.

Env (loaded from repo-root .env):
  TELEGRAM_LS_BOT_TOKEN  its OWN BotFather bot — Telegram allows only one long-poller per
                         token, so this must differ from hedge_bot's TELEGRAM_BOT_TOKEN.
  TELEGRAM_CHAT_ID       authorized chat/group id(s); comma-separated to serve several.

Run: ``uv run python -m scripts.ls_bot``

MUST NOT: place/modify orders or touch the data lake — read-only reporting only.
Keep UNIVERSE in sync with the agent's LS universe (.claude/agents/trading-advisor.md, §0).
"""

from __future__ import annotations

import io
from datetime import datetime, timezone

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless: render to PNG, no display
matplotlib.rcParams["font.family"] = "Liberation Sans"  # clean, professional sans (house style)
import matplotlib.pyplot as plt  # noqa: E402 - must follow backend selection

from src.common.config import load_config, require_env  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.common.telegram import drain_backlog, poll_updates, send, send_photo  # noqa: E402
from src.data.fetching.binance.backfill import _get_json  # noqa: E402
from src.data.fetching.binance.fetcher import fetch_recent_bars  # noqa: E402
from src.signals.jason import compute_jason_vectorized_cs  # noqa: E402
from src.signals.jason_trend_score import trend_scores  # noqa: E402

_log = get_logger("ls_bot")

# 1h bars drive the JASON support/resistance z-scores + 2-week realized vol.
ST_TF, ST_WINDOW, ST_DAYS = "1h", 336, 20      # ~2 weeks
MIN_BARS = 30                                  # below this vol is not scored
# JASON z_up/z_dn on 1h bars — matches the dashboard / compute_features.
JASON_FREQ, JASON_LOOKBACK, JASON_BARS = "1h", 10, 320
# Trend score = JasonAnalytics Trend Table methodology on 4h bars (src.signals.jason_trend_score):
# short = 7d half-life, long = 28d half-life, cross-sectionally normalized over the universe.
# Need window_days + JASON lookback of 4h history; fetch a margin beyond that.
TREND_TF, TREND_DAYS, TREND_WINDOW_DAYS = "4h", 130, 90

# Regime (3x3 grid on ST/LT trend scores), resting-state delta, and $ sizing.
PORTFOLIO_NAV = 700_000                          # $; RSD / VS-RSD are $ = pct * NAV
VS_RSD_VOL_REF = 0.50                            # VS-RSD scales RSD by ref_vol / instrument_vol
RSD_BY_REGIME = {1: -0.05, 2: -0.03, 3: 0.0, 4: 0.07, 5: 0.09}  # regime -> resting-state delta
_HOURS_PER_YEAR = 24 * 365                       # 1h-bar vol annualization (perps trade 24/7)

# The LS watchlist (keep in sync with the agent's §0). DODOUSDT is not listed on fapi;
# SKHYNIXUSDT is hynix (SKHYUSDT is a different, near-empty symbol).
UNIVERSE = [
    "COMPUSDT", "ZECUSDT", "CCUSDT", "XRPUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "AAVEUSDT",
    "BCHUSDT", "PAXGUSDT", "BNBUSDT", "NEARUSDT", "ICPUSDT", "UNIUSDT",
    "PENDLEUSDT", "RUNEUSDT", "LDOUSDT", "JTOUSDT", "POLUSDT", "SKYUSDT", "JUPUSDT", "VVVUSDT",
    "HYPEUSDT", "TAOUSDT", "BILLUSDT", "MAGMAUSDT", "GENIUSUSDT", "INJUSDT", "AEROUSDT", 
    "PARTIUSDT", "BTWUSDT", "CLUSDT", "XAGUSDT",
    "GOOGLUSDT", "METAUSDT", "AMZNUSDT", "MSFTUSDT", "NVDAUSDT", "AAPLUSDT", "TSLAUSDT",
    "SKHYNIXUSDT",
]

# CLARITY Act event basket — section 3 of the G20 note (20 names, ranked by unpriced delta).
# Any name not listed on fapi is skipped at runtime and reported in the caption.
CLARITY = [
    "UNIUSDT", "AAVEUSDT", "ENAUSDT", "SKYUSDT", "LDOUSDT", "ONDOUSDT", "HYPEUSDT", "JUPUSDT",
    "SOLUSDT", "ETHFIUSDT", "AEROUSDT", "XRPUSDT", "HBARUSDT", "RAYUSDT", "CAKEUSDT", "XLMUSDT",
    "ADAUSDT", "LINKUSDT", "PUMPUSDT", "ETHUSDT",
]

# Selectable universes for the /ls command: "/ls" (default) or "/ls clarity".
UNIVERSES = {"default": UNIVERSE, "clarity": CLARITY}


# --------------------------------------------------------------------------- #
# Trend + JASON ranking
# --------------------------------------------------------------------------- #
def _jason_z(bars: pd.DataFrame, ticker: str) -> tuple[float | None, float | None]:
    """JASON support/resistance z-scores (z_up, z_dn) on 1h bars — same call as
    compute_features. Needs the trailing lookback window; returns (None, None) if a
    ticker is too new or JASON errors, so one bad symbol never sinks the table."""
    if bars.empty or len(bars) < JASON_LOOKBACK * 24 + 1:
        return None, None
    try:
        feat = compute_jason_vectorized_cs(
            bars[["open", "high", "low", "close"]].tail(JASON_BARS).copy(),
            ticker.upper(), JASON_FREQ, lookback=JASON_LOOKBACK, asset_class="crypto",
        )
    except Exception as exc:  # noqa: BLE001 - degrade, never crash the report
        _log.warning("jason failed for %s: %s", ticker, exc)
        return None, None
    if feat.empty:
        return None, None
    last = feat.iloc[-1]
    zu, zd = last.get("z_up"), last.get("z_dn")
    return (None if pd.isna(zu) else float(zu)), (None if pd.isna(zd) else float(zd))


def _signal(lt: float | None, z_up: float | None, z_dn: float | None) -> str:
    """Actionable L/S tag — trade with the (LT) trend when price sits well for it.

    L = uptrend (LT>0) and price near support (z_up<1) OR far below resistance (z_dn>1.5).
    S = downtrend (LT<0) and price near resistance (z_dn<1) OR far above support (z_up>1.5).
    Empty when the trend or either z is unavailable, or the level condition isn't met."""
    if lt is None or z_up is None or z_dn is None:
        return ""
    if lt > 0 and (z_up < 1.0 or z_dn > 1.5):
        return "L"
    if lt < 0 and (z_dn < 1.0 or z_up > 1.5):
        return "S"
    return ""


def _ann_vol(close: pd.Series, window: int) -> float | None:
    """Annualized realized vol from the trailing `window` 1h log returns (fraction, e.g. 0.62)."""
    if len(close) < MIN_BARS + 1:
        return None
    log_ret = np.log(close.iloc[-window - 1:]).diff().dropna()
    sigma = float(log_ret.std())
    return sigma * np.sqrt(_HOURS_PER_YEAR) if sigma > 0 else None


def _regime(st: float | None, lt: float | None) -> float | None:
    """Continuous regime in [1, 5] from the ST/LT trend scores: 3 + (st + lt)/100. Matches the old
    discrete anchors (both -100 -> 1, both +100 -> 5, centre -> 3) but varies smoothly."""
    if st is None or lt is None:
        return None
    return min(5.0, max(1.0, 3.0 + (st + lt) / 100.0))


_REGIME_PTS = (1, 2, 3, 4, 5)


def _rsd_pct(regime: float) -> float:
    """RSD % for a (possibly fractional) regime — piecewise-linear over the RSD_BY_REGIME anchors."""
    return float(np.interp(regime, _REGIME_PTS, [RSD_BY_REGIME[k] for k in _REGIME_PTS]))


def _rsd_usd(regime: float | None) -> float | None:
    """Resting-state delta in $ = RSD% * NAV."""
    return None if regime is None else _rsd_pct(regime) * PORTFOLIO_NAV


def _vs_rsd_usd(regime: float | None, vol_ann: float | None) -> float | None:
    """Vol-scaled RSD in $ = RSD% * (ref_vol / instrument_vol) * NAV."""
    if regime is None or not vol_ann or vol_ann <= 0:
        return None
    return _rsd_pct(regime) * (VS_RSD_VOL_REF / vol_ann) * PORTFOLIO_NAV


def build_ranking(
    cfg: dict, universe: list[str] = UNIVERSE
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Compute ST/LT trend + JASON z for the universe; return (ranked_df, skipped, short_history)."""
    rest = cfg["markets"]["perps"]["rest_url"]
    info = _get_json(rest, "/fapi/v1/exchangeInfo", {})
    live = {
        s["symbol"] for s in info["symbols"]
        if s["status"] == "TRADING" and s["contractType"] in ("PERPETUAL", "TRADIFI_PERPETUAL")
    }

    now = pd.Timestamp.now("UTC").tz_localize(None)
    min_trend_bars = TREND_WINDOW_DAYS * 6      # 6 4h-bars/day; below this the score is short-history

    # Pass 1: fetch bars per ticker (1h drives z/vol/price; 4h feeds the cross-sectional trend score).
    bars4: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict] = {}
    skipped = []
    for ticker in universe:
        if ticker not in live:
            skipped.append(ticker)
            continue
        sym = ticker.lower()
        h1 = fetch_recent_bars(cfg, sym, ST_TF, now - pd.Timedelta(days=ST_DAYS), now)
        h4 = fetch_recent_bars(cfg, sym, TREND_TF, now - pd.Timedelta(days=TREND_DAYS), now)
        h4 = h4[h4.index < now.floor(TREND_TF)]   # drop the in-progress candle (closed-bar parity)
        z_up, z_dn = _jason_z(h1, ticker)
        meta[ticker] = {
            "price": float(h1["close"].iloc[-1]) if not h1.empty else None,
            "z_up": z_up, "z_dn": z_dn,
            "vol_2w": _ann_vol(h1["close"], ST_WINDOW) if not h1.empty else None,
            "n4": len(h4),
        }
        if not h4.empty:
            bars4[ticker] = h4

    # Cross-sectional trend scores over the whole universe (short = 7d HL, long = 28d HL).
    scores = trend_scores(bars4, freq=TREND_TF, window_days=TREND_WINDOW_DAYS) if bars4 else None

    def _score(ticker: str, col: str) -> float | None:
        if scores is None or ticker not in scores.index:
            return None
        v = scores.loc[ticker, col]
        return None if pd.isna(v) else float(v) * 100.0   # [-1,1] -> [-100,100] for display/regime

    # Pass 2: assemble rows.
    rows, short_hist = [], []
    for ticker in universe:
        if ticker in skipped:
            continue
        m = meta[ticker]
        st, lt = _score(ticker, "short"), _score(ticker, "long")
        regime = _regime(st, lt)
        is_short = lt is not None and m["n4"] < min_trend_bars
        if is_short:
            short_hist.append(ticker)
        rows.append({"ticker": ticker.replace("USDT", ""), "Price": m["price"], "LT6w": lt, "ST2w": st,
                     "Zup": m["z_up"], "Zdn": m["z_dn"], "Vol2w": m["vol_2w"], "Regime": regime,
                     "RSD": _rsd_usd(regime), "VSRSD": _vs_rsd_usd(regime, m["vol_2w"]),
                     "MaxL": None, "MaxS": None,
                     "Sig": _signal(lt, m["z_up"], m["z_dn"]), "hist": is_short})

    df = pd.DataFrame(rows)
    df["_r"] = df["Regime"].fillna(-1)        # rank by regime (most-long first)...
    df["_lt"] = df["LT6w"].fillna(-1e9)       # ...LT trend breaks ties; unscored sinks to bottom
    df = (df.sort_values(["_r", "_lt"], ascending=False)
            .drop(columns=["_r", "_lt"]).reset_index(drop=True))
    df.insert(0, "#", df.index + 1)
    return df, skipped, short_hist


# --------------------------------------------------------------------------- #
# PNG rendering
# --------------------------------------------------------------------------- #
DISPLAY = ["#", "ticker", "Price", "LT6w", "ST2w", "Zup", "Zdn", "Vol2w", "Regime",
           "RSD", "VSRSD", "MaxL", "MaxS", "Sig"]
COL_HEADERS = {"#": "#", "ticker": "Ticker", "Price": "Price", "LT6w": "LT 28d", "ST2w": "ST 7d",
               "Zup": "Z up", "Zdn": "Z dn", "Vol2w": "Vol 2w", "Regime": "Regime",
               "RSD": "RSD", "VSRSD": "VS-RSD", "MaxL": "Max L", "MaxS": "Max S", "Sig": "Sig"}
HEAT_KIND = {"LT6w": "trend", "ST2w": "trend", "Zup": "zup", "Zdn": "zdn"}
_DARK_COLS = {"#", "ticker", "Price", "Regime"}   # dark (not gray) body text


def _fmt_price(v: float) -> str:
    """Comma-separated price; decimals scale with magnitude, no scientific notation."""
    a = abs(v)
    dec = 0 if a >= 1000 else 2 if a >= 1 else 4 if a >= 0.01 else 6
    return f"{v:,.{dec}f}"


def _fmt(v: object, col: str) -> str:
    if col in ("MaxL", "MaxS"):          # placeholders — intentionally blank for now
        return ""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "NA"
    if col == "Price":
        return _fmt_price(v)
    if col in ("LT6w", "ST2w"):
        return f"{v:+.0f}"
    if col in ("Zup", "Zdn"):
        return f"{v:.2f}"
    if col == "Vol2w":
        return f"{v * 100:.0f}%"
    if col == "Regime":
        return f"{v:.1f}"
    if col in ("RSD", "VSRSD"):
        return "$0" if round(v) == 0 else f"{'+' if v > 0 else '-'}${abs(v):,.0f}"
    return str(v)


_GRAY, _DARK = "#6b7280", "#111827"


def _heat_text(value: object, kind: str) -> tuple | None:
    """Text color for a numeric cell on a gray→green/red gradient scaled by magnitude.

    House style colors the *text*, not the whole cell — but keeps the /ls signal (how long/short,
    how stretched) in the color intensity. trend saturates at |v|=100; z at |z|=2. None when NA."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    green, red, gray = (4, 120, 87), (185, 28, 28), (107, 114, 128)
    if kind == "trend":
        t = min(abs(value) / 100.0, 1.0)
        target = green if value > 0 else red if value < 0 else gray
    else:  # zup / zdn: proximity to 2
        t = min(abs(value) / 2.0, 1.0)
        target = green if kind == "zup" else red
    return tuple((gray[i] * (1 - t) + target[i] * t) / 255.0 for i in range(3))


def render_table_image(df: pd.DataFrame, universe_name: str = "default") -> bytes:
    """Render the ranking as a styled PNG (house style: Liberation Sans, horizontal rules, soft-gray
    body, dark-bold rank/ticker, green/red text on the trend/z/signal cells)."""
    disp = df.copy()
    disp["ticker"] = disp.apply(lambda r: r["ticker"] + ("*" if r["hist"] else ""), axis=1)
    text = pd.DataFrame({c: [_fmt(v, c) for v in disp[c]] for c in DISPLAY})
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    nrows, ncols = text.shape

    label = "" if universe_name == "default" else f" · {universe_name}"
    fig, ax = plt.subplots(figsize=(1.35 * ncols, 0.55 + 0.3 * (nrows + 1)))
    ax.axis("off")
    ax.set_title(f"LS trend ranking{label} — {stamp}", fontsize=13, fontweight="bold",
                 loc="left", pad=12, color=_DARK)
    tbl = ax.table(cellText=text.values, colLabels=[COL_HEADERS[c] for c in DISPLAY],
                   cellLoc="right", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(range(ncols))
    for cell in tbl.get_celld().values():          # horizontal breathing room
        cell.PAD = 0.22
        cell.set_width(cell.get_width() * 1.55)
    tbl.scale(1, 1.34)

    for (r, c), cell in tbl.get_celld().items():
        col = DISPLAY[c]
        cell.visible_edges = "B"                   # horizontal rules only
        cell.set_edgecolor("#e5e7eb")
        if r == 0:                                 # header: bold, dark bottom rule, no fill
            cell.set_facecolor("white")
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
            cell.set_text_props(ha="left" if col == "ticker" else "center")
            cell.set_edgecolor("#111827")
            cell.set_linewidth(1.3)
            continue
        data_r = r - 1
        cell.get_text().set_color(_GRAY)           # body default: soft gray
        if col in _DARK_COLS:                      # rank/ticker/regime/$ deltas: dark
            cell.get_text().set_color(_DARK)
            if col == "ticker":
                cell.set_text_props(ha="left")
                cell.get_text().set_fontweight("bold")
        elif col in HEAT_KIND:                     # trend/z: green/red text by magnitude + bold
            tc = _heat_text(df.iloc[data_r][col], HEAT_KIND[col])
            if tc is not None:
                cell.get_text().set_color(tc)
                cell.get_text().set_fontweight("bold")
        elif col in ("RSD", "VSRSD"):              # $ deltas: green long / red short, bold
            v = df.iloc[data_r][col]
            if v is not None and not pd.isna(v) and v != 0:
                cell.get_text().set_color("#047857" if v > 0 else "#b91c1c")
                cell.get_text().set_fontweight("bold")
        elif col == "Sig":                         # L/S tag: green L, red S, bold
            cell.set_text_props(ha="center")
            sig = df.iloc[data_r]["Sig"]
            if sig in ("L", "S"):
                cell.get_text().set_color("#047857" if sig == "L" else "#b91c1c")
                cell.get_text().set_fontweight("bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _caption(skipped: list[str], short_hist: list[str]) -> str:
    parts = [
        "ST=7d-HL · LT=28d-HL trend (4h, JASON Trend-Table x100) · Z=JASON 1h/10d · Vol=2w ann · ranked by regime",
        "Regime 1–5 (continuous): 3 + (ST+LT)/100",
        "RSD = resting-state Δ by regime (1:−5% 2:−3% 3:0% 4:+7% 5:+9%)",
        "VS-RSD = RSD × 50% / Vol",
        f"Assumptions: NAV ${PORTFOLIO_NAV:,.0f} · baseline vol {VS_RSD_VOL_REF:.0%}",
    ]
    if short_hist:
        parts.append("* short history (<6w), less comparable: " + ", ".join(short_hist))
    if skipped:
        parts.append("not listed on fapi: " + ", ".join(skipped))
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Telegram bot  (plumbing shared via src.common.telegram)
# --------------------------------------------------------------------------- #
def main() -> None:
    token = require_env("TELEGRAM_LS_BOT_TOKEN")
    authorized = {c.strip() for c in require_env("TELEGRAM_CHAT_ID").split(",") if c.strip()}
    cfg = load_config("data/binance.yaml")
    api = f"https://api.telegram.org/bot{token}"

    offset = drain_backlog(api)  # so a restart doesn't replay old commands
    _log.info("ls_bot started; serving /ls to chats %s", sorted(authorized))
    while True:
        updates, offset = poll_updates(api, offset)
        for upd in updates:
            msg = upd.get("message") or {}
            text = (msg.get("text") or "").strip()
            chat_id = str((msg.get("chat") or {}).get("id"))
            parts = text.split()
            cmd = parts[0].split("@")[0].lower() if parts else ""  # case-insensitive, /ls@bot ok
            if cmd != "/ls":
                continue
            if chat_id not in authorized:
                send(api, chat_id, "Not authorized.")
                _log.warning("rejected /ls from unauthorized chat %s", chat_id)
                continue
            uni_name = parts[1].lower() if len(parts) > 1 else "default"  # "/ls clarity" etc.
            if uni_name not in UNIVERSES:
                send(api, chat_id, f"Unknown universe '{uni_name}'. Options: {', '.join(UNIVERSES)}")
                continue
            send(api, chat_id, f"Computing LS trend ranking ({uni_name})…")
            try:
                df, skipped, short_hist = build_ranking(cfg, UNIVERSES[uni_name])
                send_photo(api, chat_id, render_table_image(df, uni_name))  # table image, then...
                send(api, chat_id, _caption(skipped, short_hist))          # ...the note, separately
            except Exception as exc:  # noqa: BLE001 - report over a chat, never crash the bot
                _log.exception("ls report failed")
                send(api, chat_id, f"LS report failed: {exc}")


if __name__ == "__main__":
    main()
