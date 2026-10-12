"""Telegram bot: RV>IV breach alerts (push) + a ``/vol`` Deribit implied-vol table (on demand).

Long-polls ``getUpdates`` so it can serve ``/vol`` — a table of Deribit ATM implied vols
(term structure per underlying, from Pluto's ``vsource=DERIBIT`` feed) — and ``/vol report`` —
the one-page-per-pair PDF volatility snapshot (``scripts.vol_report``) — while, on a 15-minute
timer, it still pushes the realized-vs-implied breach alerts described below.

Polls the Pluto ``TimeSeries/query`` API every 15 minutes and compares each name's
**2-week historical/realized vol** (``{sym}_H2W``) against its **1-week ATM implied
vol** (``{sym}_ATM_1W``), pushing a clean notification when

    RV > (1 + ALPHA) * IV      (ALPHA = 0.10 -> RV more than 10% above IV)

A name is announced when it crosses INTO breach, again every ``REMIND_SECONDS``
(4h) while it stays there, and once (a "recovered" note) when it drops back out.
It is deliberately NOT announced every 15-minute poll: a lasting dislocation is
worth repeating occasionally, not spamming. Edge-triggered alone was too quiet --
XRP sat in breach a whole morning after a single 07:27 message.

Universe (``UNIVERSE``)
-----------------------
Only names with a **genuine per-name VOLPUB implied-vol surface**. VOLPUB returns a
*bucketed fallback IV* for the rest — groups of unrelated names share one identical
IV series (e.g. 18 names all track 0.544) — and comparing a real RV to a proxy IV
would fire constant false breaches. This list was derived from the LS baskets
(``ls_bot.UNIVERSE`` + ``ls_bot.CLARITY``) by keeping only the names whose ATM_1W
series is unique.

Refresh ``UNIVERSE`` when VOLPUB coverage changes: fetch each candidate's
``{sym}_ATM_1W`` over ~15 days; names whose full series is identical to another's
are bucketed proxies (drop), unique series are real (keep). One-off manual check,
like the repo's other live-API probes — not a pytest.

Env (loaded from repo-root .env):
  TELEGRAM_VOL_BOT_TOKEN   its OWN BotFather bot (create it, then /start it from the
                           target chat). This bot now long-polls getUpdates for /vol, so the
                           token must be unique — only one long-poller per token (else 409).
  TELEGRAM_CHAT_ID         authorized chat/group id(s); comma-separated to serve several.
  PLUTO_API_KEY / PLUTO_API_SECRET   Pluto TimeSeries API credentials.

Run: ``uv run python -m scripts.vol_bot``

MUST NOT: place/modify orders or touch the data lake. Read-only polling + notify.
"""

from __future__ import annotations

import io
import threading
import time
from datetime import datetime, timedelta, timezone

import matplotlib
import requests
import urllib3

matplotlib.use("Agg")  # headless: render to PNG, no display
matplotlib.rcParams["font.family"] = "Liberation Sans"  # clean, professional sans (house style)
import matplotlib.pyplot as plt  # noqa: E402 - must follow backend selection

from scripts.vol_report import PAIRS, build_report  # noqa: E402
from src.common.config import require_env  # noqa: E402
from src.common.deadline import run_with_deadline  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.common.paths import REPO_ROOT  # noqa: E402
from src.common.telegram import drain_backlog, poll_updates, single_instance_lock  # noqa: E402
from src.common.pluto import JASON_LEVELS_URL, TIMESERIES_URL, get as pluto_get  # noqa: E402

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # internal host, self-signed

_GRAY, _DARK = "#6b7280", "#111827"     # house-style body / label colors

_log = get_logger("vol_bot")

PLUTO_URL = TIMESERIES_URL
JASON_URL = JASON_LEVELS_URL   # realized/pivot vols
# v1, deliberately NOT jason_levels_v2: v2 publishes only on COMPLETED 4h candles and then
# lags further, running 1-2 candles (4-12h) behind — measured 2026-08-27 15:59 UTC, v2 read
# BTC 79,440 against a live 80,329 (SOL was 5% out). No parameter reaches past it: dropping
# end_date, dropping both dates, a future end_date and every lookback from 8h to 240h all
# returned the same stale candle. v1 uses the in-progress candle and ties to the live tick.
# v1 also takes BARE ccypairs — no BINANCEF. prefix. Re-migrate once Pluto fixes v2's cadence.
ALPHA = 0.10                 # breach when RV > (1 + ALPHA) * IV
REMIND_SECONDS = 4 * 3600    # re-alert a name still in breach this often (0 = edge-triggered only)
POLL_SECONDS = 15 * 60
HSOURCE, VSOURCE = "BINANCE", "VOLPUB"
RV_SUFFIX, IV_SUFFIX = "_H2W", "_ATM_1W"    # 2W realized vs 1W ATM implied
_CHUNK_BASES = 15            # symbols per request (URL length + API latency guard)
_LOOKBACK_DAYS = 4           # request a small window; use each name's latest published row
# MUST stay "1d". An intraday candle does NOT return the Deribit ATM quote: at 15:59 UTC the 1h
# series read SOL ATM_1W 0.6144 (and its own real hours 0.63-0.66) while the 1d series read 0.6887
# — the Deribit surface viewer showed ~68% at the 9d tenor, i.e. the 1d value. Freshness comes from
# _END_PAD below, NOT from the candle size; a finer candle only changes what is being measured.
_CANDLE = "1d"

# end_date is TOMORROW, not today. The API reads a bare `%d%b%y` date as that day's 00:00, so an
# end_date of today truncates the series at MIDNIGHT and the "latest published row" is up to 24h
# stale — measured at 15:45 UTC, BTC ATM 1W read 0.4017 against a live 0.4383, and SOL ATM 1M read
# 0.7972 against 0.6235. Rows stamped past `now` repeat the live value, so asking for tomorrow is
# what puts it in range; "latest published row wins" then lands on it.
_END_PAD = timedelta(days=1)

# Names with a real (unique) VOLPUB IV surface — see module docstring for derivation
# and refresh. Bucketed-proxy names carry a fallback IV and must not be added blindly.
UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

# /vol table: Deribit ATM implied-vol surface (vsource=DERIBIT). ONLY names Pluto's DERIBIT feed
# actually carries — querying uncovered names makes the endpoint hang, so this must stay
# confirmed-covered (verified live). Re-probe before adding a name. HYPE read 0.0 across all tenors
# as of 2026-07-24; Pluto now publishes it (57.3/58.4/60.3 on 2026-08-25), but unlike every other
# name it did NOT move between the midnight and the live row in that probe — so treat HYPE as
# possibly daily-only until it is seen updating intraday.
VOL_UNIVERSE = ["BTC", "ETH", "SOL", "XRP", "AVAX", "HYPE"]
_VOL_TENORS = [("1W", "ATM_1W"), ("2W", "ATM_2W"), ("1M", "ATM_1M")]
_VOL_CHUNK = 3               # underlyings/request (×6 tenors = 18 fields; endpoint is slow on more)


def _pluto_sym(ticker: str) -> str:
    """Binance-perp ticker -> Pluto symbol: BTCUSDT -> BTCUSD, XAGUSDT -> XAGUSD."""
    return ticker[:-4] + "USD" if ticker.endswith("USDT") else ticker


def _base(ticker: str) -> str:
    """Display label: BTCUSDT -> BTC."""
    return ticker[:-4] if ticker.endswith("USDT") else ticker


# --------------------------------------------------------------------------- #
# Pluto TimeSeries fetch
# --------------------------------------------------------------------------- #
def _fetch_vols() -> dict[str, tuple[float, float]]:
    """Return {display_base: (rv, iv)} for every trusted name with published data.

    Requests are chunked (the API 404s on very long symbol lists and is slow per
    name). For each name the most recent row carrying both a realized and an
    implied vol is used; names with no usable data are omitted.
    """
    headers = {"X-Api-Key": require_env("PLUTO_API_KEY"),
               "X-Api-Secret": require_env("PLUTO_API_SECRET")}
    today = datetime.now(timezone.utc)
    params_base = {
        "start_date": (today - timedelta(days=_LOOKBACK_DAYS)).strftime("%d%b%y").upper(),
        "end_date": (today + _END_PAD).strftime("%d%b%y").upper(),
        "format": "json", "periodicity": _CANDLE, "candle": _CANDLE,
        "hsource": HSOURCE, "vsource": VSOURCE,
    }
    out: dict[str, tuple[float, float]] = {}
    for i in range(0, len(UNIVERSE), _CHUNK_BASES):
        chunk = UNIVERSE[i:i + _CHUNK_BASES]
        fields = []
        for t in chunk:
            p = _pluto_sym(t)
            fields += [f"{p}{RV_SUFFIX}", f"{p}{IV_SUFFIX}"]
        params = {**params_base, "symbols": ",".join(fields)}
        resp = pluto_get(PLUTO_URL, headers=headers, params=params, timeout=90)
        resp.raise_for_status()
        rows = resp.json()
        if not isinstance(rows, list):
            raise ValueError(f"unexpected TimeSeries payload: {type(rows).__name__}")
        for t in chunk:
            p = _pluto_sym(t)
            rv = iv = None
            for row in reversed(rows):                     # latest published row wins
                r, v = row.get(f"{p}{RV_SUFFIX}"), row.get(f"{p}{IV_SUFFIX}")
                if r and v:                                # non-null and non-zero
                    rv, iv = float(r), float(v)
                    break
            if rv is not None and iv is not None:
                out[_base(t)] = (rv, iv)
    return out


# --------------------------------------------------------------------------- #
# Notification
# --------------------------------------------------------------------------- #
def _fmt_rows(names: list[str], vols: dict[str, tuple[float, float]]) -> str:
    """Monospace table body: ticker, RV%, IV%, ratio — sorted richest-RV first."""
    rows = sorted(names, key=lambda b: -(vols[b][0] / vols[b][1]))
    lines = [f"{b:<7} RV {vols[b][0] * 100:>4.0f}%  IV {vols[b][1] * 100:>4.0f}%  ×{vols[b][0] / vols[b][1]:.2f}"
             for b in rows]
    return "<pre>" + "\n".join(lines) + "</pre>"


def _fetch_iv_surface() -> dict[str, dict[str, float]]:
    """{base: {tenor_label: iv}} — the Deribit ATM surface for VOL_UNIVERSE names that have data."""
    headers = {"X-Api-Key": require_env("PLUTO_API_KEY"),
               "X-Api-Secret": require_env("PLUTO_API_SECRET")}
    today = datetime.now(timezone.utc)
    base = {"start_date": (today - timedelta(days=_LOOKBACK_DAYS)).strftime("%d%b%y").upper(),
            "end_date": (today + _END_PAD).strftime("%d%b%y").upper(), "format": "json",
            "periodicity": _CANDLE, "candle": _CANDLE, "hsource": HSOURCE, "vsource": "DERIBIT"}
    out: dict[str, dict[str, float]] = {}
    for i in range(0, len(VOL_UNIVERSE), _VOL_CHUNK):
        chunk = VOL_UNIVERSE[i:i + _VOL_CHUNK]
        fields = [f"{u}USD_{suf}" for u in chunk for _, suf in _VOL_TENORS]
        rows = None
        for _ in range(3):                                 # the DERIBIT feed is slow / flaky
            try:
                r = pluto_get(PLUTO_URL, headers=headers,
                                 params={**base, "symbols": ",".join(fields)}, timeout=90)
                r.raise_for_status()
                rows = r.json()
                break
            except requests.exceptions.RequestException:
                time.sleep(3)
        if not isinstance(rows, list):
            continue
        for u in chunk:
            tenors: dict[str, float] = {}
            for lab, suf in _VOL_TENORS:
                key = f"{u}USD_{suf}"
                for row in reversed(rows):                  # latest published row wins
                    v = row.get(key)
                    if v:
                        tenors[lab] = float(v)
                        break
            if tenors:
                out[u] = tenors
    return out


def _fetch_jason_vols(bases: list[str]) -> dict[str, tuple]:
    """{base: (vol_up, vol_dn, period_vol)} from the JASON levels API for the given underlyings."""
    headers = {"X-Api-Key": require_env("PLUTO_API_KEY"),
               "X-Api-Secret": require_env("PLUTO_API_SECRET")}
    pairs = [f"{b}USDT" for b in bases]
    now = datetime.now(timezone.utc)
    # NO end_date ON PURPOSE. Sending it AT ALL truncates the study window to that day's 00:00,
    # whatever the format: epoch seconds, epoch millis and ISO all returned the midnight candle
    # (2026-08-31 07:19 UTC -> end 00:00, lastPrice 77,757 against a live 78,216), and %d%b%y only
    # reached 04:00. Omitted, the API runs to the CURRENT minute and lastPrice ties to the live
    # tick. start_date still sets the lookback exactly (2/7/10/20d all honoured).
    params = {"start_date": int((now - timedelta(days=10)).timestamp()),
              "exp_weight": "NONE",
              "hist_vol_calc": "OCHL", "pivot": "COMPOSITE", "candle_size": "4h",
              "alt_format": "true",
              "ccypairs": ",".join(pairs)}
    recs: object = []
    for _ in range(3):
        try:
            r = pluto_get(JASON_URL, headers=headers, params=params, timeout=90)
            r.raise_for_status()
            recs = r.json()
            break
        except requests.exceptions.RequestException:
            time.sleep(3)
    out: dict[str, tuple] = {}
    for rec in recs if isinstance(recs, list) else []:
        pair = str(rec.get("assetOrCurrencyPair", ""))
        base = pair[:-4] if pair.endswith("USDT") else pair[:-3] if pair.endswith("USD") else pair
        out[base] = (rec.get("volatilitySinceLastLowPivotAnnualized"),
                     rec.get("volatilitySinceLastHighPivotAnnualized"),
                     rec.get("volatilityForWholePeriodAnnualized"))
    return out


def _pct_whole(v: object) -> str:
    """JASON vol_up/vol_dn arrive as a whole-percent string (e.g. '38.03 %')."""
    try:
        return f"{float(str(v).replace('%', '').replace(',', '').strip()):.0f}%"
    except (TypeError, ValueError):
        return "—"


def _pct_frac(v: object) -> str:
    """JASON period vol arrives as a raw fraction (e.g. 0.328)."""
    try:
        return f"{float(v) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


# matplotlib's pyplot module keeps a global "current figure" stack that isn't thread-safe;
# /vol handlers now run on background threads (see main()), so renders must be serialized.
# Timed out (not plain `with`): a past incident saw one render hang and never release the
# lock, which silently deafened every future /vol forever (systemd still saw the process as
# "running" — the unrelated breach-poll thread kept ticking) until someone found it and sent
# SIGKILL. A stuck render must fail loud instead of wedging the bot.
_render_lock = threading.Lock()
_RENDER_TIMEOUT = 60

# Outer ceiling on the Pluto fetch steps, on top of their own retry logic (which already budgets
# up to ~14min of legitimate retries): requests' timeout= doesn't bound total call duration on a
# stalling server (see src/common/deadline.py) — this is the backstop that was missing.
_FETCH_DEADLINE = 300
_REPORT_DEADLINE = 600
_HEARTBEAT_SECONDS = 300


def render_surface_png(surface: dict[str, dict[str, float]], jvols: dict[str, tuple]) -> bytes:
    """House-style PNG (Liberation Sans, horizontal rules, soft-gray body, dark-bold labels):
    underlyings (rows) × ATM tenors + JASON vols (vol up/dn, period), sorted by 1M IV."""
    labels = [lab for lab, _ in _VOL_TENORS]
    order = sorted(surface, key=lambda u: -surface[u].get("1M", 0.0))
    headers = ["Und", *labels, "Vol up", "Vol dn", "Period"]
    cell_text = []
    for u in order:
        ivs = [f"{surface[u][lab] * 100:.0f}%" if lab in surface[u] else "—" for lab in labels]
        vu, vd, pv = jvols.get(u, (None, None, None))
        cell_text.append([u, *ivs, _pct_whole(vu), _pct_whole(vd), _pct_frac(pv)])
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    fig, ax = plt.subplots(figsize=(1.15 * len(headers), 0.55 + 0.32 * (len(order) + 1)))
    ax.axis("off")
    ax.set_title(f"Deribit ATM implied vol · {stamp}", fontsize=12, fontweight="bold",
                 loc="left", pad=10, color=_DARK)
    tbl = ax.table(cellText=cell_text, colLabels=headers, cellLoc="right", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(range(len(headers)))
    for cell in tbl.get_celld().values():
        cell.PAD = 0.22
        cell.set_width(cell.get_width() * 1.5)
    tbl.scale(1, 1.34)
    for (r, c), cell in tbl.get_celld().items():
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
        cell.get_text().set_color(_GRAY)
        if c == 0:                                     # underlying label: dark, bold, left
            cell.set_text_props(ha="left")
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _send(api: str, chat_id: str, text: str) -> None:
    try:
        requests.post(f"{api}/sendMessage",
                      data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=30)
    except requests.exceptions.RequestException as e:
        _log.warning("telegram send failed: %s", e)


def _send_photo(api: str, chat_id: str, png: bytes, caption: str | None = None) -> None:
    data = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption
    try:
        requests.post(f"{api}/sendPhoto", data=data,
                      files={"photo": ("vol.png", png, "image/png")}, timeout=30)
    except requests.exceptions.RequestException as e:
        _log.warning("telegram sendPhoto failed: %s", e)


def _send_doc(api: str, chat_id: str, path: str, caption: str | None = None) -> None:
    data = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption
    try:
        with open(path, "rb") as fh:
            requests.post(f"{api}/sendDocument", data=data,
                          files={"document": ("vol_report.pdf", fh, "application/pdf")}, timeout=60)
    except (requests.exceptions.RequestException, OSError) as e:
        _log.warning("telegram sendDocument failed: %s", e)


def _broadcast(api: str, chats: set[str], text: str) -> None:
    for chat_id in chats:
        _send(api, chat_id, text)


# --------------------------------------------------------------------------- #
# Command handlers — each runs on its own background thread (see main()) so a slow/hung
# Pluto call blocks only that one reply, never the getUpdates loop or the breach-poll timer.
# --------------------------------------------------------------------------- #
def _handle_vol_report(api: str, chat_id: str) -> None:
    _log.info("/vol report requested by chat %s", chat_id)
    _send(api, chat_id, f"Building volatility report ({len(PAIRS)} pairs)… ~2 min.")
    try:
        pdf = run_with_deadline(build_report, PAIRS, str(REPO_ROOT), deadline=_REPORT_DEADLINE)
        _send_doc(api, chat_id, pdf, "Daily volatility snapshot")
    except TimeoutError as exc:
        _log.error("/vol report failed: %s", exc)
        _send(api, chat_id, f"/vol report failed: {exc} — flag to ops.")
    except Exception as exc:  # noqa: BLE001 - report over the chat, never crash the bot
        _log.exception("/vol report failed")
        _send(api, chat_id, f"/vol report failed: {exc}")


def _handle_vol(api: str, chat_id: str) -> None:
    _log.info("/vol requested by chat %s", chat_id)
    _send(api, chat_id, "Fetching Deribit implied vols…")
    try:
        surface = run_with_deadline(_fetch_iv_surface, deadline=_FETCH_DEADLINE)
        if surface:
            jvols = run_with_deadline(_fetch_jason_vols, list(surface), deadline=_FETCH_DEADLINE)
            if not _render_lock.acquire(timeout=_RENDER_TIMEOUT):
                _log.error("/vol render lock stuck for >%ds — a prior render never released it",
                           _RENDER_TIMEOUT)
                _send(api, chat_id, "/vol failed: renderer is stuck, flag to ops.")
                return
            try:
                png = render_surface_png(surface, jvols)
            finally:
                _render_lock.release()
            _send_photo(api, chat_id, png)
        else:
            _send(api, chat_id, "No Deribit implied-vol data available right now.")
    except TimeoutError as exc:
        _log.error("/vol failed: %s — Pluto isn't responding", exc)
        _send(api, chat_id, f"/vol failed: {exc} — Pluto isn't responding, flag to ops.")
    except Exception as exc:  # noqa: BLE001 - report over the chat, never crash the bot
        _log.exception("/vol failed")
        _send(api, chat_id, f"/vol failed: {exc}")


# --------------------------------------------------------------------------- #
# Poll loop
# --------------------------------------------------------------------------- #
def _due_reminder(alerted: dict[str, float], name: str, now: float) -> bool:
    """Has this name been in breach, unmentioned, for REMIND_SECONDS?

    Timed off the LAST MESSAGE about the name, not off when it entered breach, so the reminder
    cadence stays a clean 4h whether the name just crossed or has been dislocated all day."""
    return REMIND_SECONDS > 0 and now - alerted[name] >= REMIND_SECONDS


def _breach_cycle(api: str, chats: set[str], alerted: dict[str, float], started: bool,
                  now: float | None = None) -> tuple[dict[str, float], bool]:
    """One RV>IV breach check. Returns (name -> when it was last announced, started).

    A name is announced when it crosses INTO breach, again every REMIND_SECONDS while it stays
    there, and once when it drops out. The reminder exists because a dislocation that lasts is the
    interesting one: edge-triggered alone, XRP sat in breach for a whole morning after a single
    07:27 message."""
    vols = run_with_deadline(_fetch_vols, deadline=_FETCH_DEADLINE)
    now = time.time() if now is None else now
    if not started:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        _broadcast(api, chats, f"📈 <b>vol_bot live</b> · {stamp}\n"
                   f"watching {len(vols)} names · alert RV(2W) &gt; {1 + ALPHA:.2f}×IV(1W ATM)"
                   f" · repeat every {REMIND_SECONDS // 3600}h · /vol for the surface")
        started = True
    current = {b for b, (rv, iv) in vols.items() if rv > (1 + ALPHA) * iv}
    stamp = datetime.now(timezone.utc).strftime("%H:%M UTC")

    new_breach = sorted(current - set(alerted))
    if new_breach:
        _broadcast(api, chats, f"⚠️ <b>RV &gt; IV breach</b> · {stamp} · α={ALPHA:.2f}\n"
                   + _fmt_rows(new_breach, vols))
    still = sorted(b for b in current if b in alerted and _due_reminder(alerted, b, now))
    if still:
        _broadcast(api, chats, f"⏳ <b>RV &gt; IV still breached</b> · {stamp} · α={ALPHA:.2f}\n"
                   + _fmt_rows(still, vols))
    recovered = sorted(set(alerted) - current)
    if recovered:
        _broadcast(api, chats, f"✅ <b>RV/IV recovered</b> · {stamp}\n" + _fmt_rows(recovered, vols))

    announced = set(new_breach) | set(still)
    _log.info("breach poll ok: %d names, %d breached %s%s", len(vols), len(current),
              sorted(current), f" (announced {sorted(announced)})" if announced else "")
    return {b: (now if b in announced else alerted[b]) for b in current}, started


def _breach_loop(api: str, chats: set[str]) -> None:
    """The 15-minute RV>IV push, on its OWN thread.

    This ran inline in main()'s poll loop until 2026-08-25, which is why vol_bot — alone among the
    command bots — went silently deaf "after some time": a Pluto stall inside _fetch_vols blocked
    the loop, so getUpdates stopped being called and every command was ignored until Pluto answered.
    The journal shows the signature plainly (breach polls simply stopping for 105min, then resuming
    with no restart). jason_bot cannot fail this way because it never blocks its loop on anything.
    The poll loop must only poll and dispatch; everything slow belongs on a thread with a deadline.
    """
    alerted: dict[str, float] = {}     # name -> when it was last announced
    started = False
    while True:
        try:
            alerted, started = _breach_cycle(api, chats, alerted, started)
        except TimeoutError as exc:
            _log.error("vol_bot breach poll: %s — Pluto stalled; skipping this cycle", exc)
        except Exception:  # noqa: BLE001 - a bad poll must never kill the thread
            _log.exception("vol_bot breach poll failed")
        time.sleep(POLL_SECONDS)


def main() -> None:
    lock = single_instance_lock("vol_bot")
    if lock is None:
        _log.error("another vol_bot instance is already running — refusing to start "
                   "(a duplicate poller causes Telegram 409s and silent deafness)")
        return
    token = require_env("TELEGRAM_VOL_BOT_TOKEN")
    chats = {c.strip() for c in require_env("TELEGRAM_CHAT_ID").split(",") if c.strip()}
    api = f"https://api.telegram.org/bot{token}"

    offset = drain_backlog(api)  # so a restart doesn't replay old commands

    _log.info("vol_bot started; serving /vol + RV>IV breach push every %dm "
              "(alpha=%.2f, repeat every %dh)", POLL_SECONDS // 60, ALPHA, REMIND_SECONDS // 3600)
    threading.Thread(target=_breach_loop, args=(api, chats), daemon=True).start()

    # Heartbeat: the poll loop's own liveness, logged independently of the breach thread. Without
    # it a frozen loop is invisible (systemd still reports "running") — the exact way this bot went
    # deaf unnoticed for hours.
    last_beat = 0.0

    while True:
        updates, offset = poll_updates(api, offset, poll_timeout=25)  # serve /vol commands
        if time.time() - last_beat >= _HEARTBEAT_SECONDS:
            last_beat = time.time()
            _log.info("poll loop alive (offset=%s)", offset)
        try:
            for upd in updates:
                msg = upd.get("message") or {}
                text = (msg.get("text") or "").strip()
                chat_id = str((msg.get("chat") or {}).get("id"))
                parts = text.split()
                cmd = parts[0].split("@")[0].lower() if parts else ""
                if cmd != "/vol":
                    continue
                if chat_id not in chats:
                    _send(api, chat_id, "Not authorized.")
                    continue
                arg = parts[1].lower() if len(parts) > 1 else ""
                # Dispatch to a background thread: build_report / _fetch_iv_surface /
                # _fetch_jason_vols retry against Pluto for minutes at a time when it's slow (seen
                # live as ConnectTimeout at 90-120s per attempt). Handling them inline would stall
                # this single-threaded loop's next poll_updates() call, making the bot look alive
                # (systemd, breach-poll timer) while actually deaf to every command for as long as
                # the stuck request runs.
                target = _handle_vol_report if arg == "report" else _handle_vol
                threading.Thread(target=target, args=(api, chat_id), daemon=True).start()
        except Exception:  # noqa: BLE001 - poll_updates and _breach_cycle each guard themselves;
            # this dispatch loop was the one gap — an uncaught exception here exits main()
            # outright, with recovery resting entirely on systemd's restart burst limit.
            _log.exception("vol_bot update dispatch failed")


if __name__ == "__main__":
    main()
