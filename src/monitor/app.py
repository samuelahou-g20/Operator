"""The Textual dashboard app: one titled table per group, live-updating.

Two cadences: a fast timer polls :class:`LivePrices` for the Price/Updated cells
(and staleness), while an off-thread worker periodically recomputes the trend /
JASON / volatility cells via :func:`compute_features` (JASON is numba-heavy, so it
must not run on the UI thread). On start, after a short grace, it auto-starts a
streamer for any symbol nobody is publishing.

MUST NOT: read the lake or Redis directly, or compute signals — it renders what
``features`` and ``live`` provide.
"""

from __future__ import annotations

import time

import pandas as pd
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Footer, Header, Static

from src.monitor import streamers
from src.monitor.config import DashboardConfig, Instrument, load_dashboard
from src.monitor.features import _BAR_SECONDS, Features, compute_features
from src.monitor.live import LivePrices
from src.monitor.notify import TelegramNotifier

_CATCHUP_BARS = 720           # recent bars to REST-fetch on startup to bridge lake -> now (~30d @1h)
_HOURS_PER_YEAR = 365.25 * 24  # to convert annualized realized vol -> a per-window sigma

# (key, header, width) — widths are fixed because DataTable otherwise sizes to the
# header label and truncates wider cell values (live prices, ages, pivots).
_COLUMNS = [
    ("symbol", "Symbol", 12),
    ("price", "Price", 11),
    ("abn", "Abn Move", 11),
    ("trend_st", "Trend ST", 9),
    ("trend_lt", "Trend LT", 9),
    ("z_up", "Z_up", 6),
    ("z_dn", "Z_dn", 6),
    ("json2_up", "JASON2_up", 11),
    ("json2_dn", "JASON2_dn", 11),
    ("vol_st", "Vol ST", 7),
    ("vol_lt", "Vol LT", 7),
    ("updated", "Updated", 16),
    ("feat", "Feat @", 22),
]
_DASH = Text("—", style="dim")


def _row_key(inst: Instrument) -> str:
    return f"{inst.source}|{inst.symbol}"


def _sort_value(
    col: str, symbol: str, group: str,
    rk_by_sym: dict[tuple[str, str], str], nums: dict[str, dict[str, float | None]],
    none_val: float,
) -> object:
    """Sort key for one row: the symbol string for the symbol column, else the row's numeric
    mirror value for ``col`` (``none_val`` when missing, so blanks sort last)."""
    if col == "symbol":
        return symbol
    rk = rk_by_sym.get((group, symbol))
    val = nums.get(rk, {}).get(col) if rk else None
    return none_val if val is None else val


def _fmt_price(value: float) -> str:
    if value >= 1000:
        return f"{value:,.1f}"
    if value >= 1:
        return f"{value:,.3f}"
    return f"{value:.6f}"


def _fmt_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    return f"{int(seconds // 3600)}h ago"


def _trend_cell(score: float | None) -> Text:
    """Normalized trend score in [-100,+100] with a strength color gradient: hue = direction
    (green up / red down), saturation ramps neutral-grey (no trend) -> vivid (strong)."""
    if score is None:
        return _DASH
    arrow = "▲" if score > 0 else "▼" if score < 0 else "·"
    t = min(abs(score) / 100.0, 1.0)          # 0 = flat, 1 = max strength
    strong = int(120 + 135 * t)               # 120 (grey) -> 255 (vivid)
    weak = int(120 * (1 - t))                  # 120 (grey) -> 0
    r, g, b = (weak, strong, weak) if score >= 0 else (strong, weak, weak)
    return Text(f"{arrow} {abs(score):.0f}", style=f"rgb({r},{g},{b})")


def _z_cell(value: float | None) -> Text:
    """Color a JASON z-score by magnitude; negative = price beyond the pivot."""
    if value is None:
        return _DASH
    if value < 0:
        style = "bold magenta"
    elif value < 1:
        style = "green"
    elif value < 2:
        style = "yellow"
    else:
        style = "bold red"
    return Text(f"{value:+.2f}", style=style)


def _pivot_cell(value: float | None) -> Text:
    return _DASH if value is None else Text(_fmt_price(value), style="cyan")


def _vol_cell(value: float | None) -> Text:
    if value is None:
        return _DASH
    if value < 40:
        style = "green"
    elif value < 80:
        style = "yellow"
    else:
        style = "red"
    return Text(f"{value:.1f}", style=style)


def _sigma_from_vol(vol_pct: float | None, window_sec: float) -> float | None:
    """Expected size of a ``window_sec`` move implied by an annualized realized vol (%).

    This is the STABLE long-horizon benchmark (lake hourly vol, days of history): a sqrt-time
    projection is fine here because it is the calm reference the live move is judged against,
    not the twitchy short-scale estimate."""
    if vol_pct is None or vol_pct <= 0:
        return None
    return (vol_pct / 100.0) / (_HOURS_PER_YEAR ** 0.5) * ((window_sec / 3600.0) ** 0.5)


def _abn_score(
    move: float | None, sig_short: float | None, sig_long: float | None
) -> tuple[float | None, str | None]:
    """Score a directional move on both scales; return the more extreme (|z|, tag).

    tag "S" = significant vs the recent short-horizon distribution (this period),
    "L" = significant vs the stable long-horizon vol (a longer period). Either can trigger."""
    if move is None:
        return None, None
    cands = []
    if sig_short and sig_short > 0:
        cands.append((move / sig_short, "S"))
    if sig_long and sig_long > 0:
        cands.append((move / sig_long, "L"))
    if not cands:
        return None, None
    return max(cands, key=lambda c: abs(c[0]))


def _abn_cell(z: float | None, threshold: float, tag: str | None = None) -> Text:
    """Directional move in sigma units, tagged by the scale that drove it (S/L). Highlighted
    (reverse video) once |z| crosses the alert threshold; dim when quiet."""
    if z is None:
        return _DASH
    arrow = "▲" if z > 0 else "▼" if z < 0 else "·"
    mag = abs(z)
    txt = f"{arrow} {mag:.1f}σ" + (f" {tag}" if tag else "")
    if mag >= threshold:
        return Text(txt, style="bold reverse green" if z > 0 else "bold reverse red")
    if mag >= threshold * 0.6:
        return Text(txt, style="bold green" if z > 0 else "bold red")
    return Text(txt, style="dim")


def _feat_cell(bar_ts: pd.Timestamp | None) -> Text:
    """Timestamp of the bar the features were computed from — reveals stale-lake gaps."""
    if bar_ts is None:
        return _DASH
    age = (pd.Timestamp.utcnow().tz_localize(None) - bar_ts).total_seconds()
    label = bar_ts.strftime("%m-%d %H:%M")
    if age < 7200:
        return Text(label, style="green")
    if age < 86400:
        return Text(label, style="yellow")
    return Text(f"{label}  ({int(age // 86400)}d old)", style="bold red")


class DashboardApp(App):
    """Live monitoring dashboard over a grouped universe."""

    CSS = """
    Static.group-title { padding: 1 1 0 1; text-style: bold underline; color: $accent; }
    DataTable { height: auto; margin-bottom: 1; }
    """
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Recompute features"),
    ]

    def __init__(
        self,
        conf: DashboardConfig,
        *,
        feature_interval: float = 30.0,
        auto_stream: bool = True,
        stream_grace: float = 4.0,
    ) -> None:
        super().__init__()
        self._conf = conf
        self._feature_interval = feature_interval
        self._auto_stream = auto_stream
        self._stream_grace = stream_grace
        # Abnormal-move detection + alerting state.
        self._abn_window_sec = conf.abnormal_window_min * 60.0
        self._abn_lookback_sec = conf.abnormal_lookback_min * 60.0
        self._abn_z = conf.abnormal_z
        self._alert_cooldown_sec = conf.alert_cooldown_min * 60.0
        # Keep enough ticks to hold the sigma lookback plus the scored window (+margin).
        self._live = LivePrices(
            conf.instruments, tick_history_sec=self._abn_window_sec + self._abn_lookback_sec + 120,
        )
        self._last_alert: dict[str, float] = {}   # row_key -> monotonic time of last alert
        self._notifier: TelegramNotifier | None = None
        # Preserve group order and membership.
        self._groups: dict[str, list[Instrument]] = {}
        for inst in conf.instruments:
            self._groups.setdefault(inst.group, []).append(inst)
        self._tables: dict[str, DataTable] = {}
        # Numeric mirror of sortable cells (Text can't be ordered), the active per-group sort,
        # and a (group, SYMBOL) -> row_key map to recover a row from its symbol cell when sorting.
        self._nums: dict[str, dict[str, float | None]] = {}
        self._siglong: dict[str, float | None] = {}  # row_key -> stable long-horizon window sigma
        self._sort: dict[str, tuple[str, bool]] = {}
        self._rk_by_sym: dict[tuple[str, str], str] = {
            (g, i.symbol.upper()): _row_key(i)
            for g, insts in self._groups.items() for i in insts
        }

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll():
            for group in self._groups:
                yield Static(group, classes="group-title")
                table = DataTable(zebra_stripes=True)
                self._tables[group] = table
                yield table
        yield Footer()

    def on_mount(self) -> None:
        for group, insts in self._groups.items():
            table = self._tables[group]
            for key, label, width in _COLUMNS:
                table.add_column(label, key=key, width=width)
            blanks = [_DASH] * (len(_COLUMNS) - 1)
            for inst in insts:
                table.add_row(Text(inst.symbol.upper()), *blanks, key=_row_key(inst))
        self.sub_title = "click a column header to sort"
        self._notifier = TelegramNotifier()
        self._live.start()
        self.set_interval(0.5, self._refresh_live)
        self.set_interval(self._feature_interval, self.action_refresh)
        self.action_refresh()
        self._catch_up()
        if self._auto_stream:
            self.set_timer(self._stream_grace, self._autostart_streamers)

    @work(thread=True, group="catchup", exclusive=True)
    def _catch_up(self) -> None:
        """Seed recent bars (lake's last -> now) via REST so a fresh open is current-to-hour.

        binance -> klines REST; cme -> Databento continuous historical (current to ~last hour);
        mexc -> klines REST. All bridge a stale lake so live bars splice contiguously instead of
        being rejected.
        """
        from src.common import config as cfg
        from src.data.fetching.binance import fetcher as bfetch
        from src.data.fetching.cme import fetcher as cfetch
        from src.data.fetching.mexc import fetcher as mfetch

        now = pd.Timestamp.utcnow().tz_localize(None)
        confs: dict[str, dict] = {}
        for inst in self._conf.instruments:
            bar_seconds = _BAR_SECONDS.get(inst.timeframe)
            if bar_seconds is None or inst.source not in ("binance", "cme", "mexc"):
                continue
            start = now - pd.Timedelta(seconds=bar_seconds * _CATCHUP_BARS)
            conf = confs.setdefault(inst.source, cfg.load_config(f"data/{inst.source}.yaml"))
            try:
                if inst.source == "binance":
                    bars = bfetch.fetch_recent_bars(
                        conf, inst.symbol, inst.timeframe, start, now, market=inst.asset_type,
                    )
                elif inst.source == "cme":  # end at the last complete hour so Databento has it
                    bars = cfetch.fetch_recent_continuous(
                        conf, inst.symbol, inst.timeframe, start, now.floor("h"),
                    )
                else:  # mexc klines REST
                    bars = mfetch.fetch_recent_bars(
                        conf, inst.symbol, inst.asset_type, start, now,
                    )
            except Exception:  # noqa: BLE001 - offline / API hiccup: stay lake-as-of
                continue
            if not bars.empty:
                self._live.seed(inst.source, inst.symbol, inst.timeframe, bars)
        self.call_from_thread(self.action_refresh)

    def _autostart_streamers(self) -> None:
        missing: dict[str, list[str]] = {}
        for inst in self._conf.instruments:
            if self._live.latest(inst.source, inst.symbol) is None:
                missing.setdefault(inst.source, []).append(inst.symbol)
        if missing:
            streamers.start_missing(missing)

    def _set(self, table: DataTable, key: str, column: str, value: Text) -> None:
        table.update_cell(key, column, value)

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """Click a header to sort that group's table: first click descending, click again toggles."""
        group = next((g for g, t in self._tables.items() if t is event.data_table), None)
        if group is None:
            return
        col = event.column_key.value
        prev = self._sort.get(group)
        reverse = (not prev[1]) if (prev and prev[0] == col) else (col != "symbol")
        self._sort[group] = (col, reverse)
        self._apply_sort(group)

    def _apply_sort(self, group: str) -> None:
        """Re-order a group's rows by the active sort column, using the numeric mirror.

        The sort key only sees cell values, so we recover each row from its symbol cell
        (column 0, unique within a table) and look up its numeric value. None sorts last."""
        state = self._sort.get(group)
        if state is None:
            return
        col, reverse = state
        none_val = float("-inf") if reverse else float("inf")  # Nones last in both directions

        def key(cells: tuple) -> object:
            return _sort_value(
                col, cells[0].plain, group, self._rk_by_sym, self._nums, none_val,
            )

        self._tables[group].sort(key=key, reverse=reverse)

    def _refresh_live(self) -> None:
        now = pd.Timestamp.utcnow().tz_localize(None)
        for group, insts in self._groups.items():
            table = self._tables[group]
            for inst in insts:
                key = _row_key(inst)
                nums = self._nums.setdefault(key, {})
                tick = self._live.latest(inst.source, inst.symbol)
                if tick is None:
                    self._set(table, key, "price", _DASH)
                    self._set(table, key, "updated", Text("no stream", style="dim"))
                    nums["price"] = nums["updated"] = None
                    continue
                price, ts = tick
                age = (now - ts).total_seconds()
                stale = age > self._conf.stale_after
                pstyle = "dim" if stale else "white"
                self._set(table, key, "price", Text(_fmt_price(price), style=pstyle))
                age_txt = _fmt_age(age) + (" (stale)" if stale else "")
                self._set(table, key, "updated", Text(age_txt, style="red" if stale else "dim"))
                nums["price"], nums["updated"] = price, age
                self._update_abnormal(table, inst, key)

    def _update_abnormal(self, table: DataTable, inst: Instrument, key: str) -> None:
        """Score the directional window move on both scales; flag + alert when |z| crosses.

        Short scale = the recent empirical distribution of window-moves (this period); long
        scale = the stable lake vol projected to the window (a longer period). Either triggers."""
        move = self._live.recent_move(inst.source, inst.symbol, self._abn_window_sec)
        sig_short = self._live.move_sigma(
            inst.source, inst.symbol, self._abn_window_sec, self._abn_lookback_sec,
        )
        z, tag = _abn_score(move, sig_short, self._siglong.get(key))
        self._nums.setdefault(key, {})["abn"] = z
        self._set(table, key, "abn", _abn_cell(z, self._abn_z, tag))
        if z is not None and abs(z) >= self._abn_z:
            self._maybe_alert(inst, key, move, z, tag)

    def _maybe_alert(
        self, inst: Instrument, key: str, move: float, z: float, tag: str | None
    ) -> None:
        now = time.monotonic()
        if now - self._last_alert.get(key, -1e18) < self._alert_cooldown_sec:
            return
        self._last_alert[key] = now
        if self._notifier is not None:
            mins = int(self._abn_window_sec // 60)
            scale = {"S": "vs recent", "L": "vs typical"}.get(tag, "")
            self._notifier.send(
                f"⚠️ {inst.symbol.upper()} [{inst.group}] moved {move * 100:+.2f}% "
                f"in {mins}m ({z:+.1f}σ {scale})"
            )

    def action_refresh(self) -> None:
        self._compute_features()

    @work(thread=True, exclusive=True)
    def _compute_features(self) -> None:
        results = {
            _row_key(i): compute_features(
                i, live_bars=self._live.recent_bars(i.source, i.symbol, i.timeframe),
            )
            for i in self._conf.instruments
        }
        self.call_from_thread(self._apply_features, results)

    def _apply_features(self, results: dict[str, Features]) -> None:
        for group, insts in self._groups.items():
            table = self._tables[group]
            for inst in insts:
                key = _row_key(inst)
                f = results[key]
                self._set(table, key, "trend_st", _trend_cell(f.trend_st_score))
                self._set(table, key, "trend_lt", _trend_cell(f.trend_lt_score))
                self._set(table, key, "z_up", _z_cell(f.z_up))
                self._set(table, key, "z_dn", _z_cell(f.z_dn))
                self._set(table, key, "json2_up", _pivot_cell(f.json2_up))
                self._set(table, key, "json2_dn", _pivot_cell(f.json2_dn))
                self._set(table, key, "vol_st", _vol_cell(f.vol_st_pct))
                self._set(table, key, "vol_lt", _vol_cell(f.vol_lt_pct))
                self._set(table, key, "feat", _feat_cell(f.bar_ts))
                self._siglong[key] = _sigma_from_vol(f.vol_lt_pct, self._abn_window_sec)
                self._nums.setdefault(key, {}).update(
                    trend_st=f.trend_st_score, trend_lt=f.trend_lt_score,
                    z_up=f.z_up, z_dn=f.z_dn, json2_up=f.json2_up, json2_dn=f.json2_dn,
                    vol_st=f.vol_st_pct, vol_lt=f.vol_lt_pct,
                    feat=(f.bar_ts.value if f.bar_ts is not None else None),
                )
            self._apply_sort(group)


def _silence_tqdm() -> None:
    """Disable tqdm process-wide: JASON draws progress bars to stderr, which scribble
    over the Textual screen on every recompute and make cells look blank/garbled."""
    from functools import partialmethod

    import tqdm
    tqdm.tqdm.__init__ = partialmethod(tqdm.tqdm.__init__, disable=True)


def _quiet_console_logging() -> None:
    """Route logs to file only. Console handlers created before the app starts hold the
    real terminal stderr and bypass Textual's stderr capture, corrupting the screen;
    file handlers (RotatingFileHandler) are kept."""
    import logging
    from logging.handlers import RotatingFileHandler

    for name in ["", *logging.root.manager.loggerDict]:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
                logger.removeHandler(handler)


def run(config_path: str = "dashboard.yaml", *, auto_stream: bool = True) -> None:
    import os

    _silence_tqdm()
    os.environ["OPERATOR_LOG_FILE_ONLY"] = "1"  # loggers created later (streamer) -> file only
    _quiet_console_logging()                 # strip console handlers already created (monitor)
    DashboardApp(load_dashboard(config_path), auto_stream=auto_stream).run()
