"""Orchestrates one token's ZP: fetch (``sources``) -> compute (``compute``), with the
independent fetch sections run concurrently. ``checks.checks()`` is deliberately NOT called here
— it takes the result plus optional ``dmc``/``ticks`` inputs a caller may want to supply
separately (see its docstring), so the Operator tool built on top of ``run_zp`` calls it itself.

MUST NOT: place/modify/cancel orders, call Telegram, or write the lake.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pandas as pd

from src.common.logging import get_logger
from src.the_operator.zp import compute, config, sources
from src.the_operator.zp.compute import ZPResult

_log = get_logger("zp")

_MARKOUT_WINDOWS = [60, 120, 180, 300, 600]
# Markout reference resolution per reference venue, from the source report's MD_CONFIG: the bar
# size is the horizon's resolution, so a coarse bar makes every markout read long.
_MD_MARKOUT_INTERVAL = {"binance": "1s", "gate": "10s", "gateio": "10s"}
_FINE_INTERVAL_MAX_HOURS = 24     # 1s bars over a longer window is tens of thousands of candles
_GAMMA_DASH_HOURS = 24


def _fetch_market_volume(cfg: dict, start: datetime, end: datetime, warnings: list[str]) -> pd.DataFrame:
    """Per-venue 1-minute klines for every configured (pair, exchange), reshaped into the tidy
    ``(exchange, pair, timestamp, notional_usd)`` rows ``compute.captures`` expects
    (``notional_usd = volume * (O+H+L+C)/4``, per the reference `fetch_captures`). A venue whose
    klines fail is skipped with a warning, never fatal to the whole capture section."""
    rows = []
    for entry in cfg.get("exchanges", []):
        try:
            k = sources.fetch_klines(entry["exchange"], entry["pair"], "1m", start, end)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"market volume klines failed for {entry['exchange']} {entry['pair']}: {e}")
            continue
        if k is None or k.empty:
            continue
        k = k.reset_index()
        k["exchange"] = entry["exchange"]
        k["pair"] = entry["pair"]
        k["notional_usd"] = k["volume"] * (k["open"] + k["high"] + k["low"] + k["close"]) / 4
        rows.append(k[["timestamp", "exchange", "pair", "notional_usd"]])
    if not rows:
        return pd.DataFrame(columns=["timestamp", "exchange", "pair", "notional_usd"])
    return pd.concat(rows, ignore_index=True)


def run_zp(token: str, start: datetime, end: datetime, desk: str, *, want_ob: bool = True,
           want_dmc: bool = True, warnings: list[str] | None = None) -> ZPResult:
    """Run one token's ZP over ``[start, end]``. ``desk`` is the Operator's desk name (mm2/mm3/...)
    and doubles as the orderbook query's ``product_area``. The independent fetch sections (
    reference klines, orderbook snapshots, gamma-dash, per-venue market-volume klines) run in a
    ``ThreadPoolExecutor``; every section's failure becomes an entry in ``warnings``, never an
    exception — a partial ZP still comes back."""
    warnings = warnings if warnings is not None else []
    cfg = config.token_config(token)
    pairs = config.get_pairs(token)

    trades = sources.fetch_trades(token, start, end, warnings)

    def _md_klines():
        md_pair, md_exchange = cfg.get("md_pair"), cfg.get("md_exchange")
        if not md_pair or not md_exchange:
            warnings.append(f"no md_pair/md_exchange configured for {token}; skipping price OHLC/markouts.")
            return None
        try:
            return sources.fetch_klines(md_exchange, md_pair, "1m", start, end)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"reference market klines failed for {md_pair}@{md_exchange}: {e}")
            return None

    def _markout_klines():
        """The markout reference series, at the venue's finest useful interval.

        A markout reads the price at ``fill + N seconds`` off the bar ceil-aligned to that
        instant, so the bar size IS the horizon's resolution: on 1-minute bars a "60s markout" is
        really 60-120s and every horizon reads long. The source report uses 1s on binance and 10s
        on gate for exactly this reason."""
        md_pair, md_exchange = cfg.get("md_pair"), cfg.get("md_exchange")
        if not md_pair or not md_exchange:
            return None
        interval = _MD_MARKOUT_INTERVAL.get(str(md_exchange).lower(), "1m")
        hours = (end - start).total_seconds() / 3600
        if interval == "1s" and hours > _FINE_INTERVAL_MAX_HOURS:
            warnings.append(f"markouts on 1m bars (window {hours:.0f}h > {_FINE_INTERVAL_MAX_HOURS}h "
                            "is too long for 1s bars) — horizons are coarse, read them as ranges.")
            interval = "1m"
        try:
            return sources.fetch_klines(md_exchange, md_pair, interval, start,
                                        end + timedelta(seconds=max(_MARKOUT_WINDOWS)))
        except Exception as e:  # noqa: BLE001 - fall back to the 1m series the OHLC already needs
            warnings.append(f"markout reference klines ({interval}) failed: {e}; using 1m bars, "
                            "so the horizons are coarse.")
            return None

    def _ob():
        if not want_ob:
            return None
        return sources.fetch_ob_snapshots(pairs, desk, start, end, warnings)

    def _dmc():
        if not want_dmc:
            return None
        names = cfg.get("deployment_names") or [cfg.get("g20_name", token).upper()]
        # gamma-dash is richer (live vol history + the real ladder) so it wins when reachable;
        # it's SSO-gated from this host, so the quoting boxes' own settings are the fallback.
        dmc = sources.fetch_gamma_dash(names, _GAMMA_DASH_HOURS, start, end, warnings)
        if dmc is not None:
            return dmc
        return sources.fetch_strategy_params(token, desk, warnings)

    def _market_vol():
        return _fetch_market_volume(cfg, start, end, warnings)

    def _ticks():
        # Only useful next to the quoter's min spread, so it rides with the dmc section.
        return sources.fetch_ticksize(token, warnings) if want_dmc else None

    with ThreadPoolExecutor(max_workers=6) as pool:
        f_md, f_ob, f_dmc, f_mkt, f_tick, f_mo = (
            pool.submit(_md_klines), pool.submit(_ob), pool.submit(_dmc),
            pool.submit(_market_vol), pool.submit(_ticks), pool.submit(_markout_klines))
        md_klines, ob_raw, dmc_data, market_vol = f_md.result(), f_ob.result(), f_dmc.result(), f_mkt.result()
        ticks, mo_klines = f_tick.result(), f_mo.result()

    last_px = (float(trades.loc[trades["exch_ts"] == trades["exch_ts"].max(), "price_usd"].mean())
               if not trades.empty else float("nan"))
    mm_summary, other_summary, total_summary = compute.summary_tables(trades, last_px)
    price_ohlc = compute.period_ohlc(md_klines)
    # _md_klines() always fetches "1m" bars (see below) — the interval realized_vol_ann_pct needs.
    realized_vol_ann_pct = compute.realized_vol_ann_pct(md_klines, 60.0)

    mm_trades = trades[trades["acct_type"] == "MM"] if not trades.empty else trades
    try:
        ref = mo_klines if mo_klines is not None else md_klines   # fine bars when we got them
        mo_per_venue, mo_per_trade = (compute.markouts(mm_trades, ref, tuple(_MARKOUT_WINDOWS))
                                      if ref is not None else (None, None))
    except Exception as e:  # noqa: BLE001
        warnings.append(f"markouts failed: {e}")
        mo_per_venue = mo_per_trade = None

    try:
        cap, cap_ts = compute.captures(trades, market_vol)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"captures failed: {e}")
        cap = cap_ts = None

    widths = widths_ts = compliance = compliance_ts = None
    if ob_raw is not None and not ob_raw.empty:
        try:
            widths, widths_ts = compute.ob_widths(ob_raw)
            compliance = compute.ob_compliance(ob_raw)
            compliance_ts = compute.ob_compliance_buckets(ob_raw)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"orderbook widths/compliance failed: {e}")

    try:
        dpnl = compute.delta_pnl(trades, md_klines)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"delta/pnl failed: {e}")
        dpnl = None

    result = ZPResult(
        token=token, start=start, end=end, trades=trades,
        mm_summary=mm_summary, other_summary=other_summary, total_summary=total_summary,
        price_ohlc=price_ohlc, markouts=mo_per_venue, markout_windows=list(_MARKOUT_WINDOWS),
        markout_trades=mo_per_trade, captures=cap, captures_ts=cap_ts,
        ob_widths=widths, ob_widths_ts=widths_ts, ob_compliance=compliance,
        ob_compliance_ts=compliance_ts, delta_pnl_ts=dpnl, gamma_dash=dmc_data,
        realized_vol_ann_pct=realized_vol_ann_pct, ticks=ticks,
        headline={}, warnings=warnings,
    )
    result.headline = compute.headline(result)
    return result
