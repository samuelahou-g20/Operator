"""Lightweight data-quality checks for daily fetches (ARCHITECTURE §12).

`run_checks` inspects a canonical DataFrame and logs warnings to `logs/quality/`.
Checks: ts monotonicity & duplicates, OHLC consistency, non-negative volume, and
gaps vs an expected bar frequency.

MUST NOT: run during bulk backfills (cost/OOM unrealistic there), mutate data,
block a write, or raise on a quality failure — warnings only.
"""

from __future__ import annotations

import pandas as pd

from src.common import logging as common_logging

_quality_log = common_logging.get_logger("quality")


def check_monotonic_unique(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    ts = df["ts"]
    dupes = int(ts.duplicated().sum())
    if dupes:
        issues.append(f"{dupes} duplicate ts")
    if not ts.is_monotonic_increasing:
        issues.append("ts not monotonically increasing")
    return issues


def check_ohlc(df: pd.DataFrame) -> list[str]:
    if not {"open", "high", "low", "close"} <= set(df.columns):
        return []
    issues: list[str] = []
    bad_high = int((df["high"] < df[["open", "close", "low"]].max(axis=1)).sum())
    bad_low = int((df["low"] > df[["open", "close", "high"]].min(axis=1)).sum())
    if bad_high:
        issues.append(f"{bad_high} rows with high < max(open,close,low)")
    if bad_low:
        issues.append(f"{bad_low} rows with low > min(open,close,high)")
    return issues


def check_nonneg_volume(df: pd.DataFrame) -> list[str]:
    if "volume" not in df.columns:
        return []
    n = int((df["volume"] < 0).sum())
    return [f"{n} rows with negative volume"] if n else []


def check_gaps(df: pd.DataFrame, freq: str | None) -> list[str]:
    """Gaps vs an expected bar frequency (crypto is 24/7, so freq alone suffices).

    Futures session-calendar gaps come with the futures milestone; for now this is
    the freq-based check ARCHITECTURE §12 calls for.
    """
    if freq is None or df.empty:
        return []
    ts = pd.to_datetime(df["ts"]).sort_values()
    expected = pd.date_range(ts.iloc[0], ts.iloc[-1], freq=freq)
    missing = len(expected) - ts.nunique()
    return [f"{missing} missing {freq} bars (have {ts.nunique()} of {len(expected)})"] if missing > 0 else []


def run_checks(
    df: pd.DataFrame,
    name: str,
    *,
    freq: str | None = None,
    logger=None,
) -> list[str]:
    """Run the applicable checks, log each issue as a warning, return the issues."""
    log = logger or _quality_log
    issues: list[str] = []
    issues += check_monotonic_unique(df)
    issues += check_ohlc(df)
    issues += check_nonneg_volume(df)
    issues += check_gaps(df, freq)
    for issue in issues:
        log.warning("%s: %s", name, issue)
    return issues
