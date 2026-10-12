"""Nightly snapshot of the ICP-MM aggregator workbook -> data/state/icp_snapshot.jsonl.

The ICP book's daily P&L needs YESTERDAY's delta, spot, vols and option book. The workbook
holds none of that: every cell is live and only the 'FX P&L' tab persists anything (balances
and rates, hard-stamped nightly at ~23:31 UTC by the desk's looper). This job stamps the
missing state at that same cutoff, so the day-over-day terms (delta x price move, option
value change) become computable from the first snapshot forward.

Reads the aggregate cells the sheet itself trusts -- `Dashboard!E74` is the desk's canonical
delta and ALREADY INCLUDES option delta (`= E73 + GammaSlide!Q15 + GammaSlide!AD36`), so a
P&L built on it must add only gamma/vega/theta for the options, never revalue them in full --
plus both option row-blocks raw, so greeks can be recomputed without re-reading the sheet.

Append-only, one JSON object per run; re-running on a date replaces that date's row rather
than appending a duplicate.

MUST NOT: write the sheet, compute P&L, place orders, or touch the data lake.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone

import gspread

from src.common import paths
from src.common.logging import get_logger

_log = get_logger("icp_snapshot")

SHEET_KEY = "1NcgtFY4RcaASVrJGftTcbEvIz-7IQKvQcLQg1thEedk"
_SCOPE = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
_KEY = os.environ.get("GOOGLE_AGG_SA_KEY") or str(paths.REPO_ROOT / ".gsa_key.json")
OUT = paths.STATE_DIR / "icp_snapshot.jsonl"

# field -> A1 cell. Aggregates, not re-derivations: if the desk restructures a block, the
# totals it points its own P&L at are what stay meaningful.
SCALARS = {
    "sheet_update_ts":   "Dashboard!B2",     # the looper's heartbeat -- stale => stale snapshot
    "pnl_itd":           "Dashboard!C2",
    "spot":              "Dashboard!D2",
    "fees_day":          "Dashboard!W2",
    "delta_spot_coins":  "Dashboard!E73",    # spot + HL perp, no options
    "delta_total_coins": "Dashboard!E74",    # E73 + option delta  <- the P&L anchor
    "trading_pnl_day":   "Dashboard!M67",    # sum of the per-account daily trading P&L
    "buy_qty_day":       "Dashboard!N69",
    "sell_qty_day":      "Dashboard!O69",
    "vol_1":             "GammaSlide!F5",    # applied to the far expiry
    "vol_2":             "GammaSlide!H5",    # applied to the near expiry
    "tau_1":             "GammaSlide!F11",
    "tau_2":             "GammaSlide!F15",
    "opt_pv_owned":      "GammaSlide!P15",
    "opt_delta_owned":   "GammaSlide!Q15",
    "opt_pv_internal":   "GammaSlide!AC36",
    "opt_delta_internal": "GammaSlide!AD36",
    "theta_day":         "GammaSlide!AN20",  # by full D+1 revaluation, not a Taylor theta
    "delta_bleed_day":   "GammaSlide!AN18",
    "adj_s29":           "GammaSlide!S29",   # two stray cells inside the desk's ITD P&L (E78);
    "adj_s34":           "GammaSlide!S34",   # captured so a day-over-day diff can prove they cancel
}

# (field, range, column headers) -- Option/Expiry/Strike/Size(coin)/Size(USD) in both blocks.
BOOKS = [
    ("options_owned", "GammaSlide!I4:M13"),
    ("options_internal", "GammaSlide!V26:Z34"),
]
_BOOK_COLS = ("type", "expiry", "strike", "size_coin", "size_usd")

# Which vol cell prices which line. The sheet wires this POSITIONALLY -- verified from the d1
# formulas: rows 4-12 read $F$5/$F$11, row 13 reads $H$5/$F$15. Recorded per line so the P&L
# never has to re-derive it, and so a line moving row (which silently reprices it) is visible.
_VOL2_ROW = 13
_OWNED_FIRST_ROW = 4


def _num(v):
    """UNFORMATTED_VALUE gives real numbers; keep anything else (text, blank) as-is."""
    return v if isinstance(v, (int, float)) or v in (None, "") else v


def _date(serial):
    """Sheets serial -> ISO date. Their epoch is 1899-12-30; a bare serial in the file would
    be one more thing to decode by hand at 2am."""
    if not isinstance(serial, (int, float)):
        return serial
    return (datetime(1899, 12, 30, tzinfo=timezone.utc) + timedelta(days=serial)).date().isoformat()


def read_snapshot(sheet) -> dict:
    """One batched read of every cell the daily P&L needs."""
    ranges = list(SCALARS.values()) + [r for _, r in BOOKS]
    got = sheet.values_batch_get(ranges, params={"valueRenderOption": "UNFORMATTED_VALUE"})
    vals = [vr.get("values") or [[]] for vr in got["valueRanges"]]

    snap: dict = {}
    for (field, _cell), block in zip(SCALARS.items(), vals):
        row = block[0] if block else []
        snap[field] = _num(row[0]) if row else None

    for (field, _rng), block in zip(BOOKS, vals[len(SCALARS):]):
        lines = []
        for i, row in enumerate(block):
            rec = dict(zip(_BOOK_COLS, [_num(c) for c in row] + [None] * len(_BOOK_COLS)))
            rec["expiry"] = _date(rec["expiry"])
            if field == "options_owned":
                rec["vol_cell"] = "vol_2" if i + _OWNED_FIRST_ROW == _VOL2_ROW else "vol_1"
            if rec["size_coin"]:                      # skip the block's empty/zero rows
                lines.append(rec)
        snap[field] = lines
    return snap


def vol_history(rows: list[dict]) -> dict[str, dict]:
    """Per vol cell: the mark in force, when it last changed, and how many days ago.

    The marks are entered by hand, so they go STALE rather than wrong: 60% left untouched
    while IV drifts reads clean every single day and then takes the whole correction at once,
    on a book where one vol point is worth ~14 days of trading P&L. Days-since-change is the
    only warning that exists, so it belongs on every report that quotes an option number.

    ``days`` is None until the series is long enough to have seen a change -- reported as
    "unchanged for at least N days", never as a confident zero.
    """
    out: dict[str, dict] = {}
    for cell in ("vol_1", "vol_2"):
        seen = [r for r in rows if r.get(cell) is not None]
        if not seen:
            continue
        latest = seen[-1]
        changed_on, prior = None, None
        for older, newer in zip(reversed(seen[:-1]), reversed(seen[1:])):
            if abs(newer[cell] - older[cell]) > 1e-9:   # literals, but don't trip on float noise
                changed_on, prior = newer["date"], older[cell]
                break
        span = (_days(latest["date"], changed_on) if changed_on
                else _days(latest["date"], seen[0]["date"]))
        out[cell] = {"vol": latest[cell], "changed_on": changed_on, "prior": prior,
                     "days": span, "bounded": changed_on is None}
    return out


def _days(later: str, earlier: str) -> int:
    return (date.fromisoformat(later) - date.fromisoformat(earlier)).days


def _load(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def write_snapshot(snap: dict, path=OUT) -> None:
    """Append, replacing any existing row for the same date (re-run safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in _load(path) if r.get("date") != snap["date"]]
    rows.append(snap)
    rows.sort(key=lambda r: r["date"])
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def describe_vol(h: dict) -> str:
    """One human line about a mark's age, honest about a series too short to prove anything."""
    at_least = "at least " if h["bounded"] else ""
    if h["days"] == 0 and not h["bounded"]:
        return f"re-marked TODAY from {h['prior']:.4g}"
    return f"unchanged for {at_least}{h['days']}d" + (
        f" (last re-marked {h['changed_on']} from {h['prior']:.4g})" if h["changed_on"] else "")


def live_snapshot() -> dict:
    """Read the sheet now, without writing. Lets a report close a window at the current moment
    instead of the last stamped night."""
    gc = gspread.service_account(filename=_KEY, scopes=_SCOPE)
    now = datetime.now(timezone.utc)
    snap = {"ts": now.isoformat(timespec="seconds"), "date": now.date().isoformat()}
    snap.update(read_snapshot(gc.open_by_key(SHEET_KEY)))
    return snap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print the snapshot, write nothing")
    args = ap.parse_args()

    snap = live_snapshot()

    if args.dry_run:
        print(json.dumps(snap, indent=2))
        return
    write_snapshot(snap)
    _log.info("snapshot %s: spot=%s delta=%s opt_pv=%s -> %s", snap["date"], snap["spot"],
              snap["delta_total_coins"], snap["opt_pv_owned"], OUT)
    for cell, h in vol_history(_load(OUT)).items():
        _log.info("%s = %.4g, %s", cell, h["vol"], describe_vol(h))


if __name__ == "__main__":
    main()
