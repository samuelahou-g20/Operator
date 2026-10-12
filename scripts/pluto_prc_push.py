"""Push the Pluto MM2GAMMA_TOTAL risk view into the MM2GAMMA-MM sheet's [Pluto-PRC-Summary] tab.

Two Pluto calls (`detailed=positions`, `detailed=greeks`) and one sheet write. The tab is written
as ONE contiguous rectangle -- headers, data and the side metadata block in a single request.

`detailed=positions` is still needed even though no position-level tab is written: Pluto's `pv`
is a single portfolio-wide scalar, so per-biprod PV exists only by summing the positions.

Layout follows the sheet's own convention for machine-written `[...]` tabs (see `[Positions]`,
`[Adjustments]`): headers in row 1, data from row 2, and a label/value metadata block off to the
right carrying `Requested At`. The Dashboard QUERYs `$A:$I` by biprod (PV -> "Live Options MTM",
Delta (Coin) -> "Options Delta (Coin)"), so:

  - there is deliberately NO total row -- a SUM over the range would double-count;
    the portfolio total lives in the metadata block instead
  - the tab is CLEARED AND REWRITTEN ATOMICALLY. `updateCells` with an unbounded range clears
    every cell the payload does not cover, in the same request, so a book that shrinks (they
    roll daily, and POL appeared mid-session) leaves no stale row -- with no separate clear call
    and no instant where the tab is empty for a Dashboard formula to read.

A failed cycle is not fatal: the tabs keep the last good data (the write is one atomic request,
so there is no half-written state) and the timer retries a minute later. The metadata block's
`Requested At` is how staleness is spotted. Only a SUSTAINED outage is worth waking anyone for,
so consecutive failures are counted across runs in a state file and the MM2 desk chat is
messaged once the streak passes ALERT_AFTER.

MUST NOT: place orders, write the Dashboard, or touch the data lake. Writes only that one tab.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import gspread
import urllib3

from src.common.config import require_env
from src.common.logging import get_logger
from src.common.paths import REPO_ROOT, STATE_DIR
from src.common.telegram import send
from src.common.pluto import PORTFOLIO_URL, get as pluto_get

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # Pluto internal self-signed

_log = get_logger("pluto_prc")

PLUTO_URL = PORTFOLIO_URL
PORTFOLIO = "MM2GAMMA_TOTAL"
# Omitting `riskreport` silently applies Pluto's DEFAULT model, which moves delta ~10%.
RISK_REPORT = "STICKY STRIKE FITTED"

SHEET_KEY = "11Ne3nFRAg6I90qHpyWbxcKwgu7uo_8NLB5ragX7f9gU"
# Writing needs the full scope; agg_bot/icp_snapshot use spreadsheets.readonly.
_SCOPE = ["https://www.googleapis.com/auth/spreadsheets"]
_KEY = str(REPO_ROOT / ".gsa_key.json")
TAB_SUM = "[Pluto-PRC-Summary]"

# Each run is its own process, so the consecutive-failure count has to live on disk.
FAIL_STATE = STATE_DIR / "pluto_prc_failures.json"
ALERT_AFTER = 5        # "more than 5 in a row" -> the 6th consecutive failure alerts
_GAP = 1               # blank columns between the data block and the metadata block

# `_values` is 40 unlabelled floats. These seven indices were established by reconciling column
# sums against the labelled aggregate across all three biprods at once -- see
# docs/pluto_portfolio_api.md. The other 33 slots stay out until they are identified the same way.
GREEKS = [("PV", 1), ("Delta (Coin)", 2), ("Delta ($)", 4), ("Gamma (Coin/1%)", 5),
          ("Gamma ($)", 7), ("Vega ($/pt)", 8), ("Theta ($/day)", 11)]

# PV sums every leg of a biprod EXCEPT the intra-desk loan book. `counterparty` is the field
# Pluto's UI labels Source. The loans are what made NEAR's PV read -329,741 on a long call book:
# a -200,000 NEAR CASH1 at -374,400. They are funding owed internally, not book value.
#
# Filtering by counterparty rather than by position code, because code does not identify them --
# POL's own hedge is a CASH1 too (DB-MM2GAMMA_POL, -3,690,000) and must stay in.
EXCLUDE_COUNTERPARTIES = {"DB-MM2_GAMMA_LOANS"}

# Two different scopes, deliberately:
#   PV      -> summed per biprod from the positions.
#   the rest-> Pluto's own per-biprod aggregate, i.e. exactly what the Pluto summary tab shows.
# PV has to be summed because Pluto's `pv` is one portfolio-wide scalar with no per-biprod split;
# everything else it already reports per biprod.
# Delta is the reason the rest is not filtered. CASH1 legs are spot/loan coin balances and their
# delta IS the notional, so they are real exposure: POL's hedge is booked CASH1 (net -2,765,199)
# rather than FX, and filtering left POL reading +2.37m against Pluto's -398,407.
AGG_FIELD = {"Delta (Coin)": "deltas", "Delta ($)": "delta_usds", "Gamma (Coin/1%)": "gammas",
             "Gamma ($)": "gamma_usds", "Vega ($/pt)": "vegas", "Theta ($/day)": "thetas"}
SUM_COLS = ["Biprod", "Spot Ref"] + [h for h, _ in GREEKS]


# --------------------------------------------------------------------------- #
# Pluto
# --------------------------------------------------------------------------- #
def fetch() -> tuple[list, dict]:
    """(positions, greeks) -- exactly two Pluto calls."""
    h = {"X-Api-Key": require_env("PLUTO_API_KEY"),
         "X-Api-Secret": require_env("PLUTO_API_SECRET")}
    q = {"portfolio": PORTFOLIO, "riskreport": RISK_REPORT}

    def get(detailed):
        r = pluto_get(PLUTO_URL, headers=h, timeout=60,
                         params={**q, "detailed": detailed})
        r.raise_for_status()
        return r.json()

    return get("positions"), get("greeks")


# Column -> decimals. Pluto returns float noise on exact quantities (200000.00000001563) and
# 17 digits on money; rounding is display only and never changes a reconciliation at 2dp.
_ROUND = {"Reval Vol": 6, "Official Vol": 6, "Spot Ref": 8,
          **{h: 2 for h, _ in GREEKS}}


def _cell(v, decimals=None):
    """None -> blank; ISO timestamps -> date only (unambiguous, and sorts correctly as text)."""
    if v is None:
        return ""
    if isinstance(v, str) and len(v) >= 10 and v[4] == "-" and v[7] == "-" and v.endswith("Z"):
        return v[:10]
    if decimals is not None and isinstance(v, float):
        return round(v, decimals)
    return v


def _pv(entry) -> float:
    return entry["greek"]["_values"][1] or 0.0


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #
def _grid(headers: list, rows: list[list], meta: list[tuple]) -> list[list]:
    """One rectangle: headers row 1, data from row 2, metadata `_GAP` columns to the right.

    Sized to the data -- everything outside it is cleared by the write itself, so there is no
    padding to guess at."""
    width = len(headers) + _GAP + 2
    grid = [[""] * width for _ in range(max(1 + len(rows), len(meta)))]
    for j, h in enumerate(headers):
        grid[0][j] = h
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            grid[i][j] = _cell(v, _ROUND.get(headers[j]))
    for i, (k, v) in enumerate(meta):
        grid[i][len(headers) + _GAP] = k
        grid[i][len(headers) + _GAP + 1] = _cell(v)
    return grid


def _meta(label: str, extra: list[tuple]) -> list[tuple]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return [("metric", label), ("Portfolio", f"*{PORTFOLIO}"), ("RiskReport", RISK_REPORT),
            ("Requested At", now), *extra]


def legs_of(pos: list, biprod: str) -> list:
    """Every leg of a biprod -- options, forwards, cash and the intra-desk loans alike."""
    return [e for e in pos if e["posn"]["biprod"] == biprod]


def summary_block(pos: list, g: dict) -> list[list]:
    """One row per biprod: PV summed over all its legs, every other column Pluto's own aggregate.

    A biprod with no positions keeps its row at zero rather than vanishing -- the Dashboard's
    QUERY would return 0 either way, and a visible zero says "nothing here" instead of "not
    checked"."""
    biprods = sorted(g.get("deltas") or {})
    rows = []
    for b in biprods:
        legs = legs_of(pos, b)
        row = [b, (g.get("spotrefs") or {}).get(b)]
        for head, i in GREEKS:
            row.append(sum(e["greek"]["_values"][i] or 0.0 for e in legs) if head == "PV"
                       else (g.get(AGG_FIELD[head]) or {}).get(b))
        rows.append(row)
    meta = _meta("Pluto PRC Summary",
                 [("Biprods", len(biprods)),
                  ("Positions", len(pos)),
                  # Both, so the invariant stays visible: the per-biprod column must add up to
                  # Pluto's own portfolio scalar, give or take the book moving between calls.
                  ("Summed PV", sum(_pv(e) for e in pos)),
                  ("Portfolio PV (Pluto)", g.get("pv"))])
    return _grid(SUM_COLS, rows, meta)


# --------------------------------------------------------------------------- #
def _value(v) -> dict:
    """A cell as a typed ExtendedValue. An empty dict means "no value", which `updateCells`
    treats as a clear -- that is what wipes the previous cycle's longer book."""
    if v == "" or v is None:
        return {}
    if isinstance(v, bool):            # before int: bool is an int subclass
        return {"boolValue": v}
    if isinstance(v, (int, float)):
        return {"numberValue": v}
    return {"stringValue": str(v)}


def push(blocks: dict[str, list[list]]) -> None:
    """Clear-and-write. One `updateCells` request per tab, one HTTP call.

    `userEnteredValue` (not `formattedValue`) with typed numbers is the equivalent of RAW: values
    land as real numbers for the Dashboard's QUERY/SUM, and nothing re-parses the date or
    `official_ref` strings the way USER_ENTERED would.
    """
    sh = gspread.service_account(filename=_KEY, scopes=_SCOPE).open_by_key(SHEET_KEY)
    requests_ = []
    for tab, grid in blocks.items():
        requests_.append({"updateCells": {
            # No end indices -> the range is the whole tab, so every cell the rows below do not
            # cover is cleared in this same request.
            "range": {"sheetId": sh.worksheet(tab).id,
                      "startRowIndex": 0, "startColumnIndex": 0},
            "fields": "userEnteredValue",
            "rows": [{"values": [{"userEnteredValue": _value(c)} for c in row]} for row in grid],
        }})
    sh.batch_update({"requests": requests_})


def _read_state() -> dict:
    try:
        return json.loads(FAIL_STATE.read_text())
    except (OSError, ValueError):
        return {"streak": 0, "alerted": False, "since": None, "last_error": None}


def _write_state(st: dict) -> None:
    FAIL_STATE.parent.mkdir(parents=True, exist_ok=True)
    FAIL_STATE.write_text(json.dumps(st))


def alert_mm2(text: str) -> None:
    """Push-only message to the MM2 desk chat, reusing TELEGRAM_BOT_TOKEN the way
    flow_alert_bot does -- no poller, so no 409 risk on the shared token."""
    api = f"https://api.telegram.org/bot{require_env('TELEGRAM_BOT_TOKEN')}"
    for chat in (c.strip() for c in require_env("TELEGRAM_MM2_CHAT_ID").split(",")):
        if chat:
            send(api, chat, text)


def on_failure(exc: Exception) -> None:
    """Count the streak and alert once per outage -- not once a minute, which would be spam."""
    st = _read_state()
    st["streak"] += 1
    st["last_error"] = f"{type(exc).__name__}: {exc}"[:300]
    st["since"] = st["since"] or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    _log.error("push failed (%d in a row): %s", st["streak"], st["last_error"])

    if st["streak"] > ALERT_AFTER and not st["alerted"]:
        try:
            alert_mm2(f"\u26a0\ufe0f Pluto PRC push failing\n"
                      f"{st['streak']} consecutive failures since {st['since']}\n"
                      f"{TAB_SUM} is STALE\n"
                      f"last error: {st['last_error']}")
            st["alerted"] = True
        except Exception as e:                       # a broken alert must not mask the outage
            _log.error("could not alert MM2: %s", e)
    _write_state(st)


def on_success(pos: list, g: dict) -> None:
    st = _read_state()
    if st["streak"]:
        _log.info("recovered after %d consecutive failures", st["streak"])
    _write_state({"streak": 0, "alerted": False, "since": None, "last_error": None})
    _log.info("pushed %d positions / %d biprods to %s", len(pos), len(g.get("deltas") or {}),
              SHEET_KEY)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print both blocks, write nothing")
    args = ap.parse_args()

    if args.dry_run:
        pos, g = fetch()
        grid = summary_block(pos, g)
        print(f"=== {TAB_SUM}  {len(grid)}x{len(grid[0])}")
        for i, row in enumerate(grid, 1):
            if any(str(c) != "" for c in row):
                print(f"  {i:>2} {[c for c in row]}")
        return 0

    # Any failure -- Pluto or Sheets -- leaves the tabs on their last good data and is retried
    # by the next timer tick, so it is counted rather than raised.
    try:
        pos, g = fetch()          # positions: the only source of per-biprod PV
        push({TAB_SUM: summary_block(pos, g)})
    except Exception as exc:
        on_failure(exc)
        return 1
    on_success(pos, g)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
