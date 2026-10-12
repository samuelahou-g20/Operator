"""ICP-only daily P&L for the MM2 desk, from two icp_snapshot rows.

The ICP-MM workbook reports one P&L for a book that also holds BTC/ETH/EUR/CHF, so its number
moves when those move against USD. This rebuilds the ICP part alone:

    delta   D0 * (S1 - S0)          D0 = Dashboard!E74, which ALREADY INCLUDES option delta
    gamma   0.5 * G0 * (S1 - S0)^2
    vega    v0 * (sigma1 - sigma0)  only non-zero when someone re-marks by hand
    theta   the sheet's own D+1 revaluation, prorated
    trading Dashboard!M67 -- today's fills, matched spread + residual marked to spot
    fees    Dashboard!W2

Because D0 carries option delta, the options contribute ONLY the higher-order terms; revaluing
them in full would double-count option-delta x the move. The FX legs never enter -- the
workbook's own 'FX P&L' tab owns those, and excluding them is the entire point.

Greeks are recomputed from the snapshot's own strikes/vols/taus with r=q=0, matching the
sheet's Black-Scholes exactly. The option block is reconciled against its actual PV change, so
a Taylor expansion breaking down (a large move, or a line near expiry) shows up as a residual
instead of hiding in the total.

MUST NOT: write the sheet, place orders, or touch the data lake.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime

from src.common import paths

SNAPSHOTS = paths.STATE_DIR / "icp_snapshot.jsonl"
_TAU = {"vol_1": "tau_1", "vol_2": "tau_2"}


def load(path=SNAPSHOTS) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no snapshots yet at {path} -- run scripts.icp_snapshot first")
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _phi(x: float) -> float:
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def line_greeks(line: dict, row: dict) -> tuple[float, float]:
    """(gamma, vega) for one option line at ``row``'s spot/vol/tau.

    Gamma is coins of delta per $1 of spot; vega is $ per 1.00 of vol (not per point). Both are
    sign-carrying via ``size_coin`` and identical for a put and a call, so the type is not
    needed -- the sheet prices puts by parity off the same d1.
    """
    sigma = row[line["vol_cell"]]
    tau = row[_TAU[line["vol_cell"]]]
    S, K, q = row["spot"], line["strike"], line["size_coin"]
    if not (sigma and tau and S and K):
        return 0.0, 0.0
    d1 = (math.log(S / K) + (sigma * sigma / 2) * tau) / (sigma * math.sqrt(tau))
    return _phi(d1) / (S * sigma * math.sqrt(tau)) * q, S * _phi(d1) * math.sqrt(tau) * q


def book_greeks(row: dict) -> tuple[float, float]:
    """(gamma, vega) summed over the owned option block."""
    per = [line_greeks(ln, row) for ln in row.get("options_owned") or []]
    return sum(g for g, _ in per), sum(v for _, v in per)


def _opt_pv(row: dict) -> float:
    return (row.get("opt_pv_owned") or 0.0) + (row.get("opt_pv_internal") or 0.0)


def pnl(r0: dict, r1: dict) -> dict:
    """Decompose the ICP P&L over the window (r0, r1]."""
    ds = r1["spot"] - r0["spot"]
    # Fractional, not whole days: flooring charges a 15-minute window a full day of decay
    # (~$1,840 here) and buries it in the option residual.
    dt = (datetime.fromisoformat(r1["ts"])
          - datetime.fromisoformat(r0["ts"])).total_seconds() / 86400
    gamma0, _ = book_greeks(r0)

    vega = 0.0
    remarks = []
    for cell in ("vol_1", "vol_2"):
        dsig = (r1.get(cell) or 0.0) - (r0.get(cell) or 0.0)
        if abs(dsig) > 1e-9:
            v = sum(line_greeks(ln, r0)[1] for ln in r0.get("options_owned") or []
                    if ln["vol_cell"] == cell)
            vega += v * dsig
            remarks.append(f"{cell} {r0[cell]:.4g}->{r1[cell]:.4g}")

    terms = {
        "delta_spot": r0["delta_spot_coins"] * ds,
        "delta_options": r0["opt_delta_owned"] * ds,
        "gamma": 0.5 * gamma0 * ds * ds,
        "vega": vega,
        "theta": (r0.get("theta_day") or 0.0) * dt,
        "trading": r1.get("trading_pnl_day") or 0.0,
        "fees": r1.get("fees_day") or 0.0,
    }
    # The sheet's E74 is E73 + option delta; splitting it here is presentation only, so the two
    # halves must still sum back to it or the snapshot is inconsistent with itself.
    anchor = r0["delta_total_coins"] * ds
    split = terms["delta_spot"] + terms["delta_options"] + (r0.get("opt_delta_internal") or 0.0) * ds

    taylor = terms["delta_options"] + terms["gamma"] + terms["vega"] + terms["theta"]
    return {
        **terms,
        "total": sum(terms.values()),
        "d_spot": ds,
        "days": dt,
        "remarks": remarks,
        "delta_check": anchor - split,
        "option_taylor": taylor,
        "option_actual": _opt_pv(r1) - _opt_pv(r0),
        "option_residual": (_opt_pv(r1) - _opt_pv(r0)) - taylor,
        "unmodelled_internal": bool(r0.get("options_internal")),
    }


def _vol_note(rows: list[dict], upto: str) -> str:
    from scripts.icp_snapshot import describe_vol, vol_history
    hist = vol_history([r for r in rows if r["date"] <= upto])
    return "  ".join(f"{c} {h['vol']:.4g} ({describe_vol(h)})" for c, h in hist.items())


def render(r0: dict, r1: dict, p: dict, rows: list[dict]) -> str:
    ln = [f"ICP  {r0['date']} -> {r1['date']}   spot {r0['spot']:.4f} -> {r1['spot']:.4f} "
          f"({p['d_spot']:+.4f})", ""]
    for label, key in [("delta (spot)", "delta_spot"), ("delta (options)", "delta_options"),
                       ("gamma", "gamma"), ("vega", "vega"), ("theta", "theta"),
                       ("trading", "trading"), ("fees", "fees")]:
        flag = "  <- RE-MARK " + ", ".join(p["remarks"]) if key == "vega" and p["remarks"] else ""
        ln.append(f"  {label:<16}{p[key]:>14,.0f}{flag}")
    ln += ["  " + "-" * 30, f"  {'TOTAL':<16}{p['total']:>14,.0f}", ""]
    ln.append(f"  option block: taylor {p['option_taylor']:,.0f} vs actual "
              f"{p['option_actual']:,.0f}  residual {p['option_residual']:,.0f}")
    ln.append("  " + _vol_note(rows, r1["date"]))
    if p["days"] < 0.9:
        ln.append(f"  !! window is {p['days'] * 24:.1f}h, but trading/fees are the sheet's "
                  f'"Today (UTC)" counters -- they cover 00:00 UTC to now, a longer window')
    if abs(p["delta_check"]) > 1.0:
        ln.append(f"  !! delta split misses E74 by {p['delta_check']:,.0f} -- snapshot inconsistent")
    if p["unmodelled_internal"]:
        ln.append("  !! internal option block is non-empty; its gamma/vega are NOT modelled")
    return "\n".join(ln)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="the day to close (default: the newest snapshot)")
    ap.add_argument("--live", action="store_true",
                    help="use a fresh sheet read as the end of the window instead of a snapshot")
    args = ap.parse_args()

    rows = load()
    if args.live:
        from scripts.icp_snapshot import live_snapshot
        r1 = live_snapshot()
        rows = rows + [r1]
    else:
        r1 = next((r for r in reversed(rows) if not args.date or r["date"] == args.date), None)
        if r1 is None:
            raise SystemExit(f"no snapshot for {args.date}")
    prior = [r for r in rows if r["ts"] < r1["ts"]]
    if not prior:
        raise SystemExit(
            f"nothing stamped before {r1['ts']} -- a daily P&L needs two snapshots.\n"
            f"  have {len(rows)} row(s); the next lands at 23:58 UTC "
            f"(systemctl --user list-timers icp-snapshot.timer)\n"
            f"  meanwhile: --live closes the window at a fresh sheet read instead")
    print(render(prior[-1], r1, pnl(prior[-1], r1), rows))


if __name__ == "__main__":
    main()
