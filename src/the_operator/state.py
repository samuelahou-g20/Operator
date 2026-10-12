"""Durable gateway state: the live orders being watched for fills, and unconfirmed proposals.

Both lived only in memory, so a restart dropped them silently. The fills watchlist is the one that
matters: an order the Operator placed stopped being watched mid-flight, and its fill was never
announced — the trader is left believing a message is coming. Proposals are kept too, though a
120 s confirm TTL means only a fast restart saves one; an expired proposal is dropped on load
rather than resurrected.

One small JSON file, rewritten whole on every change (a few dozen entries at most, so there is
nothing to gain from an append log and a lot to lose from a half-written one). Saving is
best-effort: a state file that cannot be written is a log line, never a failed trade.

MUST NOT: hold policy, reconstruct a proposal the journal does not corroborate, or be the record
of what happened — that is ``journal``, which is append-only and never rewritten.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

from src.common.logging import get_logger
from src.common.paths import LOGS_DIR
from src.the_operator.contract import Context
from src.the_operator.policy import CONFIRM_TTL_S, Pending

_log = get_logger("operator")

PATH = LOGS_DIR / "operator" / "state.json"


def dump(pending: dict[str, Pending], fills: dict[str, dict], path=None) -> None:
    """Write the whole state. Never raises."""
    path = path or PATH
    payload = {
        "saved_at": time.time(),
        "pending": [{"nonce": p.nonce, "ctx": asdict(p.ctx), "tool": p.tool,
                     "proposal": p.proposal, "created": p.created,
                     "notional_usd": p.notional_usd} for p in pending.values()],
        "fills": fills,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)        # atomic: a crash mid-write leaves the previous state intact
    except OSError as e:
        _log.warning("could not save operator state: %s", e)


def load(path=None, now: float | None = None) -> tuple[dict[str, Pending], dict[str, dict]]:
    """``(pending, fills)`` from disk, expired proposals dropped. Never raises."""
    path = path or PATH
    now = time.time() if now is None else now
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, {}
    pending: dict[str, Pending] = {}
    for row in payload.get("pending") or []:
        try:
            if now - float(row["created"]) > CONFIRM_TTL_S:
                continue          # it expired while we were down; a trader must ask again
            pending[row["nonce"]] = Pending(row["nonce"], Context(**row["ctx"]), row["tool"],
                                            row["proposal"], float(row["created"]),
                                            float(row.get("notional_usd") or 0.0))
        except (KeyError, TypeError, ValueError) as e:
            _log.warning("dropping unreadable pending proposal from state: %s", e)
    fills = payload.get("fills") or {}
    if not isinstance(fills, dict):
        fills = {}
    if pending or fills:
        _log.info("restored operator state: %d pending, %d watched order(s)", len(pending),
                  len(fills))
    return pending, fills
