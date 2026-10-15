"""The durable record of every /jwap parent: one append-only JSONL file per parent.

Each event is one JSON line, flushed and fsync'd before the caller acts on it, so the file is the
truth after any crash or restart. ``trade_bot`` writes an ``intent`` line BEFORE placing a child and
rebuilds every unfinished parent from its file at start-up (``replay``).

Event kinds (``kind``): ``created``, ``jason``, ``slice`` (every scheduled slice, fired or not),
``intent``, ``placed``, ``failed``, ``retry``, ``fill``, ``cancel``, ``resumed``, ``stopped``,
``done``.

MUST NOT: rewrite or delete a line, place/cancel orders, call any API, or build reports.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.adapters.coinroutes.trade import TERMINAL
from src.common.paths import STATE_DIR

RECORD_DIR = STATE_DIR / "jwap"


def next_id(root: Path | None = None) -> int:
    """One more than the highest id on disk, so "#3" means one parent for good -- a per-process
    counter restarted at 1 and reused ids across restarts."""
    d = root or RECORD_DIR
    ids = [int(f.stem) for f in d.glob("*.jsonl") if f.stem.isdigit()] if d.exists() else []
    return max(ids, default=0) + 1


def child_id(pid: int, entry: int, seq: int) -> str:
    """Our own CoinRoutes ``client_order_id`` for slice ``seq``: lets a restart or a retry ask
    CoinRoutes whether the order exists instead of guessing. ``entry`` (epoch s) keeps it unique
    even if the record folder were ever lost and ids restarted."""
    return f"jwap-{pid}-{entry}-{seq}"


def _path(pid: int, root: Path | None) -> Path:
    return (root or RECORD_DIR) / f"{pid}.jsonl"


def append(pid: int, kind: str, root: Path | None = None, **fields) -> dict:
    """Write one event and fsync it. Raises on any I/O failure: the caller must not act on an
    event it could not record."""
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "kind": kind,
           "pid": pid, **fields}
    path = _path(pid, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, default=str, ensure_ascii=False) + "\n"
    with path.open("ab+") as fh:
        fh.seek(0, os.SEEK_END)
        if fh.tell():
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":       # a crash tore the last line: never glue onto it
                line = "\n" + line
        fh.write(line.encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())
    return rec


def load(pid: int, root: Path | None = None) -> list[dict]:
    """Every event of one parent, in order. A torn last line (crash mid-write) is skipped."""
    out = []
    for line in _path(pid, root).read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def unfinished(root: Path | None = None) -> list[int]:
    """Ids of parents with no ``done`` event, oldest first."""
    d = root or RECORD_DIR
    if not d.exists():
        return []
    ids = sorted(int(f.stem) for f in d.glob("*.jsonl") if f.stem.isdigit())
    return [i for i in ids if not any(e.get("kind") == "done" for e in load(i, d))]


def replay(events: list[dict]) -> dict:
    """Rebuild a parent's state from its events: the ``created`` parameters plus progress.

    ``children`` maps client_order_id -> {seq, placed_at, terminal, last, cancel_sent} for every child with a
    ``placed`` event. ``unconfirmed`` lists intents with neither ``placed`` nor ``failed`` -- the
    crash window between writing the intent and hearing back from CoinRoutes."""
    created = next(e for e in events if e["kind"] == "created")
    st = {**created["params"], "next": 0, "fired": 0, "late": 0, "errors": 0,
          "children": {}, "oids": [], "unconfirmed": [], "stopped": False}
    intents: dict[str, dict] = {}
    for e in events:
        k = e["kind"]
        if k == "slice":
            st["next"] = max(st["next"], e["seq"] + 1)
            if e["result"] in ("late", "bot_down"):
                st["late"] += 1
        elif k == "intent":
            intents[e["cid"]] = e
        elif k == "placed":
            intents.pop(e["cid"], None)
            st["fired"] += 1
            st["oids"].append(e["cid"])
            st["children"][e["cid"]] = {"seq": e["seq"], "placed_at": e["placed_at"],
                                        "terminal": False, "last": None, "cancel_sent": False}
        elif k == "failed":
            intents.pop(e["cid"], None)
            st["errors"] += 1
        elif k == "fill":
            ch = st["children"].get(e["cid"])
            if ch is not None:
                ch["last"] = e["order"]
                ch["terminal"] = str(e["order"].get("order_status")) in TERMINAL
        elif k == "cancel":
            ch = st["children"].get(e["cid"])
            if ch is not None and e.get("ok"):
                ch["cancel_sent"] = True
        elif k == "stopped":
            st["stopped"] = True
    st["unconfirmed"] = list(intents.values())
    return st
