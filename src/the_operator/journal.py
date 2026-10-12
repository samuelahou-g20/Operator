"""Append-only decision journal: one JSON line per event, one file per UTC day.

Event kinds: ``request`` (who asked what), ``turn`` (one model response: stop reason, usage,
API request id), ``tool_call`` (proposed call + policy verdict), ``tool_result`` (outcome, timing,
attachment digests), ``reply`` (what was sent), ``admin`` (halt/resume/reset), ``error``.
Attachment bytes are never written; their name, size and sha256 are.

MUST NOT: rewrite or delete lines, hold state other than the file lock, or be bypassed by any
module that runs a tool or calls the model.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.common.paths import LOGS_DIR
from src.the_operator.contract import Attachment

TEXT_CLIP = 20_000
DEFAULT_ROOT = LOGS_DIR / "operator" / "journal"


def clip(text: str, limit: int = TEXT_CLIP) -> str:
    return text if len(text) <= limit else text[:limit] + f"…[+{len(text) - limit} chars]"


def describe_attachments(atts: list[Attachment]) -> list[dict]:
    return [{"filename": a.filename, "kind": a.kind, "bytes": len(a.data),
             "sha256": hashlib.sha256(a.data).hexdigest()[:16]} for a in atts]


class Journal:
    def __init__(self, root: Path = DEFAULT_ROOT) -> None:
        self.root = root
        self._lock = threading.Lock()

    def _path(self, day: datetime) -> Path:
        return self.root / f"{day:%Y-%m-%d}.jsonl"

    def write(self, kind: str, request_id: str, **fields) -> dict:
        now = datetime.now(timezone.utc)
        rec = {"ts": now.isoformat(timespec="milliseconds"), "kind": kind,
               "request_id": request_id, **fields}
        line = json.dumps(rec, default=str, ensure_ascii=False)
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with self._path(now).open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return rec

    def find(self, request_id: str, days: int = 3) -> list[dict]:
        """Every event of one request, oldest first, scanning the last ``days`` daily files."""
        out: list[dict] = []
        today = datetime.now(timezone.utc)
        for back in range(days - 1, -1, -1):
            path = self._path(today - timedelta(days=back))
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("request_id") == request_id:
                    out.append(rec)
        return out
