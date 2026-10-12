"""Journal is append-only JSONL, one file per UTC day, searchable by request id."""

from __future__ import annotations

import json

from src.the_operator.contract import Attachment
from src.the_operator.journal import Journal, clip, describe_attachments


def test_write_appends_and_find_returns_in_order(tmp_path):
    j = Journal(tmp_path)
    j.write("request", "abc", text="hi")
    j.write("tool_call", "abc", tool="agg")
    j.write("request", "other", text="no")
    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    lines = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert [r["kind"] for r in lines] == ["request", "tool_call", "request"]
    assert [r["kind"] for r in j.find("abc")] == ["request", "tool_call"]
    assert j.find("missing") == []


def test_attachments_described_not_stored():
    d = describe_attachments([Attachment("a.png", b"\x89PNG....", "photo")])
    assert d[0]["filename"] == "a.png" and d[0]["bytes"] == 8 and len(d[0]["sha256"]) == 16
    assert "data" not in d[0]


def test_clip_marks_truncation():
    assert clip("abc", 10) == "abc"
    out = clip("x" * 30, 10)
    assert out.startswith("x" * 10) and "+20 chars" in out
