"""Offline tests for the durable /jwap record (scripts/jwap_record.py)."""

from __future__ import annotations

import scripts.jwap_record as jrec


def test_ids_continue_from_the_highest_on_disk(tmp_path):
    assert jrec.next_id(tmp_path) == 1
    jrec.append(1, "created", root=tmp_path, params={})
    jrec.append(7, "created", root=tmp_path, params={})
    assert jrec.next_id(tmp_path) == 8


def test_child_ids_carry_the_parent_start_so_a_lost_folder_cannot_collide():
    assert jrec.child_id(3, 1_760_000_000, 12) == "jwap-3-1760000000-12"


def test_a_torn_last_line_is_skipped_and_never_glued_to_the_next(tmp_path):
    jrec.append(1, "created", root=tmp_path, params={"n": 2})
    with (tmp_path / "1.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"kind": "slice", "se')                      # crash mid-write
    jrec.append(1, "stopped", root=tmp_path, by="u")
    kinds = [e["kind"] for e in jrec.load(1, tmp_path)]
    assert kinds == ["created", "stopped"]


def test_unfinished_lists_parents_without_done(tmp_path):
    for pid in (1, 2, 3):
        jrec.append(pid, "created", root=tmp_path, params={})
    jrec.append(2, "done", root=tmp_path)
    assert jrec.unfinished(tmp_path) == [1, 3]


def _events(*extra):
    return [{"kind": "created", "params": {"id": 1, "n": 10, "size": 1000.0}}, *extra]


def test_replay_rebuilds_progress_children_and_unanswered_intents():
    st = jrec.replay(_events(
        {"kind": "slice", "seq": 0, "result": "gate_closed"},
        {"kind": "slice", "seq": 1, "result": "fired"},
        {"kind": "intent", "cid": "a", "seq": 1},
        {"kind": "placed", "cid": "a", "seq": 1, "placed_at": 100.0},
        {"kind": "fill", "cid": "a", "order": {"order_status": "open", "executed_qty": "5"}},
        {"kind": "slice", "seq": 2, "result": "late"},
        {"kind": "slice", "seq": 3, "result": "fired"},
        {"kind": "intent", "cid": "b", "seq": 3},
        {"kind": "slice", "seq": 4, "result": "fired"},
        {"kind": "intent", "cid": "c", "seq": 4},
        {"kind": "failed", "cid": "c", "seq": 4, "stage": "place"},
    ))
    assert (st["next"], st["fired"], st["late"], st["errors"]) == (5, 1, 1, 1)
    assert st["oids"] == ["a"] and st["children"]["a"]["terminal"] is False
    assert st["children"]["a"]["last"]["executed_qty"] == "5"
    assert [i["cid"] for i in st["unconfirmed"]] == ["b"]       # the crash window


def test_replay_marks_a_closed_child_final_and_keeps_stop():
    st = jrec.replay(_events(
        {"kind": "placed", "cid": "a", "seq": 0, "placed_at": 1.0},
        {"kind": "fill", "cid": "a", "order": {"order_status": "closed"}},
        {"kind": "stopped", "by": "u"},
    ))
    assert st["children"]["a"]["terminal"] is True and st["stopped"] is True


def test_error_status_is_not_final():
    """`error` is transient on CoinRoutes (an order reported it, then closed 100% filled)."""
    st = jrec.replay(_events(
        {"kind": "placed", "cid": "a", "seq": 0, "placed_at": 1.0},
        {"kind": "fill", "cid": "a", "order": {"order_status": "error"}},
    ))
    assert st["children"]["a"]["terminal"] is False
