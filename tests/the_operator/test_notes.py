"""Desk notes: written only after a confirm, read back into the prompt, never across desks."""

from __future__ import annotations

import pytest

from src.the_operator import desk, execution, notes, policy
from src.the_operator.contract import Context, Risk
from src.the_operator.tools import chat

CTX = Context("r1", "mm3", "-1", "7", "u7", is_trader=True)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A throwaway config/operator dir — tests must never append to the real playbooks."""
    operator_dir = tmp_path / "operator"
    operator_dir.mkdir()
    monkeypatch.setattr(notes, "DIR", operator_dir)
    monkeypatch.setattr(desk, "CONFIG_DIR", tmp_path)
    (operator_dir / "mm3.md").write_text("Desk notes (mm3): hand-written line.\n")
    return operator_dir


def test_a_note_round_trips_with_its_author_and_date(store):
    notes.add("mm3", "zeyi_g20", "NEAR is quoted by mm5, not mm3.", ts=1_767_225_600.0)
    got = notes.read("mm3")
    assert len(got) == 1 and got[0].author == "zeyi_g20"
    assert got[0].line().startswith("- (2026-01-01, zeyi_g20) NEAR is quoted")


def test_notes_append_below_the_marker_and_never_touch_what_a_human_wrote(store):
    notes.add("mm3", "u7", "first")
    notes.add("mm3", "u8", "second")
    text = (store / "mm3.md").read_text()
    assert text.startswith("Desk notes (mm3): hand-written line.")
    assert text.count(notes.MARKER) == 1
    assert text.index("first") < text.index("second")
    assert [n.text for n in notes.read("mm3")] == ["first", "second"]


def test_a_desk_with_no_playbook_file_yet_still_takes_a_note(store):
    notes.add("mm5", "u7", "a fact")
    assert (store / "mm5.md").read_text().startswith(notes.MARKER)
    assert [n.text for n in notes.read("mm5")] == ["a fact"]


def test_a_file_with_no_marker_has_no_notes(store):
    assert notes.read("mm3") == []          # hand-written prose alone is not a note


def test_notes_are_per_desk(store):
    notes.add("mm3", "u7", "mm3 fact")
    notes.add("mm2", "u8", "mm2 fact")
    assert [n.text for n in notes.read("mm3")] == ["mm3 fact"]
    assert [n.text for n in notes.read("mm2")] == ["mm2 fact"]


def test_the_appended_block_labels_notes_as_hearsay(store):
    notes.add("mm3", "u7", "a thing")
    text = (store / "mm3.md").read_text()
    assert "not desk policy" in text and "notes above win" in text


def test_a_line_someone_hand_edited_badly_is_skipped_not_fatal(store):
    notes.add("mm3", "u7", "good")
    with notes.path("mm3").open("a", encoding="utf-8") as fh:
        fh.write("- not a note at all\n- (nonsense, x) bad date\n")
    assert [n.text for n in notes.read("mm3")] == ["good"]


def test_desk_facts_carries_the_notes(store):
    notes.add("mm3", "zeyi_g20", "LAPTOP quotes only on Binance.")
    facts = desk.desk_facts("mm3")
    assert "LAPTOP quotes only on Binance." in facts
    assert facts.index("Desk: MM3") < facts.index("LAPTOP quotes only")   # after the facts


def test_remember_only_proposes_and_writes_nothing(store):
    res = chat._remember(CTX, {"note": "NEAR sits on mm5.", "reason": "zeyi said so"})
    assert res.proposal["kind"] == "remember" and res.proposal["note"] == "NEAR sits on mm5."
    assert notes.read("mm3") == []                 # nothing on disk until a human confirms
    assert "REMEMBER on mm3" in res.text


def test_remember_is_a_confirmed_write_not_an_execute():
    assert chat.REMEMBER.risk is Risk.WRITE
    verdict = policy.check(chat.REMEMBER, {"note": "x", "reason": "y"}, CTX, armed=True)
    assert verdict.allowed and verdict.mode == "confirm"


def test_an_empty_or_oversized_note_is_refused(store):
    assert chat._remember(CTX, {"note": "  ", "reason": "r"}).proposal is None
    long = chat._remember(CTX, {"note": "x" * (notes.MAX_CHARS + 1), "reason": "r"})
    assert long.proposal is None and "keep it under" in long.text


def test_execution_writes_the_confirmed_note(store):
    p = policy.Pending("n1", CTX, "remember", {"note": "NEAR sits on mm5.", "reason": "r"}, 0.0)
    reply, tracked = execution.execute(p)
    assert tracked is None and "Noted for mm3" in reply
    assert [n.text for n in notes.read("mm3")] == ["NEAR sits on mm5."]
    assert notes.read("mm3")[0].author == "u7"


def test_an_unwritable_store_is_reported_not_raised(store, monkeypatch):
    def boom(*a, **k):
        raise OSError("read-only file system")
    monkeypatch.setattr(notes, "add", boom)
    p = policy.Pending("n1", CTX, "remember", {"note": "x", "reason": "r"}, 0.0)
    assert "could not write the desk note" in execution.execute(p)[0]
