"""Chat -> desk pins: disjoint, explicit, never defaulted."""

from __future__ import annotations

import pytest

from src.the_operator import desk


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for d in desk.DESKS.values():
        monkeypatch.delenv(d.chat_env, raising=False)
    monkeypatch.delenv("TELEGRAM_OPERATOR_CHAT_ID", raising=False)
    monkeypatch.delenv("OPERATOR_ADMINS", raising=False)
    monkeypatch.setattr(desk, "load_dotenv", lambda: None)   # the real .env must not leak in


def test_pins_map_each_chat_to_its_desk(monkeypatch):
    monkeypatch.setenv("TELEGRAM_MM3_CHAT_ID", "-1, -2")
    monkeypatch.setenv("TELEGRAM_MM2_CHAT_ID", "-3")
    assert desk.desk_by_chat() == {"-1": "mm3", "-2": "mm3", "-3": "mm2"}


def test_overlapping_pins_refused(monkeypatch):
    monkeypatch.setenv("TELEGRAM_MM3_CHAT_ID", "-1")
    monkeypatch.setenv("TELEGRAM_TREASURY_CHAT_ID", "-1")
    with pytest.raises(RuntimeError, match="pinned to both"):
        desk.desk_by_chat()


def test_unpinned_chat_has_no_desk(monkeypatch):
    monkeypatch.setenv("TELEGRAM_MM3_CHAT_ID", "-1")
    assert desk.desk_by_chat().get("-9") is None


def test_served_and_admins_default_empty():
    assert desk.served_chats() == set() and desk.admins() == set()


def test_cr_token_absent_for_pluto_only_desks():
    with pytest.raises(KeyError):
        desk.cr_token("options")


def test_desk_facts_lists_roster_and_playbook(monkeypatch, tmp_path):
    monkeypatch.setattr(desk, "load_config", lambda _p: {
        "mm3": {"strategy": {"DNX": "s1", "ELA": "s2"}, "perp_strategy": "hedge-j",
                "bots": ["*MM3_JASONDH"]}})
    (tmp_path / "operator").mkdir()
    (tmp_path / "operator" / "mm3.md").write_text("Playbook line.")
    monkeypatch.setattr(desk, "CONFIG_DIR", tmp_path)
    facts = desk.desk_facts("mm3")
    assert "DNX, ELA" in facts and "hedge-j" in facts and "MM3_JASONDH" in facts
    assert facts.endswith("Playbook line.")
