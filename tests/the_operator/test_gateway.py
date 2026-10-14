"""Gateway parsing, chat gating, admin kill switch and history TTL (no Telegram, no model)."""

from __future__ import annotations

import pytest

from src.the_operator import agent, gateway
from src.the_operator.contract import Attachment, Context, Risk, ToolResult, ToolSpec
from src.the_operator.journal import Journal

SCHEMA = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
SPECS = [ToolSpec("agg", "d", SCHEMA, Risk.READ, lambda c, a: ToolResult("x")),
         ToolSpec("mm2only", "d", SCHEMA, Risk.READ, lambda c, a: ToolResult("x"), frozenset({"mm2"}))]


def test_parse_routes_subcommands_and_free_text():
    assert gateway.parse("/operator help") == ("help", "")
    assert gateway.parse("/Operator@bot why abc12") == ("why", "abc12")
    assert gateway.parse("/operator show the agg") == ("ask", "show the agg")
    assert gateway.parse("/operator") == ("ask", "")
    assert gateway.parse("/agg") is None and gateway.parse("") is None


def test_parse_tolerates_punctuation_stuck_to_the_command():
    """A trader typed "/operator, please do a zp" and got silence — it read as an unknown command."""
    assert gateway.parse("/operator, please do a zp") == ("ask", "please do a zp")
    assert gateway.parse("/operator: status") == ("status", "")
    assert gateway.parse("/operator@bot, show the agg") == ("ask", "show the agg")
    assert gateway.parse("/operators do a zp") is None       # still not our command


def test_chunks_split_on_newlines_under_limit():
    text = "\n".join(["line"] * 50)
    parts = gateway.chunks(text, limit=60)
    assert all(len(p) <= 60 for p in parts) and "\n".join(parts) == text


@pytest.mark.parametrize("rest", ["8bde", "req 8bde", "(req 8bde)", "REQ 8bde", ""])
def test_request_id_arg_tolerates_the_ack_wording(rest):
    assert gateway.request_id_arg(rest) == ("8bde" if rest else "")


def _gw(tmp_path, sent, buttons=None, cleared=None):
    def send(api, chat_id, text, **kw):
        sent.append((chat_id, text))

    def send_buttons(api, chat_id, text, rows):
        (buttons if buttons is not None else []).append((chat_id, text, rows))
        return 555

    def clear(api, chat_id, mid):
        (cleared if cleared is not None else []).append((chat_id, mid))
    return gateway.Gateway("api", None, SPECS, Journal(tmp_path), {"-1": "mm3", "-2": "mm2"},
                           {"-1", "-2"}, {"7"}, {"mm3": {"7", "8"}}, send=send, send_photo=None,
                           send_document=None, send_buttons=send_buttons,
                           answer_callback=lambda *a: None, clear_buttons=clear)


def _tap(chat, data, user):
    return {"callback_query": {"id": "cq1", "data": data, "from": {"id": int(user), "username": f"u{user}"},
                               "message": {"message_id": 555, "chat": {"id": int(chat)}}}}


def _upd(chat, text, user="1"):
    return {"message": {"text": text, "chat": {"id": int(chat)},
                        "from": {"id": int(user), "username": f"u{user}"}}}


def test_unserved_and_unpinned_chats_are_refused(tmp_path):
    sent = []
    gw = _gw(tmp_path, sent)
    gw.handle_update(_upd("-9", "/operator status"))
    assert "not enabled" in sent[-1][1]
    gw.served.add("-9")
    gw.handle_update(_upd("-9", "/operator status"))
    assert "not pinned" in sent[-1][1]


def test_is_trader_comes_from_the_desk_allowlist(tmp_path, monkeypatch):
    gw = _gw(tmp_path, [])
    gw.traders = {"mm3": {"7"}}
    seen = []
    monkeypatch.setattr(gateway.threading, "Thread",
                        lambda target, args, **kw: type("T", (), {"start": lambda self: seen.append(args[0])})())
    gw.handle_update(_upd("-1", "/operator buy dnx", user="7"))
    gw.handle_update(_upd("-1", "/operator buy dnx", user="1"))
    gw.handle_update(_upd("-2", "/operator buy dnx", user="7"))   # mm2 has no allowlist here
    assert [c.is_trader for c in seen] == [True, False, False]


def test_status_lists_only_that_desks_tools(tmp_path):
    sent = []
    gw = _gw(tmp_path, sent)
    gw.handle_update(_upd("-1", "/operator status"))
    assert "<b>Tools</b> (1) agg" in sent[-1][1] and "mm2only" not in sent[-1][1]
    gw.handle_update(_upd("-2", "/operator status"))
    assert "mm2only" in sent[-1][1]


def test_halt_resume_admin_only_and_journaled(tmp_path):
    sent = []
    gw = _gw(tmp_path, sent)
    gw.handle_update(_upd("-1", "/operator halt", user="1"))
    assert gw.armed and "not an Operator admin" in sent[-1][1]
    gw.handle_update(_upd("-1", "/operator halt", user="7"))
    assert not gw.armed and "HALTED" in sent[-1][1]
    gw.handle_update(_upd("-1", "/operator resume", user="7"))
    assert gw.armed
    admin_events = [e for e in gw.journal.find("-") if e["kind"] == "admin"]
    assert [e["allowed"] for e in admin_events] == [False, True, True]


def test_halted_request_is_refused_before_the_model(tmp_path, monkeypatch):
    sent = []
    gw = _gw(tmp_path, sent)
    gw.armed = False
    monkeypatch.setattr(gateway.agent, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model called")))
    gw.ask(Context("r1", "mm3", "-1", "1", "u1"), "show agg")
    assert "halted" in sent[-1][1]
    assert [e["kind"] for e in gw.journal.find("r1")] == ["request"]


def test_history_expires_after_ttl():
    h = gateway.History()
    h.add([{"role": "user", "content": "a"}], now=1000.0)
    assert len(h.messages(1000.0 + 60)) == 1
    assert h.messages(1000.0 + gateway.HISTORY_TTL_S + 1) == []


def test_history_keeps_whole_recent_exchanges():
    h = gateway.History()
    for i in range(gateway.HISTORY_EXCHANGES + 2):
        h.add([{"role": "user", "content": str(i)}, {"role": "assistant", "content": "r"}], now=1.0)
    msgs = h.messages(1.0)
    assert len(msgs) == gateway.HISTORY_EXCHANGES * 2 and msgs[0]["content"] == "2"


def test_confirm_flow_requester_only_once_and_executes(tmp_path, monkeypatch):
    sent, executed = [], []
    gw = _gw(tmp_path, sent)
    monkeypatch.setattr(gateway.execution, "execute",
                        lambda p: (executed.append(p) or "✅ Placed order", {"oid": "o1", "base": "DNX", "side": "buy"}))
    monkeypatch.setattr(gateway.threading, "Thread",
                        lambda target, args, **kw: type("T", (), {"start": lambda self: target(*args)})())
    ctx = Context("r1", "mm3", "-1", "7", "u7", is_trader=True)
    nonce = gw.propose(ctx, "place_order", {"command": "/trade dnx buy"})
    assert len(nonce) == 6 and nonce in gw.pending
    gw.handle_update(_upd("-1", f"/operator confirm {nonce}", user="1"))     # not the requester
    assert "only the requester" in sent[-1][1] and executed == []
    gw.handle_update(_upd("-2", f"/operator confirm {nonce}", user="7"))     # wrong chat
    assert "another chat" in sent[-1][1] and executed == []
    gw.handle_update(_upd("-1", f"/operator confirm {nonce}", user="7"))
    assert executed and executed[0].nonce == nonce and sent[-1][1] == "✅ Placed order"
    assert nonce not in gw.pending
    assert gw.fills["o1"]["chat_id"] == "-1" and gw.fills["o1"]["desk"] == "mm3"   # watched for fill
    gw.handle_update(_upd("-1", f"/operator confirm {nonce}", user="7"))     # a nonce fires once
    assert "No pending proposal" in sent[-1][1] and len(executed) == 1
    kinds = [e["kind"] for e in gw.journal.find("r1")]
    assert kinds == ["pending", "confirm", "confirm", "confirm", "execute"]


def test_confirm_expired_and_halted_are_refused(tmp_path, monkeypatch):
    sent = []
    gw = _gw(tmp_path, sent)
    ctx = Context("r1", "mm3", "-1", "7", "u7", is_trader=True)
    nonce = gw.propose(ctx, "place_order", {})
    gw.pending[nonce] = gateway.policy.Pending(nonce, ctx, "place_order", {},
                                               gw.pending[nonce].created - gateway.policy.CONFIRM_TTL_S - 1)
    gw.handle_update(_upd("-1", f"/operator confirm {nonce}", user="7"))
    assert "expired" in sent[-1][1]
    nonce = gw.propose(ctx, "place_order", {})
    gw.armed = False
    gw.handle_update(_upd("-1", f"/operator confirm {nonce}", user="7"))
    assert "halted" in sent[-1][1] and nonce in gw.pending


def test_buttons_follow_the_reply_and_a_tap_confirms_with_the_same_gate(tmp_path, monkeypatch):
    sent, buttons, cleared, executed = [], [], [], []
    gw = _gw(tmp_path, sent, buttons, cleared)
    monkeypatch.setattr(gateway.execution, "execute", lambda p: (executed.append(p) or "✅ Placed order", None))
    monkeypatch.setattr(gateway.threading, "Thread",
                        lambda target, args, **kw: type("T", (), {"start": lambda self: target(*args)})())
    ctx = Context("r1", "mm3", "-1", "7", "u7", is_trader=True)
    nonce = gw.propose(ctx, "place_order", {"command": "/trade dnx buy"})
    gw.send_confirm_buttons(ctx)
    assert len(buttons) == 1 and buttons[0][0] == "-1" and nonce in buttons[0][1]
    assert buttons[0][2] == [[("✅ Confirm", f"confirm:{nonce}"), ("❌ Abort", f"abort:{nonce}")]]
    gw.handle_update(_tap("-1", f"confirm:{nonce}", user="1"))      # someone else taps
    assert "only the requester" in sent[-1][1] and executed == [] and cleared == []
    gw.handle_update(_tap("-1", f"confirm:{nonce}", user="7"))      # the requester taps
    assert executed and cleared == [("-1", 555)] and nonce not in gw.pending
    gw.handle_update(_tap("-9", f"confirm:{nonce}", user="7"))      # unserved chat: ignored
    assert len(executed) == 1


def test_abort_tap_and_slash_both_retire_the_buttons(tmp_path):
    sent, buttons, cleared = [], [], []
    gw = _gw(tmp_path, sent, buttons, cleared)
    ctx = Context("r1", "mm3", "-1", "7", "u7", is_trader=True)
    nonce = gw.propose(ctx, "place_order", {})
    gw.send_confirm_buttons(ctx)
    gw.handle_update(_tap("-1", f"abort:{nonce}", user="7"))
    assert sent[-1][1].startswith("Aborted") and cleared == [("-1", 555)]
    nonce = gw.propose(ctx, "place_order", {})
    gw.send_confirm_buttons(ctx)
    gw.handle_update(_upd("-1", f"/operator abort {nonce}", user="7"))
    assert cleared[-1] == ("-1", 555) and nonce not in gw.pending


def test_four_eyes_needs_a_second_desk_trader_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway.policy, "FOUR_EYES_USD", 10_000)
    sent, buttons, executed = [], [], []
    gw = _gw(tmp_path, sent, buttons)
    monkeypatch.setattr(gateway.execution, "execute", lambda p: (executed.append(p) or "✅ Placed", None))
    monkeypatch.setattr(gateway.threading, "Thread",
                        lambda target, args, **kw: type("T", (), {"start": lambda self: target(*args)})())
    ctx = Context("r1", "mm3", "-1", "7", "u7", is_trader=True)
    nonce = gw.propose(ctx, "place_order", {"notional_usd": 10_001})
    gw.send_confirm_buttons(ctx)
    assert "a second mm3 trader must confirm" in buttons[0][1]
    gw.handle_update(_tap("-1", f"confirm:{nonce}", user="7"))      # requester: refused
    assert "four-eyes" in sent[-1][1] and executed == []
    gw.handle_update(_tap("-1", f"confirm:{nonce}", user="1"))      # not a trader: refused
    assert "only a mm3 trader" in sent[-1][1] and executed == []
    gw.handle_update(_tap("-1", f"confirm:{nonce}", user="8"))      # second trader: executes
    assert len(executed) == 1 and nonce not in gw.pending


def test_halt_drops_every_pending_proposal(tmp_path):
    sent, buttons, cleared = [], [], []
    gw = _gw(tmp_path, sent, buttons, cleared)
    ctx = Context("r1", "mm3", "-1", "7", "u7", is_trader=True)
    for _ in range(2):
        gw.propose(ctx, "place_order", {})
    gw.send_confirm_buttons(ctx)
    gw.handle_update(_upd("-1", "/operator halt", user="7"))
    assert gw.pending == {} and "Dropped 2 pending proposal(s); 0 remain" in sent[-1][1] and len(cleared) == 2


def test_abort_is_requester_only(tmp_path):
    sent = []
    gw = _gw(tmp_path, sent)
    nonce = gw.propose(Context("r1", "mm3", "-1", "7", "u7", True), "place_order", {})
    gw.handle_update(_upd("-1", f"/operator abort {nonce}", user="1"))
    assert "Nothing to abort" in sent[-1][1] and nonce in gw.pending
    gw.handle_update(_upd("-1", f"/operator abort {nonce}", user="7"))
    assert sent[-1][1].startswith("Aborted") and nonce not in gw.pending


def test_deliver_sends_verbatim_cards_before_the_reply(tmp_path):
    sent = []
    gw = _gw(tmp_path, sent)
    gw.deliver(Context("r1", "mm3", "-1", "7", "u7"), "my take", [], ["📡 card one", "card two"])
    assert [t for _c, t in sent] == ["📡 card one", "card two", "my take"]


def test_room_talk_is_recorded_for_served_chats_only(tmp_path):
    gateway.chatlog._log.clear()
    gw = _gw(tmp_path, [])
    gw.handle_update({"message": {"text": "zec is ripping", "chat": {"id": -1}, "date": 1_800_000_000,
                                  "from": {"id": 5, "username": "trader"}}})
    gw.handle_update({"message": {"text": "ignored", "chat": {"id": -9}, "from": {"id": 5, "username": "t"}}})
    gw.handle_update({"message": {"text": "bot noise", "chat": {"id": -1}, "from": {"id": 6, "is_bot": True}}})
    assert [(ln.user, ln.text) for ln in gateway.chatlog._log["-1"]] == [("trader", "zec is ripping")]
    assert "-9" not in gateway.chatlog._log


def test_format_why_renders_the_trail(tmp_path):
    j = Journal(tmp_path)
    j.write("request", "r1", user_name="sam", desk="mm3", text="agg")
    j.write("tool_call", "r1", tool="agg", args={}, allowed=True, reason="read")
    j.write("tool_result", "r1", tool="agg", ok=True, ms=12)
    j.write("reply", "r1", chars=40, attachments=1)
    out = gateway.format_why(j.find("r1"))
    assert "request by sam on mm3: agg" in out and "call agg" in out and "12ms" in out
    assert gateway.format_why([]).startswith("No journal entries")


def test_execution_and_abort_land_in_the_history_the_model_sees(tmp_path, monkeypatch):
    """The order id is born outside the model's loop; a follow-up must still be able to read it."""
    sent, seen = [], []
    gw = _gw(tmp_path, sent)
    monkeypatch.setattr(gateway.execution, "execute", lambda p: ("✅ Placed order `o1`", None))
    monkeypatch.setattr(gateway.threading, "Thread",
                        lambda target, args, **kw: type("T", (), {"start": lambda self: target(*args)})())
    monkeypatch.setattr(gateway.agent, "run",
                        lambda client, ctx, text, history, *a, **k:
                        seen.append(history) or gateway.agent.Reply("ok", turns=[]))
    ctx = Context("r1", "mm3", "-1", "7", "u7", is_trader=True)

    nonce = gw.propose(ctx, "place_order", {"command": "/trade dnx buy"})
    gw.handle_update(_tap("-1", f"confirm:{nonce}", user="7"))
    gw.ask(Context("r2", "mm3", "-1", "7", "u7", is_trader=True), "cancel that order")
    assert "`o1`" in seen[-1][-1]["content"] and seen[-1][-1]["role"] == "assistant"

    nonce = gw.propose(ctx, "place_order", {"command": "/trade dnx buy"})
    gw.handle_update(_tap("-1", f"abort:{nonce}", user="7"))
    gw.ask(Context("r3", "mm3", "-1", "7", "u7", is_trader=True), "is it still live?")
    assert f"Aborted {nonce}" in seen[-1][-1]["content"]


def test_the_gateway_starts_empty_and_persists_every_change(tmp_path, monkeypatch):
    """Restoring is main()'s job; a constructed Gateway must never inherit a live state file."""
    sent, saved = [], []
    gw = _gw(tmp_path, sent)
    assert gw.pending == {} and gw.fills == {}
    monkeypatch.setattr(gateway.state, "dump",
                        lambda pending, fills: saved.append((set(pending), dict(fills))))
    monkeypatch.setattr(gateway.execution, "execute",
                        lambda p: ("✅ Placed order `o1`", {"oid": "o1", "base": "DNX"}))
    monkeypatch.setattr(gateway.threading, "Thread",
                        lambda target, args, **kw: type("T", (), {"start": lambda self: target(*args)})())
    ctx = Context("r1", "mm3", "-1", "7", "u7", is_trader=True)
    nonce = gw.propose(ctx, "place_order", {"command": "/trade dnx buy"})
    assert saved[-1][0] == {nonce}                       # proposal saved
    gw.handle_update(_tap("-1", f"confirm:{nonce}", user="7"))
    assert saved[-1][0] == set()                         # nonce consumed
    assert "o1" in saved[-1][1]                          # and the live order is watched on disk


def test_an_unlisted_user_is_refused_before_the_model_runs(tmp_path, monkeypatch):
    """Going into a real desk chat means everyone in it can type: the gate is the user list."""
    sent = []
    gw = _gw(tmp_path, sent)
    gw.users = {"7"}
    monkeypatch.setattr(gateway.agent, "run",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("model called")))
    gw.handle_update(_upd("-1", "/operator show the agg", user="999"))
    assert "not on the Operator's user list" in sent[-1][1] and "999" in sent[-1][1]
    assert [e["kind"] for e in gw.journal.find("-")][-1:] == ["refused"]


def test_an_empty_user_list_serves_everyone(tmp_path, monkeypatch):
    """Pilot behaviour: unset means unrestricted, and main() logs a warning about it."""
    sent, seen = [], []
    gw = _gw(tmp_path, sent)
    assert gw.users == set()
    monkeypatch.setattr(gateway.threading, "Thread",
                        lambda target, args, **kw: type("T", (), {"start": lambda self: target(*args)})())
    monkeypatch.setattr(gateway.agent, "run",
                        lambda *a, **k: seen.append(1) or gateway.agent.Reply("ok", turns=[]))
    gw.handle_update(_upd("-1", "/operator show the agg", user="999"))
    assert seen == [1]


def test_the_user_gate_does_not_block_admins_from_halting(tmp_path):
    sent = []
    gw = _gw(tmp_path, sent)
    gw.users = {"7"}
    gw.handle_update(_upd("-1", "/operator halt", user="999"))
    assert "not on the Operator's user list" in sent[-1][1] and gw.armed


def test_ask_posts_a_tools_table_before_the_models_reply(tmp_path, monkeypatch):
    sent = []
    gw = _gw(tmp_path, sent)
    gw._send_photo = lambda api, chat_id, data, caption=None: sent.append((chat_id, f"photo {caption}"))

    def run(client, ctx, text, history, specs, facts, journal, is_armed, propose, post=None):
        post([Attachment("agg.png", b"png", "photo", caption="MM3 · 33 rows")], [])
        return agent.Reply("PV is up on QI", [], [{"role": "user", "content": text}], "end_turn")
    monkeypatch.setattr(gateway.agent, "run", run)
    monkeypatch.setattr(gateway.desk, "desk_facts", lambda d: "")
    gw.ask(Context("r1", "mm3", "-1", "7", "u7"), "show agg")
    texts = [t for _c, t in sent]
    assert texts[0].startswith("Working on it")
    assert texts[1:] == ["photo MM3 · 33 rows", "PV is up on QI"]
