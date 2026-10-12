"""Every tool invocation is gated, journaled, and never raises into the runner."""

from __future__ import annotations

from src.the_operator import executor
from src.the_operator.contract import Attachment, Context, Risk, ToolResult, ToolSpec
from src.the_operator.journal import Journal

CTX = Context("r1", "mm3", "-100", "42", "sam")
SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"],
          "additionalProperties": False}


def _spec(name, handler, risk=Risk.READ, desks=None):
    return ToolSpec(name, "d", SCHEMA, risk, handler, desks)


def test_available_filters_by_desk():
    specs = [_spec("x", None), _spec("y", None, desks=frozenset({"mm2"}))]
    assert [s.name for s in executor.available(specs, "mm3")] == ["x"]


def test_allowed_call_runs_handler_diverts_attachments_and_journals(tmp_path):
    j, sink = Journal(tmp_path), []
    att = Attachment("t.png", b"png", "photo", caption="cap")
    spec = _spec("agg", lambda c, a: ToolResult(f"rows for {a['a']}", [att]))
    tool = executor.build_tools([spec], CTX, sink, j, lambda: True)[0]
    assert "strict" not in tool.to_dict() and tool.to_dict()["name"] == "agg"
    out = tool.call({"a": "DNX"})
    assert out.startswith("rows for DNX") and "t.png" in out and "png" not in out.split("\n")[0][9:]
    assert sink == [att]
    kinds = [e["kind"] for e in j.find("r1")]
    assert kinds == ["tool_call", "tool_result"]
    assert j.find("r1")[1]["attachments"][0]["filename"] == "t.png"


def test_denied_call_never_reaches_handler(tmp_path):
    j, calls = Journal(tmp_path), []
    spec = _spec("m", lambda c, a: calls.append(1), risk=Risk.MONEY)
    tool = executor.build_tools([spec], CTX, [], j, lambda: True)[0]
    out = tool.call({"a": "x"})
    assert out.startswith("DENIED by policy") and calls == []
    assert j.find("r1")[0]["allowed"] is False


def test_kill_switch_read_live_per_call(tmp_path):
    armed = {"v": True}
    spec = _spec("agg", lambda c, a: ToolResult("ok"))
    tool = executor.build_tools([spec], CTX, [], Journal(tmp_path), lambda: armed["v"])[0]
    assert tool.call({"a": "x"}) == "ok"
    armed["v"] = False
    assert tool.call({"a": "x"}).startswith("DENIED")


TRADER = Context("r1", "mm3", "-100", "42", "sam", is_trader=True)
MONEY_SCHEMA = {"type": "object", "properties": {"reason": {"type": "string"}},
                "required": ["reason"], "additionalProperties": False}


def test_dry_run_money_call_is_stamped_and_proposal_journaled(tmp_path, monkeypatch):
    monkeypatch.setattr(executor.policy, "TOOL_MODES", {"place_scalp": "dry_run"})
    j = Journal(tmp_path)
    spec = ToolSpec("place_scalp", "d", MONEY_SCHEMA, Risk.MONEY,
                    lambda c, a: ToolResult("SCALP PREVIEW", proposal={"kind": "place_scalp"}))
    tool = executor.build_tools([spec], TRADER, [], j, lambda: True)[0]
    assert tool.to_dict()["strict"] is True      # money tools are the strict ones
    out = tool.call({"reason": "sam asked"})
    assert out.startswith("SCALP PREVIEW") and "DRY RUN: nothing was sent" in out
    res = j.find("r1")[1]
    assert res["mode"] == "dry_run" and res["proposal"] == {"kind": "place_scalp"}


def test_confirm_mode_registers_the_proposal_and_stamps_the_nonce(tmp_path):
    j, registered = Journal(tmp_path), []

    def propose(ctx, tool, proposal):
        registered.append((ctx.request_id, tool, proposal))
        return "ab12cd"

    cards = []
    spec = ToolSpec("place_order", "d", MONEY_SCHEMA, Risk.MONEY,
                    lambda c, a: ToolResult("ORDER PREVIEW", proposal={"command": "/trade dnx buy"}))
    tool = executor.build_tools([spec], TRADER, [], j, lambda: True, propose, cards)[0]
    out = tool.call({"reason": "sam asked"})
    assert registered == [("r1", "place_order", {"command": "/trade dnx buy"})]
    assert "PENDING CONFIRMATION" in out and "code ab12cd" in out and "DRY RUN" not in out
    assert cards == ["ORDER PREVIEW"]           # delivered verbatim by the gateway
    res = j.find("r1")[1]
    assert res["mode"] == "confirm" and res["nonce"] == "ab12cd"


def test_confirm_mode_refusal_registers_nothing(tmp_path):
    registered = []
    spec = ToolSpec("place_order", "d", MONEY_SCHEMA, Risk.MONEY,
                    lambda c, a: ToolResult("❌ exceeds the cap"))
    tool = executor.build_tools([spec], TRADER, [], Journal(tmp_path), lambda: True,
                                lambda *a: registered.append(a) or "x")[0]
    out = tool.call({"reason": "r"})
    assert out == "❌ exceeds the cap" and registered == []


def test_verbatim_result_goes_to_extras_and_tells_the_model(tmp_path):
    extras = []
    spec = _spec("buzz", lambda c, a: ToolResult("📡 BUZZ card", verbatim=True))
    tool = executor.build_tools([spec], CTX, [], Journal(tmp_path), lambda: True, None, None, extras)[0]
    out = tool.call({"a": "x"})
    assert extras == ["📡 BUZZ card"] and out.startswith("📡 BUZZ card") and "ALREADY been sent" in out


def test_handler_exception_becomes_error_text(tmp_path):
    j = Journal(tmp_path)

    def boom(c, a):
        raise RuntimeError("pluto down")

    tool = executor.build_tools([_spec("v", boom)], CTX, [], j, lambda: True)[0]
    out = tool.call({"a": "x"})
    assert out == "ERROR: v failed: RuntimeError: pluto down"
    res = j.find("r1")[1]
    assert res["ok"] is False and "pluto down" in res["error"]
