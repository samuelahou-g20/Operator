"""agent.run drives a (fake) tool runner: journals turns, mirrors history, shapes the reply."""

from __future__ import annotations

from types import SimpleNamespace as NS

from src.the_operator import agent
from src.the_operator.contract import Attachment, Context, Risk, ToolResult, ToolSpec
from src.the_operator.journal import Journal

CTX = Context("r1", "mm3", "-100", "42", "sam")
SCHEMA = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
USAGE = NS(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0)


def _msg(content, stop):
    return NS(content=content, stop_reason=stop, model="m", usage=USAGE, _request_id="req_x",
              stop_details=None)


class FakeRunner:
    """Yields a tool_use turn, runs the tool the way the SDK would, then a final text turn."""

    def __init__(self, tools, script):
        self.tools = {t.to_dict()["name"]: t for t in tools}
        self.script = script
        self._pending = None

    def __iter__(self):
        for step in self.script:
            if step[0] == "tool":
                name = step[1]
                out = self.tools[name].call({})
                self._pending = {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu1", "content": out}]}
                yield _msg([NS(type="tool_use", name=name, id="tu1", input={})], "tool_use")
            else:
                self._pending = None
                yield _msg([NS(type="text", text=step[1])], step[2])

    def generate_tool_call_response(self):
        return self._pending


def _client(script, capture):
    def tool_runner(**kw):
        capture.update(kw)
        return FakeRunner(kw["tools"], script)
    return NS(beta=NS(messages=NS(tool_runner=tool_runner)))


def test_run_journals_turns_mirrors_history_and_diverts_attachments(tmp_path):
    j, cap = Journal(tmp_path), {}
    att = Attachment("a.png", b"x", "photo")
    specs = [ToolSpec("agg", "d", SCHEMA, Risk.READ, lambda c, a: ToolResult("table", [att]))]
    client = _client([("tool", "agg"), ("text", "PV is 1.2m", "end_turn")], cap)
    reply = agent.run(client, CTX, "show agg", [], specs, "Desk: MM3.", j, lambda: True)
    assert reply.text == "PV is 1.2m" and reply.attachments == [att]
    # history mirror: user, assistant(tool_use), user(tool_result), assistant(text)
    assert [t["role"] for t in reply.turns] == ["user", "assistant", "user", "assistant"]
    assert reply.turns[0]["content"] == "[sam] show agg"
    kinds = [e["kind"] for e in j.find("r1")]
    assert kinds == ["turn", "tool_call", "tool_result", "turn"] or \
        kinds == ["tool_call", "tool_result", "turn", "turn"]
    assert cap["model"] == agent.MODEL and cap["max_iterations"] == agent.MAX_ITERATIONS
    assert cap["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "Desk: MM3." in cap["system"][1]["text"]


def test_verbatim_tool_card_is_returned_as_extra_and_the_model_still_replies(tmp_path):
    j = Journal(tmp_path)
    specs = [ToolSpec("buzz", "d", SCHEMA, Risk.READ, lambda c, a: ToolResult("📡 BUZZ card", verbatim=True))]
    client = _client([("tool", "buzz"), ("text", "ZEC is the one that matters for us.", "end_turn")], {})
    reply = agent.run(client, CTX, "what's trending", [], specs, "", j, lambda: True)
    assert reply.extras == ["📡 BUZZ card"] and reply.text == "ZEC is the one that matters for us."


def test_history_is_passed_before_the_new_turn(tmp_path):
    cap = {}
    client = _client([("text", "ok", "end_turn")], cap)
    prior = [{"role": "user", "content": "[sam] hi"}, {"role": "assistant", "content": "hello"}]
    agent.run(client, CTX, "again", prior, [], "", Journal(tmp_path), lambda: True)
    assert cap["messages"][:2] == prior and cap["messages"][2]["content"] == "[sam] again"


def test_tools_not_on_desk_are_not_offered(tmp_path):
    cap = {}
    specs = [ToolSpec("only_mm2", "d", SCHEMA, Risk.READ, lambda c, a: ToolResult("x"),
                      frozenset({"mm2"}))]
    agent.run(_client([("text", "ok", "end_turn")], cap), CTX, "q", [], specs, "",
              Journal(tmp_path), lambda: True)
    assert cap["tools"] == []


def test_money_card_short_circuits_the_paraphrase_turn(tmp_path):
    j, cap = Journal(tmp_path), {}
    trader = Context("r1", "mm3", "-100", "42", "sam", is_trader=True)
    schema = {"type": "object", "properties": {"reason": {"type": "string"}},
              "required": ["reason"], "additionalProperties": False}
    specs = [ToolSpec("place_order", "d", schema, Risk.MONEY,
                      lambda c, a: ToolResult("🟢 BUY DNX card", proposal={"command": "/trade dnx buy"}))]

    class Runner(FakeRunner):
        def __iter__(self):
            out = self.tools["place_order"].call({"reason": "r"})
            self._pending = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": out}]}
            yield _msg([NS(type="tool_use", name="place_order", id="tu1", input={"reason": "r"})], "tool_use")
            raise AssertionError("a second model turn must not happen after a card")

    def tool_runner(**kw):
        cap.update(kw)
        return Runner(kw["tools"], [])
    client = NS(beta=NS(messages=NS(tool_runner=tool_runner)))
    reply = agent.run(client, trader, "buy dnx", [], specs, "", j, lambda: True, lambda c, t, p: "ab12cd")
    assert reply.text == "🟢 BUY DNX card" and reply.stop_reason == "card"
    assert [t["role"] for t in reply.turns] == ["user", "assistant", "user", "assistant"]
    assert reply.turns[-1]["content"] == "🟢 BUY DNX card"
    assert cap["output_config"] == {"effort": agent.EFFORT}
    assert [e["kind"] for e in j.find("r1")][-1] == "card"


def test_refusal_and_iteration_cap_are_reported(tmp_path):
    j = Journal(tmp_path)
    r = agent.run(_client([("text", "", "refusal")], {}), CTX, "q", [], [], "", j, lambda: True)
    assert r.text.startswith("The model declined")
    specs = [ToolSpec("agg", "d", SCHEMA, Risk.READ, lambda c, a: ToolResult("t"))]
    r = agent.run(_client([("tool", "agg")], {}), CTX, "q", [], specs, "", j, lambda: True)
    assert "without a final answer" in r.text


def test_system_prompt_carries_the_clock_outside_the_cached_block(tmp_path):
    """Without a clock the model cannot answer anything anchored to a time ("since 00:00 UTC");
    with it in the CACHED block, every request would miss the prompt cache."""
    from datetime import datetime, timezone
    blocks = agent.system_blocks(CTX, "Desk: MM3.")
    assert blocks[0]["cache_control"] == {"type": "ephemeral"} and "Current time" not in blocks[0]["text"]
    assert f"{datetime.now(timezone.utc):%Y-%m-%d}" in blocks[1]["text"] and "UTC" in blocks[1]["text"]
