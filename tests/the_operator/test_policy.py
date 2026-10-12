"""Policy verdicts and confirm rules are pure functions (risk, desk, trader, kill switch, args, time)."""

from __future__ import annotations

import pytest

from src.the_operator import policy
from src.the_operator.contract import Context, Risk, ToolResult, ToolSpec

CTX = Context("r1", "mm3", "-100", "42", "sam")
TRADER = Context("r1", "mm3", "-100", "42", "sam", is_trader=True)
SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}, "reason": {"type": "string"}},
          "required": ["a", "reason"], "additionalProperties": False}


def _spec(risk=Risk.READ, desks=None, name="t"):
    return ToolSpec(name, "d", SCHEMA, risk, lambda c, a: ToolResult("ok"), desks)


def test_read_tool_executes():
    v = policy.check(_spec(), {"a": "x"}, CTX, armed=True)
    assert v.allowed and v.mode == "execute"


def test_kill_switch_beats_everything():
    v = policy.check(_spec(), {"a": "x"}, CTX, armed=False)
    assert not v.allowed and v.mode == "denied" and "halted" in v.reason


def test_control_is_still_denied():
    """Nothing starts or stops an external system yet; CONTROL has no mode in any phase."""
    v = policy.check(_spec(Risk.CONTROL), {"a": "x", "reason": "r"}, TRADER, armed=True)
    assert not v.allowed and Risk.CONTROL.value in v.reason


def test_write_is_confirmed_and_open_to_anyone_on_the_desk():
    """A desk note joins every later prompt, so it takes a confirm — but not the trader allowlist,
    which guards money."""
    v = policy.check(_spec(Risk.WRITE), {"a": "x", "reason": "r"}, CTX, armed=True)
    assert v.allowed and v.mode == "confirm"


def test_money_needs_confirm_for_a_trader_with_a_reason():
    v = policy.check(_spec(Risk.MONEY), {"a": "x", "reason": "sam asked"}, TRADER, armed=True)
    assert v.allowed and v.mode == "confirm"


def test_scalp_is_confirm_in_phase3():
    v = policy.check(_spec(Risk.MONEY, name="place_scalp"), {"a": "x", "reason": "r"}, TRADER, armed=True)
    assert v.allowed and v.mode == "confirm"
    assert policy.TOOL_MODES == {} and policy.MONEY_STATUS is None


def test_money_denied_for_non_trader():
    v = policy.check(_spec(Risk.MONEY), {"a": "x", "reason": "r"}, CTX, armed=True)
    assert not v.allowed and "trader allowlist" in v.reason


@pytest.mark.parametrize("reason", ["", "   ", None])
def test_money_denied_without_a_reason(reason):
    v = policy.check(_spec(Risk.MONEY), {"a": "x", "reason": reason}, TRADER, armed=True)
    assert not v.allowed and "reason" in v.reason


def test_desk_restriction_enforced():
    v = policy.check(_spec(desks=frozenset({"mm2"})), {"a": "x"}, CTX, armed=True)
    assert not v.allowed and "mm3" in v.reason
    assert policy.check(_spec(desks=frozenset({"mm3"})), {"a": "x"}, CTX, armed=True).allowed


def test_unexpected_argument_denied():
    v = policy.check(_spec(), {"a": "x", "zz": 1}, CTX, armed=True)
    assert not v.allowed and "zz" in v.reason


# ---- confirm ------------------------------------------------------------------------------------
P = policy.Pending("ab12cd", TRADER, "place_order", {"command": "/trade dnx buy"}, created=1000.0)


def test_confirm_by_requester_in_time_executes():
    v = policy.confirm_check(P, "-100", "42", 1000.0 + 30, armed=True)
    assert v.allowed and v.mode == "execute"


def test_confirm_refused_for_other_user_other_chat_expiry_and_halt():
    assert "only the requester" in policy.confirm_check(P, "-100", "99", 1010.0, armed=True).reason
    assert "another chat" in policy.confirm_check(P, "-200", "42", 1010.0, armed=True).reason
    assert "expired" in policy.confirm_check(P, "-100", "42", 1000.0 + policy.CONFIRM_TTL_S + 1,
                                            armed=True).reason
    assert "halted" in policy.confirm_check(P, "-100", "42", 1010.0, armed=False).reason


BIG = policy.Pending("big001", TRADER, "place_order", {}, created=1000.0, notional_usd=10_001)


def test_four_eyes_is_off_by_default_requester_confirms_any_size():
    assert policy.FOUR_EYES_USD is None and not policy.needs_four_eyes(BIG)
    assert policy.four_eyes_note(1e9, "mm3") is None
    v = policy.confirm_check(BIG, "-100", "42", 1010.0, armed=True, desk_traders={"42", "99"})
    assert v.allowed and "requester" in v.reason


def test_four_eyes_when_enabled(monkeypatch):
    monkeypatch.setattr(policy, "FOUR_EYES_USD", 10_000)
    v = policy.confirm_check(BIG, "-100", "42", 1010.0, armed=True, desk_traders={"42", "99"})
    assert not v.allowed and "four-eyes" in v.reason
    ok = policy.confirm_check(BIG, "-100", "99", 1010.0, armed=True, desk_traders={"42", "99"})
    assert ok.allowed and "second" in ok.reason
    no = policy.confirm_check(BIG, "-100", "77", 1010.0, armed=True, desk_traders={"42", "99"})
    assert not no.allowed and "only a mm3 trader" in no.reason
    at = policy.Pending("n", TRADER, "place_order", {}, 1000.0, notional_usd=10_000)
    assert not policy.needs_four_eyes(at) and policy.needs_four_eyes(BIG)
    assert "👥 $10,001 is above the $10,000" in policy.four_eyes_note(10_001, "mm3")


def test_limit_drift_rule():
    assert policy.limit_drift_ok("0.001536", "0.001536")
    assert policy.limit_drift_ok("0.001536", "0.001548")          # ~0.8%, a few DNX ticks
    assert not policy.limit_drift_ok("0.001536", "0.001700")      # ~11%
    assert policy.limit_drift_ok(None, None) and not policy.limit_drift_ok(None, "1")
