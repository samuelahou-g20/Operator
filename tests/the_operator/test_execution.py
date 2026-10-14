"""execution.execute re-checks, keeps the confirmed limit, applies previewed ladders, and is the
only place a trader is built (offline)."""

from __future__ import annotations

import inspect

from scripts import trade_bot as tb
from src.the_operator import execution, policy
from src.the_operator.contract import Context

CTX = Context("r1", "mm3", "-1", "7", "sam", is_trader=True)


class FakeTrader:
    calls: list[dict] = []

    def __init__(self, api_token):
        self.api_token = api_token

    def place(self, **kw):
        FakeTrader.calls.append(kw)
        return {"client_order_id": "oid-1"}

    def cancel(self, oid):
        FakeTrader.calls.append({"cancel": oid})

    def get_order(self, oid):
        return {"executed_qty": 0}


def _plan(limit="0.0015", status="open"):
    return {"order": {"strategy": "S", "currency_pair": "DNX-USDT", "side": "buy", "quantity": 1000,
                      "quantity_currency": "USDT", "aggression": "passive"},
            "base": "DNX", "limit": limit, "touch": 0.0015, "status": status,
            "place_kwargs": {"strategy": "S", "currency_pair": "DNX-USDT", "side": "buy",
                             "quantity": 1000, "quantity_currency": "USDT", "aggression": "passive",
                             "order_status": status, "limit_price": limit, "post_only": None,
                             "max_post_size": 1.0, "extra": None}}


def _pending(limit="0.0015", tool="place_order", proposal=None):
    proposal = proposal or {"command": "/trade dnx buy size=1000usd passive",
                            "place_kwargs": _plan(limit)["place_kwargs"]}
    return policy.Pending("ab12cd", CTX, tool, proposal, 0.0)


def _wire(monkeypatch, fresh_plan=None, err=None):
    FakeTrader.calls = []
    execution._traders.clear()
    monkeypatch.setattr(execution, "cr_token", lambda desk: "tok")
    monkeypatch.setattr(execution, "CoinRoutesClient", lambda api_token: object())
    monkeypatch.setattr(execution, "CoinRoutesTrader", FakeTrader)
    seen = {}

    def prepare(text, client, desk):
        seen["text"] = text
        return (None, err) if err else (fresh_plan or _plan(), None)
    monkeypatch.setattr(tb, "prepare_order", prepare)
    return seen


def test_place_rechecks_and_sends_with_the_status_trade_would_use(monkeypatch):
    seen = _wire(monkeypatch)
    reply, tracked = execution.execute(_pending())
    assert seen["text"] == "/trade dnx buy size=1000usd passive"
    assert len(FakeTrader.calls) == 1 and FakeTrader.calls[0]["order_status"] == "open"
    assert reply.startswith("✅ Placed order `oid-1`") and "Staged" not in reply
    assert tracked == {"oid": "oid-1", "base": "DNX", "side": "buy"}     # watched for its fill


def test_requested_staged_status_is_kept(monkeypatch):
    _wire(monkeypatch, fresh_plan=_plan(status="staged"))
    reply, tracked = execution.execute(_pending())
    assert FakeTrader.calls[0]["order_status"] == "staged" and "release it in the CoinRoutes UI" in reply
    assert tracked is None


def test_place_refused_when_recheck_refuses(monkeypatch):
    _wire(monkeypatch, err="❌ $1,000,000 exceeds the $50,000 cap.")
    reply, tracked = execution.execute(_pending())
    assert reply.startswith("❌ not placed") and "cap" in reply and FakeTrader.calls == [] and tracked is None


def test_place_refused_when_market_dislocated(monkeypatch):
    _wire(monkeypatch, fresh_plan=_plan(limit="0.0017"))          # +13%
    reply, _ = execution.execute(_pending(limit="0.0015"))
    assert "market moved more than 5%" in reply and FakeTrader.calls == []


def test_ordinary_drift_places_at_the_confirmed_limit(monkeypatch):
    _wire(monkeypatch, fresh_plan=_plan(limit="0.001548"))        # +0.8%: the DNX case
    reply, _ = execution.execute(_pending(limit="0.001536"))
    assert FakeTrader.calls[0]["limit_price"] == "0.001536"       # what the human confirmed
    assert FakeTrader.calls[0]["max_post_size"] == 1.0            # but fresh book-derived fields
    assert "touch moved since the preview" in reply and "0.001548" in reply


def test_scalp_applies_the_previewed_ladder_after_a_mid_check(monkeypatch):
    _wire(monkeypatch)
    applied = {}
    monkeypatch.setattr(tb, "touch_on", lambda c, pair, side, pxv: 0.00153 if side == "sell" else 0.00154)
    monkeypatch.setattr(tb, "apply_ladder", lambda want, c, t, **kw: applied.update(want=want, **kw) or ["sides"])
    monkeypatch.setattr(tb, "render_scalper_text", lambda head, sides, kmap: f"{head} applied {sides}")
    want = {"buy": [(0.00146, 2000, "usd")], "sell": [(0.00161, 2000, "usd")]}
    pr = {"command": "/scalper dnx 2000 1", "base": "DNX", "pair": "DNX-USDT", "strategy": "S",
          "pxv": ["gateio"], "venues": None, "mid": 0.001535, "want": want,
          "kmap": {0.00146: 1, 0.00161: 1}, "head": "DNX scalper"}
    reply, tracked = execution.execute(_pending(tool="place_scalp", proposal=pr))
    assert applied["want"] is want and applied["strategy"] == "S" and applied["pxv"] == ["gateio"]
    assert reply == "DNX scalper applied ['sides']" and tracked is None


def test_scalp_refused_when_mid_dislocated(monkeypatch):
    _wire(monkeypatch)
    monkeypatch.setattr(tb, "touch_on", lambda c, pair, side, pxv: 0.0020)
    monkeypatch.setattr(tb, "apply_ladder", lambda *a, **k: (_ for _ in ()).throw(AssertionError("applied")))
    pr = {"command": "x", "base": "DNX", "pair": "DNX-USDT", "strategy": "S", "pxv": [], "venues": None,
          "mid": 0.001535, "want": {}, "kmap": {}, "head": "h"}
    reply, _ = execution.execute(_pending(tool="place_scalp", proposal=pr))
    assert reply.startswith("❌ ladder not applied") and "0.002" in reply


def test_cancel_goes_through_handle_cancel(monkeypatch):
    _wire(monkeypatch)
    reply, tracked = execution.execute(_pending(tool="cancel_order", proposal={"order_id": "x-9"}))
    assert FakeTrader.calls == [{"cancel": "x-9"}] and reply.startswith("✅ Cancelled order `x-9`")
    assert tracked is None


def test_unknown_tool_and_non_trading_desk_refused(monkeypatch):
    _wire(monkeypatch)
    assert execution.execute(_pending(tool="nope"))[0].startswith("❌ nope has no execution")
    p = policy.Pending("n", Context("r", "options", "-1", "7", "s", True), "place_order", {}, 0.0)
    assert execution.execute(p)[0].startswith("❌ options is not a trading desk")
    assert FakeTrader.calls == []


def test_fills_tick_delegates_to_trade_bot_and_survives_errors(monkeypatch):
    _wire(monkeypatch)
    seen = {}
    monkeypatch.setattr(tb, "poll_pending", lambda api, traders, fills: seen.update(api=api, n=len(traders), fills=fills))
    fills = {"oid-1": {"desk": "mm3"}}
    execution.fills_tick("api", fills)
    assert seen == {"api": "api", "n": len(tb.DESKS), "fills": fills}
    monkeypatch.setattr(tb, "poll_pending", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    execution.fills_tick("api", fills)      # must not raise


def test_only_execution_imports_the_trader():
    from src.the_operator import agent, executor, gateway, tools
    from src.the_operator.tools import trading
    for mod in (agent, executor, gateway, tools, trading):
        assert "CoinRoutesTrader" not in inspect.getsource(mod).split('"""', 2)[2], mod.__name__


def test_only_execution_can_reach_the_box_write_route():
    """The boxes take that POST with NO authentication, so our own call sites are the only rail."""
    from src.the_operator import agent, executor, gateway, rapid, tools
    from src.the_operator.tools import engine
    for mod in (agent, executor, gateway, tools, engine):
        assert "push_param" not in inspect.getsource(mod).split('"""', 2)[2], mod.__name__
    # And the client exposes exactly one writer.
    assert [n for n in dir(rapid) if n.startswith("push")] == ["push_param"]


# --------------------------------------------------------------------------------------
# mm_rebound: the push merges into the LIVE group, and never fires on a stale band.
# --------------------------------------------------------------------------------------
def _rebound_pending(**over):
    target = {"box": "tk1", "url": "http://box", "calculator": "LM2Strategy:REPARTI:PARTI-USDT:BINANCE",
              "exchange": "BINANCE", "ccypair": "PARTI-USDT", "old": [0.0212, 0.0223],
              "price_min": 0.0258, "price_max": 0.0272}
    target.update(over)
    return policy.Pending("n1", Context("r1", "mm3", "-1", "7", "u7", is_trader=True), "mm_rebound",
                          {"kind": "mm_rebound", "token": "PARTI", "program": "mm3",
                           "how": "recentred on mid", "reason": "r", "targets": [target]}, 0.0)


def test_rebound_merges_into_the_live_group_so_nothing_else_is_dropped(monkeypatch):
    pushed = []
    monkeypatch.setattr(execution.rapid, "live_band",
                        lambda url, calc: ({"enabled": True, "stop_both_sides": True,
                                            "price_min": 0.0212, "price_max": 0.0223}, 0.0265))
    monkeypatch.setattr(execution.rapid, "push_param",
                        lambda url, calc, attr, value: pushed.append((calc, attr, value)) or {"ok": True})
    reply = execution.execute(_rebound_pending())[0]
    calc, attr, value = pushed[0]
    assert attr == "price_boundaries"
    assert value == {"enabled": True, "stop_both_sides": True,          # preserved, not dropped
                     "price_min": 0.0258, "price_max": 0.0272}          # and updated
    assert "✅ Rebounded PARTI on 1 of 1" in reply


def test_rebound_refuses_a_venue_whose_price_left_the_confirmed_band(monkeypatch):
    pushed = []
    monkeypatch.setattr(execution.rapid, "live_band",
                        lambda url, calc: ({"enabled": True, "price_min": 0.0212,
                                            "price_max": 0.0223}, 0.040))   # moved far since preview
    monkeypatch.setattr(execution.rapid, "push_param",
                        lambda *a: pushed.append(a) or {"ok": True})
    reply = execution.execute(_rebound_pending())[0]
    assert pushed == [] and "outside the band you confirmed" in reply
    assert "0 of 1" in reply


def test_one_failing_venue_does_not_stop_the_others(monkeypatch):
    pending = _rebound_pending()
    pending.proposal["targets"].append({**pending.proposal["targets"][0], "exchange": "OKX",
                                        "calculator": "LM2Strategy:REPARTI:PARTI-USDT:OKX"})
    monkeypatch.setattr(execution.rapid, "live_band",
                        lambda url, calc: ({"price_min": 0.0212, "price_max": 0.0223}, 0.0265))

    def push(url, calc, attr, value):
        if calc.endswith("BINANCE"):
            raise TimeoutError("box timed out")
        return {"ok": True}
    monkeypatch.setattr(execution.rapid, "push_param", push)
    reply = execution.execute(pending)[0]
    assert "1 of 2" in reply and "OKX" in reply and "TimeoutError" in reply


def test_rebound_never_raises_out_of_execute(monkeypatch):
    def boom(url, calc):
        raise RuntimeError("registry exploded")
    monkeypatch.setattr(execution.rapid, "live_band", boom)
    reply = execution.execute(_rebound_pending())[0]
    assert "0 of 1" in reply and "RuntimeError" in reply


def _set_pending(old=6.0, value=4.0):
    return policy.Pending("n2", Context("r1", "mm3", "-1", "7", "u7", is_trader=True),
                          "mm_set_param",
                          {"kind": "mm_set_param", "token": "PARTI", "program": "mm3",
                           "attribute": "ob_dynamic", "field": "base_min_spread_bps",
                           "value": value, "reason": "r",
                           "targets": [{"box": "tk1", "url": "http://box", "exchange": "BINANCE",
                                        "ccypair": "PARTI-USDT", "old": old,
                                        "calculator": "LM2Strategy:REPARTI:PARTI-USDT:BINANCE"}]},
                          0.0)


def test_set_param_merges_into_the_live_group(monkeypatch):
    pushed = []
    monkeypatch.setattr(execution.rapid, "live_group",
                        lambda url, calc, attr: {"base_min_spread_bps": 6.0, "levels": 4})
    monkeypatch.setattr(execution.rapid, "push_param",
                        lambda url, calc, attr, value: pushed.append((attr, value)) or {"ok": True})
    reply = execution.execute(_set_pending())[0]
    assert pushed == [("ob_dynamic", {"base_min_spread_bps": 4.0, "levels": 4})]
    assert "✅ Set ob_dynamic.base_min_spread_bps = 4.0 on 1 of 1" in reply


def test_set_param_refuses_when_someone_else_moved_the_field(monkeypatch):
    pushed = []
    monkeypatch.setattr(execution.rapid, "live_group",
                        lambda url, calc, attr: {"base_min_spread_bps": 9.0})   # not the 6.0 shown
    monkeypatch.setattr(execution.rapid, "push_param", lambda *a: pushed.append(a))
    reply = execution.execute(_set_pending())[0]
    assert pushed == [] and "changed to 9.0 since the preview" in reply and "0 of 1" in reply


def test_a_trader_is_built_for_every_trade_bot_desk_with_its_own_key(monkeypatch):
    """trade_bot desks the Operator does not serve (MM4, OTC) once raised KeyError here, which
    failed every placement on every desk."""
    execution._traders.clear()
    monkeypatch.setattr(execution, "CoinRoutesTrader", FakeTrader)
    monkeypatch.setattr(execution, "require_env", lambda var: f"key:{var}")
    built = execution.traders()
    assert set(built) == set(tb.DESKS)
    assert all(built[d].api_token == f"key:{cfg['cr_token_env']}" for d, cfg in tb.DESKS.items())
    execution._traders.clear()
