"""Releases a confirmed money proposal to CoinRoutes and watches live orders for fills. The ONLY
Operator module that may import ``CoinRoutesTrader``.

``execute`` is called by the gateway after ``policy.confirm_check`` allowed a ``Pending``. It does
not trust the preview blindly: a ``place_order`` is re-run through ``trade_bot.prepare_order``
(fresh caps, venues, max-post), refused if the market moved beyond ``policy.LIMIT_DRIFT_MAX``, and
otherwise sent at the LIMIT THE HUMAN CONFIRMED through ``trade_bot.place_prepared`` with the
status a typed ``/trade`` would use (or ``policy.MONEY_STATUS`` when a phase forces one). A
``place_scalp`` re-reads the mid, refuses on the same dislocation rule, then applies the
PREVIEWED ladder through ``trade_bot.apply_ladder``. Cancels go through ``trade_bot.handle_cancel``.
A ``move_order`` re-reads the order (still open, still at the price the human saw), re-reads the
touch when it was a join (same dislocation rule), then modifies the limit in place. If CoinRoutes
refuses the modify, the order is left where it was: rebuilding an arbitrary order from its record
could place something different from what was resting, so the trader decides what happens next.
``fills_tick`` is one pass of ``trade_bot.poll_pending`` over the Operator's live orders.

An ``mm_rebound`` goes to the rapid-engine boxes instead: each target's CURRENT ``price_boundaries``
group is re-read and the confirmed min/max merged into it (the box replaces the whole group, so a
partial dict would drop ``enabled``/``stop_both_sides``), and a venue whose price has since left the
confirmed band is skipped — pushing a band the price is already outside leaves the strategy stopped.
This is the only module that may call ``rapid.push_param``. A confirmed ``remember`` writes its
desk note here for the same reason: the handler proposes, the human confirms, this module acts.

MUST NOT: be reachable from a tool handler or from the model (only the gateway calls it), bypass
``prepare_order`` / ``apply_ladder``, or construct a trader anywhere but ``traders()``.
"""

from __future__ import annotations

from scripts import trade_bot as tb
from src.adapters.coinroutes.client import CoinRoutesClient
from src.adapters.coinroutes.trade import TERMINAL, CoinRoutesTrader
from src.common.config import require_env
from src.common.logging import get_logger
from src.the_operator import notes, policy, rapid, volume
from src.the_operator.policy import Pending

_log = get_logger("operator")
_traders: dict[str, CoinRoutesTrader] = {}


def cr_token(desk: str) -> str:
    """A trading desk's CoinRoutes key from trade_bot's own desk table, as /trade reads it. The
    Operator's desk list only covers the desks it serves; looking a trade_bot desk up there broke
    every placement the day trade_bot gained one (MM4)."""
    return require_env(tb.DESKS[desk]["cr_token_env"])


def traders() -> dict[str, CoinRoutesTrader]:
    """One trader per trading desk, built lazily on that desk's own key."""
    for desk in tb.DESKS:
        if desk not in _traders:
            _traders[desk] = CoinRoutesTrader(api_token=cr_token(desk))
    return _traders


def prime_venues() -> None:
    """Fill trade_bot's strategy-venue cache for every trading desk's books, one CoinRoutes listing
    per key. The first order on a strategy after a restart otherwise waits ~5 s for that listing;
    this fetches exactly what that order would have, only earlier. Run once, off the main thread."""
    by_key: dict[str, set[str]] = {}
    for cfg in tb.DESKS.values():
        names = {*cfg["strategy"].values(), cfg.get("perp_strategy"), cfg.get("spot_strategy")}
        by_key.setdefault(cfg["cr_token_env"], set()).update(n for n in names if n)
    for env, names in by_key.items():
        try:
            tb.prime_strategy_venues(CoinRoutesClient(api_token=require_env(env)), sorted(names))
        except Exception:  # noqa: BLE001 - priming only saves time; an order still resolves itself
            _log.exception("could not prime strategy venues on %s", env)


def fills_tick(api: str, fills: dict) -> None:
    """One non-blocking pass over the Operator's live orders (``{oid: tracked}``): terminal
    statuses are announced in their chat and dropped, exactly as /trade does."""
    try:
        tb.poll_pending(api, traders(), fills)
    except Exception:  # noqa: BLE001 - the watcher must survive a bad tick
        _log.exception("fills tick failed")


def execute(p: Pending) -> tuple[str, dict | None]:
    """Release one confirmed proposal. Returns (chat reply, tracked live order or None). Never raises."""
    if p.tool == "remember":
        try:
            note = notes.add(p.ctx.desk, p.ctx.user_name, p.proposal["note"])
            return f"📝 Noted for {p.ctx.desk}: {note.text}", None
        except OSError as exc:
            return f"❌ could not write the desk note: {exc}", None
    if p.tool == "volume_snooze":
        day = volume.snooze(p.ctx.desk, p.proposal["token"], p.ctx.user_name,
                            p.proposal["reason"])
        return (f"🔕 {p.proposal['token']} volume alerts are off for {p.ctx.desk} "
                f"until the end of {day:%d %b} UTC."), None
    if p.tool in ("mm_rebound", "mm_set_param"):
        try:
            return (_rebound(p) if p.tool == "mm_rebound" else _set_param(p)), None
        except Exception as exc:  # noqa: BLE001 - report over chat, never kill the gateway worker
            _log.exception("execution of %s (%s) failed", p.tool, p.nonce)
            return f"❌ {p.tool} failed: {type(exc).__name__}: {str(exc)[:200]}", None
    desk_cfg = tb.DESKS.get(p.ctx.desk)
    if desk_cfg is None:
        return f"❌ {p.ctx.desk} is not a trading desk", None
    try:
        token = cr_token(p.ctx.desk)
        if p.tool == "place_order":
            return _place(p, desk_cfg, token)
        if p.tool == "place_scalp":
            return _scalp(p, token), None
        if p.tool == "cancel_order":
            return tb.handle_cancel(p.proposal["order_id"], traders()[p.ctx.desk]), None
        if p.tool == "move_order":
            return _move(p, token)
        return f"❌ {p.tool} has no execution path in phase {policy.PHASE}", None
    except Exception as exc:  # noqa: BLE001 - report over chat, never kill the gateway worker
        _log.exception("execution of %s (%s) failed", p.tool, p.nonce)
        return f"❌ {p.tool} failed: {type(exc).__name__}: {str(exc)[:200]}", None


def _place(p: Pending, desk_cfg: dict, token: str) -> tuple[str, dict | None]:
    plan, err = tb.prepare_order(p.proposal["command"], CoinRoutesClient(api_token=token), desk_cfg)
    if err is not None:
        return f"❌ not placed — re-check at confirm time refused it: {err}", None
    previewed = p.proposal["place_kwargs"]["limit_price"]
    if not policy.limit_drift_ok(previewed, plan["limit"]):
        return (f"❌ not placed — the market moved more than {policy.LIMIT_DRIFT_MAX:.0%} since the "
                f"preview (limit would now be {plan['limit']} vs {previewed}). Ask again for a fresh preview."), None
    # The human confirmed THIS limit, so this limit is what goes out: a limit order never fills
    # worse than its limit, so a touch that moved since the preview changes fill odds, not price
    # risk. Caps, venues and max-post were re-derived above from the fresh book.
    fresh_limit = plan["limit"]
    plan["place_kwargs"]["limit_price"] = previewed
    plan["limit"] = previewed
    if policy.MONEY_STATUS is not None:
        plan["place_kwargs"]["order_status"] = policy.MONEY_STATUS
        plan["status"] = policy.MONEY_STATUS
    _log.warning("EXECUTE %s by %s (%s) on %s: %s [%s]", p.nonce, p.ctx.user_name, p.ctx.user_id,
                 p.ctx.desk, p.proposal["command"], plan["status"])
    reply, tracked = tb.place_prepared(plan, traders()[p.ctx.desk])
    if fresh_limit != previewed:
        reply += f"\n(touch moved since the preview: a fresh limit would be {fresh_limit}; yours is {previewed})"
    if plan["status"] == "staged":
        reply += "\nStaged: release it in the CoinRoutes UI."
    return reply, tracked


def _scalp(p: Pending, token: str) -> str:
    pr = p.proposal
    client = CoinRoutesClient(api_token=token)
    bid, ask = tb.touch_on(client, pr["pair"], "sell", pr["pxv"]), tb.touch_on(client, pr["pair"], "buy", pr["pxv"])
    mid = (bid + ask) / 2 if (bid and ask) else (bid or ask)
    if not mid:
        return f"❌ no consolidated mid for {pr['pair']} right now — ladder not applied."
    if not policy.limit_drift_ok(pr["mid"], mid):
        return (f"❌ ladder not applied — mid moved from {pr['mid']:g} to {mid:g} since the preview "
                f"(more than {policy.LIMIT_DRIFT_MAX:.0%}). Ask again for a fresh preview.")
    _log.warning("EXECUTE %s by %s (%s) on %s: %s", p.nonce, p.ctx.user_name, p.ctx.user_id,
                 p.ctx.desk, pr["command"])
    sides = tb.apply_ladder(pr["want"], client, traders()[p.ctx.desk], base=pr["base"], pair=pr["pair"],
                            strategy=pr["strategy"], pxv=pr["pxv"], venues=pr["venues"])
    return tb.render_scalper_text(pr["head"], sides, pr["kmap"])


def _move(p: Pending, token: str) -> tuple[str, dict | None]:
    pr, trader = p.proposal, traders()[p.ctx.desk]
    oid, limit = pr["order_id"], pr["limit_price"]
    rec = trader.get_order(oid)
    status = str(rec.get("order_status"))
    if status in TERMINAL:
        return f"❌ not moved — order `{oid}` is {status}.", None
    if float(rec.get("limit_price") or 0) != float(pr["from_price"]):
        return (f"❌ not moved — order `{oid}` now rests at {rec.get('limit_price')}, not the "
                f"{pr['from_price']} you saw. Ask again for a fresh preview."), None
    if pr["join"]:
        opp = "sell" if pr["side"] == "buy" else "buy"
        fresh = tb.touch_on(CoinRoutesClient(api_token=token), pr["pair"], opp, pr["venues"])
        if not policy.limit_drift_ok(limit, fresh):
            return (f"❌ not moved — the touch is now {fresh} vs {limit} at the preview (more than "
                    f"{policy.LIMIT_DRIFT_MAX:.0%}, or no price). Ask again for a fresh preview."), None
    _log.warning("EXECUTE %s by %s (%s) on %s: move %s %s -> %s", p.nonce, p.ctx.user_name,
                 p.ctx.user_id, p.ctx.desk, oid, pr["from_price"], limit)
    try:
        trader.modify(oid, limit_price=limit)
    except Exception as exc:  # noqa: BLE001 - modify is best-effort at CoinRoutes; say so, don't replace
        return (f"❌ CoinRoutes refused the move: {str(exc)[:200]}\nOrder `{oid}` still rests at "
                f"{pr['from_price']}. Cancel and place a new one if you still want it moved."), None
    return (f"✅ Moved order `{oid}` {pr['from_price']} → {limit}",
            {"oid": oid, "base": pr["base"], "side": pr["side"]})


def _rebound(p: Pending) -> str:
    """Push each confirmed price band, re-reading the live group first so nothing else is lost."""
    pr = p.proposal
    _log.warning("EXECUTE %s by %s (%s) on %s: rebound %s (%s), %d target(s)", p.nonce,
                 p.ctx.user_name, p.ctx.user_id, p.ctx.desk, pr["token"], pr["how"],
                 len(pr["targets"]))
    done, failed = [], []
    for t in pr["targets"]:
        try:
            live = rapid.live_band(t["url"], t["calculator"])
            if live is None:
                failed.append(f"`{t['exchange']}` {t['ccypair']}: no live price_boundaries to update")
                continue
            band, mid = live
            if mid is not None and not (t["price_min"] <= mid <= t["price_max"]):
                failed.append(f"`{t['exchange']}` {t['ccypair']}: price moved to {mid:.8g}, outside "
                              f"the band you confirmed — not pushed")
                continue
            band.update(price_min=t["price_min"], price_max=t["price_max"])
            rapid.push_param(t["url"], t["calculator"], "price_boundaries", band)
            done.append(f"`{t['exchange']}` {t['ccypair']} → "
                        f"`{t['price_min']:.8g}–{t['price_max']:.8g}`")
        except Exception as exc:  # noqa: BLE001 - one venue failing must not stop the others
            _log.exception("rebound push failed for %s", t["calculator"])
            failed.append(f"`{t['exchange']}` {t['ccypair']}: {type(exc).__name__}: {str(exc)[:120]}")
    out = [f"✅ Rebounded {pr['token']} on {len(done)} of {len(pr['targets'])} strategies"]
    out += [f"• {d}" for d in done]
    if failed:
        out += ["", "❌ Not pushed:"] + [f"• {f}" for f in failed]
    return "\n".join(out)


def _set_param(p: Pending) -> str:
    """Set one field inside one parameter group, merging into the live group on each strategy."""
    pr = p.proposal
    attribute, field, value = pr["attribute"], pr["field"], pr["value"]
    _log.warning("EXECUTE %s by %s (%s) on %s: set %s.%s=%r on %s, %d target(s)", p.nonce,
                 p.ctx.user_name, p.ctx.user_id, p.ctx.desk, attribute, field, value, pr["token"],
                 len(pr["targets"]))
    done, failed = [], []
    for t in pr["targets"]:
        try:
            group = rapid.live_group(t["url"], t["calculator"], attribute)
            if group is None or field not in group:
                failed.append(f"`{t['exchange']}` {t['ccypair']}: no live {attribute}.{field} "
                              f"to update")
                continue
            if group[field] != t["old"]:
                # Somebody else moved it between the preview and the confirm; the human approved a
                # specific change, not whatever this now is.
                failed.append(f"`{t['exchange']}` {t['ccypair']}: changed to {group[field]!r} "
                              f"since the preview (you confirmed {t['old']!r}) — not pushed")
                continue
            group[field] = value
            rapid.push_param(t["url"], t["calculator"], attribute, group)
            done.append(f"`{t['exchange']}` {t['ccypair']}: {t['old']} → {value}")
        except Exception as exc:  # noqa: BLE001 - one venue failing must not stop the others
            _log.exception("set_param push failed for %s", t["calculator"])
            failed.append(f"`{t['exchange']}` {t['ccypair']}: {type(exc).__name__}: {str(exc)[:120]}")
    out = [f"✅ Set {attribute}.{field} = {value} on {len(done)} of {len(pr['targets'])} "
           f"{pr['token']} strategies"]
    out += [f"• {d}" for d in done]
    if failed:
        out += ["", "❌ Not pushed:"] + [f"• {f}" for f in failed]
    return "\n".join(out)
