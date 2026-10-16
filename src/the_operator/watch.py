"""Background watch over the quoting books: posts what is not quoting, and when share drifts.

Two rounds on their own cadences. The quoting watch asks "is the book live"; the volume watch asks
"are we trading the right amount of it" against the desk's 5%-50% rule of thumb.

Everything else in the Operator is pull — someone has to suspect a problem before they can ask
about it. A strategy that leaves its price band or dies with an error is silent: the volume simply
stops, and the desk finds out when somebody looks. This is the one thing that speaks first.

The quoting watch posts a SNAPSHOT, not transitions: every :00 and :30, the strategies not quoting
right now, grouped by why and with the fix. "Stopped" / "quoting again" messages read as a stream
the desk had to replay in their heads; a snapshot is the state itself (the owner's call). The why
comes from the box: a STOP (ERROR) carries its validity reason, and every other stop shows as a
rule whose size multiplier is 0 (price band, low balance, descale, ...). A strategy disabled with no
rule at 0 was switched off by hand; it gets one summary line, and alone never triggers a post.

The volume watch still announces SUSTAINED transitions (``settle``) and speaks through an
``Hourly`` gate once per clock hour; the quoting watch has its own half-hourly gate. A start waits
for the next slot like any other news. The volume watch's first scan only establishes a baseline
(a log line, not a post), so a restart does not replay what was already outside the band; it stays
quiet on any coin a trader has dismissed today.

MUST NOT: change anything on a box (it only reads), decide policy, or post into a chat that is not
the one pinned to the scanned desk.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from src.common.logging import get_logger
from src.the_operator import moves, rapid, volume

_log = get_logger("operator")

INTERVAL_S = 5 * 60
MM_POST_S = 30 * 60             # the quoting snapshot goes out every :00 and :30
VOLUME_INTERVAL_S = 15 * 60     # the desk's own cadence for looking at share
VOLUME_ROUNDS = 2               # ~30 min outside the band before it is worth a message
MAX_LINES = 12          # a desk-wide event (a venue outage) must not paste 60 lines into the chat
MAX_VENUES = 3          # venues named per token line; the rest are "+N more"

# Section order, heading and the per-token fix. A rule's multiplier at 0 is the cause; the band
# comes first because a strategy stopped by its band AND low on balance is fixed by the rebound.
CAUSES = {
    "band": "⚠️ **Price outside its band**",
    "balance": "💰 **Low balance**",
    "error": "❌ **Error**",
    "descale": "📉 **Descaled**",
    "range": "📏 **Price-range rule at 0**",
    "target": "🎯 **Target-balance rule at 0**",
    "zero": "⭕ **Trading, but order size 0**",
    "unknown": "❓ **No status from the box**",
}
_RULES = (("price_boundaries", "band"), ("low_balance_policy", "balance"),
          ("descale_trigger", "descale"), ("price_range_rule", "range"),
          ("target_balance_config", "target"))


def _zero_sides(s: rapid.Strategy, rule: str) -> set[str]:
    return {side for side in ("bid", "ask") if s.adjust.get(f"{rule}_notional_multiplier_{side}") == 0}


def cause(s: rapid.Strategy) -> str | None:
    """Why ``s`` is not quoting: a ``CAUSES`` key, "off" (switched off by hand), or None (quoting)."""
    zeros = [c for rule, c in _RULES if _zero_sides(s, rule)]
    if s.errored:
        return "band" if "band" in zeros or "Out of Boundaries" in s.validity_reason else "error"
    if s.healthy:
        both_zero = (s.adjust.get("notional_multiplier_bid") == 0
                     and s.adjust.get("notional_multiplier_ask") == 0)
        return "zero" if both_zero else None
    if zeros:
        return zeros[0]
    return "unknown" if s.status in ("?", "None") else "off"


def _sides(strategies: list[rapid.Strategy], rule: str) -> str:
    sides = set().union(*(_zero_sides(s, rule) for s in strategies))
    return "both sides" if sides == {"bid", "ask"} else f"{''.join(sides)} only" if sides else ""


def _venues(strategies: list[rapid.Strategy]) -> str:
    names = sorted({s.exchange for s in strategies})
    more = f" +{len(names) - MAX_VENUES} more" if len(names) > MAX_VENUES else ""
    return ", ".join(f"`{n}`" for n in names[:MAX_VENUES]) + more


def _line(key: str, token: str, group: list[rapid.Strategy]) -> str:
    line = f"• **{token}** — {_venues(group)}"
    if key == "band":
        if _sides(group, "low_balance_policy"):
            line += " · also low balance"
        return line + f" · `/operator rebound {token}`"
    if key == "balance":
        return line + f" ({_sides(group, 'low_balance_policy')})"
    if key == "error":
        why = " ".join(next((s.validity_reason for s in group if s.validity_reason), "").split())
        return line + (f": {why[:70]}" if why else "")
    return line


def render_snapshot(program: str, strategies: list[rapid.Strategy], warnings: list[str],
                    now: datetime | None = None) -> str:
    """The half-hourly message ("" when only hand-switched-off strategies are down)."""
    by: dict[str, dict[str, list[rapid.Strategy]]] = {}
    for s in strategies:
        if (key := cause(s)) is not None:
            by.setdefault(key, {}).setdefault(s.token, []).append(s)
    down = sum(len(g) for k, toks in by.items() if k != "off" for g in toks.values())
    if not down:
        return ""
    stamp = f"{(now or datetime.now(timezone.utc)):%H:%M} UTC"
    out = [f"🔔 **MM watch · {program}** — {down} not quoting · {stamp}"]
    for key, heading in CAUSES.items():
        tokens = by.get(key)
        if not tokens:
            continue
        out += ["", heading]
        names = sorted(tokens)
        out += [_line(key, t, tokens[t]) for t in names[:MAX_LINES]]
        if len(names) > MAX_LINES:
            out.append(f"• …and {len(names) - MAX_LINES} more tokens")
    if off := by.get("off"):
        out += ["", "⏸ **Off by hand:** " + ", ".join(f"{t} ({len(g)})" for t, g in sorted(off.items()))]
    dead = [w for w in warnings if "unreachable" in w]
    if dead:
        out += ["", f"⚠️ {len(dead)} box(es) did not answer, so this list may be short."]
    return "\n".join(out)


@dataclass
class Track:
    """How long one coin has held its current state, and whether we have reported it."""

    state: str
    streak: int = 1
    announced: bool = False          # we told the chat about this one

    def observe(self, state: str) -> None:
        self.streak = self.streak + 1 if state == self.state else 1
        self.state = state


def settle(tracks: dict[str, Track], after: dict[str, str],
           rounds: int = VOLUME_ROUNDS) -> tuple[list[str], list[str]]:
    """Fold this round's states into ``tracks``; return ``(to_announce, to_clear)``.

    Only a state held long enough is reported, and a recovery only for a breach we reported. A key
    absent from ``after`` is left alone: no reading is not news."""
    to_announce, to_clear = [], []
    for calc, state in after.items():
        track = tracks.get(calc)
        if track is None:
            tracks[calc] = Track(state)
            continue
        track.observe(state)
        if state != "ok" and not track.announced and track.streak >= rounds:
            track.announced = True
            to_announce.append(calc)
        elif state == "ok" and track.announced and track.streak >= rounds:
            track.announced = False
            to_clear.append(calc)
    return to_announce, to_clear


def baseline(tracks: dict[str, Track]) -> None:
    """Treat what is already outside the band on the first scan as known, so a restart does not
    replay it. They stay marked, so a later recovery is still tracked."""
    for track in tracks.values():
        track.announced = track.state != "ok"


class Hourly:
    """One message per desk per slot (``period_s``, a clock hour by default) holding its sources'
    news; nothing is dropped.

    Scanning stays frequent so a real outage is detected within minutes, but the chat hears about
    it once per slot — detection cadence and notification cadence are different problems. The slot
    a desk is first seen in is spent, so a restart waits for the next one. ``clock`` is injectable
    so tests can move time."""

    def __init__(self, say, clock=time.time, period_s: float = 3600) -> None:
        self.say = say
        self.clock = clock
        self.period_s = period_s
        self.sources: list = []              # watchers with take(desk) -> str
        self.slot: dict[str, int] = {}       # desk -> the slot last spent
        self.lock = threading.Lock()         # watch threads hold and flush through here

    def flush(self, desk: str, chat_ids: list[str]) -> None:
        with self.lock:
            slot = int(self.clock() // self.period_s)
            if self.slot.setdefault(desk, slot) >= slot:
                return
            self.slot[desk] = slot           # spent even when empty: news at :20 waits for the next
            text = "\n\n".join(t for t in (s.take(desk) for s in self.sources) if t)
            if not text:
                return
            _log.info("watch post %s to %s: %d lines", desk, chat_ids, len(text.splitlines()))
            for chat_id in chat_ids:
                try:
                    self.say(chat_id, text)
                except Exception:  # noqa: BLE001 - one unreachable chat must not stop the others
                    _log.exception("watch could not post to %s", chat_id)


class Watcher:
    """Scans each desk every round and posts the latest snapshot through a half-hourly gate.
    ``say(chat_id, text)`` is the gateway's own sender."""

    def __init__(self, desk_chats: dict[str, list[str]], say, gate: Hourly | None = None) -> None:
        # {program: [chat ids]} — a desk can be pinned to more than one chat, and each scan is
        # done once and posted to all of them.
        self.desk_chats = desk_chats
        self.gate = gate or Hourly(say, period_s=MM_POST_S)
        self.gate.sources.append(self)
        self.latest: dict[str, tuple[list[rapid.Strategy], list[str]]] = {}   # program -> scan

    def round_once(self) -> None:
        for program, chat_ids in self.desk_chats.items():
            self._scan(program)
            self.gate.flush(program, chat_ids)

    def _scan(self, program: str) -> None:
        warnings: list[str] = []
        try:
            found = rapid.scan(program, warnings)
        except Exception:  # noqa: BLE001 - a bad round must never kill the watch
            _log.exception("mm watch scan failed for %s", program)
            return
        if not found:
            _log.warning("mm watch: %s returned no strategies (%s)", program, warnings)
            return
        with self.gate.lock:
            self.latest[program] = (found, warnings)

    def take(self, program: str) -> str:
        """The latest scan as one snapshot ("" when nothing is down). Caller holds the lock. The
        scan is consumed, so a desk whose boxes stopped answering does not repost an old list."""
        found, warnings = self.latest.pop(program, ([], []))
        return render_snapshot(program, found, warnings)


def render_volume(desk: str, flagged: list[tuple[str, float, str, str | None]]) -> str:
    """One line per coin, grouped by which side of the band it fell off, commands at the end.

    The reader already knows the rule and the window; what they need is which coin, why in a few
    words, and what to type."""
    if not flagged:
        return ""
    high = [f for f in flagged if f[1] > volume.HIGH]
    low = [f for f in flagged if f[1] < volume.LOW]        # in-band coins belong in neither
    out = [f"📉 **Volume watch · {desk}**"]
    for title, group in ((f"Above {volume.HIGH:.0%}", high), (f"Below {volume.LOW:.0%}", low)):
        if not group:
            continue
        out += ["", f"**{title}**"]
        out += [f"• **{token}** {share:.0%} — {reason}"
                for token, share, reason, _cmd in group[:MAX_LINES]]
        if len(group) > MAX_LINES:
            out.append(f"• …and {len(group) - MAX_LINES} more")
    if not high and not low:
        return ""
    commands = [cmd for t, _s, _r, cmd in flagged if cmd
                and t in {x[0] for x in high + low}]
    if commands:
        out += [""] + [f"`{c}`" for c in list(dict.fromkeys(commands))[:MAX_LINES]]
    return "\n".join(out)


class VolumeWatcher:
    """Share against the desk's rule of thumb, triaged from the aggregator sheet.

    The sheet is one cheap read that gives every coin's share; only the coins outside the band are
    then diagnosed per venue, which is the expensive part. A coin a trader has dismissed today is
    skipped entirely — including its diagnosis, so a deliberate exception costs nothing."""

    def __init__(self, desk_chats: dict[str, list[str]], say, shares_fn=None,
                 brief_fn=None, hourly: Hourly | None = None) -> None:
        self.desk_chats = desk_chats
        self.hourly = hourly or Hourly(say)
        self.hourly.sources.append(self)
        self.tracks: dict[str, dict[str, Track]] = {}
        self.held: dict[str, list[tuple]] = {}
        self._shares = shares_fn or volume.shares
        self._brief = brief_fn or volume.brief

    def _worst_venue(self, desk: str, token: str) -> tuple[str, str | None]:
        """Reason and command from the biggest venue that has an actionable one."""
        warnings: list[str] = []
        venues = self._shares(desk, token, warnings)
        if not venues:
            return "", None          # not quoted here: a non-token sheet row, or another program
        ranked = sorted(venues, key=lambda v: -(v.market or 0))
        for v in ranked:
            reason, command = self._brief(v)
            if command:
                return reason, command
        return self._brief(ranked[0])

    def round_once(self) -> None:
        for desk, chat_ids in self.desk_chats.items():
            self._scan(desk)
            self.hourly.flush(desk, chat_ids)

    def take(self, desk: str) -> str:
        """The held breaches as one block, cleared ("" when there are none). Caller holds the lock."""
        return render_volume(desk, self.held.pop(desk, []))

    def _scan(self, desk: str) -> None:
        try:
            rows = desk_shares(desk)
        except Exception:  # noqa: BLE001 - a bad round must never kill the watch
            _log.exception("volume watch could not read the %s sheet", desk)
            return
        if not rows:
            return
        states = {token: ("low" if share < volume.LOW else
                          "high" if share > volume.HIGH else "ok")
                  for token, share in rows.items() if share is not None}
        first = desk not in self.tracks
        tracks = self.tracks.setdefault(desk, {})
        breached, _cleared = settle(tracks, states, rounds=VOLUME_ROUNDS)
        if first:
            baseline(tracks)
            _log.info("volume watch baseline %s: %d coins, %d outside the band", desk,
                      len(states), sum(1 for s in states.values() if s != "ok"))
            return
        flagged = []
        for token in breached:
            if volume.is_snoozed(desk, token):
                continue
            cause, fix = self._worst_venue(desk, token)
            if cause:
                flagged.append((token, rows[token], cause, fix))
        with self.hourly.lock:
            self.held[desk] = self.held.get(desk, []) + flagged


def desk_shares(desk: str) -> dict[str, float | None]:
    """``{coin: share}`` from the desk's aggregator sheet (deferred import: gspread is heavy)."""
    from src.the_operator.tools.engine import _desk_shares

    return dict(_desk_shares(desk))


def start(desk_chats: dict[str, list[str]], say, interval_s: float = INTERVAL_S,
          volume_interval_s: float = VOLUME_INTERVAL_S) -> list[threading.Thread]:
    """Both watches, each on its own thread so a slow volume round never delays the quoting one:
    the quoting snapshot every :00 and :30, volume news once an hour. MM3 also gets the move alert,
    which posts at once (``moves``)."""
    threads = []
    for name, worker, every in (("operator-mm-watch", Watcher(desk_chats, say), interval_s),
                                ("operator-volume-watch",
                                 VolumeWatcher(desk_chats, say), volume_interval_s)):
        def loop(worker=worker, every=every) -> None:
            while True:
                worker.round_once()
                time.sleep(every)

        t = threading.Thread(target=loop, daemon=True, name=name)
        t.start()
        threads.append(t)
    if "mm3" in desk_chats:              # the calibrated tokens are MM3's
        mover = moves.MoveWatcher(desk_chats["mm3"], say)

        def move_loop() -> None:
            bar = moves.BAR.total_seconds()
            while True:
                try:
                    mover.round_once()
                except Exception:  # noqa: BLE001 - a bad round must never kill the watch
                    _log.exception("move watch round failed")
                time.sleep(bar - time.time() % bar + 20)   # just after the next bar closes

        t = threading.Thread(target=move_loop, daemon=True, name="operator-move-watch")
        t.start()
        threads.append(t)
    return threads
