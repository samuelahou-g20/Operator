"""Background watch over the quoting books: posts when market making stops, or share drifts.

Two rounds on their own cadences. The quoting watch asks "is the book live"; the volume watch asks
"are we trading the right amount of it" against the desk's 5%-50% rule of thumb. Both speak only
when something holds, and the volume one stays quiet on any coin a trader has dismissed today.

Everything else in the Operator is pull — someone has to suspect a problem before they can ask
about it. A strategy that leaves its price band or dies with an error is silent: the volume simply
stops, and the desk finds out when somebody looks. This is the one thing that speaks first.

It posts SUSTAINED transitions, not every crossing. A price sitting near its band edge crosses back
and forth many times an hour — announcing each one buried the desk in hundreds of messages a day
and taught everyone to ignore the channel. So a strategy must hold a state for ``STOPPED_ROUNDS``
consecutive scans before it is announced, and hold ``ok`` for as many before the
recovery is; a flap that reverses inside that window is never mentioned. Recovery is only reported
for a strategy whose stop was reported, so the two always come in pairs.

A strategy missing from a scan (its box did not answer) keeps its streak untouched — a box outage
must neither raise an alarm nor clear one.

Both watches speak through one ``Hourly`` gate: a desk's news goes out as ONE message, on the first
round after each :00, and a start waits for the next :00 like any other news. Two watches on rolling
hours, plus a fresh throttle and a "now on" line on every restart, read as a stream.

The first scan of a desk only establishes a baseline (a log line, not a post): dozens of
strategies are deliberately disabled at any time, and listing them on every restart is noise. State
is in memory, so a restart re-baselines — deliberate, same as ``flow_alert_bot``: the alternative is a
persisted file that goes stale while the process is down.

MUST NOT: change anything on a box (it only reads), decide policy, or post into a chat that is not
the one pinned to the scanned desk.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from src.common.logging import get_logger
from src.the_operator import moves, rapid, volume

_log = get_logger("operator")

INTERVAL_S = 5 * 60
VOLUME_INTERVAL_S = 15 * 60     # the desk's own cadence for looking at share
VOLUME_ROUNDS = 2               # ~30 min outside the band before it is worth a message
MAX_LINES = 12          # a desk-wide event (a venue outage) must not paste 60 lines into the chat
STOPPED_ROUNDS = 3      # ~15 min in a state, either way, before it is worth a message


def _state(s: rapid.Strategy) -> str:
    """The only distinction worth waking a trader for: dead, deliberately off, or quoting."""
    if s.errored:
        return "error"
    return "stopped" if not s.healthy else "ok"


def scan_desk(program: str) -> tuple[dict[str, str], dict[str, rapid.Strategy], list[str]]:
    """``({calculator: state}, {calculator: strategy}, warnings)`` for one program."""
    warnings: list[str] = []
    found = rapid.scan(program, warnings)
    return {s.name: _state(s) for s in found}, {s.name: s for s in found}, warnings


@dataclass
class Track:
    """How long one strategy has held its current state, and whether we have reported it."""

    state: str
    streak: int = 1
    announced: bool = False          # we told the chat this one stopped

    def observe(self, state: str) -> None:
        self.streak = self.streak + 1 if state == self.state else 1
        self.state = state


def settle(tracks: dict[str, Track], after: dict[str, str],
           rounds: int = STOPPED_ROUNDS) -> tuple[list[str], list[str]]:
    """Fold this round's states into ``tracks``; return ``(to_announce, to_clear)``.

    Only a state held long enough is reported, and a recovery only for a stop we reported. A
    calculator absent from ``after`` is left alone: its box did not answer, which is not news."""
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
    """Treat what is already broken on the first scan as known, so a restart does not replay it.

    Without this a pre-existing breach still reaches the announce threshold a few rounds in, and
    the first thing a restart does is paste every stopped strategy into the chat. They stay
    marked, so a later recovery is still reported."""
    for track in tracks.values():
        track.announced = track.state != "ok"


def render(program: str, went_bad: list[str], recovered: list[str],
           strategies: dict[str, rapid.Strategy]) -> str:
    """The chat message for one round's transitions ("" when there is nothing to say)."""
    if not went_bad and not recovered:
        return ""
    out = [f"🔔 **MM watch · {program}**"]
    if went_bad:
        errs = [c for c in went_bad if strategies[c].errored]
        stops = [c for c in went_bad if not strategies[c].errored]
        if errs:
            out += ["", f"❌ **Stopped in error ({len(errs)})**"]
            out += [f"• `{strategies[c].exchange}` {strategies[c].ccypair}" for c in errs[:MAX_LINES]]
            if len(errs) > MAX_LINES:
                out.append(f"• …and {len(errs) - MAX_LINES} more")
        if stops:
            out += ["", f"⚠️ **Stopped quoting ({len(stops)})** — usually the price left its band"]
            out += [f"• `{strategies[c].exchange}` {strategies[c].ccypair}" for c in stops[:MAX_LINES]]
            if len(stops) > MAX_LINES:
                out.append(f"• …and {len(stops) - MAX_LINES} more")
            tokens = sorted({strategies[c].token for c in stops})[:4]
            out.append(f"Rebound with `/operator rebound {tokens[0]}`" if len(tokens) == 1
                       else "Rebound with `/operator rebound <token>` (" + ", ".join(tokens) + ")")
    if recovered:
        out += ["", f"✅ **Quoting again ({len(recovered)})**"]
        out += [f"• `{strategies[c].exchange}` {strategies[c].ccypair}"
                for c in recovered[:MAX_LINES]]
        if len(recovered) > MAX_LINES:
            out.append(f"• …and {len(recovered) - MAX_LINES} more")
    return "\n".join(out)


class Hourly:
    """One message per desk per clock hour, holding both watches' news; nothing is dropped.

    Scanning stays frequent so a real outage is detected within minutes, but the chat hears about
    it once an hour, at :00 — detection cadence and notification cadence are different problems.
    The hour a desk is first seen in is spent, so a restart waits for the next :00. ``clock`` is
    injectable so tests can move the hour."""

    def __init__(self, say, clock=time.time) -> None:
        self.say = say
        self.clock = clock
        self.sources: list = []              # watchers with take(desk) -> str
        self.slot: dict[str, int] = {}       # desk -> the clock hour last spent
        self.lock = threading.Lock()         # both watch threads hold and flush through here

    def flush(self, desk: str, chat_ids: list[str]) -> None:
        with self.lock:
            hour = int(self.clock() // 3600)
            if self.slot.setdefault(desk, hour) >= hour:
                return
            self.slot[desk] = hour           # spent even when empty: news at :20 waits for :00
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
    """One round per desk per interval; news is held for ``Hourly``. ``say(chat_id, text)`` is the
    gateway's own sender."""

    def __init__(self, desk_chats: dict[str, list[str]], say, hourly: Hourly | None = None) -> None:
        # {program: [chat ids]} — a desk can be pinned to more than one chat, and each scan is
        # done once and posted to all of them.
        self.desk_chats = desk_chats
        self.hourly = hourly or Hourly(say)
        self.hourly.sources.append(self)
        self.tracks: dict[str, dict[str, Track]] = {}  # program -> {calculator: Track}
        self.held: dict[str, tuple[list[str], list[str]]] = {}   # program -> (bad, recovered)
        self.latest: dict[str, dict[str, rapid.Strategy]] = {}   # program -> last scan

    def round_once(self) -> None:
        for program, chat_ids in self.desk_chats.items():
            self._scan(program)
            self.hourly.flush(program, chat_ids)

    def _scan(self, program: str) -> None:
        try:
            after, strategies, warnings = scan_desk(program)
        except Exception:  # noqa: BLE001 - a bad round must never kill the watch
            _log.exception("mm watch scan failed for %s", program)
            return
        if not after:
            _log.warning("mm watch: %s returned no strategies (%s)", program, warnings)
            return
        first = program not in self.tracks
        tracks = self.tracks.setdefault(program, {})
        went_bad, recovered = settle(tracks, after)
        if first:
            baseline(tracks)
            _log.info("mm watch baseline %s: %d strategies, %d not quoting", program,
                      len(after), sum(1 for st in after.values() if st != "ok"))
            return
        with self.hourly.lock:
            bad, rec = self.held.get(program, ([], []))
            self.held[program] = (bad + went_bad, rec + recovered)
            self.latest[program] = strategies

    def take(self, program: str) -> str:
        """The held news as one block, cleared ("" when there is none). Caller holds the lock."""
        bad, rec = self.held.pop(program, ([], []))
        strategies = self.latest.get(program, {})
        return render(program, [c for c in bad if c in strategies],
                      [c for c in rec if c in strategies], strategies)


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
    """Both watches, each on its own thread so a slow volume round never delays the quoting one,
    posting through one shared ``Hourly`` so a desk gets one message an hour, not one per watch.
    MM3 also gets the move alert, which posts at once (``moves``)."""
    hourly = Hourly(say)
    threads = []
    for name, worker, every in (("operator-mm-watch", Watcher(desk_chats, say, hourly), interval_s),
                                ("operator-volume-watch",
                                 VolumeWatcher(desk_chats, say, hourly=hourly), volume_interval_s)):
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
