"""Offline tests for vol_bot's RV>IV breach rule and its 4-hourly reminder (no Pluto, no Telegram)."""

from __future__ import annotations

import pytest

from scripts import vol_bot as vb


@pytest.fixture
def sent(monkeypatch):
    """Capture broadcasts and pin the vols the cycle sees."""
    out: list[str] = []
    monkeypatch.setattr(vb, "_broadcast", lambda api, chats, text: out.append(text))
    return out


def _vols(**kw):
    """name -> (rv, iv). Defaults sit comfortably below the threshold."""
    base = {"BTC": (0.30, 0.30), "ETH": (0.40, 0.40)}
    base.update(kw)
    return base


def _cycle(monkeypatch, sent, vols, alerted, now, started=True):
    monkeypatch.setattr(vb, "run_with_deadline", lambda fn, deadline=None: vols)
    return vb._breach_cycle("api", {"1"}, alerted, started, now=now)


# --------------------------------------------------------------------------- #
# the threshold
# --------------------------------------------------------------------------- #
def test_alpha_is_ten_percent():
    assert vb.ALPHA == pytest.approx(0.10)


def test_breach_needs_rv_above_one_point_one_times_iv(monkeypatch, sent):
    #                       ratio 1.05 -> quiet,      ratio 1.15 -> breach
    vols = _vols(BTC=(0.315, 0.30), ETH=(0.46, 0.40))
    alerted, _ = _cycle(monkeypatch, sent, vols, {}, now=0.0)
    assert set(alerted) == {"ETH"}
    assert len(sent) == 1 and "breach" in sent[0]


def test_eth_at_one_point_one_eight_now_breaches(monkeypatch, sent):
    """The live case that prompted the change: ETH RV 0.4949 vs IV 0.4206 was silent at α=0.20."""
    alerted, _ = _cycle(monkeypatch, sent, _vols(ETH=(0.4949, 0.4206)), {}, now=0.0)
    assert "ETH" in alerted


def test_btc_at_one_point_zero_two_still_stays_quiet(monkeypatch, sent):
    """α=0.10 is not α=0 — a name 2% over IV must not flip in and out every poll."""
    alerted, _ = _cycle(monkeypatch, sent, _vols(BTC=(0.3123, 0.3048)), {}, now=0.0)
    assert alerted == {} and sent == []


# --------------------------------------------------------------------------- #
# the reminder
# --------------------------------------------------------------------------- #
def test_reminder_is_four_hours():
    assert vb.REMIND_SECONDS == 4 * 3600


def test_a_persisting_breach_stays_quiet_before_four_hours(monkeypatch, sent):
    vols = _vols(ETH=(0.50, 0.40))
    alerted, _ = _cycle(monkeypatch, sent, vols, {}, now=0.0)
    sent.clear()
    for t in (900.0, 3_600.0, 4 * 3600 - 1):          # next poll, 1h, one second short
        alerted, _ = _cycle(monkeypatch, sent, vols, alerted, now=t)
    assert sent == []


def test_a_persisting_breach_is_repeated_after_four_hours(monkeypatch, sent):
    """Edge-triggered alone was too quiet: XRP sat in breach a whole morning on one message."""
    vols = _vols(ETH=(0.50, 0.40))
    alerted, _ = _cycle(monkeypatch, sent, vols, {}, now=0.0)
    sent.clear()
    alerted, _ = _cycle(monkeypatch, sent, vols, alerted, now=4 * 3600)
    assert len(sent) == 1 and "still breached" in sent[0]


def test_the_reminder_clock_restarts_from_the_last_message(monkeypatch, sent):
    """Timed off the last message, not off entry — so the cadence stays a clean 4h."""
    vols = _vols(ETH=(0.50, 0.40))
    alerted, _ = _cycle(monkeypatch, sent, vols, {}, now=0.0)
    alerted, _ = _cycle(monkeypatch, sent, vols, alerted, now=4 * 3600)
    sent.clear()
    alerted, _ = _cycle(monkeypatch, sent, vols, alerted, now=4 * 3600 + 900)   # 15m later
    assert sent == []
    alerted, _ = _cycle(monkeypatch, sent, vols, alerted, now=8 * 3600)
    assert len(sent) == 1 and "still breached" in sent[0]


def test_a_new_breach_is_immediate_not_held_for_the_reminder(monkeypatch, sent):
    """The reminder must not delay a name crossing in."""
    alerted, _ = _cycle(monkeypatch, sent, _vols(ETH=(0.50, 0.40)), {}, now=0.0)
    sent.clear()
    alerted, _ = _cycle(monkeypatch, sent, _vols(ETH=(0.50, 0.40), BTC=(0.40, 0.30)),
                        alerted, now=900.0)
    assert len(sent) == 1 and "breach" in sent[0] and "BTC" in sent[0] and "ETH" not in sent[0]


# --------------------------------------------------------------------------- #
# recovery
# --------------------------------------------------------------------------- #
def test_dropping_out_of_breach_sends_one_recovered_note(monkeypatch, sent):
    alerted, _ = _cycle(monkeypatch, sent, _vols(ETH=(0.50, 0.40)), {}, now=0.0)
    sent.clear()
    alerted, _ = _cycle(monkeypatch, sent, _vols(), alerted, now=900.0)
    assert alerted == {} and len(sent) == 1 and "recovered" in sent[0]


def test_a_recovered_name_breaching_again_alerts_again(monkeypatch, sent):
    vols_hot = _vols(ETH=(0.50, 0.40))
    alerted, _ = _cycle(monkeypatch, sent, vols_hot, {}, now=0.0)
    alerted, _ = _cycle(monkeypatch, sent, _vols(), alerted, now=900.0)
    sent.clear()
    alerted, _ = _cycle(monkeypatch, sent, vols_hot, alerted, now=1_800.0)
    assert len(sent) == 1 and "breach" in sent[0]


def test_a_quiet_poll_says_nothing(monkeypatch, sent):
    alerted, _ = _cycle(monkeypatch, sent, _vols(), {}, now=0.0)
    assert alerted == {} and sent == []


def test_the_startup_note_states_the_rule(monkeypatch, sent):
    _cycle(monkeypatch, sent, _vols(), {}, now=0.0, started=False)
    assert "1.10" in sent[0] and "4h" in sent[0]
