"""Offline tests for `/log`. No Google Sheets, no Telegram — the sheet write is never exercised."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import scripts.log_bot as lb

_NOW = datetime(2026, 9, 4, 10, 30, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# parse_qty — the tokens-vs-dollars distinction
# --------------------------------------------------------------------------- #
def test_qty_is_tokens_unless_suffixed_usd():
    assert lb.parse_qty("100") == (100.0, "tok")
    assert lb.parse_qty("10000usd") == (10000.0, "usd")
    assert lb.parse_qty("10kusd") == (10000.0, "usd")
    assert lb.parse_qty("1,500") == (1500.0, "tok")


def test_qty_never_infers_usd_from_magnitude():
    """10000 is a plausible token count for a cheap coin AND a plausible dollar amount for an
    expensive one — the two differ by the price, so only the suffix may decide."""
    assert lb.parse_qty("10000")[1] == "tok"
    assert lb.parse_qty("0.5")[1] == "tok"


@pytest.mark.parametrize("raw", ["0", "-5", "abc", "", "usd"])
def test_qty_refuses_non_positive_or_unparseable(raw):
    with pytest.raises(ValueError):
        lb.parse_qty(raw)


# --------------------------------------------------------------------------- #
# resolve_amounts — both sides of the trade must be stored
# --------------------------------------------------------------------------- #
def test_token_entry_multiplies_and_usd_entry_divides():
    assert lb.resolve_amounts(100, "tok", 1905.2) == (100, pytest.approx(190520.0))
    tokens, notional = lb.resolve_amounts(10000, "usd", 1905.2)
    assert tokens == pytest.approx(5.2487927) and notional == 10000


def test_the_two_entry_forms_agree_on_the_same_trade():
    """$190,520 of ETH at 1905.2 IS 100 ETH — entering it either way must store the same row."""
    a = lb.resolve_amounts(100, "tok", 1905.2)
    b = lb.resolve_amounts(190520.0, "usd", 1905.2)
    assert a[0] == pytest.approx(b[0]) and a[1] == pytest.approx(b[1])


# --------------------------------------------------------------------------- #
# parse_log — strict positional, trader required
# --------------------------------------------------------------------------- #
def test_parse_log_reads_all_five_fields():
    assert lb.parse_log("/log buy eth 100 1905.2 sam") == {
        "side": "buy", "token": "ETH", "qty": 100.0, "unit": "tok", "price": 1905.2,
        "trader": "sam", "traded_at": None}      # None -> "traded now", filled by build_row


def test_parse_log_accepts_how_traders_say_the_side():
    for word in ("buy", "bought", "b"):
        assert lb.parse_log(f"/log {word} eth 100 1905.2 sam")["side"] == "buy"
    for word in ("sell", "sold", "s"):
        assert lb.parse_log(f"/log {word} eth 100 1905.2 sam")["side"] == "sell"


def test_parse_log_tolerates_an_at_sign_on_the_price():
    assert lb.parse_log("/log sell eth 100 @1905.2 sam")["price"] == 1905.2


@pytest.mark.parametrize("text", [
    "/log buy eth 100 1905.2",              # no trader — a row nobody can reconcile
    "/log buy eth 100 1905.2 sam extra",    # 7th arg must parse as a trade time
    "/log buy eth 100 1905.2 sam 09:35 more",  # strict arity
    "/log foo eth 100 1905.2 sam",          # bad side
    "/log buy eth 100 0 sam",               # zero price
    "/log buy eth 100 -5 sam",              # negative price
    "/log buy eth abc 1905.2 sam",          # unparseable qty
])
def test_parse_log_refuses_bad_input(text):
    with pytest.raises(ValueError):
        lb.parse_log(text, _NOW)


# --------------------------------------------------------------------------- #
# build_row / confirmation
# --------------------------------------------------------------------------- #
def test_row_matches_the_header_and_stores_both_amounts():
    cmd = lb.parse_log("/log sell eth 10000usd 1905.2 sam")
    row = lb.build_row(cmd, "mm3", "samh", "-100123", _NOW)
    d = dict(zip(lb.HEADER, row))
    assert len(row) == len(lb.HEADER)
    assert d["side"] == "sell" and d["token"] == "ETH"
    assert d["qty"] == pytest.approx(5.2487927777)      # tokens, derived
    assert d["notional_usd"] == 10000.0                 # dollars, as typed
    assert d["entered_as"] == "usd"
    assert d["trader"] == "sam" and d["logged_by"] == "samh"
    assert d["trade_ts_utc"] == "2026-09-04 10:30:00"      # no time given -> traded now
    assert d["logged_ts_utc"] == "2026-09-04 10:30:00"


def test_confirmation_echoes_what_was_stored_not_what_was_typed():
    """A usd-entered trade must confirm in TOKENS, so a mis-parse is visible immediately."""
    cmd = lb.parse_log("/log sell eth 10000usd 1905.2 sam")
    out = lb.confirmation(lb.build_row(cmd, "mm3", "samh", "-1", _NOW))
    assert "sell 5.2488 ETH" in out and "$10,000.00" in out and "entered as usd" in out


def test_desk_of_falls_back_to_the_chat_id(monkeypatch):
    """Gating is on the shared allowlist, so a chat need not be a desk chat — the row still needs
    an identifiable origin rather than a blank."""
    for var in ("TELEGRAM_MM3_CHAT_ID", "TELEGRAM_MM2_CHAT_ID", "TELEGRAM_TREASURY_CHAT_ID",
                "TELEGRAM_OPTIONS_CHAT_ID", "TELEGRAM_MM5_CHAT_ID"):
        monkeypatch.setenv(var, "-999")
    assert lb.desk_of("-12345") == "-12345"
    monkeypatch.setenv("TELEGRAM_MM3_CHAT_ID", "-12345,-777")
    assert lb.desk_of("-12345") == "mm3"


# --------------------------------------------------------------------------- #
# trade time vs log time
# --------------------------------------------------------------------------- #
def test_trade_and_log_times_are_separate_columns():
    """A trader logging an hour late must not produce a row that looks like an hour-late trade."""
    cmd = lb.parse_log("/log sell comp 1000 20.98 rami 09:35", _NOW)
    d = dict(zip(lb.HEADER, lb.build_row(cmd, "mm2", "samh", "-1", _NOW)))
    assert d["trade_ts_utc"] == "2026-09-04 09:35:00"
    assert d["logged_ts_utc"] == "2026-09-04 10:30:00"


def test_no_trade_time_means_it_happened_now():
    cmd = lb.parse_log("/log sell comp 1000 20.98 rami")
    d = dict(zip(lb.HEADER, lb.build_row(cmd, "mm2", "samh", "-1", _NOW)))
    assert d["trade_ts_utc"] == d["logged_ts_utc"] == "2026-09-04 10:30:00"


def test_trade_time_accepts_seconds_and_a_full_date():
    assert lb.parse_trade_time("09:35:20", _NOW).strftime("%Y-%m-%d %H:%M:%S") == \
        "2026-09-04 09:35:20"
    assert lb.parse_trade_time("2026-09-01-14:20", _NOW).strftime("%Y-%m-%d %H:%M") == \
        "2026-09-01 14:20"
    assert lb.parse_trade_time("2026/09/01 14:20".replace(" ", "-"), _NOW).day == 1


def test_a_bare_future_time_is_refused_not_read_as_today():
    """At 10:30 someone typing 23:00 means last night. Recording a trade that has not happened
    yet would corrupt any ordering built on this column, so it must be said with a date."""
    with pytest.raises(ValueError, match="future"):
        lb.parse_trade_time("23:00", _NOW)


def test_a_bare_time_within_clock_skew_is_allowed():
    """Two minutes of tolerance, so a slightly fast clock does not reject a legitimate 'just now'."""
    assert lb.parse_trade_time("10:31", _NOW).hour == 10


@pytest.mark.parametrize("raw", ["25:00", "nonsense", "9h35", ""])
def test_unparseable_trade_time_is_refused(raw):
    with pytest.raises(ValueError):
        lb.parse_trade_time(raw, _NOW)


def test_confirmation_shows_both_times_only_when_they_differ():
    late = lb.parse_log("/log sell comp 1000 20.98 rami 09:35", _NOW)
    assert "traded" in lb.confirmation(lb.build_row(late, "mm2", "s", "-1", _NOW))
    now = lb.parse_log("/log sell comp 1000 20.98 rami")
    assert "traded" not in lb.confirmation(lb.build_row(now, "mm2", "s", "-1", _NOW))


# --------------------------------------------------------------------------- #
# Trader defaults to the sender
# --------------------------------------------------------------------------- #
def test_trader_defaults_to_the_sender():
    cmd = lb.parse_log("/log buy eth 100 1905.2", _NOW, default_trader="sam")
    assert cmd["trader"] == "sam"
    assert cmd["traded_at"] is None              # nothing trailing -> traded now


def test_an_explicit_trader_still_wins_over_the_sender():
    """Logging on someone else's behalf must stay possible — that is the whole audit trail."""
    cmd = lb.parse_log("/log buy eth 100 1905.2 kim", _NOW, default_trader="sam")
    assert cmd["trader"] == "kim"


def test_a_lone_trailing_time_is_a_time_not_a_trader():
    cmd = lb.parse_log("/log sell eth 100 1905.2 09:35", _NOW, default_trader="sam")
    assert cmd["trader"] == "sam"
    assert cmd["traded_at"].hour == 9 and cmd["traded_at"].minute == 35


def test_a_lone_trailing_date_is_a_time_not_a_trader():
    cmd = lb.parse_log("/log sell eth 100 1905.2 2026-09-06-09:35", _NOW, default_trader="sam")
    assert cmd["trader"] == "sam"
    assert cmd["traded_at"].date().isoformat() == "2026-09-06"


def test_a_lone_future_time_is_refused_not_filed_as_a_trader_name():
    """The disambiguation is by shape, so parse_trade_time still gets to reject a future time
    instead of it silently becoming a trader called '23:00'."""
    with pytest.raises(ValueError, match="future"):
        lb.parse_log("/log sell eth 100 1905.2 23:00", _NOW, default_trader="sam")


def test_both_trader_and_time_still_parse_together():
    cmd = lb.parse_log("/log sell eth 100 1905.2 kim 09:35", _NOW, default_trader="sam")
    assert cmd["trader"] == "kim"
    assert cmd["traded_at"].hour == 9


def test_no_trader_and_no_sender_is_still_refused():
    """A sender with no Telegram username gives nothing to fall back on — better to refuse than
    to write an ownerless row."""
    with pytest.raises(ValueError):
        lb.parse_log("/log buy eth 100 1905.2", _NOW, default_trader="")


@pytest.mark.parametrize("text", [
    "/log buy eth 100",                          # too few even with a default trader
    "/log buy eth 100 1905.2 kim 09:35 more",    # still strict arity at the top end
])
def test_arity_stays_strict_with_the_default(text):
    with pytest.raises(ValueError):
        lb.parse_log(text, _NOW, default_trader="sam")
