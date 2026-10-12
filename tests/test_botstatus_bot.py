"""Offline tests for botstatus_bot: roster filtering, the MISSING row, error-caption handling,
desk routing/refusal, and the read-only rail (no network — requests.get is stubbed)."""

from __future__ import annotations

import inspect
import os
import re

import pandas as pd
import pytest
import requests

from scripts import botstatus_bot as bs

_MM3 = ["*MM3_JASONDH", "*MM3_REGIME_TRADING_CANIO"]
_TREASURY = ["*TREASURY_PROP_JASONDH", "*TREASURYPROP Complex futures"]


def _item(name, *, type_="JasonDHBot", status="ACTIVE", op="", conn="OK", err=""):
    return {"name": name, "runtimeName": name.lstrip("*"), "ownerType": "Firm", "type": type_,
            "status": status, "operationalStatus": op, "connectionStatus": conn,
            "lastError": err, "hasCube": True}


class _Resp:
    status_code = 200

    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


@pytest.fixture
def spy(monkeypatch):
    """Stubs requests.get and records the URL/params the bot asked for."""
    seen = {"payload": {"success": True, "count": 0, "items": []}}

    def _get(url, headers=None, verify=None, timeout=None, params=None):
        seen["url"], seen["params"], seen["timeout"] = url, params, timeout
        return _Resp(seen["payload"])

    monkeypatch.setenv("PLUTO_API_KEY", "k")
    monkeypatch.setenv("PLUTO_API_SECRET", "s")
    monkeypatch.setattr(requests, "get", _get)
    return seen


# --------------------------------------------------------------------------- #
# Read-only rail — the reason this bot exists in this shape
# --------------------------------------------------------------------------- #
def test_only_read_endpoints_are_wired():
    """No BotManager WRITE path may exist in this module. `GET /status` and `GET /{bot}/config` are
    both Read; start/stop/operations and any config write would let a chat message halt or re-tune a
    live trading bot. This scans the source for every URL template that can be built off the
    BotManager base, so adding a third one fails here rather than in production."""
    assert bs._STATUS_URL == f"{bs._BOTMANAGER}/status"
    assert bs._CONFIG_URL == f"{bs._BOTMANAGER}/{{}}/config"
    src = inspect.getsource(bs).replace(bs.__doc__ or "", "")   # the docstring names the writes
    # `{{}}` is how the f-string source spells the literal `{}` placeholder in _CONFIG_URL.
    assert set(re.findall(r'_BOTMANAGER\}([^"]*)"', src)) == {"/status", "/{{}}/config"}
    for verb in ("requests.post", "requests.put", "requests.delete", "requests.patch"):
        assert verb not in src, f"write verb {verb!r} appeared in botstatus_bot"


def test_config_url_encodes_the_star_and_spaces(spy):
    """Every bot name starts with a literal `*` and two contain spaces — an unencoded name would
    hit the wrong path (or none)."""
    spy["payload"] = {"success": True, "secretsRedacted": True, "config": {"tabs": []}}
    bs.fetch_config("*TREASURYPROP Complex futures")
    assert spy["url"] == f"{bs._BOTMANAGER}/%2ATREASURYPROP%20Complex%20futures/config"
    assert spy["timeout"] == bs._HTTP_TIMEOUT


def test_fetch_sends_one_roster_glob_and_a_timeout(spy):
    """One call covers the desk (the API honours only the first `name` value, so repeating the
    param cannot work), and every call is bounded."""
    bs.fetch_status(bs.roster_glob(_MM3))
    assert spy["url"] == bs._STATUS_URL
    assert spy["params"]["name"] == "*MM3_*"
    assert spy["params"]["include_inactive"] == "true"
    assert spy["timeout"] == bs._HTTP_TIMEOUT


def test_roster_glob_covers_names_with_spaces_and_no_shared_suffix():
    """`*TREASURYPROP Complex futures` has a space and no `_` after TREASURY — the glob must still
    cover it, and a roster with nothing in common must degrade to `*`, never to a partial match."""
    assert bs.roster_glob(_TREASURY) == "*TREASURY*"
    assert bs.roster_glob(["*MM3_JASONDH", "*MM2_JASON_DH"]) == "*MM*"
    assert bs.roster_glob(["*A", "B"]) == "*"


# --------------------------------------------------------------------------- #
# Roster filtering + the MISSING row
# --------------------------------------------------------------------------- #
def test_only_rostered_bots_are_shown():
    """The API returns ~172 firm-wide bots; a glob can over-match. Only the desk roster is shown."""
    items = {n: _item(n) for n in [*_MM3, "*MM3_SOMETHING_ELSE", "*THALEX_DH"]}
    df = bs.build_rows(_MM3, items)
    assert list(df["Bot"]) == ["MM3_JASONDH", "MM3_REGIME_TRADING_CANIO"]


def test_a_bot_the_api_omits_still_gets_a_missing_row():
    """Silently dropping it would hide exactly the failure this bot exists to catch."""
    df = bs.build_rows(_MM3, {"*MM3_JASONDH": _item("*MM3_JASONDH")})
    assert list(df["Bot"]) == ["MM3_JASONDH", "MM3_REGIME_TRADING_CANIO"]
    assert list(df["Status"]) == ["ACTIVE", bs._MISSING]


def test_rows_keep_roster_order_and_strip_the_firm_star():
    df = bs.build_rows(_TREASURY, {n: _item(n, status="NOT_LOADED") for n in _TREASURY})
    assert list(df["Bot"]) == ["TREASURY_PROP_JASONDH", "TREASURYPROP Complex futures"]


def test_missing_fields_render_as_blanks_not_none():
    """operationalStatus/connectionStatus are routinely empty; `None` would print in the table."""
    df = bs.build_rows(["*X"], {"*X": {"name": "*X", "status": "NOT_LOADED"}})
    assert list(df.iloc[0]) == ["X", "", "NOT_LOADED", "", ""]


# --------------------------------------------------------------------------- #
# Error caption
# --------------------------------------------------------------------------- #
def test_error_lines_collapse_whitespace_and_truncate():
    """lastError is free text and often a multi-line stack trace — a raw one would blow Telegram's
    caption limit and the photo would arrive with no caption at all."""
    long_err = "Error 08/28/2026 12:46:27:\nRejected SPOT update for SWEATUSD  " + "x" * 400
    lines = bs.error_lines(_MM3, {"*MM3_JASONDH": _item("*MM3_JASONDH", err=long_err)})
    assert len(lines) == 1
    assert "\n" not in lines[0]
    assert "update for SWEATUSD x" in lines[0]           # collapsed, not just stripped
    assert len(lines[0]) <= len("MM3_JASONDH: ") + bs._ERR_CHARS


def test_error_lines_skips_bots_without_an_error():
    items = {n: _item(n) for n in _MM3}
    items["*MM3_JASONDH"]["lastError"] = "boom"
    assert bs.error_lines(_MM3, items) == ["MM3_JASONDH: boom"]


def test_caption_reports_no_errors_and_counts_missing():
    df = bs.build_rows(_MM3, {"*MM3_JASONDH": _item("*MM3_JASONDH")})
    cap = bs.build_caption("MM3 Bots", df, [])
    assert "2 bots · 1 ACTIVE · 1 MISSING" in cap
    assert "No errors reported." in cap


def test_caption_stays_within_the_telegram_limit():
    """Enough errored bots would still overflow 1024 chars even after the per-bot cap."""
    roster = [f"*B{i}" for i in range(20)]
    items = {n: _item(n, err="y" * 400) for n in roster}
    cap = bs.build_caption("MM3 Bots", bs.build_rows(roster, items), bs.error_lines(roster, items))
    assert len(cap) <= bs._CAPTION_LIMIT


# --------------------------------------------------------------------------- #
# Desk routing — an unpinned chat is refused, never defaulted
# --------------------------------------------------------------------------- #
def _pins(monkeypatch, mm3="", mm2="", treasury=""):
    for var, val in [("TELEGRAM_MM3_CHAT_ID", mm3), ("TELEGRAM_MM2_CHAT_ID", mm2),
                     ("TELEGRAM_TREASURY_CHAT_ID", treasury)]:
        monkeypatch.setenv(var, val)


def _desk_map() -> dict[str, str]:
    """The chat->desk map main() builds, exercised without starting a poller."""
    pins = [("mm3", {c.strip() for c in (os.environ.get("TELEGRAM_MM3_CHAT_ID") or "").split(",") if c.strip()}),
            ("mm2", {c.strip() for c in (os.environ.get("TELEGRAM_MM2_CHAT_ID") or "").split(",") if c.strip()}),
            ("treasury", {c.strip() for c in (os.environ.get("TELEGRAM_TREASURY_CHAT_ID") or "").split(",") if c.strip()})]
    return {c: name for name, chats in pins for c in chats}


def test_each_pinned_chat_maps_to_its_own_desk(monkeypatch):
    _pins(monkeypatch, mm3="-100,-101", mm2="-200", treasury="-300")
    m = _desk_map()
    assert (m["-100"], m["-101"], m["-200"], m["-300"]) == ("mm3", "mm3", "mm2", "treasury")


def test_an_unpinned_chat_is_refused(monkeypatch):
    """No default desk: a chat that isn't pinned gets nothing, not mm3's roster."""
    _pins(monkeypatch, mm3="-100")
    assert _desk_map().get("-999") is None


def test_main_rejects_a_chat_pinned_to_two_desks(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOTSTATUS_BOT_TOKEN", "t")
    monkeypatch.setattr(bs, "single_instance_lock", lambda name: object())
    _pins(monkeypatch, mm3="-100", mm2="-100")
    with pytest.raises(RuntimeError, match="pinned to both"):
        bs.main()


# --------------------------------------------------------------------------- #
# Config roster
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("desk", ["mm3", "mm2", "treasury"])
def test_every_desk_has_a_roster_in_config(desk):
    roster = bs.desk_roster(desk)
    assert roster and all(isinstance(n, str) and n for n in roster)


def test_desk_roster_raises_for_a_desk_without_one():
    with pytest.raises(RuntimeError, match="no `bots` roster"):
        bs.desk_roster("options")


# --------------------------------------------------------------------------- #
# Config fetch — the payload is nested one level deeper than it looks
# --------------------------------------------------------------------------- #
def _tab(name, fields):
    return {"name": name, "fields": [{"name": n, "value": v, "type": t} for n, v, t in fields]}


_STRAT_FIELDS = [("Active", "true", "Boolean"), ("Biprod", "UNIUSD", "String"),
                 ("Hedge Instrument", "UNIUSDT", "String"),
                 ("Liquidity Provider", "BINANCEF", "String"),
                 ("Max Execution", "3$", "String"),
                 ("GateOpen-RestingStateDelta", "0$", "String"),
                 ("GateOpen-ActivationDeadBand", "100$", "String")]


def test_fetch_config_reads_tabs_from_the_nested_config_mapping(spy):
    """`config` is a MAPPING (type_of_bot/name/class_name/tabs), not the tab list itself — reading
    `config` as a list is the natural mistake and yields a TypeError deep in the renderer."""
    spy["payload"] = {"success": True, "botName": "*X", "secretsRedacted": True,
                      "config": {"type_of_bot": "JasonDHBot", "name": "", "class_name": "",
                                 "tabs": [_tab("General", [("A", "1", "Double")])]}}
    tabs, redacted = bs.fetch_config("*X")
    assert [t["name"] for t in tabs] == ["General"]
    assert redacted is True


def test_fetch_config_rejects_an_unexpected_shape(spy):
    spy["payload"] = {"success": True, "config": [{"name": "General", "fields": []}]}
    with pytest.raises(RuntimeError, match="unexpected BotManager config payload"):
        bs.fetch_config("*X")


def test_fetch_config_reports_unredacted_secrets(spy):
    """`secretsRedacted` must be surfaced, not assumed — the caption warns when it is false."""
    spy["payload"] = {"success": True, "secretsRedacted": False, "config": {"tabs": []}}
    _tabs, redacted = bs.fetch_config("*X")
    assert redacted is False
    assert "secretsRedacted=false" in bs._config_caption("*X", "0 tabs", 0, redacted)


# --------------------------------------------------------------------------- #
# Bot resolution — desk-scoped, never a guess between two live bots
# --------------------------------------------------------------------------- #
def test_resolve_bot_matches_a_case_insensitive_substring():
    """The owner will not type `*TREASURY_PROP_JASONDH`."""
    assert bs.resolve_bot(_TREASURY, "jasondh")[0] == "*TREASURY_PROP_JASONDH"
    assert bs.resolve_bot(_TREASURY, "Complex")[0] == "*TREASURYPROP Complex futures"


def test_resolve_bot_never_reaches_outside_the_calling_desk():
    """The firm runs ~172 bots; a treasury chat must not be able to name an MM2 one."""
    name, candidates = bs.resolve_bot(_TREASURY, "mm2_jason")
    assert name is None and candidates == []


def test_resolve_bot_reports_candidates_instead_of_guessing():
    roster = ["*MM2_FAKE_GAMMA_HEDGER", "*MM2_FAKE_GAMMA_HEDGER_OKX", "*MM2_JASON_DH"]
    name, candidates = bs.resolve_bot(roster, "gamma")
    assert name is None
    assert candidates == ["*MM2_FAKE_GAMMA_HEDGER", "*MM2_FAKE_GAMMA_HEDGER_OKX"]


def test_an_exact_name_wins_over_a_longer_one_containing_it():
    """Without this `*MM2_FAKE_GAMMA_HEDGER` would be permanently unaddressable — its whole name is
    a substring of `*MM2_FAKE_GAMMA_HEDGER_OKX`."""
    roster = ["*MM2_FAKE_GAMMA_HEDGER", "*MM2_FAKE_GAMMA_HEDGER_OKX"]
    assert bs.resolve_bot(roster, "mm2_fake_gamma_hedger")[0] == "*MM2_FAKE_GAMMA_HEDGER"


def test_resolve_bot_rejects_an_empty_query():
    assert bs.resolve_bot(_TREASURY, "  ") == (None, [])


# --------------------------------------------------------------------------- #
# Strategy summary
# --------------------------------------------------------------------------- #
def test_strategy_tabs_sort_numerically_not_lexicographically():
    """The API returns tabs lexicographically, which puts Strategy10 between Strategy1 and
    Strategy2 (verified live on the treasury JasonDH)."""
    tabs = [_tab(n, []) for n in ["Strategy1", "Strategy10", "Strategy2", "General", "Strategy9"]]
    assert [t["name"] for t in bs.strategy_tabs(tabs)] == \
        ["Strategy1", "Strategy2", "Strategy9", "Strategy10"]


def test_strategy_tabs_ignores_lookalike_names():
    tabs = [_tab(n, []) for n in ["Strategy1", "Strategy1Extra", "StrategyX", "VenueBasis-X"]]
    assert [t["name"] for t in bs.strategy_tabs(tabs)] == ["Strategy1"]


def test_strategy_rows_carry_the_decision_relevant_fields():
    df = bs.build_strategy_rows([_tab("Strategy2", _STRAT_FIELDS)])
    assert list(df.columns) == ["Strategy", "Biprod", "Active", "Hedge Instr", "LP", "Max Exec",
                                "Resting Δ", "Deadband"]
    # sizing is rendered in thousands ("3$" is $3,000) — see money()
    assert list(df.iloc[0]) == ["Strategy2", "UNIUSD", "true", "UNIUSDT", "BINANCEF", "$3k",
                                "$0k", "$100k"]


def test_a_strategy_tab_missing_a_field_renders_blank_not_an_error():
    df = bs.build_strategy_rows([_tab("Strategy8", [("Active", "false", "Boolean")])])
    assert list(df.iloc[0]) == ["Strategy8", "", "false", "", "", "", "", ""]


def test_a_bot_with_no_strategy_tabs_yields_an_empty_frame():
    """`*TREASURYPROP NOTIFIER` and `*TREASURYPROP Complex futures` genuinely have none — the
    handler answers with a pointer to `tabs`, not an empty picture."""
    assert bs.build_strategy_rows([_tab("Notifications", [])]).empty


# --------------------------------------------------------------------------- #
# Tab views
# --------------------------------------------------------------------------- #
_TABS = [_tab("General", [("A", "1", "Double")]), _tab("LP-OKX", [("B", "2", "String")]),
         _tab("Strategy1", _STRAT_FIELDS), _tab("VenueBasis-BINANCEF_BTCUSDT", []),
         _tab("VenueBasis-BINANCEF_ICPUSDT", [])]


def test_find_tab_is_case_insensitive_and_accepts_a_unique_substring():
    assert bs.find_tab(_TABS, "general")["name"] == "General"
    assert bs.find_tab(_TABS, "STRATEGY1")["name"] == "Strategy1"
    assert bs.find_tab(_TABS, "okx")["name"] == "LP-OKX"


def test_find_tab_refuses_an_ambiguous_substring():
    assert bs.find_tab(_TABS, "venuebasis") is None


def test_close_tab_names_suggests_instead_of_dumping_all_58():
    near = bs.close_tab_names(_TABS, "venuebasis")
    assert near[:2] == ["VenueBasis-BINANCEF_BTCUSDT", "VenueBasis-BINANCEF_ICPUSDT"]
    assert len(near) <= 6
    assert bs.close_tab_names(_TABS, "generl") == ["General"]      # fuzzy, not just substring


def test_build_tab_rows_counts_fields():
    df = bs.build_tab_rows(_TABS)
    assert list(df.iloc[0]) == ["General", "1"]
    assert list(df.iloc[3]) == ["VenueBasis-BINANCEF_BTCUSDT", "0"]


def test_enum_types_are_truncated_per_cell():
    """A `type` is either a scalar type or the comma-separated list of allowed values, which runs
    to 156 chars live — untruncated it would push Value off the image."""
    enum = "STICKYSTRIKE,STICKYSTRIKE2,STICKYSTRIKEOFFICIAL,STICKYDELTA,LN,DYNAMIC,DYNAMIC60"
    df = bs.build_field_rows(_tab("Revaluation", [("Model", "x" * 200, enum)]))
    assert len(df.iloc[0]["Type"]) <= bs._TYPE_CHARS
    assert df.iloc[0]["Type"].endswith("…")
    assert len(df.iloc[0]["Value"]) <= bs._VAL_CHARS


def test_field_values_are_whitespace_collapsed():
    df = bs.build_field_rows(_tab("General", [("A", "one\ntwo   three", "String")]))
    assert df.iloc[0]["Value"] == "one two three"


# --------------------------------------------------------------------------- #
# Row cap + layout
# --------------------------------------------------------------------------- #
def test_cap_rows_reports_what_it_dropped():
    """A silently short table reads as a complete one — the drop must reach the caption."""
    df = pd.DataFrame({"Tab": [str(i) for i in range(bs._MAX_ROWS + 7)]})
    capped, dropped = bs.cap_rows(df)
    assert len(capped) == bs._MAX_ROWS and dropped == 7
    cap = bs._config_caption("*X", "67 tabs", dropped, True)
    assert "7 more row(s) not shown" in cap


def test_cap_rows_leaves_a_short_table_alone():
    df = pd.DataFrame({"Tab": ["a", "b"]})
    capped, dropped = bs.cap_rows(df)
    assert len(capped) == 2 and dropped == 0
    assert "not shown" not in bs._config_caption("*X", "2 tabs", dropped, True)


def test_two_column_folds_a_tall_table_and_keeps_every_row():
    df = bs.build_tab_rows([_tab(f"T{i}", []) for i in range(9)])
    out = bs.two_column(df)
    assert len(out) == 5 and list(out.columns) == ["Tab", "Fields", "Tab ", "Fields "]
    assert set(out["Tab"]) | set(out["Tab "]) == {f"T{i}" for i in range(9)} | {""}


# --------------------------------------------------------------------------- #
# money(): Pluto stores sizing params in THOUSANDS
# --------------------------------------------------------------------------- #
def test_money_makes_the_thousands_unit_explicit():
    """`25$` is $25,000, not $25 — owner-confirmed, and the CircuitBreaker tab spells the same
    convention out as `Change in PV(in K)`. The digits must stay as Pluto shows them so a row still
    ties to the Pluto UI; only the unit is added."""
    assert bs.money("25$") == "$25k"
    assert bs.money("0$") == "$0k"
    assert bs.money("150$") == "$150k"
    assert bs.money("1,250$") == "$1,250k"
    assert bs.money("-80$") == "$-80k"


def test_money_passes_through_anything_that_is_not_a_money_value():
    """Only the `<number>$` shape is a money field. A bare number, an enum, a bool or an instrument
    must survive untouched — mislabelling `BINANCEF` as money would be worse than not formatting."""
    for raw in ("BINANCEF", "true", "0.1", "UNIUSDT", "AGGRESSIVE", "", "5.0"):
        assert bs.money(raw) == raw
    assert bs.money(None) == ""


def test_strategy_rows_render_sizing_in_thousands():
    tabs = [{"name": "Strategy1", "fields": [
        {"name": "Active", "value": "true"}, {"name": "Biprod", "value": "BTCUSD"},
        {"name": "Max Execution", "value": "25$"},
        {"name": "GateOpen-RestingStateDelta", "value": "150$"},
        {"name": "GateOpen-ActivationDeadBand", "value": "350$"}]}]
    row = bs.build_strategy_rows(tabs).iloc[0]
    assert row["Max Exec"] == "$25k" and row["Resting Δ"] == "$150k" and row["Deadband"] == "$350k"
    assert row["Biprod"] == "BTCUSD" and row["Active"] == "true"
