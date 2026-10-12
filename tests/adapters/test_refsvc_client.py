"""Offline tests for the RDS asset-location client (live API checks are manual, per repo rules)."""

from __future__ import annotations

from src.adapters.refsvc.client import RefSvc


def _loc(loc_id: int, desk: str | None, strategy: str | None, venue: str | None = None,
         invalidated: str | None = None) -> dict:
    tags: dict = {}
    if desk is not None:
        tags["desk"] = [desk]
    if strategy is not None:
        tags["strategy"] = [strategy]
    return {"location_id": loc_id, "tags": tags, "venue": venue,
            "invalidation_time_utc": invalidated}


def _svc(locs: list[dict]) -> RefSvc:
    rs = RefSvc()
    rs._locations = locs          # bypass the HTTP fetch; locations() returns the cache
    return rs


def test_venues_by_tag_ignores_the_strategy_tag():
    """The treasury prop book has every strategy tag null, so the category-driven location_ids can
    never resolve it — venues_by_tag must not filter on strategy."""
    rs = _svc([_loc(1, "PROP", None, "BINANCE"), _loc(2, "PROP", None, "DERIBIT"),
               _loc(3, "MM3", "GAMMA", "BINANCE")])
    assert rs.venues_by_tag("desk", "PROP") == {1: "binance", 2: "deribit"}
    assert rs.location_ids("PROP", "all") == []          # the bug this works around


def test_venues_by_tag_selects_the_prop_book_by_its_client_tag():
    """The book is identified by `client=N009-TREASURY_PROP` (3 accounts), NOT by `desk=PROP`,
    which spans 44 accounts across other books too."""
    locs = [_loc(1, "PROP", None, "BINANCE"), _loc(2, "PROP", None, "BYBIT"),
            _loc(3, "PROP", None, "MEXC")]
    locs[0]["tags"]["client"] = ["N009-TREASURY_PROP"]
    locs[1]["tags"]["client"] = ["N009-TREASURY_PROP"]
    locs[2]["tags"]["client"] = ["SOMEONE-ELSE"]
    rs = _svc(locs)
    assert rs.venues_by_tag("client", "N009-TREASURY_PROP") == {1: "binance", 2: "bybit"}
    assert len(rs.venues_by_tag("desk", "PROP")) == 3      # the over-broad filter it replaces


def test_venues_by_tag_matches_case_insensitively():
    """The tag's own case is inconsistent (MM3/PROP but Treasury), which defeats upper()-only."""
    rs = _svc([_loc(1, "Treasury", None, "OKX")])
    assert rs.venues_by_tag("desk", "treasury") == {1: "okx"}
    assert rs.venues_by_tag("desk", "TREASURY") == {1: "okx"}


def test_venues_by_tag_drops_invalidated_locations_and_tolerates_a_missing_venue():
    rs = _svc([_loc(1, "PROP", None, "BYBIT"),
               _loc(2, "PROP", None, "BYBIT", invalidated="2026-01-01T00:00:00Z"),
               _loc(3, "PROP", None, None)])
    assert rs.venues_by_tag("desk", "PROP") == {1: "bybit", 3: ""}
