"""Read-only Reference Data Service client (ARCHITECTURE §9 — thin service translation).

RDS is the registry of **AssetLocations**: every venue account the firm trades, tagged with
``desk`` (MM3/MM2/OTC/...), ``strategy`` (MM/GAMMA/HEDGING/HEDGERV2/CUSTODY), ``client`` and
``entity``. Its ``location_id`` is the key the internal trade service reports fills against, so
this adapter is what turns "the MM3 gamma book" into the id list a fill query needs.

MUST NOT: place/modify/cancel orders, hold desk or PnL logic, or open the data lake.
"""

from __future__ import annotations

import os

import requests

from src.common.config import load_dotenv, require_env
from src.common.logging import get_logger

_log = get_logger("refsvc")

DEFAULT_BASE_URL = "https://refsvc.prod.g20api.com/ref-svc"

# Category -> the RDS `strategy` tags it covers. Deliberately a mapping, not an identity: MM2
# carries BOTH `HEDGING` (1 account) and `HEDGERV2` (130), so a category that only looked for
# `HEDGING` would report 1/131 of that desk's hedge book. CUSTODY is not trading and is excluded
# from every category, including `all`.
CATEGORY_TAGS: dict[str, set[str]] = {
    "mm": {"MM"},
    "gamma": {"GAMMA"},
    "hedge": {"HEDGING", "HEDGERV2"},
}
CATEGORY_TAGS["all"] = set().union(*CATEGORY_TAGS.values())

# Reference-data corrections: mandates whose accounts carry a stale `desk` tag. Keyed by the
# `client` tag (the mandate id, e.g. `H021-LAPTOP` for strategy `mm-h021_laptop-gamma-j`) rather
# than by location_id, so a mandate that opens another venue is covered with no code change --
# a list of ids goes stale the moment it does.
#
# This is a correction, not policy: RDS is the record and it is simply behind. DELETE an entry
# the day reference data is fixed, and verify with `RefSvc._tag(loc, "desk")`.
#   H021-LAPTOP  LAPTOP  traded by MM3, all 19 accounts still tagged desk=MM2 (listed 2026-09-09)
#   C236-CXT     CXT     moved to MM4 2026-09-10, 15 accounts still tagged desk=MM3
CLIENT_DESK: dict[str, str] = {"H021-LAPTOP": "MM3", "C236-CXT": "MM4"}


class RefSvc:
    """Read-only client for the RDS asset-location registry."""

    def __init__(self, *, base_url: str = DEFAULT_BASE_URL,
                 session: requests.Session | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self._token: str | None = None
        self._locations: list[dict] | None = None

    def token(self) -> str:
        """A JWT for the read API: a pre-minted ``REFSVC_TOKEN`` if set, else minted from
        ``REFSVC_USER``/``REFSVC_PASSWORD``. Cached for the client's lifetime."""
        if self._token:
            return self._token
        load_dotenv()          # REFSVC_TOKEN is read directly, so .env must be loaded FIRST —
        # otherwise the first call sees nothing, falls through to the password path and raises,
        # and only later calls (after require_env has loaded .env as a side effect) succeed.
        tok = os.environ.get("REFSVC_TOKEN")
        if not tok:
            user = os.environ.get("REFSVC_USER") or "prod_system_reader"
            r = self.session.post(f"{self.base_url}/token", verify=False, timeout=30,
                                  data={"username": user,
                                        "password": require_env("REFSVC_PASSWORD")})
            r.raise_for_status()
            tok = r.json()["access_token"]
        self._token = tok
        return tok

    def locations(self, *, refresh: bool = False) -> list[dict]:
        """Every AssetLocation record (cached — the registry changes on the order of days)."""
        if self._locations is None or refresh:
            r = self.session.get(f"{self.base_url}/location", verify=False, timeout=60,
                                 headers={"Authorization": f"Bearer {self.token()}"})
            r.raise_for_status()
            self._locations = r.json()
        return self._locations

    @classmethod
    def _desk_of(cls, loc: dict) -> str | None:
        """The desk that actually trades this account: the ``CLIENT_DESK`` correction when the
        mandate has one, else the RDS ``desk`` tag. Everything that scopes by desk must go
        through this, or a corrected mandate is both missing from its real desk and present on
        the stale one."""
        return CLIENT_DESK.get(cls._tag(loc, "client") or "") or cls._tag(loc, "desk")

    @staticmethod
    def _tag(loc: dict, key: str) -> str | None:
        v = (loc.get("tags") or {}).get(key) or []
        return v[0] if v else None

    def location_ids(self, desk: str, category: str) -> list[int]:
        """Active ``location_id``s for one desk and trade category (``mm``/``gamma``/``hedge``/
        ``all``). Raises ValueError on an unknown category.

        Tag-driven, not a fixed list: a mandate onboarded with the right tags is picked up with no
        code change — but one tagged wrongly, or left untagged, is silently invisible here."""
        tags = CATEGORY_TAGS.get(category.lower())
        if tags is None:
            raise ValueError(f"unknown category {category!r}; use {'/'.join(CATEGORY_TAGS)}")
        return sorted({loc["location_id"] for loc in self.locations()
                       if self._desk_of(loc) == desk.upper()
                       and self._tag(loc, "strategy") in tags
                       and loc.get("invalidation_time_utc") is None})

    def venues_by_tag(self, key: str, value: str) -> dict[int, str]:
        """``{location_id: venue}`` for every ACTIVE location whose ``key`` tag equals ``value``
        (lowercase venue; empty string when the record carries none).

        Tag-agnostic because the registry identifies a book differently depending on which one:
        the MM desks carry a ``strategy`` tag that ``location_ids`` filters on, while the treasury
        prop book carries none at all — its 3 accounts are identified by ``client=N009-TREASURY_PROP``
        instead. Matching on the wrong tag silently returns either nothing or far too much.

        Comparison is case-insensitive: the tag's own case is inconsistent (``MM3``/``PROP`` but
        ``Treasury``), which defeats an upper()-only test.
        """
        out: dict[int, str] = {}
        for loc in self.locations():
            if str(self._tag(loc, key) or "").upper() != value.upper():
                continue
            if loc.get("invalidation_time_utc"):
                continue
            out[loc["location_id"]] = str(loc.get("venue") or "").lower()
        return out

    def category_of(self, desk: str) -> dict[int, str]:
        """``{location_id: category}`` for one desk, so a fill can be labelled by where it landed."""
        tag_to_cat = {t: c for c, ts in CATEGORY_TAGS.items() if c != "all" for t in ts}
        out: dict[int, str] = {}
        for loc in self.locations():
            if self._desk_of(loc) != desk.upper() or loc.get("invalidation_time_utc"):
                continue
            cat = tag_to_cat.get(self._tag(loc, "strategy"))
            if cat:
                out[loc["location_id"]] = cat
        return out
