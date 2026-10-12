"""Load instrument universe files (``config/universe/{source}.yaml``).

``load_universe(source)`` returns ``{symbol: instrument_dict}`` with each
instrument's ``session`` key resolved from the file's top-level ``sessions``
templates into its descriptive text. ``get_instrument`` is a convenience lookup.

MUST NOT: define instrument fields in code (they live in YAML), fetch data, or
decide what data_types exist (that is discovered via ``query.available()``).
"""

from __future__ import annotations

from typing import Any

from src.common import config


def load_universe(source: str) -> dict[str, dict[str, Any]]:
    """Return ``{symbol: fields}`` for ``source`` with sessions resolved."""
    raw = config.load_config(f"universe/{source}.yaml")
    sessions = raw.get("sessions", {})

    universe: dict[str, dict[str, Any]] = {}
    for symbol, fields in raw.items():
        if symbol == "sessions":
            continue
        if not isinstance(fields, dict):
            raise ValueError(f"universe entry {source}:{symbol} is not a mapping")
        inst = dict(fields)
        session_key = inst.get("session")
        if session_key is not None:
            if session_key not in sessions:
                raise ValueError(
                    f"{source}:{symbol} references unknown session {session_key!r}; "
                    f"defined: {sorted(sessions)}"
                )
            inst["session"] = sessions[session_key]
        universe[symbol] = inst
    return universe


def get_instrument(source: str, symbol: str) -> dict[str, Any]:
    """Return a single instrument's fields, or raise KeyError if unknown."""
    universe = load_universe(source)
    try:
        return universe[symbol]
    except KeyError:
        raise KeyError(
            f"instrument {source}:{symbol} not in universe ({sorted(universe)})"
        ) from None
