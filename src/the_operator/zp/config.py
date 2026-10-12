"""Token configuration for the ZP quoting review.

Loads ``config/zp/tokens.yaml`` and shapes one token's entry the way the frozen reference
(``research/zp_reference/mmanalytics_extract/trade_analysis_core.py``) shaped its ``config.yaml``:
``exchanges`` expanded from ``{pair: [EXCHANGE, ...]}`` to a flat ``[{pair, exchange}, ...]``, plus
``g20_name``/``deployment_names`` defaults.

MUST NOT: fetch data, call any adapter, or write anything — YAML load + reshape only.
"""

from __future__ import annotations

from typing import Any

from src.common.config import load_config

_TOKENS_CONFIG_PATH = "zp/tokens.yaml"


def _expand_exchanges(exchanges_dict: Any) -> list[dict[str, str]]:
    """``{pair: [EXCHANGE, ...]}`` -> ``[{'pair': pair, 'exchange': exch}, ...]``. Ported from the
    reference ``_expand_exchanges``; a list is passed through unchanged (already flat)."""
    flat: list[dict[str, str]] = []
    if isinstance(exchanges_dict, dict):
        for pair, exch_list in exchanges_dict.items():
            for exch in exch_list:
                flat.append({"pair": pair, "exchange": exch})
    elif isinstance(exchanges_dict, list):
        flat = exchanges_dict
    return flat


def token_config(token: str) -> dict[str, Any]:
    """One token's ZP config (case-insensitive), reshaped like the reference loader.

    Raises ``KeyError`` (with the list of known tokens) for a token not in
    ``config/zp/tokens.yaml``.
    """
    raw = load_config(_TOKENS_CONFIG_PATH)
    key = token.upper()
    if key not in raw:
        known = ", ".join(sorted(raw))
        raise KeyError(f"unknown ZP token {token!r}; add it to config/zp/tokens.yaml (known: {known})")
    cfg = dict(raw[key])  # shallow copy — never mutate the loaded config
    cfg.setdefault("g20_name", key.lower())
    cfg["exchanges"] = _expand_exchanges(cfg.get("exchanges", {}))
    cfg.setdefault("deployment_names", [cfg.get("g20_name", key.lower()).upper()])
    return cfg


def get_pairs(token: str) -> list[str]:
    """Sorted unique pairs from a token's (expanded) exchanges list."""
    return sorted({e["pair"] for e in token_config(token).get("exchanges", [])})
