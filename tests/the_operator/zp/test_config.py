"""Tests for zp.config — token config load/expand against the real config/zp/tokens.yaml."""

from __future__ import annotations

import pytest

from src.the_operator.zp import config


def test_token_config_expands_exchanges_and_defaults():
    cfg = config.token_config("icp")   # case-insensitive
    assert cfg["md_pair"] == "ICP-USDT"
    assert cfg["md_exchange"] == "binance"
    assert isinstance(cfg["exchanges"], list)
    assert {"pair": "ICP-USDT", "exchange": "BINANCE"} in cfg["exchanges"]
    assert cfg["g20_name"] == "icp"
    assert cfg["deployment_names"] == ["ICP"]


def test_get_pairs_sorted_unique():
    pairs = config.get_pairs("ICP")
    assert pairs == sorted(set(pairs))
    assert "ICP-USDT" in pairs


def test_unknown_token_raises_keyerror_with_hint():
    with pytest.raises(KeyError, match="unknown ZP token"):
        config.token_config("NOT_A_REAL_TOKEN")
