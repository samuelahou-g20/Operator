"""ToolSpec rejects schemas the strict API would reject (offline)."""

from __future__ import annotations

import pytest

from src.the_operator.contract import Risk, ToolResult, ToolSpec


def _h(ctx, args):
    return ToolResult("ok")


def test_strict_schema_accepted():
    ToolSpec("t", "d", {"type": "object", "properties": {"a": {"type": "string"}},
                        "required": ["a"], "additionalProperties": False}, Risk.READ, _h)


def test_missing_additional_properties_false_rejected():
    with pytest.raises(ValueError, match="additionalProperties"):
        ToolSpec("t", "d", {"type": "object", "properties": {}, "required": []}, Risk.READ, _h)


def test_every_property_must_be_required():
    with pytest.raises(ValueError, match="required"):
        ToolSpec("t", "d", {"type": "object", "properties": {"a": {"type": "string"}},
                            "required": [], "additionalProperties": False}, Risk.READ, _h)


def test_non_object_rejected():
    with pytest.raises(ValueError, match="object"):
        ToolSpec("t", "d", {"type": "string", "additionalProperties": False}, Risk.READ, _h)
