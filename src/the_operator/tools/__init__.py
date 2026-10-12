"""Operator tool catalogue: ``TOOLS`` is every ``ToolSpec`` the agent may be offered, in a fixed
order (a stable order keeps the cached prompt prefix stable). MUST NOT: contain logic or policy.
"""

from src.the_operator.contract import ToolSpec
from src.the_operator.tools import (
    activity, advisory, chat, discovery, engine, intel, market, quoting, trading,
)

TOOLS: list[ToolSpec] = [*market.TOOLS, *activity.TOOLS, *intel.TOOLS, *advisory.TOOLS,
                         *discovery.TOOLS, *quoting.TOOLS, *engine.TOOLS, *chat.TOOLS,
                         *trading.TOOLS]

_names = [t.name for t in TOOLS]
if len(set(_names)) != len(_names):
    raise RuntimeError(f"duplicate Operator tool names: {sorted(n for n in _names if _names.count(n) > 1)}")
