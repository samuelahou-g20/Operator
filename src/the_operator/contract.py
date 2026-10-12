"""The tool contract: the one shape every Operator tool is written against.

A ``ToolSpec`` is a name, a description the model reads, a strict JSON schema, a ``Risk`` class
and a handler ``(Context, args) -> ToolResult``. Handlers may raise; ``executor`` turns that into
an error result. ``Attachment`` is anything that goes to the chat as a photo/document rather than
as model-visible text. A MONEY tool returns its fully built action in ``ToolResult.proposal`` and
never executes it itself: whether a proposal is dry-run, confirmed or executed is the executor's
decision, driven by ``policy``.

MUST NOT: import the SDK, Telegram, or any tool module; hold policy (see ``policy``); do I/O.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class Risk(str, Enum):
    READ = "read"          # fetch + render only, no side effects anywhere
    WRITE = "write"        # writes a non-money artefact (a sheet, a note)
    MONEY = "money"        # places / amends / cancels orders
    CONTROL = "control"    # starts / stops / reconfigures an external system


@dataclass(frozen=True)
class Context:
    """Who asked, from where. Built by the gateway; tools read, never build, it."""
    request_id: str
    desk: str              # "mm3" | "mm2" | "treasury" | "options" | "mm5"
    chat_id: str
    user_id: str
    user_name: str
    is_trader: bool = False   # on the desk's trade_bot trader allowlist (gateway-computed)


@dataclass
class Attachment:
    filename: str
    data: bytes
    kind: str              # "photo" | "document"
    caption: str | None = None
    mime: str | None = None   # documents only; None = the sender's default (text/csv)


@dataclass
class ToolResult:
    """``text`` is what the model sees (keep it informative: tables as text, not just "sent");
    ``attachments`` go to the chat verbatim and are only described to the model."""
    text: str
    attachments: list[Attachment] = field(default_factory=list)
    proposal: dict | None = None   # MONEY tools: the built action, for the journal / confirm step
    verbatim: bool = False         # send to the chat as-is (before the model's reply); the model
                                   # still sees `text` and adds only what is not on the card
    card: str | None = None        # with `verbatim`, the part of `text` to send; None = all of it
                                   # (lets a tool hand the model working notes it must not repost)


Handler = Callable[[Context, dict], ToolResult]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict     # JSON schema, object type; sent with strict=True
    risk: Risk
    handler: Handler
    desks: frozenset[str] | None = None   # None = available on every desk

    def __post_init__(self) -> None:
        if self.input_schema.get("type") != "object":
            raise ValueError(f"{self.name}: input_schema must be an object schema")
        if self.input_schema.get("additionalProperties") is not False:
            raise ValueError(f"{self.name}: input_schema needs additionalProperties=false (strict)")
        props = self.input_schema.get("properties", {})
        if set(self.input_schema.get("required", [])) != set(props):
            raise ValueError(f"{self.name}: strict schemas list every property in `required` "
                             "(use a nullable type for optional fields)")
