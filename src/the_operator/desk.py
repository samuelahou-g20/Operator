"""Chat -> desk resolution and the per-desk facts the agent is told.

A chat is a desk ONLY if pinned in that desk's own env var (``TELEGRAM_MM3_CHAT_ID`` etc., the
same vars every bot uses); the pins must be disjoint and nothing defaults. Separately,
``TELEGRAM_OPERATOR_CHAT_ID`` lists the chats the Operator serves at all (the pilot roster), so
pinning a chat to a desk does not by itself enable the Operator there.

MUST NOT: default a chat to a desk, read the lake, or hold safety values (caps stay in
``scripts/trade_bot.py``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import combinations

from src.common.config import load_config, load_dotenv
from src.common.paths import CONFIG_DIR


@dataclass(frozen=True)
class Desk:
    name: str
    label: str
    chat_env: str
    cr_token_env: str | None     # CoinRoutes key for that desk's books; None = no CR access


DESKS: dict[str, Desk] = {
    "mm3": Desk("mm3", "MM3", "TELEGRAM_MM3_CHAT_ID", "CR_API_TOKEN_MM3"),
    "mm2": Desk("mm2", "MM2", "TELEGRAM_MM2_CHAT_ID", "CR_API_TOKEN_MM2"),
    "treasury": Desk("treasury", "Treasury", "TELEGRAM_TREASURY_CHAT_ID", "CR_API_TOKEN_G20JPY"),
    "options": Desk("options", "Options", "TELEGRAM_OPTIONS_CHAT_ID", None),
    "mm5": Desk("mm5", "MM5", "TELEGRAM_MM5_CHAT_ID", None),
}


def _csv_env(var: str) -> set[str]:
    return {c.strip() for c in (os.environ.get(var) or "").split(",") if c.strip()}


def desk_by_chat() -> dict[str, str]:
    """``{chat_id: desk_name}`` from the per-desk pins. Raises if a chat is pinned twice."""
    load_dotenv()
    pins = {name: _csv_env(d.chat_env) for name, d in DESKS.items()}
    for (na, a), (nb, b) in combinations(pins.items(), 2):
        if a & b:
            raise RuntimeError(f"chat(s) {sorted(a & b)} pinned to both {na} and {nb} — fix .env")
    return {chat: name for name, chats in pins.items() for chat in chats}


def served_chats() -> set[str]:
    """Chats the Operator answers in (``TELEGRAM_OPERATOR_CHAT_ID``). Empty = serves nowhere."""
    load_dotenv()
    return _csv_env("TELEGRAM_OPERATOR_CHAT_ID")


def operator_users() -> set[str]:
    """Telegram user ids allowed to use the Operator at all (``TELEGRAM_OPERATOR_USERS``).

    EMPTY MEANS EVERYONE in a served chat, which is the pilot behaviour. Set it before putting the
    Operator in a desk chat: without it every member of that group can run every read tool."""
    load_dotenv()
    return _csv_env("TELEGRAM_OPERATOR_USERS")


def admins() -> set[str]:
    """Telegram user ids allowed to halt/resume (``OPERATOR_ADMINS``)."""
    load_dotenv()
    return _csv_env("OPERATOR_ADMINS")


def traders() -> dict[str, set[str]]:
    """``{desk: {telegram user ids}}`` read from trade_bot's own per-desk allowlist vars — one
    source, so a trader the bot refuses is a trader the Operator refuses."""
    from scripts.trade_bot import DESKS as TRADE_DESKS   # deferred: heavy import, money module
    load_dotenv()
    return {name: _csv_env(cfg["traders_env"]) for name, cfg in TRADE_DESKS.items()}


def cr_token(desk: str) -> str:
    """That desk's CoinRoutes API token; raises if the desk has no CoinRoutes access."""
    env = DESKS[desk].cr_token_env
    if env is None:
        raise KeyError(f"{DESKS[desk].label} has no CoinRoutes key configured")
    load_dotenv()
    return os.environ[env]


def desk_facts(desk: str) -> str:
    """Plain-text facts about the desk for the system prompt: token roster, hedge strategy, bots,
    plus the human-curated playbook ``config/operator/{desk}.md`` if present."""
    d = DESKS[desk]
    cfg = load_config("desks.yaml").get(desk, {})
    lines = [f"Desk: {d.label} ({desk})."]
    tokens = sorted(cfg.get("strategy", {}))
    if tokens:
        lines.append(f"Spot gamma tokens ({len(tokens)}): {', '.join(tokens)}.")
    if cfg.get("perp_strategy"):
        lines.append(f"Futures/perps hedge strategy: {cfg['perp_strategy']}.")
    if cfg.get("bots"):
        lines.append("Pluto trading bots: " + ", ".join(b.lstrip("*") for b in cfg["bots"]) + ".")
    playbook = CONFIG_DIR / "operator" / f"{desk}.md"
    if playbook.is_file():
        lines.append("")
        lines.append(playbook.read_text(encoding="utf-8").strip())
    return "\n".join(lines)
