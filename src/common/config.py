"""Load YAML configs under ``config/`` and resolve credential env vars.

``load_config`` parses YAML (the single place that happens). ``require_env`` reads a
credential from the environment, transparently loading the repo-root ``.env`` first
(env vars already set always win — ``.env`` never overrides real environment).

MUST NOT: hold credentials in YAML (config references the var NAME only), or be
bypassed by modules calling ``yaml.safe_load``/``os.environ`` for secrets directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from src.common import paths

_dotenv_loaded = False


def load_config(relative_path: str | Path) -> dict[str, Any]:
    """Load ``config/{relative_path}`` as a dict (must parse to a mapping)."""
    path = paths.CONFIG_DIR / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"config {path} did not parse to a mapping (got {type(data).__name__})"
        )
    return data


def load_dotenv(path: str | Path | None = None) -> None:
    """Populate ``os.environ`` from the repo-root ``.env`` (idempotent; never overrides)."""
    global _dotenv_loaded
    if _dotenv_loaded and path is None:
        return
    env_path = Path(path) if path is not None else paths.REPO_ROOT / ".env"
    if env_path.is_file():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    if path is None:
        _dotenv_loaded = True


def require_env(name: str) -> str:
    """Return env var ``name`` (loading ``.env`` first); raise a clear error if absent."""
    load_dotenv()
    try:
        return os.environ[name]
    except KeyError:
        raise KeyError(
            f"missing required env var {name!r}; set it in the environment or in .env"
        ) from None
