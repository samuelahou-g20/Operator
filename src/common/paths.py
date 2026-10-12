"""Single source of truth for every filesystem path used by the system.

Exposes the lake/runtime directory constants and ``data_path`` which builds the
canonical lake layout (ARCHITECTURE §2.1). All paths derive from ``REPO_ROOT``.

MUST NOT: read or write files, parse config, or be bypassed — no other module
hardcodes a path; everyone obtains paths from here.
"""

from __future__ import annotations

from pathlib import Path

# repo/ is two levels up from src/common/paths.py
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STATE_DIR = DATA_DIR / "state"
LOGS_DIR = REPO_ROOT / "logs"
RESULTS_DIR = REPO_ROOT / "results"
CONFIG_DIR = REPO_ROOT / "config"


def data_path(
    stage: str,
    asset_type: str,
    source: str,
    symbol: str,
    data_type: str,
    timeframe: str | None = None,
) -> Path:
    """Directory for a ``(stage, asset_type, source, symbol, data_type)`` dataset.

    Mirrors ``data/{stage}/{asset_type}/{source}/{symbol}/{data_type}``. When
    ``data_type == "bars"`` a ``timeframe`` is required and appended as the extra
    path level (``bars/{timeframe}``) per ARCHITECTURE §2.1. Callers append the
    ``contract=``/``date=`` hive partitions and the parquet file themselves.

    Reads the module-level ``*_DIR`` constants at call time so tests can
    monkeypatch the lake root.
    """
    if stage not in ("raw", "processed"):
        raise ValueError(f"stage must be 'raw' or 'processed', got {stage!r}")
    if data_type == "bars" and timeframe is None:
        raise ValueError("timeframe is required when data_type == 'bars'")

    root = RAW_DIR if stage == "raw" else PROCESSED_DIR
    path = root / asset_type / source / symbol / data_type
    if data_type == "bars":
        path = path / timeframe
    return path
