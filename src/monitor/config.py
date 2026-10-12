"""Parse ``config/dashboard.yaml`` into a flat list of monitored instruments.

Resolves group-level ``source``/``asset_type``/``timeframe`` defaults against
per-symbol overrides, preserving group order and membership for the display.

MUST NOT: read the lake or Redis, or parse YAML directly (uses common.config).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import config as cfg


@dataclass(frozen=True)
class Instrument:
    group: str
    source: str
    symbol: str
    asset_type: str
    timeframe: str


@dataclass(frozen=True)
class DashboardConfig:
    instruments: list[Instrument]
    stale_after: float          # seconds without a live tick before a row is stale
    abnormal_window_min: float  # window (minutes) whose directional move is scored
    abnormal_lookback_min: float  # lookback for the SHORT-scale sigma (empirical window-moves)
    abnormal_z: float           # |z| threshold that flags a move as abnormal (and alerts)
    alert_cooldown_min: float   # min minutes between Telegram alerts per symbol


def load_dashboard(relative_path: str = "dashboard.yaml") -> DashboardConfig:
    """Load and resolve the dashboard universe into ordered ``Instrument`` rows."""
    return resolve_dashboard(cfg.load_config(relative_path))


def resolve_dashboard(conf: dict) -> DashboardConfig:
    """Resolve a parsed dashboard config into ordered ``Instrument`` rows."""
    defaults = conf.get("defaults", {})
    default_tf = defaults.get("timeframe", "1h")
    stale_after = float(defaults.get("stale_after", 120))

    instruments: list[Instrument] = []
    for group in conf.get("groups", []):
        name = group["name"]
        g_source = group.get("source")
        g_asset = group.get("asset_type")
        g_tf = group.get("timeframe", default_tf)
        for entry in group.get("symbols", []):
            if isinstance(entry, str):
                entry = {"symbol": entry}
            source = entry.get("source", g_source)
            asset_type = entry.get("asset_type", g_asset)
            if source is None or asset_type is None:
                raise ValueError(
                    f"group {name!r}: symbol {entry.get('symbol')!r} is missing "
                    "source/asset_type (set them on the group or the symbol)"
                )
            instruments.append(
                Instrument(
                    group=name,
                    source=source,
                    symbol=entry["symbol"],
                    asset_type=asset_type,
                    timeframe=entry.get("timeframe", g_tf),
                )
            )
    if not instruments:
        raise ValueError("dashboard config defines no symbols")
    return DashboardConfig(
        instruments=instruments,
        stale_after=stale_after,
        abnormal_window_min=float(defaults.get("abnormal_window_min", 15)),
        abnormal_lookback_min=float(defaults.get("abnormal_lookback_min", 120)),
        abnormal_z=float(defaults.get("abnormal_z", 4.0)),
        alert_cooldown_min=float(defaults.get("alert_cooldown_min", 10)),
    )
