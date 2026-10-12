"""warm_once refreshes only what expired and never raises (offline)."""

from __future__ import annotations

from src.the_operator import warm
from src.the_operator.tools import advisory, intel


def test_warm_once_reports_per_input_and_survives_failures(monkeypatch):
    monkeypatch.setattr(warm, "load_config", lambda p: {})
    monkeypatch.setattr(intel, "macro_cached", lambda focus: (("m", b""), 0.0))
    monkeypatch.setattr(advisory, "_section_universe", lambda cfg: ("LS ranking (fresh; …)", {}))
    assert warm.warm_once() == {"macro": "refreshed", "universe": "refreshed"}
    monkeypatch.setattr(intel, "macro_cached", lambda focus: (("m", b""), 1800.0))
    monkeypatch.setattr(advisory, "_section_universe", lambda cfg: (_ for _ in ()).throw(RuntimeError("binance down")))
    out = warm.warm_once()
    assert out["macro"] == "still valid (30m old)" and out["universe"].startswith("failed: RuntimeError")
