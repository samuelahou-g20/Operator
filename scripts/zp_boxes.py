"""Regenerate ``config/zp/strategy_boxes.yaml`` — the quoting boxes each program runs.

The authoritative list is the AWS SSM parameter ``/configs/deployments/deployments.yaml``
(SecureString, then base64 + zstd). It changes on the order of weeks, so the ZP reads a generated
registry from config rather than taking a boto3 + zstandard dependency on the request path — the
same pattern as ``config/universe/{source}.yaml`` and ``discover_universe --write``.

Run (needs the ``aws`` CLI with credentials for account 550630135186, and ``zstd``):

    uv run python -m scripts.zp_boxes --write

MUST NOT: place orders, touch the data lake, or write anything but that one config file.
"""

from __future__ import annotations

import argparse
import base64
import shutil
import subprocess
import sys

import yaml

from src.common.logging import get_logger
from src.common.paths import CONFIG_DIR

_log = get_logger("zp_boxes")

PARAM = "/configs/deployments/deployments.yaml"
REGION = "ap-northeast-1"
OUT = CONFIG_DIR / "zp" / "strategy_boxes.yaml"
# Programs that quote; the rest of the parameter is infrastructure (refsvc, trdsvc, hedging, ...).
QUOTING_PROGRAMS = ("mm1", "mm2", "mm3", "mm4", "mm5", "perp")

_HEADER = """# Quoting boxes per program: the LM2Strategy deployments whose /rest/calculators/params the ZP
# reads for each venue's dynamic-mode settings (levels, base min/max spread, vol scaling factors,
# reference vol). This replaces gamma-dash, which is behind Entra SSO from a server.
#
# GENERATED — do not hand-edit. Refresh from the authoritative AWS SSM parameter with:
#     uv run python -m scripts.zp_boxes --write
"""


def fetch_deployments() -> dict:
    """The SSM parameter, decrypted and decompressed, as a dict. Raises with a clear message."""
    for tool in ("aws", "zstd"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} is not on PATH; see this module's docstring")
    raw = subprocess.run(
        ["aws", "ssm", "get-parameter", "--name", PARAM, "--with-decryption",
         "--region", REGION, "--query", "Parameter.Value", "--output", "text"],
        capture_output=True, text=True, timeout=120, check=True).stdout.strip()
    if raw.startswith("ZSTD:"):
        blob = subprocess.run(["zstd", "-dc"], input=base64.b64decode(raw[len("ZSTD:"):]),
                              capture_output=True, timeout=60, check=True).stdout
    else:
        blob = raw.encode()
    return yaml.safe_load(blob)


def build(deployments: dict) -> dict:
    """``{program: {box: "http://ip:port"}}`` for the quoting programs only."""
    out: dict[str, dict[str, str]] = {}
    for program in QUOTING_PROGRAMS:
        boxes = deployments.get(program) or {}
        rows = {name: f"http://{b['ip_address']}:{b['port']}"
                for name, b in sorted(boxes.items()) if b and b.get("ip_address") and b.get("port")}
        if rows:
            out[program] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help=f"write {OUT}, else print to stdout")
    args = ap.parse_args()
    registry = build(fetch_deployments())
    text = _HEADER + yaml.safe_dump(registry, sort_keys=True, default_flow_style=False)
    if not args.write:
        sys.stdout.write(text)
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    _log.info("wrote %s: %s", OUT, {p: len(b) for p, b in registry.items()})


if __name__ == "__main__":
    main()
