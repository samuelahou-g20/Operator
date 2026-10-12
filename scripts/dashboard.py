"""CLI: launch the live monitoring dashboard.

    uv run python -m scripts.dashboard
    uv run python -m scripts.dashboard --config dashboard.yaml

Live price + freshness come from the Redis trade stream. By default the dashboard
auto-starts a streamer for any symbol nobody is publishing; pass --no-stream to
rely only on externally-run streamers. Feature columns come from the persisted lake.
"""

from __future__ import annotations

import argparse

from src.monitor.app import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Live monitoring dashboard.")
    parser.add_argument(
        "--config", default="dashboard.yaml",
        help="dashboard config under config/ (default: dashboard.yaml)",
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="do not auto-start streamers; use only externally-run ones",
    )
    args = parser.parse_args()
    run(args.config, auto_stream=not args.no_stream)


if __name__ == "__main__":
    main()
