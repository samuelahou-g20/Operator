"""CLI: summarize recent data-quality warnings from logs/quality/.

    uv run python -m scripts.quality_report --tail 50
"""

from __future__ import annotations

import argparse

from src.common import paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize recent quality warnings.")
    parser.add_argument("--tail", type=int, default=50, help="max lines to show")
    args = parser.parse_args()

    qdir = paths.LOGS_DIR / "quality"
    if not qdir.exists():
        print("no quality logs yet")
        return

    lines: list[str] = []
    for log_file in sorted(qdir.glob("*.log")):
        with log_file.open("r", encoding="utf-8", errors="replace") as fh:
            lines.extend(ln.rstrip("\n") for ln in fh if "WARNING" in ln)

    if not lines:
        print("no quality warnings recorded")
        return
    for line in lines[-args.tail:]:
        print(line)
    print(f"\n{len(lines)} total warning(s); showing last {min(args.tail, len(lines))}")


if __name__ == "__main__":
    main()
