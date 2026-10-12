"""Daily client sentiment brief: live X sentiment -> FROZEN G-20 .docx -> one-page PDF.

Pulls the X sentiment (same source as sentiment_bot), scores it into structured per-theme data
with Claude, fills the FROZEN colleagues' template (templates/g20_sentiment_brief.docx — the
approved format, never restyled in code), and converts to PDF via LibreOffice. Because LibreOffice
lays the doc out slightly larger than Word, the auto-fit loop tightens the CONTENT (not the format)
and re-measures the real converted PDF until LibreOffice itself renders exactly one page. Read-only.

Run: ``uv run python -m scripts.daily_sentiment_report [--outdir .]``
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from datetime import datetime, timezone

import anthropic

from scripts.sentiment_bot import fetch_following, fetch_themes, structured_digest
from src.common.config import require_env
from src.common.logging import get_logger
from src.reporting import g20_fill

_log = get_logger("daily_sentiment_report")

# Char budgets per field (starting point); the auto-fit loop tightens them until it fits 1 page.
_BUDGETS = {"abstract": 320, "line": 84, "contrib": 42, "net": 190, "trend": 145,
            "voices": 200, "takeaway": 195}


def _clip(s: str, n: int) -> str:
    """Trim to <= n chars at a word boundary."""
    s = str(s).strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0].rstrip(",.;·") + "…"


def build_report(data: dict, budgets: dict = _BUDGETS) -> dict:
    """Map the structured digest into the template's `report` dict (KPI strip = Net + 4 notable)."""
    net = float(data["net_score"])
    notable = sorted(data["themes"], key=lambda t: abs(float(t["score"])), reverse=True)[:4]
    kpis = [(f"{net:+.2f}", "Net mood", g20_fill.NAVY)] + [
        (f"{float(t['score']):+.2f}", t["name"], g20_fill._score_hex(float(t["score"])))
        for t in notable]
    themes = [{"name": t["name"], "score": float(t["score"]),
               "line": _clip(t["line"], budgets["line"]),
               "contributors": _clip(t.get("contributors", ""), budgets["contrib"])}
              for t in data["themes"]]
    return {
        "abstract": _clip(data["abstract"], budgets["abstract"]),
        "net_line": _clip(data["net_line"], budgets["net"]), "themes": themes, "kpis": kpis,
        "trending": _clip(data["trending"], budgets["trend"]),
        "voices": _clip(data["voices"], budgets["voices"]),
        "takeaway": _clip(data["takeaway"], budgets["takeaway"]),
    }


def _page_count(pdf: str) -> int:
    out = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", out)
    return int(m.group(1)) if m else 99


def render_one_page(data: dict, outdir: str) -> str:
    """Fill the FROZEN template + convert via LibreOffice, tightening CONTENT (never the format)
    until LibreOffice lays it out on exactly one page. Page count is measured on the real file."""
    budgets = dict(_BUDGETS)
    docx = pdf = ""
    for attempt in range(8):
        docx = g20_fill.fill(build_report(data, budgets), f"{outdir}/_g20_try.docx")
        pdf = g20_fill.to_pdf(docx, outdir)              # reliable convert (no stale-file race)
        n = _page_count(pdf)
        _log.info("attempt %d: LibreOffice renders %d page(s)", attempt + 1, n)
        if n <= 1:
            break
        budgets = {k: max(20, int(v * 0.85)) for k, v in budgets.items()}
    final_pdf = f"{outdir}/g20_sentiment_daily.pdf"
    final_docx = f"{outdir}/g20_sentiment_daily.docx"
    shutil.move(pdf, final_pdf)
    shutil.move(docx, final_docx)
    return final_pdf


def main() -> None:
    ap = argparse.ArgumentParser(description="daily X sentiment client brief (PDF)")
    ap.add_argument("--outdir", default=".", help="output directory (default: repo root)")
    args = ap.parse_args()

    client = anthropic.Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))
    stamp = datetime.now(timezone.utc).strftime("%d %B %Y")
    _log.info("building daily sentiment brief for %s", stamp)

    themed = fetch_themes()
    followed = fetch_following()
    total = sum(len(t) for _, t in themed) + len(followed)
    _log.info("pulled %d tweets (est $%.2f); scoring…", total, total * 0.005)

    data = structured_digest(client, themed, followed, stamp)
    if len(data.get("themes", [])) != len(themed):
        raise ValueError(f"expected {len(themed)} themes, got {len(data.get('themes', []))}")

    pdf = render_one_page(data, args.outdir)
    _log.info("wrote %s (%d page; net %.2f)", pdf, _page_count(pdf), float(data["net_score"]))
    print(pdf)


if __name__ == "__main__":
    main()
