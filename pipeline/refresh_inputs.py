#!/usr/bin/env python3
"""
Best-effort daily input refresher.

Pulls the tier-1 event pages listed in data/sources.json from Liquipedia
(politely: rate-limited, cached, proper User-Agent) so the match parser can
extract today's fixtures and results into data/{cs2,lol}_inputs.json and
data/results.json.

Status: the fetch + cache layer is live and compliant. The HTML->fixtures
parser is the active daily-iteration work; until it lands, this module fetches
and caches the pages (proving the legal feed) and leaves the committed inputs
untouched, so the pipeline still publishes. It never raises into the cron.
"""
from __future__ import annotations

import json
from pathlib import Path

from .sources import liquipedia

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "sources.json"


def refresh() -> dict:
    cfg = json.loads(SOURCES.read_text(encoding="utf-8"))
    report = {}
    for game in ("cs2", "lol"):
        wiki = cfg[game]["wiki"]
        fetched = []
        for page in cfg[game].get("events", []):
            try:
                html = liquipedia.page_html(wiki, page, cache_hours=1.0)
                fetched.append({"page": page, "chars": len(html)})
            except Exception as e:
                fetched.append({"page": page, "error": str(e)})
        report[game] = fetched
    # TODO(parser): map cached HTML -> upcoming fixtures + finished results,
    # write data/{game}_inputs.json (tier-1 only) and merge into data/results.json.
    return report


if __name__ == "__main__":
    print(json.dumps(refresh(), indent=2))
