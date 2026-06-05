#!/usr/bin/env python3
"""
Auto-grade pending predictions from Liquipedia — no manual results entry.

For each game, fetch the watched tier-1 event pages (data/sources.json), parse
their completed matches, and match each still-ungraded prediction by team name.
Returns {match_id: {result:'a'|'b', score_a, score_b}} which grade_pending then
records (winner + map score). Manual data/results.json still works and takes
precedence, so a human can always override.
"""
from __future__ import annotations

import json
from pathlib import Path

from .sources import liquipedia
from .sources import results_parser as rp

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "sources.json"


def _load_sources() -> dict:
    if SOURCES.exists():
        return json.loads(SOURCES.read_text(encoding="utf-8"))
    return {}


def auto_results(log: dict, games=("cs2", "lol")) -> dict:
    """Parse watched event pages and auto-grade pending predictions by team name."""
    sources = _load_sources()
    pending = [p for p in log["predictions"] if p.get("result") is None]
    out: dict[str, dict] = {}
    for game in games:
        s = sources.get(game, {})
        wiki, events = s.get("wiki"), s.get("events", [])
        game_pending = [p for p in pending if p["game"] == game]
        if not wiki or not events or not game_pending:
            continue
        parsed: list[dict] = []
        for ev in events:
            try:
                html = liquipedia.page_html(wiki, ev.replace(" ", "_"), cache_hours=1.0)
                parsed += rp.parse_event(html)
            except Exception:
                continue  # network/parse hiccup: skip this page, try the rest
        if not parsed:
            continue
        for p in game_pending:
            r = rp.winner_for(p["team_a"], p["team_b"], parsed)
            if r:
                out[p["match_id"]] = r
    return out
