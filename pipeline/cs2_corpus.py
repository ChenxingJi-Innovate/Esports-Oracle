#!/usr/bin/env python3
"""
Build a CS2 historical match corpus from Liquipedia tier-1 event pages.

Why this exists: unlike LoL (Oracle's Elixir ships a ready feature table), CS2
has no public engineered-feature dataset we can use. Liquipedia gives us match
*results*; we derive features ourselves (see cs2_features.py). This module is
step 1: politely scrape a curated set of recent S-tier event pages and emit a
flat, dated, deduplicated match table.

Each event's structure differs (Swiss stages, group stages, playoffs live on
varied subpages), so rather than hardcode subpage names we fetch the base page
and auto-discover same-event subpage links, then parse matchlists from each.
The Liquipedia client throttles 30s between parse requests and caches on disk,
so a full build is slow on first run and instant on re-run.

Output: data/processed/cs2_matches.csv
    date, ts, team_a, team_b, score_a, score_b, winner, event

Run:  python -m pipeline.cs2_corpus            # full build (slow, cached)
      python -m pipeline.cs2_corpus --events   # list the event seed pages
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from .sources import liquipedia
from .sources.cs2_corpus_parser import parse_event_dated, parse_league_matches

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data" / "processed" / "cs2_matches.csv"
WIKI = "counterstrike"

# Curated recent S-tier CS2 events (real Liquipedia base-page titles, verified
# against Category:S-Tier_Tournaments). Future-dated or unplayed events simply
# yield no completed matches and cost nothing but an existence check. Extend
# this list as new majors/seasons complete; the corpus grows monotonically.
EVENT_SEEDS = [
    "Intel Extreme Masters/2026/Cologne",
    "Intel Extreme Masters/2026/Beijing",
    "Intel Extreme Masters/2026/Atlanta",
    "Intel Extreme Masters/2026/Rio",
    "Intel Extreme Masters/2026/Kraków",
    "PGL/2026/Astana",
    "PGL/2025/Bucharest",
    "PGL/2025/Cluj-Napoca",
    "ESL/Pro League/Season 25",
    "ESL/Pro League/Season 24",
    "StarLadder/StarSeries/2026/Fall",
    "StarLadder/StarSeries/2025/Budapest",
    "BLAST/Open/2026/Spring",
    "BLAST/Open/2025/Fall",
    "BLAST/Bounty/2026/Summer",
    "BLAST/Bounty/2026/Winter",
    "BLAST/Bounty/2025/Fall",
    "BLAST/Major/2025/Austin",
    "BLAST/Premier/2026/Frequent Flyers",
    "Skyesports/Championship/2025",
    "Esports World Cup/2025",
    "FISSURE/Playground/2",
]

# How many auto-discovered subpages to follow per event (caps request budget;
# each followed page is a 30s parse request).
MAX_SUBPAGES = 5

# Only follow subpages whose name signals they hold matchlists. This keeps the
# request budget tight: ancillary pages (Statistics, Prize Pool, Teams, Maps)
# are never parsed. Tier-1 CS2 matches live under these stage names.
_STAGE_WORDS = re.compile(
    r'\b(Stage|Group|Playoff|Playoffs|Main Event|Opening|Elimination|'
    r'Bracket|Knockout|Swiss|Final|Finals|Round)\b', re.I)

# Discover internal links that are subpages of the SAME event, e.g. from
# "Intel Extreme Masters/2026/Cologne" -> ".../Stage 1", ".../Playoffs".
_HREF = re.compile(r'href="/counterstrike/([^"#?]+)"')


def _subpages(base: str, html: str) -> list[str]:
    prefix = base.replace(" ", "_") + "/"
    found = []
    for raw in _HREF.findall(html):
        page = raw.replace("_", " ")
        if page.replace(" ", "_").startswith(prefix) and page != base:
            # one level deeper only (e.g. ".../Stage 1", not ".../Stage 1/Foo")
            tail = page[len(base) + 1:]
            if "/" not in tail and page not in found and _STAGE_WORDS.search(tail):
                found.append(page)
    return found[:MAX_SUBPAGES]


def _match_key(m: dict) -> tuple:
    return (m["ts"], frozenset(n.lower() for n in m["teams"]),
            m["scores"][0], m["scores"][1])


def build(seeds: list[str] | None = None, wiki: str = WIKI) -> list[dict]:
    # wiki lets the same scraper serve another tactical FPS on Liquipedia
    # (e.g. "valorant"); the match-card markup is identical (round-score maps).
    seeds = seeds or EVENT_SEEDS
    seen: set[tuple] = set()
    rows: list[dict] = []
    for base in seeds:
        try:
            exists = liquipedia.page_exists(wiki, base)
        except Exception as e:
            # A transient network error on the cheap existence check must not
            # abort the whole corpus; everything fetched so far is cached, so we
            # just skip this seed and a later re-run will pick it up.
            print(f"  [warn] exists-check {base}: {e}", file=sys.stderr)
            continue
        if not exists:
            print(f"  [skip] missing: {base}", file=sys.stderr)
            continue
        pages = [base]
        try:
            base_html = liquipedia.page_html(wiki, base, cache_hours=24.0)
            pages += _subpages(base, base_html)
        except Exception as e:
            print(f"  [warn] base fetch {base}: {e}", file=sys.stderr)
            base_html = ""
        for page in pages:
            try:
                html = base_html if page == base else liquipedia.page_html(
                    wiki, page, cache_hours=24.0)
                # brackets (playoffs) + match-card wikitables (group stage)
                matches = parse_event_dated(html) + parse_league_matches(html)
            except Exception as e:
                print(f"  [warn] {page}: {e}", file=sys.stderr)
                continue
            added = 0
            for m in matches:
                k = _match_key(m)
                if k in seen:
                    continue
                seen.add(k)
                rows.append({
                    "date": m["date"], "ts": m["ts"],
                    "team_a": m["teams"][0], "team_b": m["teams"][1],
                    "score_a": m["scores"][0], "score_b": m["scores"][1],
                    "winner": m["winner_name"], "event": base,
                })
                added += 1
            print(f"  {page}: +{added} matches (total {len(rows)})", file=sys.stderr)
    rows.sort(key=lambda r: r["ts"])
    return rows


def write_csv(rows: list[dict], out_csv: Path = OUT_CSV) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "date", "ts", "team_a", "team_b", "score_a", "score_b", "winner", "event"])
        w.writeheader()
        w.writerows(rows)
    return out_csv


if __name__ == "__main__":
    if "--events" in sys.argv:
        for e in EVENT_SEEDS:
            print(e)
        sys.exit(0)
    rows = build()
    path = write_csv(rows)
    print(f"\nWrote {len(rows)} matches -> {path}")
    if rows:
        print(f"date range: {rows[0]['date']} .. {rows[-1]['date']}")
