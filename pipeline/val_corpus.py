#!/usr/bin/env python3
"""
Valorant historical match corpus from Liquipedia VCT tier-1 event pages.

Valorant is a tactical FPS with the same Liquipedia match-card markup as CS2
(round-score maps, BoX series), so we reuse the CS2 scraper and dated parser
wholesale and only swap the wiki + event list. Output mirrors cs2_matches.csv,
so the shared feature/case-base engine (cs2_features / cs2_case_based) consumes
it unchanged via their matches_csv parameter.

Output: data/processed/val_matches.csv
Run:    python -m pipeline.val_corpus        # full build (slow, cached)
"""
from __future__ import annotations

from pathlib import Path

from . import cs2_corpus

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data" / "processed" / "val_matches.csv"
WIKI = "valorant"

# Curated recent VCT tier-1 events (real Liquipedia base-page titles follow the
# "VCT/<year>/..." scheme). Future/unplayed events yield no completed matches and
# cost only an existence check; extend as Masters/Champions/league splits finish.
EVENT_SEEDS = [
    "VCT/2026/Masters/Bangkok",
    "VCT/2026/Americas League/Stage 1",
    "VCT/2026/EMEA League/Stage 1",
    "VCT/2026/Pacific League/Stage 1",
    "VCT/2026/China League/Stage 1",
    "VCT/2025/Champions",
    "VCT/2025/Masters/Toronto",
    "VCT/2025/Masters/Bangkok",
    "VCT/2025/Americas League/Stage 2",
    "VCT/2025/EMEA League/Stage 2",
    "VCT/2025/Pacific League/Stage 2",
    "VCT/2025/China League/Stage 2",
    "VCT/2025/Americas League/Stage 1",
    "VCT/2025/EMEA League/Stage 1",
    "VCT/2025/Pacific League/Stage 1",
    "VCT/2025/China League/Stage 1",
]


def build() -> list[dict]:
    return cs2_corpus.build(seeds=EVENT_SEEDS, wiki=WIKI)


def write_csv(rows: list[dict]) -> Path:
    return cs2_corpus.write_csv(rows, out_csv=OUT_CSV)


if __name__ == "__main__":
    rows = build()
    path = write_csv(rows)
    print(f"\nWrote {len(rows)} Valorant matches -> {path}")
    if rows:
        print(f"date range: {rows[0]['date']} .. {rows[-1]['date']}")
