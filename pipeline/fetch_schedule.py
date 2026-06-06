#!/usr/bin/env python3
"""
Auto-populate the daily slate from Liquipedia, so the cron predicts the REAL
upcoming fixtures instead of a hand-typed inputs file.

For a small watchlist of currently-running tier-1 events (CS2 + LoL), we pull
every scheduled match whose start time falls in [today, today+window], skip the
ones already finished, and write them into data/cs2_inputs.json /
data/lol_inputs.json in the exact shape the pipelines expect.

Feature derivation (the part that makes the picks real, not 50/50):
  - CS2: rank/form/h2h come from the case-base state we already build from
    Liquipedia history (cs2_features). Elo ordering -> pseudo rank; recent
    win rate -> form; prior meetings -> h2h. Unknown teams fall back to neutral.
  - LoL: Oracle's Elixir feeds the kNN, so LoL fixtures are flagged
    `predict: case_based` and lol_pipeline scores them with the 69% OE kNN
    rather than the manual linear inputs.

Network is best-effort: any fetch that fails leaves the existing inputs intact,
so a flaky run never wipes a good slate. On GitHub Actions (clean network) this
runs cleanly every morning as daily.py's first step.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .sources import liquipedia

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CS2_INPUTS = DATA / "cs2_inputs.json"
LOL_INPUTS = DATA / "lol_inputs.json"

# Currently-running tier-1 events to watch. Each entry: (wiki, page, default_fmt).
# Keep this list current (it is the only manual upkeep); matches and dates are
# pulled live. Swiss/opening stages are BO1, later stages BO3, finals BO5.
WATCH_CS2 = [
    ("counterstrike", "Intel Extreme Masters/2026/Cologne/Stage 2", "BO3"),
    ("counterstrike", "Intel Extreme Masters/2026/Cologne/Stage 3", "BO3"),
    ("counterstrike", "Intel Extreme Masters/2026/Cologne/Playoffs", "BO3"),
]
WATCH_LOL = [
    ("leagueoflegends", "LPL/2026/Split 2/Playoffs", "BO5"),
    ("leagueoflegends", "LCK/2026/Split 2/Playoffs", "BO5"),
]

DEFAULT_WINDOW_DAYS = 2  # today + next 2

_MATCH_BLOCK = re.compile(
    r'(brkts-popup-header-left.*?|brkts-matchlist-match\b.*?)'
    r'(?=brkts-popup-header-left|brkts-matchlist-match\b|$)', re.S)
_TS = re.compile(r'data-timestamp="(\d+)"')
_ARIA = re.compile(r'aria-label="([^"]+)"')


def _distinct_teams(block: str) -> list[str]:
    names, seen = [], []
    for o in _ARIA.findall(block):
        if not seen or seen[-1] != o:
            seen.append(o)
    # collapse to the first two distinct names
    for n in seen:
        if n not in names:
            names.append(n)
        if len(names) == 2:
            break
    return names


def _fetch_matches(wiki: str, page: str, fmt: str,
                   today: date, window: int) -> list[dict]:
    """Upcoming (not finished) matches on `page` within the date window."""
    try:
        html = liquipedia.page_html(wiki, page, cache_hours=1.0)
    except Exception as e:
        print(f"  [warn] fetch {page}: {e}")
        return []
    lo = today
    hi = today + timedelta(days=window)
    out = []
    for block in _MATCH_BLOCK.findall(html):
        ts_m = _TS.search(block)
        if not ts_m:
            continue
        teams = _distinct_teams(block)
        if len(teams) < 2 or "TBD" in teams:
            continue
        if 'data-finished="finished"' in block:
            continue  # already played
        t = datetime.fromtimestamp(int(ts_m.group(1)), tz=timezone.utc)
        if not (lo <= t.date() <= hi):
            continue
        out.append({
            "scheduled_at": t.isoformat(),
            "date": t.date().isoformat(),
            "team_a": teams[0], "team_b": teams[1],
            "event": page.split("/")[0] if "/" in page else page,
            "page": page, "fmt": fmt,
        })
    return out


# --------------------------------------------------------------------------- #
# CS2: derive linear-model features from the case-base state
# --------------------------------------------------------------------------- #
def _cs2_team_features(state: dict) -> dict:
    """Map corpus state -> per-team {rank, form, player} via Elo ordering."""
    elo = state.get("elo", {})
    order = sorted(elo, key=lambda t: -elo[t])           # strongest first
    rank = {name: i + 1 for i, name in enumerate(order)}
    form = state.get("form", {})
    feats = {}
    for name, rating in elo.items():
        d = form.get(name)
        winrate = (sum(d) / len(d)) if d else 0.5
        feats[name] = {
            "rank": rank.get(name, len(order) + 20),     # unknown -> weak rank
            "form": round(winrate, 3),
            "player": round((rating - 1500.0) / 400.0, 3),
        }
    return feats


def _norm(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def build_cs2_inputs(today: date, window: int = DEFAULT_WINDOW_DAYS) -> dict | None:
    from . import cs2_features
    _, state = cs2_features.build(return_state=True)
    if not state:
        return None
    tf = _cs2_team_features(state)
    by_norm = {_norm(k): v for k, v in tf.items()}
    h2h = state.get("h2h", {})

    matches = []
    for wiki, page, fmt in WATCH_CS2:
        for m in _fetch_matches(wiki, page, fmt, today, window):
            na, nb = _norm(m["team_a"]), _norm(m["team_b"])
            fa = by_norm.get(na, {"rank": 60, "form": 0.5, "player": 0.0})
            fb = by_norm.get(nb, {"rank": 60, "form": 0.5, "player": 0.0})
            ha = h2h.get((na, nb)); hb = h2h.get((nb, na))
            h2a = (ha[0] / ha[1]) if ha and ha[1] else 0.5
            h2b = (hb[0] / hb[1]) if hb and hb[1] else 0.5
            matches.append({
                "match_id": f"{na}-{nb}-{m['date']}",
                "event": m["event"], "tier": 1, "fmt": m["fmt"],
                "scheduled_at": m["scheduled_at"],
                "team_a": {"name": m["team_a"], "rank": fa["rank"], "form": fa["form"],
                           "map_edge": 0.5, "player": fa["player"], "h2h": round(h2a, 3)},
                "team_b": {"name": m["team_b"], "rank": fb["rank"], "form": fb["form"],
                           "map_edge": 0.5, "player": fb["player"], "h2h": round(h2b, 3)},
            })
    return {
        "_comment": ("AUTO-GENERATED by fetch_schedule.py from Liquipedia. "
                     "rank/form/h2h derived from the CS2 case-base (Elo ordering, "
                     "recent win rate, prior meetings). Edit WATCH_CS2 to change "
                     "which events are tracked; matches/dates are pulled live."),
        "date": today.isoformat(), "matches": matches,
    }


def build_lol_inputs(today: date, window: int = DEFAULT_WINDOW_DAYS) -> dict | None:
    matches = []
    for wiki, page, fmt in WATCH_LOL:
        for m in _fetch_matches(wiki, page, fmt, today, window):
            na, nb = _norm(m["team_a"]), _norm(m["team_b"])
            matches.append({
                "match_id": f"{na}-{nb}-{m['date']}",
                "event": m["event"], "league": "LPL" if "LPL" in m["page"] else (
                    "LCK" if "LCK" in m["page"] else m["event"]),
                "fmt": m["fmt"], "scheduled_at": m["scheduled_at"],
                "predict": "case_based",       # score via OE kNN, not manual inputs
                "team_a": {"name": m["team_a"]},
                "team_b": {"name": m["team_b"]},
            })
    return {
        "_comment": ("AUTO-GENERATED by fetch_schedule.py from Liquipedia. LoL "
                     "fixtures are scored by the Oracle's Elixir kNN (predict: "
                     "case_based); edit WATCH_LOL to track different splits."),
        "date": today.isoformat(),
        "allowed_leagues": ["LPL", "LCK", "LEC", "MSI", "WLDs", "Worlds"],
        "matches": matches,
    }


def refresh(today: date | None = None, window: int = DEFAULT_WINDOW_DAYS) -> dict:
    """Best-effort: write whatever we can fetch; never wipe a good slate on
    failure or an empty pull (the existing inputs stay)."""
    today = today or date.today()
    summary = {}
    for label, builder, path in (
        ("cs2", build_cs2_inputs, CS2_INPUTS),
        ("lol", build_lol_inputs, LOL_INPUTS),
    ):
        try:
            payload = builder(today, window)
        except Exception as e:
            print(f"  [warn] {label} schedule build failed: {e}")
            payload = None
        if payload and payload.get("matches"):
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                            encoding="utf-8")
            summary[label] = len(payload["matches"])
        else:
            summary[label] = f"kept existing ({'no fixtures' if payload else 'fetch failed'})"
    return summary


if __name__ == "__main__":
    import sys
    d = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    print(json.dumps(refresh(d), indent=2, ensure_ascii=False))
