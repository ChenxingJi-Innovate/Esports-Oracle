#!/usr/bin/env python3
"""
Build a static "upcoming fixtures" schedule for the front-end.

The daily inputs (data/cs2_inputs.json, data/lol_inputs.json) only carry a
single root-level `date` (the slate is curated one day at a time), so we group
the loaded matches under that date and emit a small JSON the schedule page can
render with no backend.

If/when match-level dates are added to the inputs, extend `_match_date` to read
them and this module will naturally fan the matches across multiple days.

Output shape (app/data/schedule.json):
{
  "generated_for": "2026-06-05",
  "window_days": 4,                         # today + next 3
  "next_dates": [
    {
      "date": "2026-06-05",
      "cs2_matches": [ {match_id, event, team_a, team_b, p_a, fmt, confidence}, ... ],
      "lol_matches": [ ... ]
    },
    ...
  ]
}

We never invent matches: a date only appears if it carries at least one real
fixture from the inputs (joined with the published predictions for win %).
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
APP_DATA = ROOT / "app" / "data"
CS2_INPUTS = DATA / "cs2_inputs.json"
LOL_INPUTS = DATA / "lol_inputs.json"
PREDICTIONS = APP_DATA / "predictions.json"


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _input_date(inputs: dict, fallback: date) -> date:
    """Root-level slate date, tolerant of missing/invalid values."""
    raw = inputs.get("date")
    if isinstance(raw, str):
        try:
            return date.fromisoformat(raw)
        except ValueError:
            pass
    return fallback


def _pred_index(predictions: dict) -> dict[str, dict]:
    """match_id -> published prediction (carries p_a, confidence, names)."""
    slate = (predictions or {}).get("slate", {})
    index: dict[str, dict] = {}
    for game in ("cs2", "lol"):
        for m in slate.get(game, []) or []:
            mid = m.get("match_id")
            if mid:
                index[mid] = m
    return index


def _normalise_match(raw: dict, pred: dict | None) -> dict:
    """Join a raw input match with its prediction into a render-ready row."""
    team_a = raw.get("team_a", {})
    team_b = raw.get("team_b", {})
    name_a = team_a.get("name") if isinstance(team_a, dict) else str(team_a)
    name_b = team_b.get("name") if isinstance(team_b, dict) else str(team_b)
    pred = pred or {}
    return {
        "match_id": raw.get("match_id"),
        "event": raw.get("event") or pred.get("event") or "",
        "league": raw.get("league"),
        "fmt": raw.get("fmt") or pred.get("fmt") or "",
        "team_a": pred.get("team_a") or name_a or "TBD",
        "team_b": pred.get("team_b") or name_b or "TBD",
        "p_a": pred.get("p_a"),
        "confidence": pred.get("confidence"),
    }


def schedule_3days(today_date: date) -> dict:
    """Group loaded CS2 + LoL fixtures by date, covering today + next 3 days.

    Returns a dict ready to publish as app/data/schedule.json. Only dates that
    actually carry fixtures are emitted, and only within the 4-day window
    [today, today+3].
    """
    cs2_inputs = _load_json(CS2_INPUTS)
    lol_inputs = _load_json(LOL_INPUTS)
    predictions = _load_json(PREDICTIONS)
    preds = _pred_index(predictions)

    window_start = today_date
    window_end = today_date + timedelta(days=3)

    # date -> {"cs2_matches": [...], "lol_matches": [...]}
    by_date: dict[str, dict] = {}

    def collect(inputs: dict, game_key: str) -> None:
        slate_date = _input_date(inputs, today_date)
        if slate_date < window_start or slate_date > window_end:
            return
        for raw in inputs.get("matches", []) or []:
            row = _normalise_match(raw, preds.get(raw.get("match_id")))
            day = slate_date.isoformat()
            bucket = by_date.setdefault(day, {"cs2_matches": [], "lol_matches": []})
            bucket[game_key].append(row)

    collect(cs2_inputs, "cs2_matches")
    collect(lol_inputs, "lol_matches")

    next_dates = [
        {"date": day, **by_date[day]}
        for day in sorted(by_date.keys())
    ]

    return {
        "generated_for": today_date.isoformat(),
        "window_days": 4,
        "next_dates": next_dates,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat(),
                    help="run date (YYYY-MM-DD); defaults to today")
    args = ap.parse_args()
    out = schedule_3days(date.fromisoformat(args.date))
    APP_DATA.mkdir(parents=True, exist_ok=True)
    (APP_DATA / "schedule.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"dates": [d["date"] for d in out["next_dates"]]}, indent=2))


if __name__ == "__main__":
    main()
