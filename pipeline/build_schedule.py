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
  "window_days": 5,                         # yesterday + today + next 3
  "next_dates": [
    {
      "date": "2026-06-05",
      "cs2_matches": [ {match_id, event, team_a, team_b, p_a, fmt, confidence,
                        time?, scheduled_at?, league?}, ... ],
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
import logging
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

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


def _normalise_match(raw: dict, pred: dict | None, match_id: str | None = None) -> dict:
    """Join a raw input match with its prediction into a render-ready row.

    `match_id` is passed in so we can warn when a curated fixture has no
    matching prediction (a real error: dropped row, failed prediction, etc.)
    rather than silently emitting a null model line.
    """
    if match_id and pred is None:
        logger.warning("Match %s has no prediction (missing from predictions.json)", match_id)
    team_a = raw.get("team_a", {})
    team_b = raw.get("team_b", {})
    name_a = team_a.get("name") if isinstance(team_a, dict) else str(team_a)
    name_b = team_b.get("name") if isinstance(team_b, dict) else str(team_b)
    pred = pred or {}
    row = {
        "match_id": raw.get("match_id"),
        "event": raw.get("event") or pred.get("event") or "",
        "fmt": raw.get("fmt") or pred.get("fmt") or "",
        "team_a": pred.get("team_a") or name_a or "TBD",
        "team_b": pred.get("team_b") or name_b or "TBD",
        "p_a": pred.get("p_a"),
        "confidence": pred.get("confidence"),
    }
    # Optional, backward-compatible scheduling fields. Only emit when present
    # so existing time-less fixtures stay clean and the frontend can omit them.
    league = raw.get("league")
    if league:
        row["league"] = league
    time_str = raw.get("time")
    if time_str:
        row["time"] = time_str
    scheduled_at = raw.get("scheduled_at")
    if scheduled_at:
        row["scheduled_at"] = scheduled_at
    return row


def schedule_3days(today_date: date) -> dict:
    """Group loaded CS2 + LoL fixtures by date, covering yesterday + next 3 days.

    Returns a dict ready to publish as app/data/schedule.json. Only dates that
    actually carry fixtures are emitted, and only within the 5-day window
    [today-1, today+3].

    The window starts at yesterday because the inputs are curated once per day
    while this schedule is rebuilt daily: when the cron runs on day N but the
    slate is still dated N-1, a forward-only window would drop every fixture
    and blank the page. If even the tolerant window yields nothing (e.g. a very
    stale slate), we fall back to the most recent available slate date so the
    page is never needlessly blank. We never invent matches: a date only
    appears if it carries at least one real curated fixture.
    """
    cs2_inputs = _load_json(CS2_INPUTS)
    lol_inputs = _load_json(LOL_INPUTS)
    predictions = _load_json(PREDICTIONS)
    preds = _pred_index(predictions)

    window_start = today_date - timedelta(days=1)
    window_end = today_date + timedelta(days=3)

    games = (
        (cs2_inputs, "cs2_matches"),
        (lol_inputs, "lol_matches"),
    )

    def collect(in_window: bool) -> dict[str, dict]:
        """Group fixtures by their slate date.

        in_window=True keeps only dates inside [window_start, window_end].
        in_window=False ignores the window (used for the stale-slate fallback).
        """
        by_date: dict[str, dict] = {}
        for inputs, game_key in games:
            slate_date = _input_date(inputs, today_date)
            if in_window and (slate_date < window_start or slate_date > window_end):
                continue
            matches = inputs.get("matches", []) or []
            if not matches:
                continue
            day = slate_date.isoformat()
            for raw in matches:
                mid = raw.get("match_id")
                row = _normalise_match(raw, preds.get(mid), match_id=mid)
                bucket = by_date.setdefault(day, {"cs2_matches": [], "lol_matches": []})
                bucket[game_key].append(row)
        return by_date

    by_date = collect(in_window=True)

    # Fallback: nothing in the tolerant window (stale slate). Surface the most
    # recent available slate date so the page still shows real fixtures.
    if not by_date:
        all_dates = collect(in_window=False)
        if all_dates:
            most_recent = max(all_dates.keys())
            by_date = {most_recent: all_dates[most_recent]}
            logger.warning(
                "No fixtures in window [%s, %s]; falling back to most recent slate %s",
                window_start.isoformat(), window_end.isoformat(), most_recent)

    next_dates = [
        {"date": day, **by_date[day]}
        for day in sorted(by_date.keys())
    ]

    return {
        "generated_for": today_date.isoformat(),
        "window_days": (window_end - window_start).days + 1,
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
