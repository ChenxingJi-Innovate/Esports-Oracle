#!/usr/bin/env python3
"""
The self-improvement heart of the tool.

Every day the pipeline:
  1. records the predictions it makes (append_predictions),
  2. the next day, looks up which of those matches now have results and grades
     them (grade_pending), updating a rolling accuracy / Brier / calibration
     record so the model's *real* hit-rate is visible over time,
  3. auto-purges anything older than the rolling window (purge_old).

The log is plain JSON committed to the repo, so history is versioned for free
and the web UI can chart "how right has this model actually been".
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

WINDOW_DAYS = 365


def _today() -> date:
    # injected by callers in tests; real runs pass date.today()
    return date.today()


def load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"predictions": [], "accuracy": {}}


def save(path: Path, log: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


def append_predictions(log: dict, day: str, game: str, preds: list[dict]) -> None:
    """preds: [{match_id, event, team_a, team_b, p_a, fmt, confidence}]"""
    have = {p["match_id"] for p in log["predictions"]}
    for p in preds:
        if p["match_id"] in have:
            continue
        log["predictions"].append({
            "match_id": p["match_id"],
            "date": day,
            "game": game,
            "event": p["event"],
            "team_a": p["team_a"],
            "team_b": p["team_b"],
            "p_a": round(p["p_a"], 4),
            "fmt": p.get("fmt", "BO3"),
            "confidence": p.get("confidence", ""),
            "features": p.get("features"),   # feature vector -> becomes a training row once graded
            "pick": p["team_a"] if p["p_a"] >= 0.5 else p["team_b"],
            "result": None,        # filled by grade_pending: "a" | "b"
            "correct": None,        # filled by grade_pending: bool
        })


def grade_pending(log: dict, results: dict[str, str]) -> int:
    """results maps match_id -> winner 'a'|'b'. Returns count newly graded."""
    graded = 0
    for p in log["predictions"]:
        if p["result"] is None and p["match_id"] in results:
            winner = results[p["match_id"]]
            p["result"] = winner
            picked_a = p["p_a"] >= 0.5
            p["correct"] = (winner == "a") == picked_a
            graded += 1
    return graded


def purge_old(log: dict, today: date, window: int = WINDOW_DAYS) -> int:
    cutoff = today - timedelta(days=window)
    before = len(log["predictions"])
    log["predictions"] = [
        p for p in log["predictions"]
        if datetime.fromisoformat(p["date"]).date() >= cutoff
    ]
    return before - len(log["predictions"])


def training_examples(log: dict, game: str) -> list[dict]:
    """Graded predictions with a stored feature vector become labeled rows.
    label = 1 if the first-listed team (team_a) actually won."""
    rows = []
    for p in log["predictions"]:
        if p["game"] == game and p.get("features") and p.get("result") is not None:
            rows.append({"features": p["features"], "label": 1 if p["result"] == "a" else 0})
    return rows


def recompute_accuracy(log: dict) -> dict:
    """Rolling scorecard, overall and per game, only over graded predictions."""
    def stats(rows: list[dict]) -> dict:
        graded = [r for r in rows if r["correct"] is not None]
        n = len(graded)
        if not n:
            return {"graded": 0, "accuracy": None, "brier": None}
        hits = sum(1 for r in graded if r["correct"])
        # Brier on the probability assigned to the actual winner
        brier = 0.0
        for r in graded:
            outcome_a = 1.0 if r["result"] == "a" else 0.0
            brier += (r["p_a"] - outcome_a) ** 2
        return {
            "graded": n,
            "accuracy": round(hits / n, 4),
            "brier": round(brier / n, 4),
        }

    by_game = {}
    for g in sorted({p["game"] for p in log["predictions"]}):
        by_game[g] = stats([p for p in log["predictions"] if p["game"] == g])
    acc = {"overall": stats(log["predictions"]), "by_game": by_game}
    log["accuracy"] = acc
    return acc
