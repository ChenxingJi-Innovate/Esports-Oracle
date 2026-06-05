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
            "p_map": round(p.get("p_map_a", p["p_a"]), 4),  # per-map prob, for calibration
            "fmt": p.get("fmt", "BO3"),
            "confidence": p.get("confidence", ""),
            "features": p.get("features"),   # feature vector -> becomes a training row once graded
            "pick": p["team_a"] if p["p_a"] >= 0.5 else p["team_b"],
            "result": None,        # filled by grade_pending: "a" | "b"
            "map_score": None,      # filled by grade_pending when known: [maps_a, maps_b]
            "correct": None,        # filled by grade_pending: bool
        })


def grade_pending(log: dict, results: dict) -> int:
    """results maps match_id -> winner 'a'|'b', OR -> {result, score_a, score_b}.
    The dict form (from the Liquipedia auto-parser) also records the map score.
    Returns count newly graded."""
    graded = 0
    for p in log["predictions"]:
        if p["result"] is None and p["match_id"] in results:
            r = results[p["match_id"]]
            if isinstance(r, dict):
                winner = r["result"]
                if r.get("score_a") is not None and r.get("score_b") is not None:
                    p["map_score"] = [r["score_a"], r["score_b"]]
            else:
                winner = r
            p["result"] = winner
            picked_a = p["p_a"] >= 0.5
            p["correct"] = (winner == "a") == picked_a
            graded += 1
    return graded


def map_training_examples(log: dict, game: str) -> list[dict]:
    """Expand each graded match into PER-MAP rows for calibration: a 2-0 gives
    two team_a-win rows, a 2-1 gives two win + one loss, etc. Round-score Bo1s
    (e.g. 13-8) count as a single map to the winner. More rows + truer signal
    than one series row, so the model learns map-result vs odds correlation."""
    rows = []
    for p in log["predictions"]:
        if p["game"] != game or not p.get("features") or p.get("result") is None:
            continue
        ms = p.get("map_score")
        if ms and isinstance(ms[0], int) and isinstance(ms[1], int) and max(ms) <= 3:
            wins_a, wins_b = ms[0], ms[1]            # real map counts (Bo3/Bo5)
        else:
            wins_a, wins_b = (1, 0) if p["result"] == "a" else (0, 1)  # Bo1 / unknown
        rows += [{"features": p["features"], "label": 1}] * wins_a
        rows += [{"features": p["features"], "label": 0}] * wins_b
    return rows


def calibration(log: dict, game: str, bins: int = 4) -> list[dict]:
    """Predicted per-map prob vs realized per-map win rate, bucketed. This is the
    'map result vs odds' curve: if the 60-70% bucket realizes at 50%, we're hot."""
    pts = []  # (p_map_for_favorite, won?) per map
    for p in log["predictions"]:
        if p["game"] != game or p.get("result") is None or p.get("p_map") is None:
            continue
        ms = p.get("map_score")
        pm = p["p_map"]  # prob team_a wins a map
        if ms and isinstance(ms[0], int) and isinstance(ms[1], int) and max(ms) <= 3:
            seq = [1] * ms[0] + [0] * ms[1]
        else:
            seq = [1] if p["result"] == "a" else [0]
        for won_a in seq:
            # express from the favorite's view so buckets are >= 0.5
            pts.append((pm, won_a) if pm >= 0.5 else (1 - pm, 1 - won_a))
    if not pts:
        return []
    out = []
    for i in range(bins):
        lo, hi = 0.5 + i * 0.5 / bins, 0.5 + (i + 1) * 0.5 / bins
        b = [w for pmf, w in pts if (lo <= pmf < hi or (i == bins - 1 and pmf == 1.0))]
        if b:
            out.append({"bucket": f"{lo:.2f}-{hi:.2f}", "n": len(b),
                        "predicted": round((lo + hi) / 2, 3),
                        "realized": round(sum(b) / len(b), 3)})
    return out


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
