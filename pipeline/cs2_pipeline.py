#!/usr/bin/env python3
"""
CS2 tier-1 prediction pipeline.

Reads data/cs2_inputs.json (the day's tier-1 LAN slate) and produces map +
series probabilities through the shared engine. Weights are the damped
calibrated priors validated against the IEM Cologne slate (see notes in
predictor.py / the project writeup); ordering mirrors what the LoL logistic
learned (rating gap + form dominate, map edge is the CS2-specific add).

Daily-iteration TODO (each makes the number more "trained", less "prior"):
  - auto-fill team stats from Liquipedia (rank/form) + bo3.gg (map win rates),
  - fit the weights on a rolling 1-year LPDB match table instead of priors,
  - drop bookmaker odds entirely (the bo3.gg failure mode we beat).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .predictor import LinearModel, best_of, confidence_band

ROOT = Path(__file__).resolve().parents[1]

# damped priors: a disciplined step above the market, never a runaway.
_CS2_MODEL = LinearModel(
    intercept=0.0,
    weights={"rating_diff": 0.35, "form_diff": 0.22, "map_diff": 0.17,
             "player_diff": 0.17, "h2h_diff": 0.11},
    mean={k: 0.0 for k in ["rating_diff", "form_diff", "map_diff", "player_diff", "h2h_diff"]},
    std={"rating_diff": 1.0, "form_diff": 0.15, "map_diff": 0.12, "player_diff": 0.12, "h2h_diff": 0.80},
)


def _rank_rating(rank: int) -> float:
    return -math.log(max(rank, 1))


def _features(a: dict, b: dict) -> dict:
    return {
        "rating_diff": _rank_rating(a["rank"]) - _rank_rating(b["rank"]),
        "form_diff": a["form"] - b["form"],
        "map_diff": a["map_edge"] - b["map_edge"],
        "player_diff": a["player"] - b["player"],
        "h2h_diff": a["h2h"] - b["h2h"],
    }


def predict(inputs_path: Path | None = None, model: LinearModel | None = None) -> list[dict]:
    path = inputs_path or (ROOT / "data" / "cs2_inputs.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    mdl = model or _CS2_MODEL          # use the self-trained model when supplied
    out = []
    for m in data.get("matches", []):
        if m.get("tier") != 1:
            continue
        a, b = m["team_a"], m["team_b"]
        feats = _features(a, b)
        p_map = mdl.map_prob(feats)
        fmt = m.get("fmt", "BO3")
        n = {"BO1": 1, "BO3": 3, "BO5": 5}.get(fmt, 3)
        p_series = best_of(p_map, n)
        out.append({
            "match_id": m["match_id"],
            "event": m["event"],
            "team_a": a["name"],
            "team_b": b["name"],
            "p_map_a": round(p_map, 4),
            "p_a": round(p_series, 4),
            "fmt": fmt,
            "confidence": confidence_band(p_series, fmt),
            "features": {k: round(v, 4) for k, v in feats.items()},
        })
    return out


if __name__ == "__main__":
    for p in predict():
        print(f"{p['event']}: {p['team_a']} vs {p['team_b']}  "
              f"{p['team_a']} {p['p_a']*100:.1f}% ({p['fmt']}, {p['confidence']})")
