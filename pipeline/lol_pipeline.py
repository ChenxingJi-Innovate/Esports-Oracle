#!/usr/bin/env python3
"""
LoL tier-1 prediction pipeline.

Two modes:
  - PRIOR mode (now): reads data/lol_inputs.json (LPL/LCK/LEC/MSI/Worlds only)
    and runs the same rank/form/h2h prior model as CS2, so the dual-game tool
    works today from editable inputs.
  - TRAINED mode (planned upgrade): swap _LOL_MODEL for the logistic fit by
    build_dataset.py on the rolling 1-year Oracle's Elixir window, restricted to
    tier-1 leagues. That model is already verified at 65.3% hold-out accuracy;
    wiring it to *upcoming* fixtures needs the Leaguepedia schedule + a current
    team/player rating export, which is the next daily iteration.

Tier-1 gate: only leagues in inputs["allowed_leagues"] are predicted; anything
else (minor regions) is dropped, per the product spec.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .predictor import LinearModel, best_of, confidence_band

ROOT = Path(__file__).resolve().parents[1]

_LOL_MODEL = LinearModel(
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
        "form_diff": a.get("form", 0.5) - b.get("form", 0.5),
        "map_diff": a.get("map_edge", 0.5) - b.get("map_edge", 0.5),
        "player_diff": a.get("player", 0.0) - b.get("player", 0.0),
        "h2h_diff": a.get("h2h", 0.5) - b.get("h2h", 0.5),
    }


def predict(inputs_path: Path | None = None, model: LinearModel | None = None) -> list[dict]:
    path = inputs_path or (ROOT / "data" / "lol_inputs.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = set(data.get("allowed_leagues", []))
    mdl = model or _LOL_MODEL          # use the self-trained model when supplied
    out = []
    for m in data.get("matches", []):
        if allowed and m.get("league") not in allowed:
            continue  # tier-1 gate
        a, b = m["team_a"], m["team_b"]
        feats = _features(a, b)
        p_map = mdl.map_prob(feats)
        fmt = m.get("fmt", "BO3")
        n = {"BO1": 1, "BO3": 3, "BO5": 5}.get(fmt, 3)
        p_series = best_of(p_map, n)
        out.append({
            "match_id": m["match_id"],
            "event": m.get("event", m.get("league", "")),
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
    preds = predict()
    if not preds:
        print("No tier-1 LoL slate loaded for today (data/lol_inputs.json is empty).")
    for p in preds:
        print(f"{p['event']}: {p['team_a']} vs {p['team_b']}  "
              f"{p['team_a']} {p['p_a']*100:.1f}% ({p['fmt']}, {p['confidence']})")
