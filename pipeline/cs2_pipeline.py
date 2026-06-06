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

from .predictor import LinearModel, best_of, confidence_band, scoreline

ROOT = Path(__file__).resolve().parents[1]

# damped priors: a disciplined step above the market, never a runaway.
# World rank is DE-EMPHASISED on purpose: it is a slow, lagging aggregate that
# misreads volatile teams (e.g. Liquid #26 lost 0-2 to FlyQuest #56 at IEM
# Cologne 2026 despite a 73% rank-driven call). Recent form + map-pool edge are
# more proximate predictors of a single match, so they now carry the most weight.
_CS2_MODEL = LinearModel(
    intercept=0.0,
    weights={"rating_diff": 0.18, "form_diff": 0.28, "map_diff": 0.24,
             "player_diff": 0.16, "h2h_diff": 0.12},
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

    # Lazy-load the CS2 case base (derived from Liquipedia tier-1 history) once
    # per slate. Optional: if the corpus/feature table is missing, the linear
    # pick stands alone, exactly like the LoL pipeline's OE case base.
    case_base = None
    cs2_case_based = None
    try:
        from . import cs2_case_based as _cb
        cs2_case_based = _cb
        case_base = _cb.load_case_base()
    except Exception:
        case_base = None

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
        pred = {
            "match_id": m["match_id"],
            "event": m["event"],
            "team_a": a["name"],
            "team_b": b["name"],
            "p_map_a": round(p_map, 4),
            "p_a": round(p_series, 4),
            "fmt": fmt,
            "confidence": confidence_band(p_series, fmt),
            "scoreline": scoreline(p_map, fmt),
            "features": {k: round(v, 4) for k, v in feats.items()},
        }
        # Case-based reasoning: top-5 similar past maps when at least one team is
        # in the CS2 history base. Append-only; never breaks the core pick.
        if case_base is not None and cs2_case_based is not None:
            try:
                sim = cs2_case_based.similar_matches_for_live(
                    a["name"], b["name"], case_base)
                if sim:
                    pred["similar_matches"] = sim["matches"]
                    pred["case_based"] = {
                        "p_blue_knn": sim["p_blue_knn"],
                        "k": sim["k"],
                        "outcome_rate": sim["outcome_rate"],
                        "note": sim["note"],
                    }
            except Exception:
                pass  # reasoning is append-only; never break the core pick
        out.append(pred)
    return out


if __name__ == "__main__":
    for p in predict():
        print(f"{p['event']}: {p['team_a']} vs {p['team_b']}  "
              f"{p['team_a']} {p['p_a']*100:.1f}% ({p['fmt']}, {p['confidence']})")
