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
from datetime import date
from pathlib import Path

from .predictor import LinearModel, best_of, confidence_band, scoreline

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
    # Auto-fetched fixtures are name-only (the OE kNN drives their pick), so
    # every linear feature must tolerate missing keys with a neutral default.
    return {
        "rating_diff": _rank_rating(a.get("rank", 20)) - _rank_rating(b.get("rank", 20)),
        "form_diff": a.get("form", 0.5) - b.get("form", 0.5),
        "map_diff": a.get("map_edge", 0.5) - b.get("map_edge", 0.5),
        "player_diff": a.get("player", 0.0) - b.get("player", 0.0),
        "h2h_diff": a.get("h2h", 0.5) - b.get("h2h", 0.5),
    }


def predict(inputs_path: Path | None = None, model: LinearModel | None = None,
            as_of_date: date | None = None) -> list[dict]:
    path = inputs_path or (ROOT / "data" / "lol_inputs.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = set(data.get("allowed_leagues", []))
    mdl = model or _LOL_MODEL          # use the self-trained model when supplied
    as_of = as_of_date or date.today()

    # Lazy-load the OE case base once for this slate. It is optional: if the
    # module/CSV is unavailable the logistic pick stands alone.
    case_base = None
    try:
        from . import case_based
        case_base = case_based.load_case_base()
    except Exception:
        case_based = None  # type: ignore

    out = []
    for m in data.get("matches", []):
        if allowed and m.get("league") not in allowed:
            continue  # tier-1 gate
        a, b = m["team_a"], m["team_b"]
        feats = _features(a, b)

        # Compute the OE kNN first: it doubles as reasoning AND, for
        # auto-fetched fixtures (predict: case_based) that carry no hand-entered
        # rank/form, as the actual pick (the grounded ~69% model beats a 50/50
        # linear read on empty inputs).
        sim = None
        if case_base is not None:
            try:
                sim = case_based.similar_matches_for_live(m, as_of, case_base=case_base)
            except Exception:
                sim = None

        use_knn = m.get("predict") == "case_based" and sim is not None
        p_map = sim["p_blue_knn"] if use_knn else mdl.map_prob(feats)
        fmt = m.get("fmt", "BO3")
        n = {"BO1": 1, "BO3": 3, "BO5": 5}.get(fmt, 3)
        p_series = best_of(p_map, n)
        pred = {
            "match_id": m["match_id"],
            "event": m.get("event", m.get("league", "")),
            "team_a": a["name"],
            "team_b": b["name"],
            "p_map_a": round(p_map, 4),
            "p_a": round(p_series, 4),
            "fmt": fmt,
            "confidence": confidence_band(p_series, fmt),
            "scoreline": scoreline(p_map, fmt),
            "pick_source": "case_based" if use_knn else "linear",
            "scheduled_at": m.get("scheduled_at"),
            "date": (m.get("scheduled_at") or "")[:10] or m.get("date"),
            "features": {k: round(v, 4) for k, v in feats.items()},
        }
        # Attach the top-5 similar matches as the auditable "why" whenever both
        # teams are OE-grounded (independent of which model drove the pick).
        if sim:
            pred["similar_matches"] = sim["matches"]
            pred["case_based"] = {
                "p_blue_knn": sim["p_blue_knn"],
                "k": sim["k"],
                "outcome_rate": sim["outcome_rate"],
            }
        out.append(pred)
    return out


if __name__ == "__main__":
    preds = predict()
    if not preds:
        print("No tier-1 LoL slate loaded for today (data/lol_inputs.json is empty).")
    for p in preds:
        print(f"{p['event']}: {p['team_a']} vs {p['team_b']}  "
              f"{p['team_a']} {p['p_a']*100:.1f}% ({p['fmt']}, {p['confidence']})")
