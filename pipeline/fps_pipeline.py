#!/usr/bin/env python3
"""
Shared tier-1 prediction engine for the tactical FPS games (CS2 + Valorant).

Both games read a daily inputs file of {team_a, team_b, rank/form/map_edge/
player/h2h}, push it through the same linear map model + best_of() series
conversion + scoreline() map score, and attach top-5 similar historical maps
from that game's case base as reasoning. The only per-game differences are the
inputs path, the corpus the case base is built from, and a display label, so
cs2_pipeline / val_pipeline are thin wrappers over predict() here.

LoL is intentionally NOT folded in: lol_pipeline has a different signature
(as_of_date, leakage-aware kNN that can drive the pick, pick_source) and a
genuinely different feature source (Oracle's Elixir), so unifying it would be
over-generalizing.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .predictor import LinearModel, best_of, confidence_band, scoreline

ROOT = Path(__file__).resolve().parents[1]

# Damped calibrated priors: a disciplined step above the market, never a
# runaway. World rank is DE-EMPHASISED on purpose (a slow, lagging aggregate
# that misreads volatile rosters, e.g. Liquid #26 lost 0-2 to FlyQuest #56 at
# IEM Cologne 2026); recent form + map-pool edge are more proximate predictors.
DEFAULT_MODEL = LinearModel(
    intercept=0.0,
    weights={"rating_diff": 0.18, "form_diff": 0.28, "map_diff": 0.24,
             "player_diff": 0.16, "h2h_diff": 0.12},
    mean={k: 0.0 for k in ["rating_diff", "form_diff", "map_diff", "player_diff", "h2h_diff"]},
    std={"rating_diff": 1.0, "form_diff": 0.15, "map_diff": 0.12, "player_diff": 0.12, "h2h_diff": 0.80},
)

_N_MAPS = {"BO1": 1, "BO3": 3, "BO5": 5}


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


def predict(inputs_path: Path, model: LinearModel | None = None, *,
            matches_csv=None, game_label: str = "CS2", case_base=None) -> list[dict]:
    """Predict one game's slate. `case_base` may be passed in (prebuilt) to skip
    a redundant corpus build; otherwise it is lazy-loaded from `matches_csv`."""
    if not Path(inputs_path).exists():
        return []
    data = json.loads(Path(inputs_path).read_text(encoding="utf-8"))
    mdl = model or DEFAULT_MODEL

    # The case base (derived from Liquipedia tier-1 history) is optional: if the
    # corpus is missing/too small the linear pick stands alone. Reuse a prebuilt
    # one when given so daily.py doesn't rebuild it per step.
    cs2_case_based = None
    try:
        from . import cs2_case_based as _cb
        cs2_case_based = _cb
        if case_base is None:
            case_base = _cb.load_case_base(matches_csv=matches_csv)
    except Exception:
        case_base = None

    out = []
    for m in data.get("matches", []):
        if m.get("tier") not in (1, None):
            continue
        a, b = m["team_a"], m["team_b"]
        feats = _features(a, b)
        p_map = mdl.map_prob(feats)
        fmt = m.get("fmt", "BO3")
        p_series = best_of(p_map, _N_MAPS.get(fmt, 3))
        pred = {
            "match_id": m["match_id"],
            "event": m.get("event", ""),
            "team_a": a["name"], "team_b": b["name"],
            "p_map_a": round(p_map, 4), "p_a": round(p_series, 4),
            "fmt": fmt, "confidence": confidence_band(p_series, fmt),
            "scoreline": scoreline(p_map, fmt),
            "scheduled_at": m.get("scheduled_at"),
            "date": (m.get("scheduled_at") or "")[:10] or m.get("date"),
            "features": {k: round(v, 4) for k, v in feats.items()},
        }
        # Case-based reasoning: top-5 similar past maps when a team is in the
        # history base. Append-only; never breaks the core pick.
        if case_base is not None and cs2_case_based is not None:
            try:
                sim = cs2_case_based.similar_matches_for_live(a["name"], b["name"], case_base)
                if sim:
                    pred["similar_matches"] = sim["matches"]
                    pred["case_based"] = {
                        "p_blue_knn": sim["p_blue_knn"], "k": sim["k"],
                        "outcome_rate": sim["outcome_rate"],
                        "note": sim["note"].replace("CS2", game_label),
                    }
            except Exception:
                pass
        out.append(pred)
    return out
