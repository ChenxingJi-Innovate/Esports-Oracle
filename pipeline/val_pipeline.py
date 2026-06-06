#!/usr/bin/env python3
"""
Valorant tier-1 prediction pipeline.

A thin wrapper over the shared FPS engine (fps_pipeline), pointed at the
Valorant inputs + corpus. Same linear model + best_of() + scoreline() + case-
based reasoning as CS2; the case base self-improves as val_corpus grows.
"""
from __future__ import annotations

from pathlib import Path

from . import fps_pipeline
from .fps_pipeline import DEFAULT_MODEL as _VAL_MODEL  # same damped priors as CS2

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data" / "val_inputs.json"
VAL_MATCHES = ROOT / "data" / "processed" / "val_matches.csv"


def predict(inputs_path=None, model=None, case_base=None):
    return fps_pipeline.predict(inputs_path or INPUTS, model or _VAL_MODEL,
                                matches_csv=VAL_MATCHES, game_label="Valorant",
                                case_base=case_base)


if __name__ == "__main__":
    preds = predict()
    if not preds:
        print("No Valorant slate loaded (data/val_inputs.json empty or missing).")
    for p in preds:
        print(f"{p['event']}: {p['team_a']} vs {p['team_b']}  "
              f"{p['team_a']} {p['p_a']*100:.1f}% ({p['fmt']}, {p['confidence']})")
