#!/usr/bin/env python3
"""
CS2 tier-1 prediction pipeline.

Reads data/cs2_inputs.json (the day's tier-1 LAN slate) and produces map +
series probabilities through the shared FPS engine (fps_pipeline). Weights are
the damped calibrated priors validated against the IEM Cologne slate; ordering
mirrors what the LoL logistic learned (rating gap + form dominate, map edge is
the CS2-specific add). The case base self-improves as cs2_corpus grows.
"""
from __future__ import annotations

from pathlib import Path

from . import fps_pipeline
from .fps_pipeline import DEFAULT_MODEL as _CS2_MODEL  # damped priors (re-exported)

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data" / "cs2_inputs.json"


def predict(inputs_path=None, model=None, case_base=None):
    return fps_pipeline.predict(inputs_path or INPUTS, model or _CS2_MODEL,
                                matches_csv=None, game_label="CS2", case_base=case_base)


if __name__ == "__main__":
    for p in predict():
        print(f"{p['event']}: {p['team_a']} vs {p['team_b']}  "
              f"{p['team_a']} {p['p_a']*100:.1f}% ({p['fmt']}, {p['confidence']})")
