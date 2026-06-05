#!/usr/bin/env python3
"""
CS2 match predictor — IEM Cologne Major 2026.

This ports the *structure* of the verified LoL formula

    p = sigmoid(intercept + sum(weight_i * zscore(feature_i)))

to Counter-Strike 2. The LoL-specific layers (champion mastery, patch
experience, fixed blue/red side) do not transfer, so they are replaced by the
CS2 analogs that actually move map outcomes: world-ranking rating gap, recent
map form, map-pool edge after the veto, player rating gap, and a decayed
head-to-head term.

IMPORTANT honesty notes:
- Weights are calibrated PRIORS, ordered after what the LoL logistic learned
  (player form and rating gap dominated; comfort/meta secondary). They are NOT
  a regression fit on a CS2 training set -- we do not have one wired in yet.
- Each input is tagged real / est so the reader sees what is sourced vs assumed.
- Sources: HLTV world ranking (Jun 2026), Liquipedia Stage-1 records,
  egamersworld/tips.gg head-to-head, consensus bookmaker lines.
"""
from __future__ import annotations
import math

# feature weights (z-scored inputs). Ordering mirrors the LoL model:
# rating gap + form + player skill carry most signal; map edge is the CS2 add.
# Deliberately damped: a disciplined model should land a modest step above the
# market, not stack every signal into a runaway 85%+ on a mid-tier match.
W = {
    "rating_diff": 0.35,   # world-ranking strength gap (Elo-like)
    "form_diff":   0.22,   # last ~3 months map win-rate gap
    "map_diff":    0.17,   # expected edge over the likely map pool (post-veto)
    "player_diff": 0.17,   # avg player rating (HLTV 2.0 / Rating3) gap
    "h2h_diff":    0.11,   # decayed recent head-to-head, capped (rosters/maps drift)
}
INTERCEPT = 0.0

# rough population spreads to z-score the raw differentials onto a common scale
SCALE = {
    "rating_diff": 1.0,   # rating_diff already expressed in "ranking tiers"
    "form_diff":   0.15,
    "map_diff":    0.12,
    "player_diff": 0.12,
    "h2h_diff":    0.80,   # widened so a lopsided H2H informs but never dominates
}


def rank_to_rating(rank: int) -> float:
    """Convert an HLTV world rank into a smooth strength score.
    Top teams ~ a few tiers above the pack; gaps shrink deeper in the list."""
    return -math.log(rank)  # higher rank number -> lower (more negative) score


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, x))))


def map_prob(a: dict, b: dict) -> float:
    feats = {
        "rating_diff": rank_to_rating(a["rank"]) - rank_to_rating(b["rank"]),
        "form_diff":   a["form"] - b["form"],
        "map_diff":    a["map_edge"] - b["map_edge"],
        "player_diff": a["player"] - b["player"],
        "h2h_diff":    a["h2h"] - b["h2h"],
    }
    z = INTERCEPT + sum(W[k] * (feats[k] / SCALE[k]) for k in W)
    return sigmoid(z), feats


def bo3(p: float) -> float:
    """P(win best-of-3) given per-map p, assuming map independence."""
    return p ** 2 + 2 * p ** 2 * (1 - p)


def show(name_a, name_b, a, b, fmt="BO3"):
    p, feats = map_prob(a, b)
    series = bo3(p) if fmt == "BO3" else p
    print(f"\n=== {name_a} vs {name_b}  ({fmt}) ===")
    for k, v in feats.items():
        print(f"  {k:12s} {v:+.3f}  (contrib {W[k]*v/SCALE[k]:+.3f})")
    print(f"  per-map  P({name_a}) = {p*100:5.1f}%")
    print(f"  {fmt:5s}    P({name_a}) = {series*100:5.1f}%   P({name_b}) = {(1-series)*100:5.1f}%")
    return series


# -------- matchup inputs (A = first team) --------
# form: last ~3mo map win rate (0-1, est). map_edge: expected pool win rate (est).
# player: avg roster rating proxy ~ (rating-1.0). h2h: recent H2H win share vs THIS opp.

# M80 (#24) vs NRG (#33): M80 9-2 all-time, 3-0 last 12mo (6:0 maps); books M80 ~58%
m80 = {"rank": 24, "form": 0.58, "map_edge": 0.55, "player": 0.06, "h2h": 0.82}
nrg = {"rank": 33, "form": 0.50, "map_edge": 0.50, "player": 0.02, "h2h": 0.18}

# Liquid (#26) vs HEROIC (#25): near-even rank; Liquid better at THIS event
# (beat BIG), HEROIC in elimination zone / poor day 1. H2H ~ even-ish.
liquid = {"rank": 26, "form": 0.55, "map_edge": 0.52, "player": 0.05, "h2h": 0.50}
heroic = {"rank": 25, "form": 0.45, "map_edge": 0.50, "player": 0.04, "h2h": 0.50}

if __name__ == "__main__":
    print("IEM Cologne Major 2026 - Stage 1 Round 4 (advancement, BO3)")
    show("M80", "NRG", m80, nrg, "BO3")
    show("Liquid", "HEROIC", liquid, heroic, "BO3")
