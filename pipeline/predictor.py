#!/usr/bin/env python3
"""
Shared prediction engine for both games.

Same backbone the LoL model was verified on:

    p(map win) = sigmoid(intercept + sum(weight_i * zscore(feature_i)))

The LoL trainer fits weights from data (build_dataset.py). For CS2, until a
trained LPDB-backed dataset exists, weights are calibrated priors (see
cs2_pipeline.py). Both feed through this one function so the UI, the BO
conversion, and the self-grading log treat every prediction identically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, x))))


@dataclass
class LinearModel:
    """A z-scored logistic model: weights + per-feature mean/std + intercept."""
    intercept: float
    weights: dict[str, float]
    mean: dict[str, float]
    std: dict[str, float]

    def map_prob(self, feats: dict[str, float]) -> float:
        z = self.intercept
        for name, w in self.weights.items():
            raw = feats.get(name, 0.0)
            mu = self.mean.get(name, 0.0)
            sd = self.std.get(name, 1.0) or 1.0
            z += w * ((raw - mu) / sd)
        return sigmoid(z)


def best_of(p_map: float, n: int) -> float:
    """P(win a best-of-n) given per-map probability p, assuming map independence.
    n must be odd (1, 3, 5). Independence is a simplifying assumption; momentum
    and veto order make real series slightly more deterministic than this."""
    if n == 1:
        return p_map
    need = n // 2 + 1
    # P(win >= need of n maps)
    total = 0.0
    for k in range(need, n + 1):
        total += math.comb(n, k) * p_map ** k * (1 - p_map) ** (n - k)
    return total


def scoreline_probs(p_map: float, fmt: str) -> list[dict]:
    """Distribution over final map scorelines, assuming map independence (same
    assumption as best_of). Returns every reachable scoreline as
    {a, b, p} (a = team_a's maps, b = team_b's maps), sorted most likely first.

    For a first-to-`need` series, team_a winning `need`-`k` means: across the
    first need-1+k maps team_a took need-1 and the opponent took k, then team_a
    closes it out. Summing the team_a branch reproduces best_of() exactly."""
    need = {"BO1": 1, "BO3": 2, "BO5": 3}.get(fmt, 2)
    p, q = p_map, 1.0 - p_map
    out: list[dict] = []
    for k in range(need):  # opponent's map count when team_a wins
        prob = math.comb(need - 1 + k, k) * p ** need * q ** k
        out.append({"a": need, "b": k, "p": prob})
    for k in range(need):  # team_a's map count when opponent wins
        prob = math.comb(need - 1 + k, k) * q ** need * p ** k
        out.append({"a": k, "b": need, "p": prob})
    out.sort(key=lambda d: -d["p"])
    return out


def scoreline(p_map: float, fmt: str) -> dict:
    """Most-likely scoreline + the full distribution, ready to drop on a pick.
    {"pick": "2-1", "a": 2, "b": 1, "p": 0.29, "dist": [{score,p}, ...]}."""
    dist = scoreline_probs(p_map, fmt)
    top = dist[0]
    return {
        "pick": f"{top['a']}-{top['b']}",
        "a": top["a"], "b": top["b"], "p": round(top["p"], 4),
        "dist": [{"score": f"{d['a']}-{d['b']}", "p": round(d["p"], 4)} for d in dist],
    }


def confidence_band(p_series: float, fmt: str) -> str:
    """Honest confidence tag. BO1 is high variance, so we never call it 'strong'."""
    edge = abs(p_series - 0.5)
    if fmt == "BO1":
        return "lean" if edge < 0.12 else "moderate"
    if edge < 0.08:
        return "coin-flip"
    if edge < 0.18:
        return "lean"
    if edge < 0.30:
        return "moderate"
    return "strong"
