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
