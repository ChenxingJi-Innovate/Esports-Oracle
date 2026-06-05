#!/usr/bin/env python3
"""
Daily self-training ("SFT itself") for the per-game logistic model.

How it learns from each day's new results, without leakage:
  - every prediction logs the feature vector it was made from (scorelog),
  - once a match is graded, that (features, winner) pair is a training example,
  - each morning, BEFORE making new picks, the model refits on the rolling
    1-year window of graded examples and saves the new weights,
  - today's picks then use that freshly refit model — which only ever saw
    matches that are already finished.

Cold-start is handled by ridge-toward-prior: the fit is penalised for moving
away from the calibrated prior weights, in the prior's own fixed z-scale. With
few graded matches the penalty wins and the model ~= prior; as real results
accumulate the data gradient takes over and the weights become genuinely
trained. So the model upgrades itself a little every day instead of swinging
wildly on three data points.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .predictor import LinearModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "models"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def train_game(examples: list[dict], prior: LinearModel,
               anchor: float = 0.6, epochs: int = 3000) -> tuple[LinearModel, dict]:
    """examples: [{"features": {feat: raw}, "label": 1 if team_a won else 0}].
    Returns (model, meta). Fits in the prior's fixed z-scale, ridge-anchored to
    the prior weights so it warms up smoothly from the prior."""
    feats = list(prior.weights.keys())
    w_prior = np.array([prior.intercept] + [prior.weights[f] for f in feats])
    mean = np.array([prior.mean.get(f, 0.0) for f in feats])
    std = np.array([prior.std.get(f, 1.0) or 1.0 for f in feats])

    usable = [e for e in examples if e.get("features") and e.get("label") is not None]
    n = len(usable)
    if n == 0:
        return prior, {"n_train": 0, "trained": False, "shrinkage": 0.0}

    X = np.array([[e["features"].get(f, 0.0) for f in feats] for e in usable], float)
    y = np.array([float(e["label"]) for e in usable])
    Xz = np.c_[np.ones(n), (X - mean) / std]

    # data weight vs prior weight: as n grows, data dominates
    shrink = n / (n + 40.0)          # 0 -> all prior, 1 -> all data
    l2 = anchor * (1.0 - shrink) + 0.02   # strong pull to prior when n small
    w = w_prior.copy()
    lr = 0.05
    for ep in range(epochs):
        p = _sigmoid(Xz @ w)
        grad = (Xz.T @ (p - y)) / n
        grad += l2 * (w - w_prior)    # ridge TOWARD the prior, not toward 0
        w -= lr * grad
        if ep in (1000, 2000):
            lr *= 0.6

    model = LinearModel(
        intercept=float(w[0]),
        weights={f: float(w[i + 1]) for i, f in enumerate(feats)},
        mean={f: float(mean[i]) for i, f in enumerate(feats)},
        std={f: float(std[i]) for i, f in enumerate(feats)},
    )
    # in-sample sanity only (real accuracy is tracked by scorelog on hold-out days)
    acc = float(((_sigmoid(Xz @ w) >= 0.5) == y).mean())
    meta = {"n_train": n, "trained": True, "shrinkage": round(shrink, 3),
            "l2_to_prior": round(l2, 3), "in_sample_acc": round(acc, 3)}
    return model, meta


def save(game: str, model: LinearModel, meta: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / f"{game}_model.json").write_text(json.dumps({
        "intercept": model.intercept, "weights": model.weights,
        "mean": model.mean, "std": model.std, "meta": meta,
    }, indent=2), encoding="utf-8")


def load(game: str) -> tuple[LinearModel, dict] | tuple[None, None]:
    path = MODEL_DIR / f"{game}_model.json"
    if not path.exists():
        return None, None
    d = json.loads(path.read_text(encoding="utf-8"))
    return LinearModel(d["intercept"], d["weights"], d["mean"], d["std"]), d.get("meta", {})
