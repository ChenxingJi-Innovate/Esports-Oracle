#!/usr/bin/env python3
"""
Formula-verification + improvement experiment.

Loads the already-built feature matrix (data/processed/model_features.csv) and
re-fits several logistic variants on the same train/validation split the main
pipeline uses (train 2024-2025, validate 2026). Goal: confirm the published
numbers reproduce and test whether a leaner / reweighted formula beats the
shipped 8-feature model on the 2026 hold-out.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "processed" / "model_features.csv"

ALL = [
    "elo_diff",
    "team_recent_diff",
    "side_profile_diff",
    "player_form_diff",
    "player_champion_diff",
    "team_champion_diff",
    "champion_meta_diff",
    "patch_experience_diff",
]
# weak features in the shipped model (|weight| tiny or negative)
WEAK = ["side_profile_diff", "team_champion_diff", "patch_experience_diff"]
LEAN = [f for f in ALL if f not in WEAK]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def fit(x, y, l2=0.04, epochs=2400):
    x = np.c_[np.ones(len(x)), x]
    w = np.zeros(x.shape[1])
    lr = 0.08
    for e in range(epochs):
        p = sigmoid(x @ w)
        g = (x.T @ (p - y)) / len(y)
        g[1:] += l2 * w[1:]
        w -= lr * g
        if e in (600, 1200, 1800):
            lr *= 0.55
    return w


def metrics(x, y, w):
    p = sigmoid(np.c_[np.ones(len(x)), x] @ w)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    ll = float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())
    acc = float(((p >= 0.5) == y).mean())
    brier = float(((p - y) ** 2).mean())
    # 10-bin ECE
    edges = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < 9 else p <= hi)
        if m.sum():
            ece += abs(p[m].mean() - y[m].mean()) * m.sum() / len(y)
    return {"acc": acc, "logloss": ll, "brier": brier, "ece": float(ece)}


def run(df, cols, l2=0.04, epochs=2400):
    tr = df[df["year"].isin([2024, 2025])]
    va = df[df["year"] == 2026]
    trx = tr[cols].fillna(0.0).to_numpy(float)
    vax = va[cols].fillna(0.0).to_numpy(float)
    try_y = tr["target"].to_numpy(float)
    vay = va["target"].to_numpy(float)
    mu, sd = trx.mean(0), trx.std(0)
    sd[sd < 1e-6] = 1.0
    w = fit((trx - mu) / sd, try_y, l2=l2, epochs=epochs)
    return metrics((vax - mu) / sd, vay, w), dict(zip(["intercept"] + cols, w.round(4)))


def main():
    df = pd.read_csv(FEAT)
    print(f"rows={len(df)}  train={int((df.year!=2026).sum())}  valid(2026)={int((df.year==2026).sum())}\n")
    variants = {
        "shipped (8 feat)": (ALL, 0.04),
        "lean (drop 3 weak)": (LEAN, 0.04),
        "lean + stronger L2": (LEAN, 0.12),
        "elo only": (["elo_diff"], 0.04),
        "elo + form + player": (["elo_diff", "team_recent_diff", "player_form_diff"], 0.04),
    }
    rows = []
    for name, (cols, l2) in variants.items():
        m, w = run(df, cols, l2=l2)
        rows.append((name, m))
        print(f"{name:24s} acc={m['acc']:.4f}  logloss={m['logloss']:.4f}  brier={m['brier']:.4f}  ece={m['ece']:.4f}")
    best = max(rows, key=lambda r: r[1]["acc"])
    bestll = min(rows, key=lambda r: r[1]["logloss"])
    print(f"\nbest accuracy : {best[0]}  ({best[1]['acc']:.4f})")
    print(f"best logloss  : {bestll[0]}  ({bestll[1]['logloss']:.4f})")


if __name__ == "__main__":
    main()
