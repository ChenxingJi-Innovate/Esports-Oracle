#!/usr/bin/env python3
"""
Windowed backtest for the LoL logistic model.

Question this answers: how much training history do you actually need? We fix a
recent hold-out (the last full ~17 days of available data, 2026-05-15 to
2026-06-01) and, for each look-back window {1w, 1m, 3m, 6m}, train a fresh
logistic ONLY on games inside [holdout_start - window, holdout_start). Every
window is then scored on the exact same fixed hold-out. Comparing accuracy /
Brier / log-loss across windows shows whether more history helps or whether the
signal is recent-form-dominated.

Leakage safety: every training row is strictly before holdout_start, and the
z-score scaler (mean/std) is computed on the training window only, then applied
to the hold-out. The hold-out never touches the fit.

Model: identical z-scored L2-logistic backbone as scripts/build_dataset.py and
pipeline/predictor.py:

    p(blue win) = sigmoid(intercept + sum(weight_i * zscore(feature_i)))

CLI:
    python -m pipeline.backtest_windows
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data" / "processed" / "model_features.csv"
OUT_JSON = ROOT / "app" / "data" / "backtest.json"

# Same eight engineered OE features the production model uses.
FEATURES = [
    "elo_diff",
    "team_recent_diff",
    "side_profile_diff",
    "player_form_diff",
    "player_champion_diff",
    "team_champion_diff",
    "champion_meta_diff",
    "patch_experience_diff",
]

# Fixed hold-out: the last full stretch of available data. Data ends 2026-06-01.
HOLDOUT_START = pd.Timestamp("2026-05-15")
HOLDOUT_END = pd.Timestamp("2026-06-01")

# Look-back windows, each ending exactly at HOLDOUT_START.
WINDOWS = [
    ("1w", 7),
    ("1m", 30),
    ("3m", 90),
    ("6m", 182),
]


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def fit_logistic(train_x: np.ndarray, train_y: np.ndarray, l2: float = 0.04, epochs: int = 2400) -> np.ndarray:
    """Identical optimiser to scripts/build_dataset.py: batch gradient descent
    on the L2-penalised logistic loss (intercept unpenalised), with the same
    learning-rate decay schedule. Inputs are already z-scored."""
    x = np.c_[np.ones(len(train_x)), train_x]
    w = np.zeros(x.shape[1], dtype=float)
    lr = 0.08
    for epoch in range(epochs):
        p = sigmoid(x @ w)
        grad = (x.T @ (p - train_y)) / len(train_y)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
        if epoch in (600, 1200, 1800):
            lr *= 0.55
    return w


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(((p - y) ** 2).mean())


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def run_window(df: pd.DataFrame, days: int, holdout_x: np.ndarray, holdout_y: np.ndarray) -> dict | None:
    """Train on [HOLDOUT_START - days, HOLDOUT_START); score on the fixed hold-out."""
    window_start = HOLDOUT_START - pd.Timedelta(days=days)
    train = df[(df["date"] >= window_start) & (df["date"] < HOLDOUT_START)]
    n_train = int(len(train))
    if n_train == 0:
        return {"n_train": 0, "window_start": str(window_start.date())}

    train_x_raw = train[FEATURES].fillna(0.0).to_numpy(dtype=float)
    train_y = train["target"].to_numpy(dtype=float)

    # z-score scaler fit on the training window only, applied to the hold-out.
    means = train_x_raw.mean(axis=0)
    stds = train_x_raw.std(axis=0)
    stds[stds < 1e-6] = 1.0
    train_x = (train_x_raw - means) / stds
    test_x = (holdout_x - means) / stds

    w = fit_logistic(train_x, train_y)
    p_test = sigmoid(np.c_[np.ones(len(test_x)), test_x] @ w)

    return {
        "n_train": n_train,
        "window_start": str(window_start.date()),
        "accuracy": float(((p_test >= 0.5) == holdout_y).mean()),
        "brier": brier(holdout_y, p_test),
        "logloss": log_loss(holdout_y, p_test),
    }


def main() -> None:
    df = load_features()

    holdout = df[(df["date"] >= HOLDOUT_START) & (df["date"] < HOLDOUT_END)]
    holdout_x = holdout[FEATURES].fillna(0.0).to_numpy(dtype=float)
    holdout_y = holdout["target"].to_numpy(dtype=float)
    n_holdout = int(len(holdout))

    windows: list[dict] = []
    for name, days in WINDOWS:
        res = run_window(df, days, holdout_x, holdout_y)
        row = {"window": name, "days": days}
        row.update(res)
        # round metrics for a tidy payload
        for k in ("accuracy", "brier", "logloss"):
            if k in row and row[k] is not None:
                row[k] = round(row[k], 4)
        windows.append(row)

    # best = highest accuracy on the hold-out, ties broken by lower Brier.
    scored = [w for w in windows if w.get("accuracy") is not None]
    best_window = None
    if scored:
        best = max(scored, key=lambda w: (w["accuracy"], -w["brier"]))
        best_window = best["window"]

    payload = {
        "holdout": {
            "start": str(HOLDOUT_START.date()),
            "end": str(HOLDOUT_END.date()),
            "n": n_holdout,
        },
        "windows": windows,
        "best_window": best_window,
        "model": "z-scored L2-logistic (same backbone as build_dataset / predictor)",
        "features": FEATURES,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Hold-out {payload['holdout']['start']} -> {payload['holdout']['end']}  n={n_holdout}")
    for w in windows:
        if w.get("accuracy") is not None:
            print(f"  {w['window']:>3}  n_train={w['n_train']:>5}  "
                  f"acc={w['accuracy']:.4f}  brier={w['brier']:.4f}  logloss={w['logloss']:.4f}")
        else:
            print(f"  {w['window']:>3}  n_train=0 (no data in window)")
    print(f"best_window = {best_window}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
