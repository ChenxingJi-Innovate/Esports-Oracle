#!/usr/bin/env python3
"""
Case-based (kNN) reasoning for CS2, mirroring the LoL case_based.py design.

For an upcoming match it finds the k most similar HISTORICAL maps (z-scored
Euclidean distance over the derived CS2 features) and reports:
  - p = share of the k neighbors where the analogous favored side won, and
  - the top-5 nearest neighbors as the human-readable "why".

The whole point, same as LoL: not a higher number than the linear model, but a
prediction you can audit ("here are the 5 past games we're reasoning from").

Leakage guard: predict()/backtest() only consider cases strictly BEFORE the
as-of date. The case base is built by cs2_features.py, which is itself
leakage-free (each row's features are known before its own result).

Honesty note: CS2's case base is FAR smaller than LoL's (Liquipedia tier-1
results vs Oracle's Elixir's 20k+ games) and has 4 derived features vs 8
engineered ones, so expect a noisier, lower out-of-sample number. We report the
real holdout figure, whatever it is; we do NOT tune toward 80%.

Run:  python -m pipeline.cs2_case_based     # print the backtest + an example
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

from . import cs2_features

ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLS = cs2_features.FEATURE_COLS

# Neighbors for the probability estimate; top-5 surfaced as the "why". k is kept
# modest because the corpus is small; it is re-derived honestly by the backtest.
DEFAULT_K = 25
TOP_N_REASONS = 5


def _parse_day(value) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


@dataclass
class CaseBase:
    X: np.ndarray            # (n, f) z-scored feature matrix
    raw: np.ndarray          # (n, f) raw features (for inspection)
    ts: np.ndarray           # (n,) unix timestamps
    target: np.ndarray       # (n,) 1 if team_a won
    team_a: list             # (n,)
    team_b: list             # (n,)
    winner: list             # (n,)
    dates: list              # (n,) iso strings
    mean: np.ndarray
    std: np.ndarray
    state: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ts)

    def z(self, feats: dict) -> np.ndarray:
        v = np.array([feats[c] for c in FEATURE_COLS], dtype=float)
        return (v - self.mean) / self.std


def load_case_base() -> CaseBase | None:
    rows, state = cs2_features.build(return_state=True)
    if len(rows) < 30:                      # too thin to reason from
        return None
    raw = np.array([[float(r[c]) for c in FEATURE_COLS] for r in rows], dtype=float)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std[std == 0] = 1.0
    X = (raw - mean) / std
    return CaseBase(
        X=X, raw=raw,
        ts=np.array([int(r["ts"]) for r in rows]),
        target=np.array([int(r["target"]) for r in rows]),
        team_a=[r["team_a"] for r in rows],
        team_b=[r["team_b"] for r in rows],
        winner=[r["winner"] if "winner" in r else (r["team_a"] if int(r["target"]) else r["team_b"]) for r in rows],
        dates=[r["date"] for r in rows],
        mean=mean, std=std, state=state,
    )


def _neighbors(cb: CaseBase, qz: np.ndarray, before_ts: int | None, k: int):
    mask = np.ones(len(cb), dtype=bool)
    if before_ts is not None:
        mask = cb.ts < before_ts
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    d = np.linalg.norm(cb.X[idx] - qz, axis=1)
    order = idx[np.argsort(d)[:k]]
    return [(int(i), float(np.linalg.norm(cb.X[i] - qz))) for i in order]


def predict(feats: dict, cb: CaseBase, before_ts: int | None = None,
            k: int = DEFAULT_K) -> dict:
    """kNN probability that TEAM_A wins, plus the top-N supporting neighbors."""
    qz = cb.z(feats)
    nn = _neighbors(cb, qz, before_ts, k)
    if not nn:
        return {"p_a_knn": 0.5, "k": 0, "neighbors": []}
    p = float(np.mean([cb.target[i] for i, _ in nn]))
    reasons = []
    for i, dist in nn[:TOP_N_REASONS]:
        # Field names mirror the LoL neighbor record so the shared UI simBlock
        # renders CS2 too: blue_team = first-listed (team_a), red = team_b.
        reasons.append({
            "date": cb.dates[i],
            "blue_team": cb.team_a[i],
            "red_team": cb.team_b[i],
            "winner": cb.winner[i],
            "winner_side": "blue" if int(cb.target[i]) == 1 else "red",
            "target": int(cb.target[i]),
            "distance": round(dist, 4),
        })
    return {"p_a_knn": round(p, 4), "k": len(nn), "neighbors": reasons}


def similar_matches_for_live(team_a: str, team_b: str, cb: CaseBase,
                             k: int = DEFAULT_K) -> dict | None:
    """Reasoning block for an UPCOMING match (all corpus rows are in the past)."""
    feats = cs2_features.live_features(team_a, team_b, cb.state)
    if feats is None:
        return None
    res = predict(feats, cb, before_ts=None, k=k)
    if not res["neighbors"]:
        return None
    won = sum(1 for n in res["neighbors"] if n["target"] == 1)
    total = len(res["neighbors"])
    return {
        "matches": res["neighbors"],
        "p_blue_knn": res["p_a_knn"],          # P(first-listed team wins)
        "k": res["k"],
        "outcome_rate": f"{won}/{total}",
        "note": f"CS2 Elo/form/H2H kNN (k={res['k']}). Nearer = more similar on 4 derived features.",
        "features": {c: feats[c] for c in FEATURE_COLS},
    }


def backtest(holdout_frac: float = 0.2, k: int = DEFAULT_K) -> dict:
    """Time-ordered holdout: evaluate the last `holdout_frac` of maps, each
    using only strictly-earlier cases. Reports the honest out-of-sample number."""
    cb = load_case_base()
    if cb is None:
        return {"error": "case base too small; scrape more events first"}
    n = len(cb)
    cut = int(n * (1 - holdout_frac))
    correct = brier = scored = 0
    for i in range(cut, n):
        feats = {c: cb.raw[i][j] for j, c in enumerate(FEATURE_COLS)}
        res = predict(feats, cb, before_ts=int(cb.ts[i]), k=k)
        if res["k"] == 0:
            continue
        p = res["p_a_knn"]
        y = int(cb.target[i])
        correct += int((p > 0.5) == bool(y))
        brier += (p - y) ** 2
        scored += 1
    return {
        "n_cases": n,
        "holdout_maps": scored,
        "k": k,
        "accuracy": round(correct / scored, 4) if scored else None,
        "brier": round(brier / scored, 4) if scored else None,
    }


def main() -> None:
    cb = load_case_base()
    if cb is None:
        print("CS2 case base too small (need >=30 matches). "
              "Run `python -m pipeline.cs2_corpus` then `python -m pipeline.cs2_features`.")
        return
    print(f"CS2 case base: {len(cb)} maps, {cb.dates[0]} .. {cb.dates[-1]}")
    print(json.dumps(backtest(), indent=2))
    # Example reasoning on the most recent two teams seen.
    a, b = cb.team_a[-1], cb.team_b[-1]
    sim = similar_matches_for_live(a, b, cb)
    if sim:
        print(f"\nExample: {a} vs {b}  kNN P({a} win)={sim['p_blue_knn']} from {sim['k']} neighbors")
        for nbr in sim["matches"]:
            print(f"  - {nbr['date']} {nbr['blue_team']} vs {nbr['red_team']} "
                  f"-> {nbr['winner']} won (dist {nbr['distance']})")


if __name__ == "__main__":
    main()
