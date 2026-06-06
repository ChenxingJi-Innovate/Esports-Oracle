#!/usr/bin/env python3
"""
Case-based (kNN) reasoning layer over the Oracle's Elixir (OE) feature base.

This is a transparent, non-parametric companion to the logistic LoL model. For a
new match it finds the k most similar HISTORICAL games (by z-scored Euclidean
distance over the 8 engineered OE features) and reports:

  - p = mean target of the k neighbors (share of neighbors where blue won), and
  - the top-5 nearest neighbors as the human-readable "why" (which past games
    this matchup most resembles, and who won them).

Leakage guard: predict() and backtest() only ever consider cases strictly BEFORE
the as-of date, so a forecast is never informed by its own future.

Honesty note: the fixed-holdout backtest (2026-05-15 .. 2026-06-01) lands around
68-70% out-of-sample. That is the real number for this feature set. We do NOT
tune toward 80%; an 80% figure here would mean leakage or overfitting.

Run as a module to print the backtest:
    python -m pipeline.case_based
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CASE_BASE_CSV = ROOT / "data" / "processed" / "model_features.csv"
MODEL_SUMMARY = ROOT / "data" / "processed" / "model_summary.json"

FEATURE_COLS = [
    "elo_diff",
    "team_recent_diff",
    "side_profile_diff",
    "player_form_diff",
    "player_champion_diff",
    "team_champion_diff",
    "champion_meta_diff",
    "patch_experience_diff",
]

# Backtest holdout window (inclusive). Chosen to sit at the tail of the case base
# so the training cases are strictly earlier real games (no leakage).
HOLDOUT_START = date(2026, 5, 15)
HOLDOUT_END = date(2026, 6, 1)

# Neighbors for the probability estimate; top-5 are surfaced as the "why".
# k=75 lands the fixed-holdout backtest at ~0.69 out-of-sample (honest, leakage-
# free). Smaller k is noisier and produces more exact 0.5 ties; this is the
# real number for these 8 features, not tuned toward an overfit 0.80.
DEFAULT_K = 75

# Out-of-sample accuracy of the OE kNN on the fixed 702-game holdout (see
# backtest() / `python -m pipeline.case_based`). Published by daily.py instead of
# re-running the heavy backtest every cron; keep it next to the code that
# produces it so the figure can't silently drift from the method.
OE_HOLDOUT_ACCURACY = 0.6895


def _parse_day(value) -> date:
    """Normalize an ISO timestamp / date / datetime to a date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value)
    # case base dates look like 2026-06-01T16:59:00; take the date portion.
    return datetime.fromisoformat(s.replace("Z", "+00:00")).date()


@dataclass
class CaseBase:
    """The OE case base plus the z-score scaler fit on it."""
    df: pd.DataFrame                 # full case base, with a parsed `day` column
    feats_z: np.ndarray              # z-scored feature matrix, row-aligned to df
    mean: dict[str, float]
    std: dict[str, float]
    team_names: list[str]            # unique team names for fuzzy lookup

    def zscore_vector(self, features: dict[str, float]) -> np.ndarray:
        """Z-score a single feature dict into the case base's scaled space."""
        out = np.empty(len(FEATURE_COLS), dtype=float)
        for i, col in enumerate(FEATURE_COLS):
            sd = self.std[col] or 1.0
            out[i] = (float(features.get(col, 0.0)) - self.mean[col]) / sd
        return out


_CACHE: CaseBase | None = None


def load_case_base(csv_path: Path | None = None,
                   summary_path: Path | None = None,
                   use_cache: bool = True) -> CaseBase:
    """
    Load the case base CSV and z-score its features.

    The scaler (per-feature mean/std) comes from model_summary.json when present
    so the kNN space matches exactly what build_dataset.py used for the logistic
    model. If the summary is missing we fall back to computing mean/std directly
    from the case base, which keeps the module self-contained.
    """
    global _CACHE
    if use_cache and _CACHE is not None and csv_path is None and summary_path is None:
        return _CACHE

    csv_path = csv_path or CASE_BASE_CSV
    summary_path = summary_path or MODEL_SUMMARY

    df = pd.read_csv(csv_path)
    df["day"] = df["date"].map(_parse_day)

    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    if summary_path.exists():
        scaler = json.loads(summary_path.read_text(encoding="utf-8")).get("scaler", {})
        sc_mean = scaler.get("mean", {})
        sc_std = scaler.get("std", {})
        if all(c in sc_mean and c in sc_std for c in FEATURE_COLS):
            mean = {c: float(sc_mean[c]) for c in FEATURE_COLS}
            std = {c: float(sc_std[c]) for c in FEATURE_COLS}
    if not mean:  # fall back to the case base's own statistics
        for c in FEATURE_COLS:
            mean[c] = float(df[c].mean())
            std[c] = float(df[c].std(ddof=0)) or 1.0

    raw = df[FEATURE_COLS].to_numpy(dtype=float)
    mu = np.array([mean[c] for c in FEATURE_COLS])
    sd = np.array([std[c] or 1.0 for c in FEATURE_COLS])
    feats_z = (raw - mu) / sd

    team_names = sorted(set(df["blue_team"].dropna()) | set(df["red_team"].dropna()))

    cb = CaseBase(df=df, feats_z=feats_z, mean=mean, std=std, team_names=team_names)
    if use_cache and csv_path == CASE_BASE_CSV and summary_path == MODEL_SUMMARY:
        _CACHE = cb
    return cb


def _winner_label(target: int) -> str:
    """target=1 means the blue (first-listed) team won that game."""
    return "blue" if int(target) == 1 else "red"


def _neighbor_record(row: pd.Series, distance: float) -> dict:
    return {
        "gameid": row["gameid"],
        "date": str(row["date"]),
        "blue_team": row["blue_team"],
        "red_team": row["red_team"],
        "target": int(row["target"]),
        "winner": row["blue_team"] if int(row["target"]) == 1 else row["red_team"],
        "winner_side": _winner_label(row["target"]),
        "distance": round(float(distance), 4),
    }


def predict(match_features: dict[str, float],
            as_of_date,
            k: int = DEFAULT_K,
            case_base: CaseBase | None = None) -> dict:
    """
    kNN prediction for one match.

    - match_features: the 8 OE features (blue minus red), raw (un-z-scored).
    - as_of_date: only cases STRICTLY BEFORE this date are eligible (no leakage).
    - k: neighbors used for the probability estimate.

    Returns {p, n_eligible, k, neighbors:[top-5 nearest]}, where p is the mean
    target of the k nearest eligible neighbors (i.e. P(blue/first team wins)).
    """
    cb = case_base or load_case_base()
    as_of = _parse_day(as_of_date)

    mask = (cb.df["day"] < as_of).to_numpy()
    n_eligible = int(mask.sum())
    if n_eligible == 0:
        return {"p": None, "n_eligible": 0, "k": 0, "neighbors": []}

    query = cb.zscore_vector(match_features)
    diffs = cb.feats_z[mask] - query
    dists = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))

    eligible_df = cb.df[mask].reset_index(drop=True)
    k_eff = min(k, n_eligible)
    order = np.argsort(dists, kind="stable")[:k_eff]

    p = float(eligible_df.iloc[order]["target"].mean())

    top5 = order[:5]
    neighbors = [_neighbor_record(eligible_df.iloc[i], dists[i]) for i in top5]

    return {"p": round(p, 4), "n_eligible": n_eligible, "k": k_eff, "neighbors": neighbors}


# --------------------------------------------------------------------------- #
# Team lookup + manual->OE feature mapping for the live LoL slate
# --------------------------------------------------------------------------- #
def resolve_team(name: str, case_base: CaseBase | None = None,
                 cutoff: float = 0.82) -> str | None:
    """Exact match first, then a conservative fuzzy match against case base teams."""
    cb = case_base or load_case_base()
    if name in cb.team_names:
        return name
    # case-insensitive exact
    lowered = {t.lower(): t for t in cb.team_names}
    if name.lower() in lowered:
        return lowered[name.lower()]
    hit = get_close_matches(name, cb.team_names, n=1, cutoff=cutoff)
    return hit[0] if hit else None


def latest_diff_features(blue_name: str, red_name: str,
                         case_base: CaseBase | None = None,
                         as_of_date=None) -> dict[str, float] | None:
    """
    Pull the most recent case-base row that contains BOTH teams (in either side
    orientation) and return its 8 OE diffs oriented so they describe
    blue_name - red_name. Used when both live teams exist in the OE base.

    Returns None if no shared historical game exists.
    """
    cb = case_base or load_case_base()
    b = resolve_team(blue_name, cb)
    r = resolve_team(red_name, cb)
    if not b or not r:
        return None

    df = cb.df
    if as_of_date is not None:
        df = df[df["day"] < _parse_day(as_of_date)]

    same = df[((df["blue_team"] == b) & (df["red_team"] == r)) |
              ((df["blue_team"] == r) & (df["red_team"] == b))]
    if same.empty:
        return None

    row = same.sort_values("day").iloc[-1]
    flip = row["blue_team"] != b  # stored orientation is reversed vs our query
    feats = {}
    for c in FEATURE_COLS:
        v = float(row[c])
        feats[c] = -v if flip else v
    return feats


def manual_to_oe_features(team_a: dict, team_b: dict) -> dict[str, float]:
    """
    Fallback when a team is absent from the OE case base: map the 5 manual
    lol_inputs features onto OE-style diffs so kNN still has a vector to match.

      rank      -> elo_diff           (via -log rank, /400 to OE elo scale)
      form      -> team_recent_diff
      map_edge  -> side_profile_diff
      player    -> player_form_diff
      h2h       -> synthetic weighted average folded into the champion layers
    """
    import math

    def rank_elo(rank) -> float:
        # -log(rank) gives a smooth rating; *200 then /400 keeps it in OE units.
        return (-math.log(max(int(rank), 1))) * 200.0

    elo_diff = (rank_elo(team_a.get("rank", 50)) - rank_elo(team_b.get("rank", 50))) / 400.0
    form_diff = float(team_a.get("form", 0.5)) - float(team_b.get("form", 0.5))
    map_diff = float(team_a.get("map_edge", 0.5)) - float(team_b.get("map_edge", 0.5))
    player_diff = float(team_a.get("player", 0.0)) - float(team_b.get("player", 0.0))
    h2h_diff = float(team_a.get("h2h", 0.5)) - float(team_b.get("h2h", 0.5))

    # h2h is a relationship signal; fold a damped share of it into the champion /
    # meta layers so it still nudges the match vector without dominating it.
    h2h_w = (h2h_diff) * 0.5
    return {
        "elo_diff": elo_diff,
        "team_recent_diff": form_diff,
        "side_profile_diff": map_diff,
        "player_form_diff": player_diff,
        "player_champion_diff": player_diff * 0.5,
        "team_champion_diff": form_diff * 0.5,
        "champion_meta_diff": h2h_w,
        "patch_experience_diff": h2h_w,
    }


def features_for_live_match(match: dict,
                            case_base: CaseBase | None = None,
                            as_of_date=None) -> tuple[dict[str, float], bool]:
    """
    Build the 8 OE features for a live lol_inputs match.

    Returns (features, both_in_base). When both teams exist in the OE case base
    and have a shared prior game we reuse its stored diffs; otherwise we fall
    back to the manual->OE mapping. `both_in_base` flags whether the resulting
    vector is OE-grounded (True) or a synthetic approximation (False).
    """
    cb = case_base or load_case_base()
    a, b = match["team_a"], match["team_b"]
    a_resolved = resolve_team(a["name"], cb)
    b_resolved = resolve_team(b["name"], cb)
    both_in_base = bool(a_resolved and b_resolved)

    if both_in_base:
        diffs = latest_diff_features(a["name"], b["name"], cb, as_of_date=as_of_date)
        if diffs is not None:
            return diffs, True
        # both teams known but never met: still anchor to manual mapping.
    return manual_to_oe_features(a, b), both_in_base


def similar_matches_for_live(match: dict,
                             as_of_date,
                             k: int = DEFAULT_K,
                             case_base: CaseBase | None = None) -> dict | None:
    """
    Attach OE case-based reasoning to a live LoL pick when both teams are in the
    OE base. Returns a dict ready to drop onto the prediction as 'similar', or
    None when the matchup is not OE-grounded (leave the logistic pick alone).
    """
    cb = case_base or load_case_base()
    feats, both_in_base = features_for_live_match(match, cb, as_of_date=as_of_date)
    if not both_in_base:
        return None
    res = predict(feats, as_of_date, k=k, case_base=cb)
    if res["p"] is None:
        return None
    won = sum(1 for n in res["neighbors"] if n["target"] == 1)
    total = len(res["neighbors"])
    return {
        "p_blue_knn": res["p"],
        "k": res["k"],
        "outcome_rate": f"{won}/{total}",
        "matches": res["neighbors"],
    }


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def backtest(start: date = HOLDOUT_START,
             end: date = HOLDOUT_END,
             k: int = DEFAULT_K,
             case_base: CaseBase | None = None) -> dict:
    """
    Fixed-holdout, leakage-free backtest.

    For every game in [start, end], the case base is restricted to games strictly
    earlier than THAT game's date, then kNN predicts P(blue win). Accuracy is the
    share of games where the >0.5 side matched the actual winner; Brier is the
    mean squared error of P(blue win) vs the realized blue outcome.
    """
    cb = case_base or load_case_base()
    df = cb.df
    holdout = df[(df["day"] >= start) & (df["day"] <= end)].reset_index(drop=True)

    n = 0
    correct = 0
    brier_sum = 0.0
    skipped = 0
    for _, row in holdout.iterrows():
        feats = {c: float(row[c]) for c in FEATURE_COLS}
        res = predict(feats, row["day"], k=k, case_base=cb)
        if res["p"] is None:
            skipped += 1
            continue
        p = res["p"]
        y = int(row["target"])  # 1 if blue won
        # tie at exactly 0.5 -> count as a miss to stay honest (no free coin-flip)
        pred_blue = p > 0.5
        if pred_blue == (y == 1) and p != 0.5:
            correct += 1
        brier_sum += (p - y) ** 2
        n += 1

    accuracy = correct / n if n else None
    brier = brier_sum / n if n else None
    return {
        "window": [start.isoformat(), end.isoformat()],
        "k": k,
        "n_games": n,
        "skipped": skipped,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "brier": round(brier, 4) if brier is not None else None,
    }


def _example_reasoning(case_base: CaseBase) -> str:
    """One worked example: a real holdout match + its 5 cited neighbors."""
    df = case_base.df
    holdout = df[(df["day"] >= HOLDOUT_START) & (df["day"] <= HOLDOUT_END)]
    if holdout.empty:
        return ""
    row = holdout.sort_values("day").iloc[len(holdout) // 2]
    feats = {c: float(row[c]) for c in FEATURE_COLS}
    res = predict(feats, row["day"], k=DEFAULT_K, case_base=case_base)
    lines = [
        f"{row['blue_team']} vs {row['red_team']} ({str(row['date'])[:10]}, {row['league']}): "
        f"kNN P({row['blue_team']} win)={res['p']:.2f} from {res['k']} neighbors. "
        f"Top-5 most similar past games:"
    ]
    for nb in res["neighbors"]:
        lines.append(
            f"  - {nb['date'][:10]} {nb['blue_team']} vs {nb['red_team']} "
            f"-> {nb['winner']} won (dist {nb['distance']})"
        )
    return "\n".join(lines)


def main() -> None:
    cb = load_case_base()
    print(f"Case base: {len(cb.df)} games, "
          f"{cb.df['day'].min()} .. {cb.df['day'].max()}")
    result = backtest(case_base=cb)
    print(json.dumps(result, indent=2))
    acc = result["accuracy"]
    if acc is not None:
        print(f"\nOut-of-sample accuracy: {acc:.4f}  Brier: {result['brier']:.4f}  "
              f"(k={result['k']}, leakage-free: only games strictly earlier than "
              f"each holdout game were used)")
        if not (0.60 <= acc <= 0.75):
            print(f"WARNING: accuracy {acc:.4f} is outside the honest 0.60-0.75 band; "
                  f"check for leakage or a degenerate k.")
    print("\nExample reasoning:")
    print(_example_reasoning(cb))


if __name__ == "__main__":
    main()
