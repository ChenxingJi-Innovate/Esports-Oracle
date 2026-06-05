#!/usr/bin/env python3
"""
Build the local League of Legends esports prediction dataset.

The model is intentionally transparent:
- all predictive features are computed from matches that happened before the
  target game;
- the target is blue-side game win probability;
- a small L2-regularized logistic regression is trained with numpy so the
  formula can be inspected and reused in the browser.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
APP_DIR = ROOT / "app"

YEARS = [2024, 2025, 2026]
POSITIONS = ["top", "jng", "mid", "bot", "sup"]
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


@dataclass
class RateStat:
    games: int = 0
    wins: float = 0.0
    values: deque = field(default_factory=lambda: deque(maxlen=10))

    def rate(self, prior: float = 0.5, strength: float = 8.0) -> float:
        return (self.wins + prior * strength) / (self.games + strength)

    def recent(self, prior: float = 0.5, strength: float = 4.0) -> float:
        if not self.values:
            return prior
        return (sum(self.values) + prior * strength) / (len(self.values) + strength)

    def add(self, result: float) -> None:
        self.games += 1
        self.wins += float(result)
        self.values.append(float(result))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(((p - y) ** 2).mean())


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    edges = np.linspace(0, 1, bins + 1)
    rows: list[dict] = []
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": 0, "avg_pred": None, "actual": None})
            continue
        avg_pred = float(p[mask].mean())
        actual = float(y[mask].mean())
        ece += abs(avg_pred - actual) * count / len(y)
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": count, "avg_pred": avg_pred, "actual": actual})
    return float(ece), rows


def fit_logistic(train_x: np.ndarray, train_y: np.ndarray, l2: float = 0.04, epochs: int = 2400) -> np.ndarray:
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


def evaluate(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    p = sigmoid(np.c_[np.ones(len(x)), x] @ w)
    ece, bins = expected_calibration_error(y, p)
    return {
        "games": int(len(y)),
        "accuracy": float(((p >= 0.5) == y).mean()),
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "ece": ece,
        "calibration_bins": bins,
    }


def read_raw() -> pd.DataFrame:
    usecols = [
        "gameid",
        "datacompleteness",
        "league",
        "year",
        "split",
        "playoffs",
        "date",
        "game",
        "patch",
        "participantid",
        "side",
        "position",
        "playername",
        "playerid",
        "teamname",
        "teamid",
        "champion",
        "result",
        "kills",
        "deaths",
        "assists",
        "dpm",
        "earned gpm",
        "visionscore",
        "wardsplaced",
    ]
    frames = []
    for year in YEARS:
        path = RAW_DIR / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run scripts/fetch_data.py first.")
        frames.append(pd.read_csv(path, usecols=usecols, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["datacompleteness"].eq("complete")].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "gameid", "teamname", "teamid"])
    df["patch"] = df["patch"].astype(str)
    return df.sort_values(["date", "gameid", "participantid"])


def player_impact(row: pd.Series) -> float:
    deaths = max(float(row.get("deaths") or 0), 1.0)
    kda = (float(row.get("kills") or 0) + 0.7 * float(row.get("assists") or 0)) / deaths
    dpm = float(row.get("dpm") or 0)
    egpm = float(row.get("earned gpm") or 0)
    vision = float(row.get("visionscore") or 0)
    # Squash noisy stat scales into roughly [0, 1] before mixing with game result.
    stat_score = (
        0.34 * math.tanh(kda / 4.0)
        + 0.28 * math.tanh(dpm / 850.0)
        + 0.24 * math.tanh(egpm / 360.0)
        + 0.14 * math.tanh(vision / 75.0)
    )
    return 0.52 * float(row["result"]) + 0.48 * stat_score


def smoothed_record(stat: RateStat | None, prior: float = 0.5, strength: float = 8.0) -> float:
    return stat.rate(prior, strength) if stat else prior


def build_game_objects(df: pd.DataFrame) -> list[dict]:
    games: list[dict] = []
    for gameid, group in df.groupby("gameid", sort=False):
        team_rows = group[group["position"].eq("team")]
        player_rows = group[group["position"].isin(POSITIONS)]
        if len(team_rows) != 2 or len(player_rows) < 10:
            continue
        sides = {row["side"]: row for _, row in team_rows.iterrows()}
        if "Blue" not in sides or "Red" not in sides:
            continue
        players = {}
        ok = True
        for side in ["Blue", "Red"]:
            side_players = []
            side_group = player_rows[player_rows["side"].eq(side)]
            for pos in POSITIONS:
                rows = side_group[side_group["position"].eq(pos)]
                if rows.empty:
                    ok = False
                    break
                row = rows.iloc[0]
                if not isinstance(row["champion"], str) or not row["champion"]:
                    ok = False
                    break
                side_players.append(row)
            players[side] = side_players
        if not ok:
            continue
        games.append(
            {
                "gameid": gameid,
                "date": sides["Blue"]["date"],
                "year": int(sides["Blue"]["year"]),
                "league": sides["Blue"]["league"],
                "split": sides["Blue"]["split"],
                "patch": str(sides["Blue"]["patch"]),
                "blue": sides["Blue"],
                "red": sides["Red"],
                "blue_players": players["Blue"],
                "red_players": players["Red"],
                "target": int(sides["Blue"]["result"]),
            }
        )
    return sorted(games, key=lambda g: (g["date"], g["gameid"]))


def run_feature_pass(games: Iterable[dict]) -> tuple[pd.DataFrame, dict]:
    team_hist: dict[str, RateStat] = defaultdict(RateStat)
    side_hist: dict[tuple[str, str], RateStat] = defaultdict(RateStat)
    player_hist: dict[str, RateStat] = defaultdict(RateStat)
    player_champ_hist: dict[tuple[str, str], RateStat] = defaultdict(RateStat)
    team_champ_hist: dict[tuple[str, str], RateStat] = defaultdict(RateStat)
    champ_role_hist: dict[tuple[str, str], RateStat] = defaultdict(RateStat)
    team_patch_hist: dict[tuple[str, str], RateStat] = defaultdict(RateStat)
    player_patch_hist: dict[tuple[str, str], RateStat] = defaultdict(RateStat)
    team_elo: dict[str, float] = defaultdict(lambda: 1500.0)
    team_names: dict[str, str] = {}
    team_latest: dict[str, dict] = {}
    player_names: dict[str, str] = {}

    rows = []

    def side_features(team_id: str, side: str, player_rows: list[pd.Series], patch: str) -> dict:
        team_rate = team_hist[team_id].recent()
        side_rate = side_hist[(team_id, side)].rate(strength=6)
        player_form = []
        player_champ = []
        patch_exp = []
        team_champ = []
        champ_meta = []
        for player in player_rows:
            pid = player["playerid"]
            champ = player["champion"]
            role = player["position"]
            player_form.append(smoothed_record(player_hist.get(pid), strength=10))
            player_champ.append(smoothed_record(player_champ_hist.get((pid, champ)), strength=5))
            team_champ.append(smoothed_record(team_champ_hist.get((team_id, champ)), strength=7))
            champ_meta.append(smoothed_record(champ_role_hist.get((role, champ)), strength=14))
            team_games_on_patch = team_patch_hist[(team_id, patch)].games
            player_games_on_patch = player_patch_hist[(pid, patch)].games
            patch_exp.append(math.log1p(team_games_on_patch + player_games_on_patch))
        return {
            "recent": team_rate,
            "side": side_rate,
            "player_form": float(np.mean(player_form)),
            "player_champ": float(np.mean(player_champ)),
            "team_champ": float(np.mean(team_champ)),
            "champ_meta": float(np.mean(champ_meta)),
            "patch_exp": float(np.mean(patch_exp)),
        }

    for game in games:
        blue = game["blue"]
        red = game["red"]
        blue_id = blue["teamid"]
        red_id = red["teamid"]
        team_names[blue_id] = blue["teamname"]
        team_names[red_id] = red["teamname"]
        for player in game["blue_players"] + game["red_players"]:
            player_names[player["playerid"]] = player["playername"]

        bf = side_features(blue_id, "Blue", game["blue_players"], game["patch"])
        rf = side_features(red_id, "Red", game["red_players"], game["patch"])
        row = {
            "gameid": game["gameid"],
            "date": game["date"].isoformat(),
            "year": game["year"],
            "league": game["league"],
            "patch": game["patch"],
            "blue_team": blue["teamname"],
            "red_team": red["teamname"],
            "blue_teamid": blue_id,
            "red_teamid": red_id,
            "target": game["target"],
            "elo_diff": (team_elo[blue_id] - team_elo[red_id]) / 400.0,
            "team_recent_diff": bf["recent"] - rf["recent"],
            "side_profile_diff": bf["side"] - rf["side"],
            "player_form_diff": bf["player_form"] - rf["player_form"],
            "player_champion_diff": bf["player_champ"] - rf["player_champ"],
            "team_champion_diff": bf["team_champ"] - rf["team_champ"],
            "champion_meta_diff": bf["champ_meta"] - rf["champ_meta"],
            "patch_experience_diff": bf["patch_exp"] - rf["patch_exp"],
        }
        rows.append(row)

        blue_result = float(game["target"])
        red_result = 1.0 - blue_result
        expected_blue = 1.0 / (1.0 + 10 ** ((team_elo[red_id] - team_elo[blue_id]) / 400.0))
        k_blue = 34.0 / (1.0 + team_hist[blue_id].games / 80.0)
        k_red = 34.0 / (1.0 + team_hist[red_id].games / 80.0)
        team_elo[blue_id] += k_blue * (blue_result - expected_blue)
        team_elo[red_id] += k_red * (red_result - (1.0 - expected_blue))
        team_hist[blue_id].add(blue_result)
        team_hist[red_id].add(red_result)
        side_hist[(blue_id, "Blue")].add(blue_result)
        side_hist[(red_id, "Red")].add(red_result)
        team_patch_hist[(blue_id, game["patch"])].add(blue_result)
        team_patch_hist[(red_id, game["patch"])].add(red_result)

        for player, result in [(p, blue_result) for p in game["blue_players"]] + [(p, red_result) for p in game["red_players"]]:
            pid = player["playerid"]
            champ = player["champion"]
            team_id = player["teamid"]
            role = player["position"]
            impact = player_impact(player)
            player_hist[pid].add(impact)
            player_champ_hist[(pid, champ)].add(result)
            team_champ_hist[(team_id, champ)].add(result)
            champ_role_hist[(role, champ)].add(result)
            player_patch_hist[(pid, game["patch"])].add(result)

        team_latest[blue_id] = {
            "last_seen": game["date"].isoformat(),
            "league": game["league"],
            "patch": game["patch"],
            "roster": [
                {
                    "position": p["position"],
                    "playerid": p["playerid"],
                    "playername": p["playername"],
                    "champion": p["champion"],
                }
                for p in game["blue_players"]
            ],
        }
        team_latest[red_id] = {
            "last_seen": game["date"].isoformat(),
            "league": game["league"],
            "patch": game["patch"],
            "roster": [
                {
                    "position": p["position"],
                    "playerid": p["playerid"],
                    "playername": p["playername"],
                    "champion": p["champion"],
                }
                for p in game["red_players"]
            ],
        }

    state = {
        "team_hist": team_hist,
        "side_hist": side_hist,
        "player_hist": player_hist,
        "player_champ_hist": player_champ_hist,
        "team_champ_hist": team_champ_hist,
        "champ_role_hist": champ_role_hist,
        "team_patch_hist": team_patch_hist,
        "player_patch_hist": player_patch_hist,
        "team_elo": dict(team_elo),
        "team_names": team_names,
        "team_latest": team_latest,
        "player_names": player_names,
    }
    return pd.DataFrame(rows), state


def train_and_validate(features: pd.DataFrame) -> dict:
    train = features[features["year"].isin([2024, 2025])].copy()
    valid = features[features["year"].eq(2026)].copy()
    train_x_raw = train[FEATURES].fillna(0.0).to_numpy(dtype=float)
    valid_x_raw = valid[FEATURES].fillna(0.0).to_numpy(dtype=float)
    train_y = train["target"].to_numpy(dtype=float)
    valid_y = valid["target"].to_numpy(dtype=float)
    means = train_x_raw.mean(axis=0)
    stds = train_x_raw.std(axis=0)
    stds[stds < 1e-6] = 1.0
    train_x = (train_x_raw - means) / stds
    valid_x = (valid_x_raw - means) / stds

    w = fit_logistic(train_x, train_y)
    summary = {
        "features": FEATURES,
        "intercept": float(w[0]),
        "weights": {feature: float(weight) for feature, weight in zip(FEATURES, w[1:])},
        "scaler": {"mean": {f: float(v) for f, v in zip(FEATURES, means)}, "std": {f: float(v) for f, v in zip(FEATURES, stds)}},
        "train": evaluate(train_x, train_y, w),
        "validation": evaluate(valid_x, valid_y, w),
        "formula": "p(blue win)=sigmoid(intercept + sum(weight_i * zscore(feature_i)))",
    }

    ablations = []
    variants = {
        "elo only": ["elo_diff"],
        "elo + recent form": ["elo_diff", "team_recent_diff", "side_profile_diff"],
        "roster layer": ["elo_diff", "team_recent_diff", "side_profile_diff", "player_form_diff"],
        "draft layer": FEATURES,
    }
    for name, cols in variants.items():
        idx = [FEATURES.index(c) for c in cols]
        local_means = train_x_raw[:, idx].mean(axis=0)
        local_stds = train_x_raw[:, idx].std(axis=0)
        local_stds[local_stds < 1e-6] = 1.0
        tx = (train_x_raw[:, idx] - local_means) / local_stds
        vx = (valid_x_raw[:, idx] - local_means) / local_stds
        ww = fit_logistic(tx, train_y, epochs=1600)
        metrics = evaluate(vx, valid_y, ww)
        ablations.append({"name": name, "features": cols, **{k: metrics[k] for k in ["accuracy", "log_loss", "brier", "ece"]}})
    summary["ablations"] = ablations
    return summary


def rate_payload(stat: RateStat | None, prior: float = 0.5, strength: float = 8.0) -> dict:
    if not stat:
        return {"games": 0, "wins": 0, "rate": prior}
    return {"games": stat.games, "wins": round(stat.wins, 2), "rate": round(stat.rate(prior, strength), 4)}


def build_platform_payload(features: pd.DataFrame, state: dict, model: dict) -> dict:
    team_hist = state["team_hist"]
    side_hist = state["side_hist"]
    player_hist = state["player_hist"]
    player_champ_hist = state["player_champ_hist"]
    team_champ_hist = state["team_champ_hist"]
    champ_role_hist = state["champ_role_hist"]
    team_patch_hist = state["team_patch_hist"]
    team_elo = state["team_elo"]
    team_names = state["team_names"]
    team_latest = state["team_latest"]
    player_names = state["player_names"]

    active_cutoff = pd.Timestamp("2026-01-01")
    teams = []
    for team_id, hist in team_hist.items():
        latest = team_latest.get(team_id)
        if not latest:
            continue
        if pd.Timestamp(latest["last_seen"]) < active_cutoff and hist.games < 20:
            continue
        roster = latest["roster"]
        team_champs = []
        for (tid, champ), stat in team_champ_hist.items():
            if tid == team_id and stat.games >= 2:
                team_champs.append({"champion": champ, **rate_payload(stat, strength=7)})
        team_champs.sort(key=lambda x: (x["rate"], x["games"]), reverse=True)

        player_cards = []
        current_patch_exp = []
        for player in roster:
            pid = player["playerid"]
            champ_cards = []
            for (p_id, champ), stat in player_champ_hist.items():
                if p_id == pid and stat.games >= 2:
                    champ_cards.append({"champion": champ, **rate_payload(stat, strength=5)})
            champ_cards.sort(key=lambda x: (x["rate"], x["games"]), reverse=True)
            current_patch_exp.append(
                math.log1p(
                    team_patch_hist[(team_id, latest["patch"])].games
                    + state["player_patch_hist"][(pid, latest["patch"])].games
                )
            )
            player_cards.append(
                {
                    **player,
                    "form": rate_payload(player_hist.get(pid), strength=10),
                    "champions": champ_cards[:16],
                }
            )

        teams.append(
            {
                "id": team_id,
                "name": team_names[team_id],
                "league": latest["league"],
                "last_seen": latest["last_seen"],
                "patch": latest["patch"],
                "elo": round(team_elo.get(team_id, 1500.0), 1),
                "overall": rate_payload(hist),
                "recent_rate": round(hist.recent(), 4),
                "blue_side": rate_payload(side_hist.get((team_id, "Blue")), strength=6),
                "red_side": rate_payload(side_hist.get((team_id, "Red")), strength=6),
                "roster": player_cards,
                "team_champions": team_champs[:30],
                "current_patch_experience": round(float(np.mean(current_patch_exp)) if current_patch_exp else 0.0, 4),
            }
        )
    teams.sort(key=lambda t: (pd.Timestamp(t["last_seen"]), t["elo"], t["overall"]["games"]), reverse=True)

    champions_by_role = {role: [] for role in POSITIONS}
    for (role, champ), stat in champ_role_hist.items():
        if stat.games >= 5:
            champions_by_role[role].append({"champion": champ, **rate_payload(stat, strength=14)})
    for role in POSITIONS:
        champions_by_role[role].sort(key=lambda x: (x["games"], x["rate"]), reverse=True)
        champions_by_role[role] = champions_by_role[role][:80]

    source_files = []
    for year in YEARS:
        path = RAW_DIR / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"
        source_files.append({"year": year, "file": path.name, "bytes": path.stat().st_size})

    return {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "sources": {
            "primary": "Oracle's Elixir public match data CSV",
            "download_endpoint": "https://oe.datalisk.io/matchData",
            "files": source_files,
            "notes": [
                "HLTV was not crawled because its terms prohibit data mining and web scraping.",
                "Features are calculated only from games earlier than the predicted game.",
                "This MVP predicts LoL game/map outcomes; BO series probability is simulated from game win probability.",
            ],
        },
        "dataset": {
            "rows": int(sum((RAW_DIR / f"{year}_LoL_esports_match_data_from_OraclesElixir.csv").read_text(errors="ignore").count("\n") for year in YEARS)),
            "games": int(len(features)),
            "date_min": str(features["date"].min()),
            "date_max": str(features["date"].max()),
            "train_years": [2024, 2025],
            "validation_year": 2026,
        },
        "model": model,
        "teams": teams,
        "champions_by_role": champions_by_role,
        "roles": POSITIONS,
    }


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if pd.isna(value):
        return None
    return value


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    APP_DIR.mkdir(parents=True, exist_ok=True)
    df = read_raw()
    games = build_game_objects(df)
    features, state = run_feature_pass(games)
    model = train_and_validate(features)
    payload = clean_json(build_platform_payload(features, state, model))
    features.to_csv(PROCESSED_DIR / "model_features.csv", index=False)
    (PROCESSED_DIR / "model_summary.json").write_text(json.dumps(clean_json(model), indent=2, allow_nan=False), encoding="utf-8")
    (APP_DIR / "platform_data.json").write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"games": len(features), "validation": model["validation"], "teams": len(payload["teams"])}, indent=2))


if __name__ == "__main__":
    main()
