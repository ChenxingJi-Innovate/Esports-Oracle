#!/usr/bin/env python3
"""Validate the static app data and browser-side prediction formula inputs."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "app" / "platform_data.json"


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-35, min(35, value))))


def choose(n: int, k: int) -> float:
    result = 1.0
    for i in range(1, k + 1):
        result = result * (n - k + i) / i
    return result


def series_probability(p: float, games: int) -> float:
    need = games // 2 + 1
    return sum(choose(games, wins) * p**wins * (1 - p) ** (games - wins) for wins in range(need, games + 1))


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def find_rate(records: list[dict], champion: str, fallback: float = 0.5) -> float:
    for record in records or []:
        if record["champion"] == champion:
            return record["rate"]
    return fallback


def role_player(team: dict, role: str) -> dict | None:
    for player in team.get("roster", []):
        if player["position"] == role:
            return player
    return None


def role_meta(data: dict, role: str, champion: str) -> float:
    return find_rate(data["champions_by_role"].get(role, []), champion)


def side_values(data: dict, team: dict, side: str, picks: dict[str, str]) -> dict:
    side_record = team["blue_side"] if side == "Blue" else team["red_side"]
    player_form = []
    player_champ = []
    team_champ = []
    champ_meta = []
    for role in data["roles"]:
        champion = picks[role]
        player = role_player(team, role) or {}
        player_form.append(player.get("form", {}).get("rate", 0.5))
        player_champ.append(find_rate(player.get("champions", []), champion))
        team_champ.append(find_rate(team.get("team_champions", []), champion))
        champ_meta.append(role_meta(data, role, champion))
    return {
        "recent": team["recent_rate"],
        "side": side_record["rate"],
        "player_form": average(player_form),
        "player_champ": average(player_champ),
        "team_champ": average(team_champ),
        "champ_meta": average(champ_meta),
        "patch_exp": team.get("current_patch_experience", 0.0),
    }


def main() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    teams = sorted(data["teams"], key=lambda team: team["elo"], reverse=True)
    blue, red = teams[0], teams[1]
    blue_picks = {role: (role_player(blue, role) or {}).get("champion") for role in data["roles"]}
    red_picks = {role: (role_player(red, role) or {}).get("champion") for role in data["roles"]}
    if not all(blue_picks.values()) or not all(red_picks.values()):
        raise SystemExit("Missing default roster champion picks")

    bv = side_values(data, blue, "Blue", blue_picks)
    rv = side_values(data, red, "Red", red_picks)
    raw = {
        "elo_diff": (blue["elo"] - red["elo"]) / 400,
        "team_recent_diff": bv["recent"] - rv["recent"],
        "side_profile_diff": bv["side"] - rv["side"],
        "player_form_diff": bv["player_form"] - rv["player_form"],
        "player_champion_diff": bv["player_champ"] - rv["player_champ"],
        "team_champion_diff": bv["team_champ"] - rv["team_champ"],
        "champion_meta_diff": bv["champ_meta"] - rv["champ_meta"],
        "patch_experience_diff": bv["patch_exp"] - rv["patch_exp"],
    }
    model = data["model"]
    score = model["intercept"]
    for feature in model["features"]:
        score += model["weights"][feature] * ((raw[feature] - model["scaler"]["mean"][feature]) / model["scaler"]["std"][feature])
    game_p = sigmoid(score)
    payload = {
        "blue": blue["name"],
        "red": red["name"],
        "game_probability": round(game_p, 4),
        "bo3_probability": round(series_probability(game_p, 3), 4),
        "bo5_probability": round(series_probability(game_p, 5), 4),
        "validation_accuracy": round(model["validation"]["accuracy"], 4),
        "validation_log_loss": round(model["validation"]["log_loss"], 4),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
