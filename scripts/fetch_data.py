#!/usr/bin/env python3
"""Fetch public Oracle's Elixir LoL match-data CSVs used by the platform."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
API_URL = "https://oe.datalisk.io/matchData"
API_KEY = "f561197a-82ea-4e54-acd2-386979018a7a"


def request_json(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"X-Api-Key": API_KEY, "User-Agent": "Codex-LoL-Analytics-MVP/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def canonical_s3_url(url: str) -> str:
    filename = url.rsplit("/", 1)[-1]
    return f"https://s3.amazonaws.com/oracles-elixir/{filename}"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "Codex-LoL-Analytics-MVP/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2025, 2026])
    args = parser.parse_args()

    links = request_json(API_URL)
    (RAW_DIR / "match_data_links.json").write_text(json.dumps(links, indent=2), encoding="utf-8")
    by_year = {item["year"]: item for item in links}
    for year in args.years:
        if year not in by_year:
            raise SystemExit(f"No Oracle's Elixir match-data file for {year}")
        item = by_year[year]
        destination = RAW_DIR / item["name"]
        print(f"Downloading {item['name']} ({item['games']} games, updated {item['updatedAt']})")
        download(canonical_s3_url(item["link"]), destination)
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
