#!/usr/bin/env python3
"""
Derive a leakage-free CS2 feature table from the scraped match corpus.

LoL gets engineered features for free from Oracle's Elixir; CS2 does not, so we
compute the equivalent here from raw results (cs2_corpus.py -> cs2_matches.csv).
Walking matches in chronological order, we maintain per-team running state and,
for each match, emit the features known STRICTLY BEFORE that match — then update
state with the outcome. That ordering is the whole leakage guarantee: a match's
own result never informs its own feature row.

Features (team_a minus team_b, so positive = team_a favored):
    elo_diff        - classic Elo (1500 start, K=32), the backbone strength gap
    form_diff       - win rate over each team's last 10 maps (0.5 prior)
    h2h_diff        - team_a's win rate in prior meetings vs team_b (centered, 0 prior)
    rounddiff_diff  - recent avg round margin per map over last 10 (momentum/dominance)

Target: 1 if team_a won the map, else 0.

Output: data/processed/cs2_features.csv
    date, ts, team_a, team_b, target, elo_diff, form_diff, h2h_diff, rounddiff_diff, event
"""
from __future__ import annotations

import csv
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATCHES_CSV = ROOT / "data" / "processed" / "cs2_matches.csv"
OUT_CSV = ROOT / "data" / "processed" / "cs2_features.csv"

FEATURE_COLS = ["elo_diff", "form_diff", "h2h_diff", "rounddiff_diff"]

ELO_START = 1500.0
ELO_K = 32.0
FORM_WINDOW = 10


def _norm(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def build(matches_csv: Path | None = None, return_state: bool = False):
    """Feature rows in chronological order. With return_state=True, also return
    the FINAL per-team running state (elo/form/margin/h2h) after the last match,
    used to derive features for upcoming (live) matches."""
    path = matches_csv or MATCHES_CSV
    if not path.exists():
        return ([], {}) if return_state else []
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows.sort(key=lambda r: int(r["ts"]))

    elo: dict[str, float] = defaultdict(lambda: ELO_START)
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))      # 1/0 win flags
    margin: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))    # signed round diffs
    h2h: dict[tuple, list] = defaultdict(lambda: [0, 0])                         # (na,nb) -> [a_wins, total]

    out: list[dict] = []
    for r in rows:
        a, b = r["team_a"], r["team_b"]
        na, nb = _norm(a), _norm(b)
        if not na or not nb or na == nb:
            continue
        try:
            sa = int(r["score_a"]); sb = int(r["score_b"])
        except (TypeError, ValueError):
            sa = sb = None
        a_won = 1 if _norm(r["winner"]) == na else 0

        # ---- features AS OF before this match (no leakage) ----
        form_a = sum(form[na]) / len(form[na]) if form[na] else 0.5
        form_b = sum(form[nb]) / len(form[nb]) if form[nb] else 0.5
        md_a = sum(margin[na]) / len(margin[na]) if margin[na] else 0.0
        md_b = sum(margin[nb]) / len(margin[nb]) if margin[nb] else 0.0
        h = h2h[(na, nb)]
        h2h_a = (h[0] / h[1] - 0.5) if h[1] else 0.0

        out.append({
            "date": r["date"], "ts": r["ts"], "team_a": a, "team_b": b,
            "target": a_won,
            "elo_diff": round(elo[na] - elo[nb], 2),
            "form_diff": round(form_a - form_b, 4),
            "h2h_diff": round(h2h_a, 4),
            "rounddiff_diff": round(md_a - md_b, 3),
            "event": r.get("event", ""),
        })

        # ---- update running state with the outcome ----
        ea = _expected(elo[na], elo[nb])
        elo[na] += ELO_K * (a_won - ea)
        elo[nb] += ELO_K * ((1 - a_won) - (1 - ea))
        form[na].append(a_won); form[nb].append(1 - a_won)
        if sa is not None and sb is not None:
            margin[na].append(sa - sb); margin[nb].append(sb - sa)
        h2 = h2h[(na, nb)]; h2[0] += a_won; h2[1] += 1
        hr = h2h[(nb, na)]; hr[0] += (1 - a_won); hr[1] += 1

    if return_state:
        return out, {"elo": dict(elo), "form": form, "margin": margin, "h2h": h2h}
    return out


def live_features(team_a: str, team_b: str, state: dict) -> dict[str, float] | None:
    """Derive the four diff-features for an upcoming match from the corpus's
    final state. Returns None if neither team is known to the corpus (no basis
    for case-based reasoning); a single known team still yields a usable row via
    the ELO_START / neutral priors for the unknown side."""
    na, nb = _norm(team_a), _norm(team_b)
    elo, form, margin, h2h = state["elo"], state["form"], state["margin"], state["h2h"]
    if na not in elo and nb not in elo:
        return None

    def _form(n):
        d = form.get(n)
        return sum(d) / len(d) if d else 0.5

    def _margin(n):
        d = margin.get(n)
        return sum(d) / len(d) if d else 0.0

    h = h2h.get((na, nb))
    h2h_a = (h[0] / h[1] - 0.5) if h and h[1] else 0.0
    return {
        "elo_diff": round(elo.get(na, ELO_START) - elo.get(nb, ELO_START), 2),
        "form_diff": round(_form(na) - _form(nb), 4),
        "h2h_diff": round(h2h_a, 4),
        "rounddiff_diff": round(_margin(na) - _margin(nb), 3),
    }


def write_csv(rows: list[dict]) -> Path:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["date", "ts", "team_a", "team_b", "target", *FEATURE_COLS, "event"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return OUT_CSV


if __name__ == "__main__":
    rows = build()
    if not rows:
        print("No CS2 matches found. Run `python -m pipeline.cs2_corpus` first.")
    else:
        path = write_csv(rows)
        print(f"Wrote {len(rows)} feature rows -> {path}")
        print(f"date range: {rows[0]['date']} .. {rows[-1]['date']}")
