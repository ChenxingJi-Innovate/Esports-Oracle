#!/usr/bin/env python3
"""
Daily orchestrator. The single entrypoint the cron runs each morning.

Flow:
  1. grade yesterday's open predictions against any results in data/results.json
     (match_id -> "a"|"b"),
  2. recompute the rolling accuracy scorecard,
  3. purge predictions older than the 1-year window,
  4. predict today's tier-1 CS2 + LoL slate,
  5. append today's predictions to the log,
  6. publish app/data/predictions.json (today) + app/data/scorelog.json (history)
     for the static site to render.

All state is plain JSON committed back by the GitHub Action, so history is
versioned and the model's real hit-rate is always visible.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from . import scorelog, trainer, refresh_results
from .build_schedule import schedule_3days
from .cs2_pipeline import predict as predict_cs2, _CS2_MODEL
from .lol_pipeline import predict as predict_lol, _LOL_MODEL

ROOT = Path(__file__).resolve().parents[1]
APP_DATA = ROOT / "app" / "data"
SCORELOG = APP_DATA / "scorelog.json"
PREDICTIONS = APP_DATA / "predictions.json"
SCHEDULE = APP_DATA / "schedule.json"
RESULTS = ROOT / "data" / "results.json"


def _load_results() -> dict[str, str]:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text(encoding="utf-8")).get("results", {})
    return {}


def run(today: date) -> dict:
    day = today.isoformat()
    log = scorelog.load(SCORELOG)

    # 0. AUTO-SCHEDULE: pull today's real upcoming fixtures from Liquipedia into
    #    the inputs, so we predict the actual slate instead of a stale hand-typed
    #    one. Best-effort: a failed/empty fetch leaves existing inputs untouched.
    try:
        from . import fetch_schedule
        sched = fetch_schedule.refresh(today)
        print(json.dumps({"schedule_refresh": sched}))
    except Exception as e:
        print(json.dumps({"schedule_refresh_error": str(e)}))

    # 1-2. auto-grade from Liquipedia (manual results.json overrides), + scorecard
    results = {**refresh_results.auto_results(log), **_load_results()}
    graded = scorelog.grade_pending(log, results)
    # 3. rolling window
    purged = scorelog.purge_old(log, today)

    # 4. SELF-TRAIN: refit each model on the graded 1-year window, anchored to
    #    the prior. Today's picks then use a model that only saw finished games.
    priors = {"cs2": _CS2_MODEL, "lol": _LOL_MODEL}
    models, train_meta = {}, {}
    for game, prior in priors.items():
        # train on PER-MAP rows (a 2-0 is two rows), so the model learns the
        # map-result vs odds correlation, not just series win/lose.
        model, meta = trainer.train_game(scorelog.map_training_examples(log, game), prior)
        trainer.save(game, model, meta)
        models[game], train_meta[game] = model, meta

    # 5. today's tier-1 slate, using the freshly self-trained models
    cs2 = predict_cs2(model=models["cs2"])
    # LoL picks carry OE case-based reasoning (top-5 similar matches) when both
    # teams exist in the OE case base; as_of=today keeps the kNN leakage-free.
    lol = predict_lol(model=models["lol"], as_of_date=today)

    # 6. log today's picks (dedup-safe) + refresh scorecard
    scorelog.append_predictions(log, day, "cs2", cs2)
    scorelog.append_predictions(log, day, "lol", lol)
    acc = scorelog.recompute_accuracy(log)

    # 6. publish
    APP_DATA.mkdir(parents=True, exist_ok=True)
    scorelog.save(SCORELOG, log)

    # Case-based reasoning health: the HONEST out-of-sample number for the kNN
    # similar-match layer, published so the UI never mistakes it for an edge.
    # CS2 is cheap to backtest live (small corpus); LoL's fixed-window figure is
    # static (~0.689 on a 702-game holdout, see pipeline.case_based) so we cite
    # it rather than re-running the heavy backtest every cron.
    case_based_health = {"lol": {"accuracy": 0.689, "note": "OE kNN, 702-game holdout (static)"}}
    try:
        from . import cs2_case_based
        case_based_health["cs2"] = cs2_case_based.backtest()
    except Exception as e:  # corpus missing / too small -> reasoning just absent
        case_based_health["cs2"] = {"error": str(e)}

    PREDICTIONS.write_text(json.dumps({
        "generated_for": day,
        "accuracy": acc,
        "training": train_meta,
        "calibration": {g: scorelog.calibration(log, g) for g in ("cs2", "lol")},
        "case_based": case_based_health,
        "slate": {"cs2": cs2, "lol": lol},
        "notes": [
            "Tier-1 events only (CS2 big LAN; LoL LPL/LCK/LEC/MSI/Worlds).",
            "Rolling 1-year window; older predictions auto-purged.",
            "Model self-trains daily on graded results, anchored to the prior.",
            "p_a = probability the first-listed team wins the series.",
            "Case-based 'similar matches' are transparency, not the pick: the "
            "displayed win% is the calibrated model. LoL kNN ~69% out-of-sample; "
            "CS2 kNN is provisional (small Liquipedia corpus) - see case_based.",
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # 7. publish the upcoming-fixtures schedule (reads the predictions just
    #    written for win %), grouped by date over today + next 3 days.
    SCHEDULE.write_text(json.dumps(schedule_3days(today), indent=2,
                                   ensure_ascii=False), encoding="utf-8")

    summary = {
        "date": day, "graded": graded, "purged": purged,
        "cs2_matches": len(cs2), "lol_matches": len(lol),
        "overall_accuracy": acc["overall"]["accuracy"],
        "overall_graded": acc["overall"]["graded"],
        "training": {g: {"n_train": m["n_train"], "shrinkage": m["shrinkage"]}
                     for g, m in train_meta.items()},
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat(),
                    help="run date (YYYY-MM-DD); defaults to today")
    args = ap.parse_args()
    run(date.fromisoformat(args.date))


if __name__ == "__main__":
    main()
