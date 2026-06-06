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
from datetime import date, timedelta
from pathlib import Path

from . import scorelog, trainer, refresh_results
from .build_schedule import schedule_3days
from .cs2_pipeline import predict as predict_cs2, _CS2_MODEL
from .lol_pipeline import predict as predict_lol, _LOL_MODEL
from .val_pipeline import predict as predict_val, _VAL_MODEL

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


def _recent_results(log: dict, today: date, days: int = 3) -> dict[str, list]:
    """Graded predictions from the last `days` days, grouped by game, newest
    first. This is exactly the set the model self-trains on, surfaced for the
    user's 'past results' view."""
    cutoff = (today - timedelta(days=days)).isoformat()
    out: dict[str, list] = {"cs2": [], "lol": [], "val": []}
    for p in log.get("predictions", []):
        if p.get("correct") is None or p.get("date", "") < cutoff:
            continue
        g = p.get("game")
        if g in out:
            out[g].append(p)
    for g in out:
        out[g].sort(key=lambda r: r["date"], reverse=True)
    return out


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
    priors = {"cs2": _CS2_MODEL, "lol": _LOL_MODEL, "val": _VAL_MODEL}
    models, train_meta = {}, {}
    for game, prior in priors.items():
        # train on PER-MAP rows (a 2-0 is two rows), so the model learns the
        # map-result vs odds correlation, not just series win/lose.
        model, meta = trainer.train_game(scorelog.map_training_examples(log, game), prior)
        trainer.save(game, model, meta)
        models[game], train_meta[game] = model, meta

    # 5. today's tier-1 slate, using the freshly self-trained models. Build each
    # FPS case base ONCE and reuse it for both the pick and the health backtest
    # (instead of load_case_base rebuilding the corpus in each step).
    from . import cs2_case_based
    VAL_MATCHES = ROOT / "data" / "processed" / "val_matches.csv"
    cs2_cb = cs2_case_based.load_case_base()
    val_cb = cs2_case_based.load_case_base(matches_csv=VAL_MATCHES)
    cs2 = predict_cs2(model=models["cs2"], case_base=cs2_cb)
    # LoL picks carry OE case-based reasoning (top-5 similar matches) when both
    # teams exist in the OE case base; as_of=today keeps the kNN leakage-free.
    lol = predict_lol(model=models["lol"], as_of_date=today)
    val = predict_val(model=models["val"], case_base=val_cb)

    # 6. log today's picks (dedup-safe) + refresh scorecard
    scorelog.append_predictions(log, day, "cs2", cs2)
    scorelog.append_predictions(log, day, "lol", lol)
    scorelog.append_predictions(log, day, "val", val)
    acc = scorelog.recompute_accuracy(log)

    # 6. publish
    APP_DATA.mkdir(parents=True, exist_ok=True)
    scorelog.save(SCORELOG, log)

    # Case-based reasoning health: the HONEST out-of-sample number for the kNN
    # similar-match layer, published so the UI never mistakes it for an edge.
    # CS2 is cheap to backtest live (small corpus); LoL's fixed-window figure is
    # static (~0.689 on a 702-game holdout, see pipeline.case_based) so we cite
    # it rather than re-running the heavy backtest every cron.
    from .case_based import OE_HOLDOUT_ACCURACY
    case_based_health = {"lol": {"accuracy": OE_HOLDOUT_ACCURACY,
                                 "note": "OE kNN, 702-game holdout (static)"}}
    # Only backtest a game when its OWN case base built; a missing/empty corpus
    # must NOT fall through to another game's (backtest's case_base=None default
    # would otherwise reload the CS2 corpus and mislabel it).
    for game, cb in (("cs2", cs2_cb), ("val", val_cb)):
        try:
            case_based_health[game] = (cs2_case_based.backtest(case_base=cb) if cb
                                       else {"error": "corpus pending (no case base yet)"})
        except Exception as e:
            case_based_health[game] = {"error": str(e)}

    # Past-3-days graded results, per game, for the "results" UI tab. Reads the
    # same scorelog the model self-trains on, so what the user checks is exactly
    # what tuned the model.
    results_3d = _recent_results(log, today, days=3)

    PREDICTIONS.write_text(json.dumps({
        "generated_for": day,
        "window_days": 2,                 # slate covers today + next 2 days
        "accuracy": acc,
        "training": train_meta,
        "calibration": {g: scorelog.calibration(log, g) for g in ("cs2", "lol", "val")},
        "case_based": case_based_health,
        "slate": {"cs2": cs2, "lol": lol, "val": val},
        "results": results_3d,            # past 3 days graded, per game
        "notes": [
            "Tier-1 events only (CS2 LAN; LoL LPL/LCK/etc.; Valorant VCT).",
            "Auto-fetched from Liquipedia daily at 00:00 Beijing (16:00 UTC).",
            "Rolling 1-year window; older predictions auto-purged.",
            "Model self-trains daily on graded results, anchored to the prior.",
            "p_a = probability the first-listed team wins the series.",
            "Case-based 'similar matches' are transparency, not the pick: the "
            "displayed win% is the calibrated model. LoL kNN ~69% out-of-sample; "
            "CS2/Valorant kNN provisional (small Liquipedia corpus) - see case_based.",
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # 7. publish the upcoming-fixtures schedule (reads the predictions just
    #    written for win %), grouped by date over today + next 3 days.
    SCHEDULE.write_text(json.dumps(schedule_3days(today), indent=2,
                                   ensure_ascii=False), encoding="utf-8")

    summary = {
        "date": day, "graded": graded, "purged": purged,
        "cs2_matches": len(cs2), "lol_matches": len(lol), "val_matches": len(val),
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
