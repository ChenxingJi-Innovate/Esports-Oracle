#!/usr/bin/env python3
"""Self-test the shared engine + self-grading loop, then live-ping Liquipedia."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.predictor import LinearModel, best_of, confidence_band
from pipeline import scorelog, trainer
from pipeline.sources import liquipedia


def _prior():
    feats = ["rating_diff", "form_diff", "map_diff", "player_diff", "h2h_diff"]
    return LinearModel(
        intercept=0.0,
        weights={"rating_diff": 0.35, "form_diff": 0.22, "map_diff": 0.17,
                 "player_diff": 0.17, "h2h_diff": 0.11},
        mean={f: 0.0 for f in feats},
        std={"rating_diff": 1.0, "form_diff": 0.15, "map_diff": 0.12,
             "player_diff": 0.12, "h2h_diff": 0.80},
    )


def test_engine():
    m = LinearModel(
        intercept=0.0,
        weights={"rating_diff": 0.35, "form_diff": 0.22},
        mean={"rating_diff": 0.0, "form_diff": 0.0},
        std={"rating_diff": 1.0, "form_diff": 0.15},
    )
    p = m.map_prob({"rating_diff": 0.318, "form_diff": 0.08})
    assert 0.5 < p < 0.7, p
    assert abs(best_of(0.5, 3) - 0.5) < 1e-9
    assert best_of(0.61, 3) > 0.61          # favorite gains in a BO3
    assert best_of(0.61, 1) == 0.61
    assert confidence_band(0.5, "BO3") == "coin-flip"
    assert confidence_band(0.85, "BO3") == "strong"
    print(f"engine ok: per-map={p:.3f}  BO3={best_of(p,3):.3f}  band={confidence_band(best_of(p,3),'BO3')}")


def test_self_grading():
    log = {"predictions": [], "accuracy": {}}
    day = "2026-06-04"
    scorelog.append_predictions(log, day, "cs2", [
        {"match_id": "m1", "event": "IEM Cologne", "team_a": "M80", "team_b": "NRG",
         "p_a": 0.661, "fmt": "BO3", "confidence": "moderate"},
        {"match_id": "m2", "event": "IEM Cologne", "team_a": "Liquid", "team_b": "HEROIC",
         "p_a": 0.565, "fmt": "BO3", "confidence": "lean"},
    ])
    assert len(log["predictions"]) == 2
    # dedupe: re-appending the same ids does nothing
    scorelog.append_predictions(log, day, "cs2", [
        {"match_id": "m1", "event": "x", "team_a": "M80", "team_b": "NRG", "p_a": 0.9}])
    assert len(log["predictions"]) == 2
    # grade: M80 won (a), HEROIC won (b) -> 1 correct, 1 wrong
    n = scorelog.grade_pending(log, {"m1": "a", "m2": "b"})
    assert n == 2
    acc = scorelog.recompute_accuracy(log)
    assert acc["overall"]["graded"] == 2
    assert acc["overall"]["accuracy"] == 0.5
    purged = scorelog.purge_old(log, date(2026, 6, 5))
    assert purged == 0                       # both within a year
    purged = scorelog.purge_old(log, date(2027, 7, 1))
    assert purged == 2                        # now older than a year
    print(f"self-grading ok: acc={acc['overall']['accuracy']}  brier={acc['overall']['brier']}")


def test_self_training():
    prior = _prior()
    # cold start: 3 examples -> stays ~ prior (shrinkage low)
    few = [{"features": {"rating_diff": 0.3, "form_diff": 0.05, "map_diff": 0, "player_diff": 0, "h2h_diff": 0.3}, "label": 1}] * 3
    _, meta_few = trainer.train_game(few, prior)
    assert meta_few["shrinkage"] < 0.1, meta_few

    # lots of signal: team_a wins iff rating_diff>0. Symmetric range so the
    # learned intercept stays ~0. Deterministic, no RNG.
    rng = [(i - 100) / 200.0 for i in range(201)]    # -0.5 .. +0.5
    many = [{"features": {"rating_diff": r, "form_diff": 0.0, "map_diff": 0.0,
                          "player_diff": 0.0, "h2h_diff": 0.0},
             "label": 1 if r > 0 else 0} for r in rng]
    model, meta = trainer.train_game(many, prior)
    assert meta["shrinkage"] > 0.8, meta
    assert meta["in_sample_acc"] > 0.85, meta
    # learned the signal: monotonic, big margin between +0.4 and -0.4
    hi = model.map_prob({"rating_diff": 0.4, "form_diff": 0, "map_diff": 0, "player_diff": 0, "h2h_diff": 0})
    lo = model.map_prob({"rating_diff": -0.4, "form_diff": 0, "map_diff": 0, "player_diff": 0, "h2h_diff": 0})
    prior_hi = prior.map_prob({"rating_diff": 0.4, "form_diff": 0, "map_diff": 0, "player_diff": 0, "h2h_diff": 0})
    prior_lo = 1 - prior_hi
    # learned: monotonic, sharpened well past the prior's separation
    assert hi > 0.6 and lo < 0.4 and (hi - lo) > 0.2, (hi, lo)
    assert (hi - lo) > 2 * (prior_hi - prior_lo), (hi - lo, prior_hi - prior_lo)
    print(f"self-training ok: few(n=3) shrink={meta_few['shrinkage']}  "
          f"many(n=200) shrink={meta['shrinkage']} acc={meta['in_sample_acc']}  "
          f"p(+0.4)={hi:.2f} vs prior {prior_hi:.2f}, p(-0.4)={lo:.2f}")


def test_liquipedia_live():
    ok = liquipedia.page_exists("counterstrike", "Intel Extreme Masters/2026/Cologne/Stage 1")
    html = liquipedia.page_html("counterstrike", "Intel Extreme Masters/2026/Cologne/Stage 1")
    print(f"liquipedia ok: page_exists={ok}  html_chars={len(html)}  (cached for reuse)")


if __name__ == "__main__":
    test_engine()
    test_self_grading()
    test_self_training()
    try:
        test_liquipedia_live()
    except Exception as e:  # network/ratelimit should not fail the unit tests
        print(f"liquipedia live check skipped: {e}")
    print("\nALL CORE TESTS PASSED")
