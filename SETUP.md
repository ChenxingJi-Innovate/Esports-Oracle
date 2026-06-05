# Esports Oracle — daily tier-1 predictor (CS2 + LoL)

A self-updating prediction site. Every morning a GitHub Action grades
yesterday's picks, purges anything older than 1 year, predicts today's
**tier-1** CS2 + LoL slate, and commits the JSON back so Vercel redeploys.

## What runs where

```
GitHub Actions (daily cron)                     Vercel (static host)
  refresh_inputs  -> Liquipedia (polite)          serves app/  ->  dist/
  daily.py        -> grade + predict + publish     reads app/data/*.json
  git commit app/data/*.json  ──auto-deploy────►   redeploys on push
```

Git is the store: `app/data/scorelog.json` (full 1-year history, versioned)
and `app/data/predictions.json` (today's slate + rolling accuracy).

## Local use

```bash
PY=python3
$PY scripts/selftest_pipeline.py       # sanity: engine + self-grading + Liquipedia
$PY -m pipeline.daily                  # produce today's predictions.json + scorelog.json
$PY -m http.server 4173 --directory app   # open http://localhost:4173
```

Edit `data/cs2_inputs.json` / `data/lol_inputs.json` to set the day's slate
(team rank / recent form / map edge / H2H). Re-run `pipeline.daily`.

## Deploy (one-time)

1. **Push to GitHub** (new repo).
2. **Vercel**: import the repo. Build command `npm run build`, output `dist/`
   (already in `vercel.json`). Deploy — the dashboard is the root page.
3. **Enable the cron**: the workflow in `.github/workflows/daily.yml` runs at
   09:00 UTC. Give Actions write permission: repo Settings → Actions → General →
   Workflow permissions → **Read and write**. Use the "Run workflow" button to
   test it once.
4. Done. Each morning the site updates itself and the accuracy scorecard grows.

## How predictions get graded (the self-improvement loop)

`daily.py` reads `data/results.json` (`{match_id: "a"|"b"}`) and marks every
matching open prediction hit/miss, then recomputes rolling accuracy + Brier,
overall and per game. Results are filled by `refresh_inputs` (see below) or by
hand for now.

## Data sources & the legal line

- **LoL history**: Oracle's Elixir CSV (already wired in `scripts/fetch_data.py`).
- **CS2 + LoL schedules/results**: **Liquipedia MediaWiki API** via
  `pipeline/sources/liquipedia.py` — compliant by construction: custom
  User-Agent + contact, ≥2s between requests (≥30s for `parse`), gzip, on-disk
  cache. CC BY-SA attribution shown in the footer.
- **HLTV is never scraped** (their terms forbid it).
- **bo3.gg**: used as the AI baseline to beat; wire its odds only if their TOS
  permits automated access.

## Roadmap (the "keeps upgrading" part)

Each item moves a number from *prior* toward *trained*:

1. **HTML→fixtures parser** in `refresh_inputs.py`: turn cached Liquipedia event
   HTML into today's tier-1 fixtures + results (auto-fills inputs + results.json).
2. **Trained CS2 model**: fit the logistic on a rolling 1-year LPDB match table
   (apply for LiquipediaDB API access) instead of the damped priors.
3. **Trained LoL upcoming**: connect the verified Oracle's Elixir logistic
   (65.3% hold-out) to live fixtures via Leaguepedia schedule + current ratings.
4. **Drop bookmaker odds entirely** — the bo3.gg failure mode (odds-anchoring)
   we deliberately avoid.

## Tests

`scripts/selftest_pipeline.py` covers the engine, the full self-grading loop
(append / dedupe / grade / accuracy / Brier / 1-year purge), and a live
Liquipedia ping. `scripts/experiment_formula.py` re-verifies the LoL backbone.
