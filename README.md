# LoL Pro Edge

Professional League of Legends esports analysis MVP built from public Oracle's Elixir match data.

## Data Sources

- Primary: Oracle's Elixir public match-data CSVs from `https://oe.datalisk.io/matchData`.
- Local years: 2024, 2025, 2026.
- Validation split: train on 2024-2025, validate on 2026.
- Not crawled: HLTV, because its terms prohibit automated scraping/data mining.

## Model

The platform predicts blue-side game win probability with a transparent logistic formula:

```text
p(blue win) = sigmoid(intercept + sum(weight_i * zscore(feature_i)))
```

All features are computed only from games earlier than the target game:

- team Elo difference
- recent team form
- side profile
- current roster player form
- player champion mastery
- team champion comfort
- role champion meta strength
- current patch experience

The app converts game probability to BO1/BO3/BO5 series probability.

## Refresh

```bash
/Users/chenxingji/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/fetch_data.py --years 2024 2025 2026
/Users/chenxingji/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/build_dataset.py
```

## Run

```bash
python3 -m http.server 4173 --directory app
```

Then open `http://localhost:4173`.
