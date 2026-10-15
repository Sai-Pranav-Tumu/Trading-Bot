# Changelog

## 2026-10-14 - Phase 1 protocol started

- Added a sealed-fold protocol for walk-forward testing.
- Latest 2 folds are reserved for one final evaluation and must not guide development.
- Development comparisons use the older 7 folds only.
- Success is defined in advance: a change counts only if it beats cash and the same-exposure passive benchmark in at least 6 of 7 development folds.
- Phase 1 is limited to at most 3 experiments.
- Experiment A: metrics, timestamped results, and this changelog.
- Experiment B: predefined top-25 ensemble vs champion.
- Experiments C and D are intentionally not started yet.

Result:
- Development run saved to `results/walkforward_dev_20261014_054555.csv`.
- Summary saved to `results/summary_dev_20261014_054555.txt`.
- Champion success: 0/7 development folds.
- Evolved population-median success: 0/7 development folds.
- Top-25 ensemble success: 0/7 development folds.
- Honest verdict: Experiment B failed the predefined 6/7 success rule.

## 2026-10-14 - Diagnostic and Phase 2 collector scaffold

- Added `diagnose_costs.py` for diagnostic-only gross vs net reporting on development folds.
- Final folds remain sealed; final mode was not run.
- Gross/net diagnostic saved to `results/cost_diagnostic_dev_20261014_055534.csv`.
- Added `collect_daily_data.py` for point-in-time Phase 2 data collection.
- Collector supports Fear & Greed history, paginated Binance funding-rate backfill, forward-only headline collection, raw response storage, `available_at_utc`, `collected_at_utc`, and `code_version`.
- Endpoint check: Binance futures funding and Alternative.me Fear & Greed were reachable from this environment when run with network approval.
- Collector smoke test appended one Fear & Greed row and one BTCUSDT funding row. GDELT headline collection returned HTTP 429, so news reachability is not assumed.

## 2026-10-14 - Phase 2 experiments pre-registered

Final folds remain sealed. These experiments must run only on the 7 development folds until development is frozen.

Shared rules:
- A bot at decision time `T` may only use external-data rows with `available_at_utc <= T - 1 hour`.
- External features are standardized using historical training data only inside each fold, never recent or holdout/test data.
- Re-run the price-only baseline with identical seeds/settings in the same run before comparing any Phase 2 feature.
- Report both gross and net results, average trades, estimated cost, max drawdown, Sharpe-like ratio, exposure, same-exposure passive benchmark, and per-fold bootstrap confidence intervals.
- Success rule: an experiment counts only if it beats cash and beats the same-exposure passive benchmark in at least 6 of 7 development folds.
- Stop rule: if both pre-registered Phase 2 experiments fail, do not run more feature experiments on these bots/folds.
- Add one shuffled-feature control for each feature. The control uses the same feature values shuffled across days. These controls are not additional experiments; they are sanity checks for leakage/overfitting.

Experiment P2-A: funding-rate feature.
- Formula: for coin `C` and decision time `T`, `funding(C,T)` is the mean Binance funding rate settlements for `CUSDT` in the 24 hours ending at `T - 1 hour`, using only rows with `available_at_utc <= T - 1 hour`.
- If no funding settlements are available in that window, use 0 for that coin/day before standardization.
- The feature uses the existing placeholder sentiment slot; no other strategy rule changes.
- Control P2-A-shuffled: same funding feature matrix, shuffled across days with a fixed seed before training/testing.

Experiment P2-B: Fear & Greed feature.
- Formula: for all coins at decision time `T`, `fear_greed(T)` is the latest Alternative.me Fear & Greed value divided by 100 using only rows with `available_at_utc <= T - 1 hour`, forward-filled only up to a max age of 3 days. Older values become neutral 0 before standardization.
- The feature uses the existing placeholder sentiment slot; no other strategy rule changes.
- Control P2-B-shuffled: same Fear & Greed feature vector, shuffled across days with a fixed seed before training/testing.

## 2026-10-14 - Collector hardening, full backfill, and Phase 2 code

- Enforced/documented the 1-hour external-data safety lag: feature code may only use rows where `available_at_utc <= T - 1 hour`.
- Made collector writes idempotent by dataset keys.
- Added failure logging to `data/collector_runs.jsonl`.
- Added retry/backoff for GDELT and added RSS headline collection from Coindesk, Cointelegraph, and Decrypt.
- Added `COLLECTOR_SCHEDULE.md` with Windows Task Scheduler instructions.
- Ran full approved backfill for Binance funding rates and Alternative.me Fear & Greed.
- Funding backfill results:
  - ADAUSDT: 5476 rows, 2021-10-15T00:00:00+00:00 to 2026-10-14T00:00:00+00:00, 0 gaps over 12h.
  - BNBUSDT: 5476 rows, 2021-10-15T00:00:00+00:00 to 2026-10-14T00:00:00+00:00, 0 gaps over 12h.
  - BTCUSDT: 5476 rows, 2021-10-15T00:00:00+00:00 to 2026-10-14T00:00:00+00:00, 0 gaps over 12h.
  - ETHUSDT: 5476 rows, 2021-10-15T00:00:00+00:00 to 2026-10-14T00:00:00+00:00, 0 gaps over 12h.
  - SOLUSDT: 5551 rows, 2021-10-15T00:00:00+00:00 to 2026-10-14T00:00:00+00:00, 0 gaps over 12h.
  - XRPUSDT: 5476 rows, 2021-10-15T00:00:00+00:00 to 2026-10-14T00:00:00+00:00, 0 gaps over 12h.
- Fear & Greed backfill result: 3174 rows, 2018-02-01T00:00:00+00:00 to 2026-10-14T00:00:00+00:00, 2 gaps over 1.5 days, max gap 4 days.
- Idempotency check: latest Fear & Greed and BTCUSDT funding rows were skipped as duplicates on re-run.
- News check: RSS appended 88 headline rows; GDELT still failed on verification and remains logged as unreliable.
- Added `phase2_experiments.py` for the two pre-registered Phase 2 experiments.
- Phase 2 experiments were implemented but not run. Final folds remain sealed.

## 2026-10-14 - Point-in-time audit hardening before Phase 2

- Conservative Fear & Greed availability changed to `available_at_utc = value_date_utc + 1 day`.
- Existing `data/fear_greed.csv` backfill was updated to the same conservative availability rule.
- Price row mapping documented in audit: Binance daily candle date `D` closes at `D+1 00:00 UTC`, so external-data decision time is `T = D+1 00:00 UTC`, then the 1-hour safety lag is applied.
- Fear & Greed stale fallback rule confirmed in code: use raw `0.5` before standardization when no value is available or the latest value is older than 3 days.
- Funding edge-case rule confirmed in code: require at least 2 funding settlements in the 24h window, otherwise use raw neutral `0.0`.
- Added `audit_point_in_time.py` to print selected rows and assert the cutoff, stale fallback, funding edge cases, and future-poisoning behavior.
- Added baseline and shuffled-control modes to `phase2_experiments.py`; experiments still have not been run.

## 2026-10-14 - Final Phase 2 run rules frozen

- Primary metric: top-25 ensemble net return.
- A Phase 2 feature passes only if the top-25 ensemble beats cash and same-exposure passive in at least 6 of 7 development folds.
- Champion and population median are reported for context only and are not decisive.
- A real feature only counts as a pass if it also clearly beats its own shuffled-feature control.
- Each shuffled control is run with 5 fixed shuffle seeds: 101, 202, 303, 404, 505.
- Baseline, real features, and controls use identical evolution seeds/settings: seeds 1..3, 60 generations, 8 windows, 9 total folds with latest 2 sealed.
- No tuning after seeing this run's results.
- Final mode remains forbidden until explicitly approved.
- Frozen file/data hashes before the one Phase 2 dev run:
  - `phase2_experiments.py` sha256 `089a2a1236417038cbdce4aec10e192a86348a7d5c63ce994477cc2bf449704f`
  - `audit_point_in_time.py` sha256 `997896e5605a5f7d61d64f0fd3657a8bbafb1aaeefbd03128db6b6ef15f0df7a`
  - `data/funding_rates.csv` sha256 `9df0214ae82e6bb296a278a29e282d72a3134c1e3b530ec94dbe6428015cc3e0`
  - `data/fear_greed.csv` sha256 `2b92eea4e113ea5c8f803bdd3509817b43f8a62192003fc3fcf688c84320ea56`
  - `prices_cache.csv` sha256 `aed19dc9fae83e29b70dd2a54b286d398fd14a1addd711d358c6a46adc4e0b2b`

Result of one frozen Phase 2 development-fold run:
- Baseline top-25 ensemble: 0/7 success folds, mean net -2.35%.
- Funding feature: 0/7 success folds, mean net -2.45%.
- Fear & Greed feature: 0/7 success folds, mean net -2.82%.
- Funding shuffled controls: success-fold range 0..1, mean-net range -3.83%..-1.86%.
- Fear & Greed shuffled controls: success-fold range 0..2, mean-net range -2.63%..-1.17%.
- Honest verdict: both real Phase 2 features failed the pre-registered 6/7 rule and did not clearly beat controls.
- Stop rule triggered for these bots/folds: no more feature experiments on this bot design and fold set.
- Final mode was not run.
