# Profitability results summary (trigger model)
This document captures the staged profitability experiments and the final production default wiring.
Current recommended profile: use Priority2 stable combo (`open_interest_feature, participant_positioning_feature`), which in the fresh post-wiring A/B improved cumulative net return from `0.167487` to `0.476735` (`+0.309249`).

## Goal
Target was at least **20% cumulative net return** while keeping the implementation simple and controllable.

## What was tested
All runs used the same market scope (`binanceus`, `BTC/USDT`, `1h`) and were compared against the same baseline reference.

- Baseline (item 1 baseline mode)
  - Cumulative net return: **0.369998** (~37.0%)
  - Model: `/mnt/quant-data/models/trigger-models/exchange=binanceus/symbol=BTC-USDT/interval=1h/20260617T161727Z/model.json`
- Item 4 (trade less / higher confidence gating)
  - Cumulative net return: **0.183148** (~18.3%)
  - Delta vs baseline: **-0.186850**
  - Model: `/mnt/quant-data/models/trigger-models/exchange=binanceus/symbol=BTC-USDT/interval=1h/20260617T162130Z/model.json`
- Item 5 (constrained threshold optimization)
  - Cumulative net return: **0.162912** (~16.3%)
  - Delta vs baseline: **-0.207086**
  - Model: `/mnt/quant-data/models/trigger-models/exchange=binanceus/symbol=BTC-USDT/interval=1h/20260617T162425Z/model.json`
- Item 6 (cost-aware action gate)
  - Cumulative net return: **0.162912** (~16.3%)
  - Delta vs baseline: **-0.207086**
  - Model: `/mnt/quant-data/models/trigger-models/exchange=binanceus/symbol=BTC-USDT/interval=1h/20260617T162736Z/model.json`
- Item 7 (feature-combo search over Priority2 features)
  - Search summary: `/mnt/quant-data/curated/features/external/feature_combo_search/combo_search_summary.json`
  - Best combo: `open_interest_feature + participant_positioning_feature`
  - Best cumulative net return: **0.792963** (~79.3%)
  - Target check: **PASS** (79.3% > 20%)
  - Best model: `/mnt/quant-data/models/trigger-models/exchange=binanceus/symbol=BTC-USDT/interval=1h/20260617T165027Z/model.json`
## Fresh A/B validation after production wiring
This section is a fresh apples-to-apples A/B run using the same input dataset after wiring the stable default behavior.

- Baseline A (Priority2 disabled)
  - Model: `/mnt/quant-data/models/trigger-models/exchange=binanceus/symbol=BTC-USDT/interval=1h/20260617T183421Z/model.json`
  - Accuracy: `0.398931`
  - Actionable rate: `0.044969`
  - Binary actionable precision: `0.574257`
  - Binary actionable recall: `0.043155`
  - Net expectancy per bar: `0.000071694`
  - Net expectancy per actionable: `0.001594307`
  - Cumulative net return: `0.167487` (~16.7%)
  - Selected thresholds: `buy=0.004`, `sell=0.004`, `confidence=0.55`
- Variant B (Priority2 enabled, stable combo)
  - Model: `/mnt/quant-data/models/trigger-models/exchange=binanceus/symbol=BTC-USDT/interval=1h/20260617T183704Z/model.json`
  - Feature columns: `open_interest_feature, participant_positioning_feature`
  - Accuracy: `0.420303`
  - Actionable rate: `0.131790`
  - Binary actionable precision: `0.527027`
  - Binary actionable recall: `0.123320`
  - Net expectancy per bar: `0.000178625`
  - Net expectancy per actionable: `0.001355374`
  - Cumulative net return: `0.476735` (~47.7%)
  - Selected thresholds: `buy=0.005`, `sell=0.006`, `confidence=0.60`
- Delta (Variant B minus Baseline A)
  - Accuracy: `+0.021371`
  - Actionable rate: `+0.086821`
  - Binary actionable precision: `-0.047230`
  - Binary actionable recall: `+0.080165`
  - Net expectancy per bar: `+0.000106930`
  - Net expectancy per actionable: `-0.000238933`
  - Cumulative net return: `+0.309249` (**+30.9 percentage points**)

## Plain-language takeaway
Increasing conservatism alone (items 4/5/6) reduced returns versus the baseline.  
The strongest gain came from using a focused Priority2 feature subset, specifically:
- `open_interest_feature`
- `participant_positioning_feature`

## Production default now wired
Stable default production path is now configured to use that best-performing combo by default:
- `PRIORITY2_FEATURES_ENABLED=1`
- `PRIORITY2_FEATURE_COLUMNS=open_interest_feature,participant_positioning_feature`

You can still choose different combinations when needed:
- Environment override:
  - Set `PRIORITY2_FEATURE_COLUMNS` to any comma-separated subset.
- CLI override (train/predict/monitor):
  - `--priority2-feature-columns open_interest_feature,participant_positioning_feature`
  - `--priority2-feature-columns basis_feature,funding_rate_feature,participant_positioning_feature`
  - `--priority2-feature-columns all`
  - `--priority2-feature-columns stable`
- To disable Priority2 features entirely for a run:
  - `--no-priority2-features-enabled`

## Notes on artifact diagnostics
Prediction and training artifacts now record Priority2 selection diagnostics so runs are auditable:
- selected columns
- disabled columns
- Priority2 reason codes and diagnostics
