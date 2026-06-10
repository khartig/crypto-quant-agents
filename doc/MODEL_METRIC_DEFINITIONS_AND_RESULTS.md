# Model Metric Definitions and Results Snapshot
This document defines the actionable model-quality metrics and summarizes current model results to date.

## Scope and notation
- Recommendation classes: `buy`, `sell`, `hold`
- Actionable classes: `buy` or `sell`
- `pred`: model prediction
- `actual`: realized label
- `A_pred`: `pred in {buy, sell}`
- `A_true`: `actual in {buy, sell}`

## Metric definitions
| Metric | Plain-language definition | Formula |
| --- | --- | --- |
| Actionable accuracy (directional) | When the model predicts `buy`/`sell`, how often is the exact side correct? | `(count(pred=buy and actual=buy) + count(pred=sell and actual=sell)) / count(pred in {buy,sell})` |
| Binary actionable precision (non-directional) | When the model predicts `buy`/`sell`, how often is the market actually actionable at all? | `count(A_pred and A_true) / count(A_pred)` |
| Binary actionable recall (non-directional) | Of all truly actionable points, how many did the model flag as `buy`/`sell`? | `count(A_pred and A_true) / count(A_true)` |
| Per-class precision: buy | Of all predicted buys, how many were truly buy? | `count(pred=buy and actual=buy) / count(pred=buy)` |
| Per-class precision: sell | Of all predicted sells, how many were truly sell? | `count(pred=sell and actual=sell) / count(pred=sell)` |

Notes:
- Directional metrics penalize `buy` vs `sell` mismatches.
- Binary actionable metrics treat both `buy` and `sell` as actionable, so a `buy`/`sell` mismatch can still count as binary-actionable correct.
- If a denominator is zero (for example, no predicted sells), the metric is undefined (`N/A`).

## Evaluation setup used for side-by-side comparison
| Item | Value |
| --- | --- |
| Evaluation source dataset | `/mnt/quant-data/raw/exchange=binanceus/symbol=BTC-USDT/interval=1h/year=2026/month=06/ohlcv_20260610T173442Z_6mo.parquet` |
| Weekly window | `2026-06-03T17:00:00+00:00` to `2026-06-10T17:00:00+00:00` |
| Fixed evaluation label horizon | `2` bars |
| Fixed evaluation move threshold | `±0.004` |
| Evaluation rows | `167` |
| Weekly actual label distribution | `buy=44`, `sell=57`, `hold=66` |

Important:
- The side-by-side table below is computed on one fixed evaluation target (`horizon=2`, threshold=`0.004`) for apples-to-apples comparison.
- This is why values may differ from earlier single-model reports that used each model's own training thresholds.

## Model training snapshots to date
| Model | Exchange | Horizon | Buy/Sell threshold | Samples (train/test) | Train-test accuracy | Train label distribution (buy/hold/sell) | Model path |
| --- | --- | ---: | ---: | --- | ---: | --- | --- |
| Kraken high-accuracy (earlier phase) | kraken | 3 | 0.012 / 0.012 | 450 (360/90) | 82.22% | 25 / 385 / 40 | `/mnt/quant-data/models/trigger-models/exchange=kraken/symbol=BTC-USDT/interval=1h/20260610T163405Z/model.json` |
| Baseline hold-optimized | binanceus | 2 | 0.03 / 0.03 | 4174 (3339/835) | 98.32% | 10 / 4155 / 9 | `/mnt/quant-data/models/trigger-models/exchange=binanceus/symbol=BTC-USDT/interval=1h/20260610T175033Z/model.json` |
| Second actionable-optimized | binanceus | 2 | 0.005 / 0.005 | 4174 (3339/835) | 77.49% | 611 / 2895 / 668 | `/mnt/quant-data/models/trigger-models/exchange=binanceus/symbol=BTC-USDT/interval=1h/20260610T181601Z/model.json` |

## Weekly side-by-side results (fixed actionable evaluation target)
| Metric | Baseline hold-optimized | Second actionable-optimized |
| --- | ---: | ---: |
| Overall accuracy | 41.92% | 42.51% |
| Hold rate | 80.84% | 55.69% |
| Actionable rows | 32 | 74 |
| Actionable rate | 19.16% | 44.31% |
| Actionable accuracy (directional) | 34.38% | 33.78% |
| Binary actionable precision | 78.13% | 72.97% |
| Binary actionable recall | 24.75% | 53.47% |
| Buy predictions | 13 | 57 |
| Buy precision | 23.08% | 29.82% |
| Sell predictions | 19 | 17 |
| Sell precision | 42.11% | 47.06% |
| Average actionable confidence | 68.70% | 66.38% |

## Prediction distribution in the weekly benchmark window
| Model | Predicted buy | Predicted sell | Predicted hold |
| --- | ---: | ---: | ---: |
| Baseline hold-optimized | 13 | 19 | 135 |
| Second actionable-optimized | 57 | 17 | 93 |

## Interpretation of current performance
- The baseline model is highly selective (high hold-rate). It has stronger binary actionable precision, but misses many actionable opportunities (low recall).
- The second model materially increases actionable coverage and recall, while keeping overall weekly accuracy slightly higher in this fixed benchmark.
- Directional actionable accuracy is similar between the two models, but the second model emits much more actionable volume.
- Train-test accuracy alone is not sufficient to choose production behavior; class balance and weekly out-of-sample behavior are critical.

## Practical takeaway
- Use the baseline hold-optimized profile if your priority is strict selectivity and fewer signals.
- Use the second actionable-optimized profile if your priority is capturing more actionable opportunities with better recall and broader buy/sell coverage.
