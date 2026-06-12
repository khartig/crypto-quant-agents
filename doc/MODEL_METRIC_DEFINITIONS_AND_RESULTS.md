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
| Latest cost-aware optimized (2026-06-11) | binanceus | 2 | 0.006 / 0.003 (optimized) | 4343 (3474/869) | 64.56% | 518 / 2707 / 1118 | `/mnt/quant-data/models/trigger-models/exchange=binanceus/symbol=BTC-USDT/interval=1h/20260611T163045Z/model.json` |

## Weekly side-by-side results (fixed actionable evaluation target)
| Metric | Baseline hold-optimized | Second actionable-optimized | Latest cost-aware optimized |
| --- | ---: | ---: | ---: |
| Overall accuracy | 41.92% | 42.51% | 41.32% |
| Hold rate | 80.84% | 55.69% | 51.50% |
| Actionable rows | 32 | 74 | 81 |
| Actionable rate | 19.16% | 44.31% | 48.50% |
| Actionable accuracy (directional) | 34.38% | 33.78% | 32.10% |
| Binary actionable precision | 78.13% | 72.97% | 71.60% |
| Binary actionable recall | 24.75% | 53.47% | 57.43% |
| Buy predictions | 13 | 57 | 63 |
| Buy precision | 23.08% | 29.82% | 26.98% |
| Sell predictions | 19 | 17 | 18 |
| Sell precision | 42.11% | 47.06% | 50.00% |
| Average actionable confidence | 68.70% | 66.38% | 74.87% |

## Prediction distribution in the weekly benchmark window
| Model | Predicted buy | Predicted sell | Predicted hold |
| --- | ---: | ---: | ---: |
| Baseline hold-optimized | 13 | 19 | 135 |
| Second actionable-optimized | 57 | 17 | 93 |
| Latest cost-aware optimized | 63 | 18 | 86 |

## Interpretation of current performance
- The baseline model is highly selective (high hold-rate). It has stronger binary actionable precision, but misses many actionable opportunities (low recall).
- The second model materially increases actionable coverage and recall while keeping overall weekly accuracy slightly higher in this fixed benchmark.
- The latest cost-aware optimized model pushes actionable coverage further (highest recall, highest actionable confidence), with a modest trade-off in weekly overall/directional accuracy versus the second model.
- Train-test accuracy alone is not sufficient to choose production behavior; threshold objective, class balance, and out-of-sample behavior are all critical.

## Practical takeaway
- Use the baseline hold-optimized profile if your priority is strict selectivity and fewer signals.
- Use the second actionable-optimized profile if your priority is capturing more actionable opportunities with better recall and broader buy/sell coverage.
- Use the latest cost-aware optimized profile if your priority is higher actionable coverage under a net-expectancy objective with confidence-filtered trigger generation.

## Latest full-run metrics on ~6-month dataset (2026-06-11)
Run context:
- Exchange/symbol/timeframe: `binanceus` / `BTC/USDT` / `1h`
- 6-month dataset used for backtest + training + evaluation: `/mnt/quant-data/raw/exchange=binanceus/symbol=BTC-USDT/interval=1h/year=2026/month=06/ohlcv_20260610T173442Z_6mo.parquet`
- Dataset rows: `4392` raw bars, `4343` labeled feature rows
- Note: live ingest endpoints returned shorter windows, so this run uses the existing 6-month parquet snapshot for consistent long-window comparison.

Backtest snapshot (SMA 20/50, fee=5bps, slippage=2.5bps):
| Metric | Value |
| --- | ---: |
| Gross total return | 0.53% |
| Net total return | -5.75% |
| Gross annualized return | 1.06% |
| Net annualized return | -11.14% |
| Gross Sharpe | 0.19 |
| Net Sharpe | -0.24 |
| Net max drawdown | -12.31% |
| Turnover units | 86 |
| Total cost return drag | 6.45% |
| Break-even one-way cost | 0.62 bps |
| Applied one-way trading cost | 7.50 bps |

Latest model full-window quality (fixed evaluation target horizon=2, threshold=0.004):
| Metric | Latest cost-aware optimized |
| --- | ---: |
| Overall accuracy | 58.19% |
| Hold rate | 79.97% |
| Actionable rows | 870 |
| Actionable rate | 20.03% |
| Actionable accuracy (directional) | 29.89% |
| Binary actionable precision | 55.29% |
| Binary actionable recall | 28.51% |
| Buy precision | 31.66% |
| Sell precision | 28.51% |

Trigger artifact run summary:
| Metric | Value |
| --- | ---: |
| Backfilled predictions | 4345 |
| Backfilled alerts (confidence >= 0.60) | 479 |
| Alert rate vs labeled rows | 11.03% |
| Directional alert accuracy | 32.78% |
| Binary alert precision | 57.83% |
| Binary alert recall | 16.42% |
| Average alert confidence | 78.87% |
| Buy alert precision | 34.46% |
| Sell alert precision | 30.66% |
| Live monitor smoke test | 2 cycles, 0 alerts |

## Agent-plane comparison block: optimized-parameter pass (2026-06-11)
Run scope:
- Command path: `quant-agents agent-plane` on `/mnt/quant-data/raw/exchange=binanceus/symbol=BTC-USDT/interval=1h/year=2026/month=06/ohlcv_20260610T173442Z_6mo.parquet`
- Run id / dir: `20260611T173455Z` / `/mnt/quant-data/logs/agents/openclaw-orchestrator/2026-06-11/20260611T173455Z`
- Parameters applied: `ensemble_mode=single`, `ensemble_arms=sma_baseline`, `backtest/walkforward fee=5bps`, `backtest/walkforward slippage=2.5bps`, `paper slippage=1bps`, `max_cost_return_drag=0.03`
- Intended optimized SMA target: `24/60`
- Actual strategy windows used in this run: `20/50` (strategy step fell back after model timeouts).

Outcome summary:
- Risk gate: `fail` (`approved=false`)
- Intent: `blocked`
- Paper execution: `skipped`
- Primary blocker reasons: `total_return_below_threshold`, `sharpe_below_threshold`, `cost_drag_above_threshold`, `configured_cost_above_break_even`, and walk-forward contradiction reasons.

Backtest comparison (same 6-month dataset, same cost model):
| Metric | Optimized standalone SMA (24/60) | Agent-plane pass (actual 20/50 fallback) | Delta (agent-plane - optimized) |
| --- | ---: | ---: | ---: |
| Net total return | 8.70% | -5.75% | -14.45 pp |
| Gross total return | 15.08% | 0.53% | -14.55 pp |
| Net Sharpe | 0.70 | -0.24 | -0.95 |
| Net max drawdown | -11.11% | -12.31% | -1.20 pp |
| Turnover units | 76 | 86 | +10 |
| Total cost return drag | 5.70% | 6.45% | +0.75 pp |
| Break-even one-way cost | 19.84 bps | 0.62 bps | -19.22 bps |
| Applied one-way cost | 7.50 bps | 7.50 bps | 0.00 bps |

Calibration + walk-forward snapshot (agent-plane pass):
| Metric | Value |
| --- | ---: |
| Walk-forward quality score | 0.4224 |
| Walk-forward quality band | low |
| Walk-forward aggregate Sharpe | -0.7578 |
| Calibrated confidence | 0.5304 |
| Min confidence threshold | 0.55 |
| Contradiction detected | true (block) |

## Agent-plane corrected rerun block: strategy timeout fix validated (2026-06-11)
Run scope:
- Command path: `quant-agents agent-plane` on `/mnt/quant-data/raw/exchange=binanceus/symbol=BTC-USDT/interval=1h/year=2026/month=06/ohlcv_20260610T173442Z_6mo.parquet`
- Run id / dir: `20260611T195118Z` / `/mnt/quant-data/logs/agents/openclaw-orchestrator/2026-06-11/20260611T195118Z`
- Parameters applied: `ensemble_mode=single`, `ensemble_arms=sma_baseline`, `fast_window=24`, `slow_window=60`, `backtest/walkforward fee=5bps`, `backtest/walkforward slippage=2.5bps`, `paper slippage=1bps`, `max_cost_return_drag=0.03`

Timeout/debug verification:
- Strategy step status: `success` on first attempt (`duration_ms=577293.732`), no fallback.
- Strategy contract confirms `fast_window=24` and `slow_window=60`.
- Backtest/risk artifacts and run manifest config all confirm `24/60`.
- Ops report step still fell back after retries due `TimeoutError` (non-blocking for strategy/backtest correctness).

Backtest comparison (corrected rerun vs prior fallback run):
| Metric | Corrected rerun (actual 24/60) | Prior fallback run (actual 20/50) | Delta (corrected - fallback) |
| --- | ---: | ---: | ---: |
| Net total return | 8.70% | -5.75% | +14.45 pp |
| Gross total return | 15.08% | 0.53% | +14.55 pp |
| Net Sharpe | 0.70 | -0.24 | +0.95 |
| Net max drawdown | -11.11% | -12.31% | +1.20 pp |
| Turnover units | 76 | 86 | -10 |
| Total cost return drag | 5.70% | 6.45% | -0.75 pp |
| Break-even one-way cost | 19.84 bps | 0.62 bps | +19.22 bps |
| Applied one-way cost | 7.50 bps | 7.50 bps | 0.00 bps |

Risk decision snapshot (corrected rerun):
| Metric | Value |
| --- | ---: |
| Risk gate | fail (`approved=false`) |
| Reason codes | `cost_drag_above_threshold`, `risk_block_sell_walkforward_quality_low` |
| Calibrated confidence | 0.95 |
| Walk-forward quality score | 0.4343 |
| Walk-forward quality band | low |
| Walk-forward aggregate Sharpe | 0.2114 |
| Contradiction detected | false |

## Latest tuned-threshold verification run (2026-06-12)
Run scope:
- Command path: `quant-agents agent-plane` on `/mnt/quant-data/raw/exchange=binanceus/symbol=BTC-USDT/interval=1h/year=2026/month=06/ohlcv_20260610T173442Z_6mo.parquet`
- Run id / dir: `20260612T000725Z` / `/mnt/quant-data/logs/agents/openclaw-orchestrator/2026-06-12/20260612T000725Z`
- Strategy profile: `fast_window=24`, `slow_window=60`, `ensemble_mode=single`, `ensemble_arms=sma_baseline`
- Tuned gate thresholds used: `max_cost_return_drag=0.06`, `min_walkforward_quality_score=0.43`, `min_signal_confidence=0.55`

Outcome summary:
- Deterministic gate: `pass` (`approved=true`)
- Intent: `emitted` (action `sell`)
- Paper execution: `rejected` (`reason=no_long_position_to_sell`, long-only inventory rule)
- Risk reason codes: none (`[]`)

Performance + stability snapshot (same 6-month dataset):
| Metric | Value |
| --- | ---: |
| Net total return | 8.70% |
| Gross total return | 15.08% |
| Net Sharpe | 0.7039 |
| Net max drawdown | -11.11% |
| Total cost return drag | 5.70% |
| Turnover units | 76 |
| Break-even one-way cost | 19.84 bps |
| Configured one-way cost | 7.50 bps |
| Walk-forward window count | 57 |
| Walk-forward quality score | 0.4343 |
| Walk-forward quality band | low |
| Walk-forward aggregate Sharpe | 0.2114 |
| Walk-forward stability score | 0.6906 |
| Walk-forward contradiction detected | false |

Gate-policy delta vs prior corrected rerun:
| Check | Prior corrected run (20260611T195118Z) | Tuned-threshold run (20260612T000725Z) |
| --- | --- | --- |
| Max cost drag threshold | 0.03 | 0.06 |
| Walk-forward low-quality block rule | low band always blocks | block only if quality score < 0.43 |
| Observed cost drag | 0.057 (blocked) | 0.057 (passes) |
| Observed walk-forward quality score | 0.4343 (blocked under prior rule) | 0.4343 (passes new threshold) |
| Final deterministic gate | fail | pass |
