# Profitability optimization flow
## Purpose
This runbook defines the standard optimization loop for maximizing **net profitability** while controlling drawdown, execution quality, and deployment risk.
Use this flow for every new candidate model/profile before any promotion to paper-forward or limited live stages.

## Step 1 — Define promotion objective (profit-first, risk-constrained)
Primary ranking metrics:
- `execution_backtest_realized_pnl_delta_usd`
- `execution_backtest_equity_return`
- `net_expectancy_per_actionable`

Hard constraints (must pass):
- drawdown within policy thresholds
- cost drag within policy thresholds
- fill/execution quality above minimum thresholds
- incident/readiness thresholds within stage-gate limits

Selection rule:
- rank by robust performance across windows/splits (median + worst-window), not by single best run.

## Step 2 — Sweep trigger/labeling parameters
Goal:
- find labeling/threshold settings that improve execution-aware expectancy and realized PnL quality.

Command:
- `python scripts/run_labeling_objective_benchmark.py --exchange <exchange> --symbol <symbol> --timeframe <timeframe> --scenario-set extended`

Outputs:
- benchmark JSON/markdown summary
- actionable coverage vs precision frontier plot

## Step 3 — Sweep feature sets (ablation matrix)
Goal:
- measure incremental profitability and robustness of feature bundles vs baseline.

Command:
- `python scripts/run_ranked_feature_ablation.py --config scripts/ranked_feature_ablation_plan.json`

Outputs:
- split-level scenario results
- ranked scenario summary by profitability metrics
- baseline deltas for attribution

## Step 4 — Regime robustness + cost stress
Goal:
- verify candidate behavior remains acceptable under regime shifts and transaction-cost stress.

Command:
- `python scripts/evaluate_agent_regime_windows.py --exchange <exchange> --symbol <symbol> --timeframe <timeframe> --enable-cost-stress --cost-stress-multiplier 1.5 --cost-stress-multiplier 2.0 --cost-stress-multiplier 3.0`

Outputs:
- per-window profile outcomes
- cost-stress sensitivity summary
- cost-drag decomposition by profile/regime bucket/arm

## Step 5 — Final readiness gates
Goal:
- confirm execution realism behavior and deployment readiness before promotion.

Commands:
- `python scripts/run_execution_realism_stress_suite.py --exchange <exchange> --symbol <symbol> --timeframe <timeframe> --enforce-gate`
- `python scripts/run_live_readiness_stage_gate.py --exchange <exchange> --symbol <symbol> --timeframe <timeframe> --required-stage paper_forward`

Outputs:
- execution realism stress report (gateable)
- readiness stage-gate dashboard (pass/fail with blockers)

## Promotion decision protocol
Promote only if all are true:
- Step 2 and 3 produce a candidate with positive net lift vs baseline.
- Step 4 shows stable risk-adjusted behavior under stress.
- Step 5 gate checks pass for the target deployment stage.

Otherwise:
- keep current production profile unchanged,
- use failed-gate diagnostics to define the next experiment batch.

