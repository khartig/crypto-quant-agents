# Live Readiness Runbook
## Purpose
This runbook defines staged rollout criteria and rollback triggers for promoting from shadow evaluation to limited live-capital deployment.

## Stages
### 1) Shadow
Scope:
- Evaluate benchmark and manifest evidence only.
- No capital allocation.

Gate intent:
- Benchmark gate must pass.
- Operational incident rate must remain controlled.

### 2) Paper-forward
Scope:
- Execute deterministic paper intents/executions over rolling windows.
- Validate execution quality under realistic assumptions.

Gate intent:
- Shadow checks remain green.
- Sufficient paper run count for stability.
- Execution success and fill quality exceed minimum thresholds.

### 3) Limited live
Scope:
- Controlled notional exposure after sustained paper-forward stability.
- Strict rollback automation and incident handling.

Gate intent:
- Paper-forward checks remain green.
- Drawdown stays below configured limit.
- Realized PnL and benchmark cost-drag expectations remain acceptable.

## Required operational SLOs
- Benchmark gate status: `pass`
- Paper execution success rate: at or above policy threshold
- Average fill ratio: at or above policy threshold
- Incident rate: at or below policy threshold
- Max drawdown ratio: at or below policy threshold
- Realized paper PnL delta: at or above policy threshold

Thresholds are controlled through `scripts/run_live_readiness_stage_gate.py` arguments.

## Rollback triggers
Rollback to the previous stage immediately if any of these occur:
- Benchmark gate turns non-pass.
- Incident rate breaches threshold.
- Drawdown exceeds stage limit.
- Execution success/fill quality falls below stage minimums.
- Critical artifact contract failures or missing manifests.

## Stage-gate dashboard command
Use this command to generate readiness artifacts:

`python scripts/run_live_readiness_stage_gate.py --exchange <exchange> --symbol <symbol> --timeframe <timeframe> --required-stage paper_forward`

Optional strict enforcement:

`python scripts/run_live_readiness_stage_gate.py --exchange <exchange> --symbol <symbol> --timeframe <timeframe> --required-stage limited_live --fail-on-unmet-stage`

## Checklist
Before promoting stages, confirm:
- Benchmark summary and manifest artifacts are available.
- Latest execution realism stress report generated.
- Readiness report recommends target stage or higher.
- No unresolved blockers in readiness report.
- Rollback owner and on-call path confirmed for current promotion window.
