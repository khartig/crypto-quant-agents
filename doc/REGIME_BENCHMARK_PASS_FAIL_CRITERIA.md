# REGIME_BENCHMARK_PASS_FAIL_CRITERIA
## Purpose
Defines the Priority 0 go/no-go gate for strategy/model changes before promotion.
Use this with `scripts/run_regime_benchmark_gate.py`.

## Mandatory artifacts
Every benchmark run must emit and retain:
- dataset manifest JSON with source hashes and min/max timestamps
- benchmark summary JSON
- benchmark markdown summary
- metric snapshot JSON
- per-window profile artifacts for both `regime_enabled` and `regime_ablated`

## Required profiles and windows
- Required profiles: `regime_enabled`, `regime_ablated`
- Canonical windows are sourced from `scripts/regime_window_slices.json`
- All priority windows must pass minimum-bar coverage checks (no skipped priority windows)

## Required core metrics
Each benchmark report must include, for both profiles:
- approval rate
- contradiction rate
- net total return
- sharpe
- max drawdown
- total cost return drag

## Baseline delta checks (go/no-go)
Model changes are evaluated against `doc/REGIME_BENCHMARK_BASELINE.json` using the `regime_enabled` profile:
- net total return delta must be `>= 0.0`
- max drawdown delta must be `>= 0.0` (less negative or improved)
- total cost return drag delta must be `<= 0.0`
- contradiction rate delta must be `<= 0.02`
- approval rate delta must be `>= -0.05`

If any check fails, the gate result is **NO-GO**.

## Reason-code drift review
Every run must include reason-code drift versus baseline:
- normalized reason-code distribution deltas
- top absolute share changes (largest drift first)

## Baseline management
- Use `--accept-as-baseline` only after explicit review/approval.
- Updating the baseline without review invalidates downstream delta interpretations.
