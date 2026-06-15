# CONTRADICTION_POLICY_SPEC_AND_BENCHMARK_DELTA
## Scope
Priority 1 contradiction/calibration policy specification and benchmark delta summary for roadmap item 5.

Reference artifacts:
- Baseline benchmark: `doc/REGIME_BENCHMARK_BASELINE.json`
- Benchmark-mode summary: `/mnt/quant-data/logs/analysis/regime-window-evals/2026-06-15/20260615T011447Z/summary.json`
- Priority 1 matrix summary: `/mnt/quant-data/logs/analysis/regime-window-evals/2026-06-15/20260615T014120Z/summary.json`

## Contradiction policy specification (current)
### Contradiction dimensions
Policy now tracks contradiction in two dimensions:
- Directional contradiction: recommendation direction conflicts with walk-forward directional evidence.
- Quality contradiction: recommendation quality conflicts with walk-forward quality-band assessment.

Both are surfaced as explicit rates in evaluation outputs:
- `directional_contradiction_rate`
- `quality_contradiction_rate`

### Calibration controls
Calibration now separates penalty sources:
- quality penalty: `CALIBRATION_QUALITY_PENALTY_STRENGTH` (default `0.25`)
- directional contradiction penalty: `CALIBRATION_DIRECTIONAL_CONTRADICTION_PENALTY` (default `0.35`)
- cost-pressure penalty: `CALIBRATION_COST_PRESSURE_PENALTY_STRENGTH` (default `0.30`)

Additional contradiction-policy controls:
- `CALIBRATION_MAX_CONTRADICTIONS` (default `0`)
- `CALIBRATION_DIRECTIONAL_EDGE_THRESHOLD` (default `0.0`)

### Deterministic risk interactions
Risk decisions now consume contradiction/cost diagnostics explicitly, including:
- `risk_block_buy_walkforward_directional_contradiction_low`
- `risk_block_buy_walkforward_quality_contradiction_low`
- `risk_fail_buy_calibration_contradiction_limit_exceeded`
- `cost_pressure_score_above_threshold`

## Before/after benchmark framing
### Baseline (accepted 2026-06-12)
From `doc/REGIME_BENCHMARK_BASELINE.json`:
- `regime_enabled`: contradiction rate `0.0`, approval rate `0.0`
- `regime_ablated`: contradiction rate `0.0`, approval rate `0.0`

### Current benchmark snapshot (2026-06-15)
From `/mnt/quant-data/logs/analysis/regime-window-evals/2026-06-15/20260615T011447Z/summary.json`:
- `regime_enabled`: contradiction `1.0`, directional `1.0`, quality `1.0`, approval `0.0`
- `regime_ablated`: contradiction `1.0`, directional `1.0`, quality `1.0`, approval `0.0`

### Baseline-to-current deltas
`regime_enabled`:
- approval rate delta: `0.0`
- contradiction rate delta: `+1.0`
- mean net total return delta: `-0.036324`
- mean sharpe delta: `-7.334479`
- mean max drawdown delta: `-0.029805`
- mean total cost return drag delta: `-0.018735`

`regime_ablated`:
- approval rate delta: `0.0`
- contradiction rate delta: `+1.0`
- mean net total return delta: `-0.026164`
- mean sharpe delta: `-7.334479`
- mean max drawdown delta: `-0.022017`
- mean total cost return drag delta: `-0.015849`

## Interpretation
The policy refactor delivered the intended split diagnostics and clearer reason-code taxonomy, but current measured outcomes are dominated by a single runnable canonical window (`decline_2026may_to_now`).
Before/after conclusions about contradiction strictness should be treated as provisional until all canonical windows are available and re-evaluated.
