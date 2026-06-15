# COST_STRESS_TEST_REPORT_AND_THRESHOLD_UPDATES
## Scope
Priority 1 cost-awareness report for roadmap item 6.

Reference artifacts:
- Priority 1 matrix + stress summary: `/mnt/quant-data/logs/analysis/regime-window-evals/2026-06-15/20260615T014120Z/summary.json`
- Benchmark baseline: `doc/REGIME_BENCHMARK_BASELINE.json`

## Stress setup
Profiles stressed:
- `regime_v2_full`
- `regime_ablated`

Scenarios (from `cost_stress.scenarios`):
- `base`: fee `5.0` bps, slippage `2.5` bps
- `stress_x1.50`: fee `7.5` bps, slippage `3.75` bps
- `stress_x2.00`: fee `10.0` bps, slippage `5.0` bps
- `stress_x3.00`: fee `15.0` bps, slippage `7.5` bps

Coverage caveat:
- Stress results are derived from one runnable canonical window (`decline_2026may_to_now`).
- Four canonical windows were skipped due insufficient bars.

## Stress results
### `regime_v2_full`
- `base`: net `-0.110424`, sharpe `-7.334479`, max drawdown `-0.139705`, cost drag `0.051865`, cost pressure `1.400000`
- `stress_x1.50`: net `-0.133217`, sharpe `-8.910718`, max drawdown `-0.157853`, cost drag `0.077797`, cost pressure `1.470253`
- `stress_x2.00`: net `-0.155429`, sharpe `-10.451819`, max drawdown `-0.175621`, cost drag `0.103729`, cost pressure `1.593670`
- `stress_x3.00`: net `-0.198169`, sharpe `-13.413802`, max drawdown `-0.213470`, cost drag `0.155594`, cost pressure `1.840505`

`regime_v2_full` shows monotonic degradation in net return, sharpe, and drawdown as cost assumptions increase, while cost pressure rises from `1.40` to `1.84`.

### `regime_ablated`
- `base`: net `-0.110424`, sharpe `-7.334479`, max drawdown `-0.139705`, cost drag `0.051865`, cost pressure `1.400000`
- `stress_x1.50`: net `-0.103571`, sharpe `-8.910718`, max drawdown `-0.123259`, cost drag `0.059626`, cost pressure `1.469109`
- `stress_x2.00`: net `-0.124657`, sharpe `-10.451819`, max drawdown `-0.141307`, cost drag `0.081892`, cost pressure `1.592294`
- `stress_x3.00`: net `-0.155506`, sharpe `-13.413802`, max drawdown `-0.167919`, cost drag `0.119252`, cost pressure `1.838218`

For `regime_ablated`, cost pressure and cost drag increase across stress levels, with strongest degradation at `stress_x3.00`.

## Cost decomposition
### Base decomposition by profile/regime bucket (top-level summary)
From `cost_decomposition.by_profile_regime_bucket`:
- `regime_v2_full` / `range`: mean total cost drag `0.051865`, mean cost pressure `1.4`
- `regime_ablated` / `unknown`: mean total cost drag `0.051865`, mean cost pressure `1.4`

### Base decomposition by profile/regime bucket/arm
From `cost_decomposition.by_profile_regime_bucket_arm`:
- `llm_context`: `0.000000` mean arm cost drag
- `sma_baseline`: `0.008250` mean arm cost drag
- `technical_composite`: `0.123750` mean arm cost drag

### Stress decomposition aggregation
From `cost_stress.cost_decomposition.by_profile_regime_bucket`:
- `regime_v2_full` / `range`: mean total cost drag `0.097246`, mean cost pressure `1.576107` (4 stress scenarios)
- `regime_ablated` / `unknown`: mean total cost drag `0.078159`, mean cost pressure `1.574905` (4 stress scenarios)

From `cost_stress.cost_decomposition.by_profile_regime_bucket_arm`:
- `llm_context`: `0.000000` mean arm cost drag
- `sma_baseline`: `0.015469` mean arm cost drag
- `technical_composite`: `0.232031` mean arm cost drag

## Threshold updates applied
Based on stress behavior and high observed cost-pressure scores, defaults were tightened:
- `RISK_MAX_COST_RETURN_DRAG`: `0.06 -> 0.05`
- `RISK_MAX_COST_PRESSURE_SCORE`: `1.00 -> 0.95`
- `CALIBRATION_COST_PRESSURE_PENALTY_STRENGTH`: `0.20 -> 0.30`

Updated in:
- `src/quant_agents/config.py`
- `src/quant_agents/agent_plane.py`
- `.env.example`

## Conclusion
Priority 1 cost-awareness tooling is implemented and producing decomposition/stress artifacts.
Final threshold calibration confidence remains conditional on restoring full canonical window coverage and repeating the stress sweep over all required windows.
