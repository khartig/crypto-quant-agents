# PRIORITY1_ABLATION_MATRIX_REPORT
## Scope
Priority 1 component-level ablation report for roadmap item 4.

Reference artifacts:
- Priority 1 summary: `/mnt/quant-data/logs/analysis/regime-window-evals/2026-06-15/20260615T014120Z/summary.json`
- Benchmark compatibility summary: `/mnt/quant-data/logs/analysis/regime-window-evals/2026-06-15/20260615T011447Z/summary.json`

## Evaluation coverage
- Runnable canonical windows: `decline_2026may_to_now` only
- Skipped canonical windows (insufficient bars):
  - `uptrend_2025q2`
  - `flat_2025nov_to_2026jan`
  - `drawdown_2026latejan_to_mar`
  - `rebound_2026apr`

## Priority 1 ablation matrix
Reference profile: `regime_v2_full` (`regime_policy_mode=conditional_v2`, all touchpoints enabled).

All deltas below are `delta_vs_reference` from `ablation_matrix.rows`:
- `regime_v2_no_prompting`
  - approval rate delta: `0.0`
  - contradiction rate delta: `0.0`
  - directional contradiction rate delta: `0.0`
  - quality contradiction rate delta: `0.0`
  - mean cost pressure score delta: `0.0`
  - mean net total return delta: `0.0`
  - mean sharpe delta: `0.0`
  - mean max drawdown delta: `0.0`
  - mean total cost return drag delta: `0.0`
- `regime_v2_no_calibration`: all tracked deltas `0.0`
- `regime_v2_no_self_critique`: all tracked deltas `0.0`
- `regime_v2_no_risk_gate`: all tracked deltas `0.0`
- `regime_legacy_full`: all tracked deltas `0.0`
- `regime_ablated`: all tracked deltas `0.0`

## Benchmark-mode compatibility comparison
From `/mnt/quant-data/logs/analysis/regime-window-evals/2026-06-15/20260615T011447Z/summary.json` (`reference_profile=regime_enabled`):
- `regime_enabled`: baseline row (all deltas `0.0`)
- `regime_ablated` vs `regime_enabled`:
  - mean net total return delta: `+0.024860`
  - mean sharpe delta: `~0.0`
  - mean max drawdown delta: `+0.030887` (less negative)
  - mean total cost return drag delta: `-0.012114`
  - contradiction/ directional/quality contradiction deltas: `0.0`
  - mean cost pressure score delta: `0.0`

## Interpretation
Priority 1 matrix plumbing is complete and contract-valid, but current lift/drag interpretation is low-confidence because the matrix is effectively a single-window snapshot.
The matrix should be re-generated after canonical historical coverage is restored to determine which touchpoints provide consistent net benefit.
