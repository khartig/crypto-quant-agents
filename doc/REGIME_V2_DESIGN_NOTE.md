# REGIME_V2_DESIGN_NOTE
## Scope
This note captures the Priority 1 regime-policy redesign for roadmap item 4 in `doc/NEXT_STEPS_BUILD_ROADMAP.md`.

Reference artifacts:
- Baseline benchmark snapshot: `doc/REGIME_BENCHMARK_BASELINE.json`
- Benchmark-mode evaluation summary: `/mnt/quant-data/logs/analysis/regime-window-evals/2026-06-15/20260615T011447Z/summary.json`
- Priority 1 evaluation summary (matrix + stress): `/mnt/quant-data/logs/analysis/regime-window-evals/2026-06-15/20260615T014120Z/summary.json`

## Problem statement
The legacy regime contribution path primarily adjusted confidence additively, which made policy behavior difficult to reason about and harder to isolate during ablations.
Priority 1 required an explicit regime-conditional policy path plus touchpoint-level toggles and clearer diagnostics.

## Regime v2 design
### 1) Explicit conditional policy behavior
- Added `regime_policy_mode` with `legacy` and `conditional_v2`.
- In `conditional_v2`, regime confidence gates actionable recommendations directly instead of only nudging confidence.
- New configuration controls:
  - `REGIME_POLICY_MIN_ACTIONABLE_CONFIDENCE` (default `0.50`)
  - `REGIME_POLICY_TRANSITION_CONFIDENCE` (default `0.65`)

### 2) Touchpoint-level ablation controls
Regime contribution can now be enabled/disabled independently across four touchpoints:
- Prompting/recommendation generation
- Confidence calibration
- Self-critique contradiction checks
- Deterministic risk gate

This is surfaced via:
- `REGIME_TOUCHPOINT_PROMPTING_ENABLED`
- `REGIME_TOUCHPOINT_CALIBRATION_ENABLED`
- `REGIME_TOUCHPOINT_SELF_CRITIQUE_ENABLED`
- `REGIME_TOUCHPOINT_RISK_GATE_ENABLED`

### 3) Contradiction and calibration separation
Contradiction logic now separates:
- directional contradiction behavior (signal-vs-walkforward direction conflict), and
- quality contradiction behavior (walkforward quality-band degradation).

Separate calibration penalties are exposed through:
- `CALIBRATION_QUALITY_PENALTY_STRENGTH` (default `0.25`)
- `CALIBRATION_DIRECTIONAL_CONTRADICTION_PENALTY` (default `0.35`)
- `CALIBRATION_COST_PRESSURE_PENALTY_STRENGTH` (default `0.30`)

### 4) Cost-awareness integration
- Added cost-pressure diagnostics and risk gating via `cost_pressure_score`.
- Added deterministic threshold `RISK_MAX_COST_PRESSURE_SCORE` (default `0.95`).
- Evaluation outputs include cost decomposition by profile/regime bucket and by profile/regime bucket/arm.

## Default threshold updates adopted in Priority 1
- `RISK_MAX_COST_RETURN_DRAG`: `0.06 -> 0.05`
- `RISK_MAX_COST_PRESSURE_SCORE`: `1.00 -> 0.95`
- `CALIBRATION_COST_PRESSURE_PENALTY_STRENGTH`: `0.20 -> 0.30`

These are reflected in:
- `src/quant_agents/config.py`
- `src/quant_agents/agent_plane.py`
- `.env.example`

## Current evaluation snapshot (2026-06-15)
From `/mnt/quant-data/logs/analysis/regime-window-evals/2026-06-15/20260615T014120Z/summary.json`:
- Evaluated windows: `decline_2026may_to_now` (1)
- Skipped windows due insufficient bars (4):
  - `uptrend_2025q2`
  - `flat_2025nov_to_2026jan`
  - `drawdown_2026latejan_to_mar`
  - `rebound_2026apr`

`regime_v2_full` aggregate metrics (single available window):
- approval rate: `0.0`
- contradiction rate: `1.0`
- directional contradiction rate: `1.0`
- quality contradiction rate: `1.0`
- mean cost pressure score: `1.4`
- mean net total return: `-0.110424`
- mean sharpe: `-7.334479`
- mean max drawdown: `-0.139705`
- mean total cost return drag: `0.051865`

For this run, `regime_v2_full`, all v2 touchpoint ablations, `regime_legacy_full`, and `regime_ablated` produced identical aggregate metrics in the one runnable window.

## Limitations and next validation step
This design implementation is complete, but decision quality evidence is currently limited by missing historical bars for four canonical windows.
Promoting `conditional_v2` as a default policy should wait for a full canonical-window rerun after data backfill.
