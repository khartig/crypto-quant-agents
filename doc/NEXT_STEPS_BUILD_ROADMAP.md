# NEXT_STEPS_BUILD_ROADMAP
## Purpose
This document captures the prioritized future build plan after the regime-detection rollout and true-ablation comparison.
The goal is to improve out-of-sample **net** performance, reduce drawdown, and increase production readiness.

## Current baseline context
- Recent segmented evaluation showed regime-enabled logic underperforming regime-ablated on mean net return, drawdown, and cost drag.
- Approval and contradiction rates were unchanged between enabled and ablated profiles in tested windows.
- One key window (`uptrend_2025q2`) was skipped due to insufficient local bars.
- Conclusion: immediate priority is evaluation/data rigor first, then redesign regime contribution and alpha inputs.
## Implementation status snapshot (current)
- Priority 0 status: **implemented and gate-validated**.
  - Canonical windows and benchmark gate artifacts are present and CI-validated.
  - Historical coverage gaps were closed for canonical windows used by segmented evaluation.
- Priority 1 status: **implemented and validated**.
  - Regime conditional policy mode, touchpoint ablations, contradiction split, and cost-pressure policy were integrated and validated through suite runs.
  - Full-window Priority 1 evaluator evidence was generated after historical backfill unblocked all canonical windows.
- Priority 2 status: **implemented and validated**.
  - Agent-plane and evaluator paths now enforce Priority 2 feature-column selection plus Priority 2 external-data quality-gate behavior consistently with trigger-model paths.
  - OpenClaw request mapping and CLI wiring now expose Priority 2 feature-column and quality-gate controls end-to-end.
  - Roadmap item 8 deliverables are now implemented with a schema-versioned alternative-data feature module and quality/latency reporting script.
  - Roadmap item 9 deliverables are now implemented with labeling/objective benchmark + frontier plot generation and a dedicated labeling specification doc.
- Priority 3 status: **implemented (validation artifacts integrated)**.
  - Agent-plane now supports volatility/confidence/drawdown-aware paper notional sizing with fallback profile diagnostics.
  - Paper execution now persists spread/latency/liquidity/impact realism diagnostics in execution artifacts and run manifests.
  - Execution realism stress suite and live-readiness stage-gate dashboard scripts are available for repeatable validation.

## Execution protocol (strict sequencing)
- Work strictly in order: **Priority 0 → Priority 1 → Priority 2 → Priority 3**.
- Do not start Priority 1 until Priority 0 is both implemented and validation-tested.
- Every priority transition requires passing the active gate checks and preserving reproducible artifact history.

## Priority 0 implementation assets (current)
- Canonical windows config: `scripts/regime_window_slices.json`.
- Single-command benchmark gate harness: `scripts/run_regime_benchmark_gate.py`.
- Presubmit/CI validator: `scripts/validate_regime_benchmark_gate.py`.
- Baseline snapshot: `doc/REGIME_BENCHMARK_BASELINE.json`.
- Pass/fail policy spec: `doc/REGIME_BENCHMARK_PASS_FAIL_CRITERIA.md`.
- CI enforcement entrypoint: `.github/workflows/ci.yml` (`validate_regime_benchmark_gate.py`).

## Priority 0 — Immediate (highest impact, unblockers)
### 1) Close historical coverage gaps and freeze evaluation slices
- Build tasks:
  - Ingest and store continuous BTC/USDT history that fully covers bull/flat/bear windows (including 2025 Q2).
  - Lock a canonical evaluation slice set and keep it versioned.
  - Add strict data-coverage checks before segmented evaluation runs.
- Deliverables:
  - Canonical dataset manifest with min/max timestamps and hash.
  - Re-runnable window config file under `doc/` or `scripts/`.
  - Updated segmented results with zero skipped priority windows.

### 2) Standardize the evaluation harness as the decision gate
- Build tasks:
  - Keep `regime_enabled` vs `regime_ablated` as mandatory comparison profiles.
  - Add fixed benchmark outputs (JSON + markdown summary) with deterministic run IDs/artifact links.
  - Add CI/presubmit check that fails when required comparison artifacts are missing.
- Deliverables:
  - Single command for full benchmark generation.
  - Machine-readable summary with per-window and aggregate deltas.
  - Documented pass/fail criteria for model changes.

### 3) Track exact “go/no-go” metrics for every iteration
- Build tasks:
  - Require these core metrics in every benchmark report:
    - approval rate
    - contradiction rate
    - net return
    - sharpe
    - max drawdown
    - total cost drag
  - Add reason-code drift reporting and top changes versus prior baseline.
- Deliverables:
  - Versioned metric snapshots per experiment.
  - Automated delta report against the last accepted baseline.

## Priority 1 — Regime and policy redesign
### 4) Redesign regime contribution (from additive confidence to conditional policy)
- Build tasks:
  - Replace/augment current regime influence so it affects policy behavior explicitly (not just confidence adjustments).
  - Evaluate regime-conditioned thresholding or regime-specific action rules.
  - Keep ablation toggles for each regime touchpoint (prompting, calibration, self-critique, risk gate) for isolation.
- Deliverables:
  - Regime v2 design note.
  - Component-level ablation matrix with measured lift/drag.
  - Updated default profile only if net benefit is demonstrated.

### 5) Refine contradiction logic and calibration policy
- Build tasks:
  - Review contradiction thresholds by quality band to reduce “noisy block/fail” behavior.
  - Separate confidence calibration quality penalties from directional contradiction penalties.
  - Test whether contradiction policy is overly constraining in rebound windows.
- Deliverables:
  - Updated contradiction-policy spec with rationale.
  - Before/after benchmark showing effect on approvals and net outcomes.

### 6) Make cost-awareness first-class in optimization
- Build tasks:
  - Optimize for net utility (post-cost) rather than gross signal quality.
  - Add sensitivity sweeps for fee/slippage scenarios.
  - Track cost-drag decomposition by arm and by regime bucket.
- Deliverables:
  - Cost stress-test report.
  - Updated thresholds that minimize cost-induced degradation.

## Priority 2 — Feature expansion and alternative alpha
### 7) Add market-structure and derivatives features
- Build tasks:
  - Funding rate, open interest, basis, liquidation intensity, and volatility term structure.
  - Volume imbalance and momentum persistence features by horizon.
- Deliverables:
  - Feature contracts and ingestion pipeline updates.
  - Incremental lift analysis with ablations.

### 8) Add “whale” and high-signal participant proxies
- Build tasks:
  - Introduce large-transfer/on-chain flow proxies (exchange inflow/outflow, concentration spikes).
  - Track publicly-known trader proxy signals where legally/operationally feasible.
  - Convert all such signals into deterministic, timestamp-aligned feature series.
- Deliverables:
  - Alternative data feature module with schema/versioning.
  - Data quality and latency validation report.
- Implementation assets:
  - `src/quant_agents/alternative_data_features.py`
  - `scripts/validate_priority2_alternative_data_quality.py`

### 9) Improve labeling/objectives beyond simple directional framing
- Build tasks:
  - Add meta-labeling and/or triple-barrier style outcome labeling.
  - Include “trade/no-trade” quality labels to reduce low-edge participation.
  - Evaluate precision/recall trade-off for actionable signals under cost constraints.
- Deliverables:
  - Labeling specification and benchmark comparison.
  - Actionable coverage vs precision frontier plots.
- Implementation assets:
  - `doc/TRIGGER_LABELING_OBJECTIVE_SPEC.md`
  - `scripts/run_labeling_objective_benchmark.py`

## Priority 3 — Risk, execution, and deployment hardening
### 10) Upgrade position sizing and risk budget logic
- Build tasks:
  - Add volatility-targeted sizing and confidence-weighted notional controls.
  - Add drawdown-aware throttle/kill-switch logic.
  - Add regime-independent fallback sizing profile.
- Deliverables:
  - Risk budget policy document.
  - Backtest + paper-trade validation showing improved drawdown behavior.
- Implementation assets:
  - `doc/RISK_BUDGET_POLICY.md`
  - `src/quant_agents/agent_plane.py`
  - `src/quant_agents/agent_contracts.py`

### 11) Improve execution realism
- Build tasks:
  - Model spread/latency/partial fill impacts more explicitly.
  - Add stress scenarios for slippage shocks and low-liquidity periods.
- Deliverables:
  - Execution realism test suite.
  - Cost drag vs execution assumption sensitivity report.
- Implementation assets:
  - `src/quant_agents/paper_trading.py`
  - `scripts/run_execution_realism_stress_suite.py`

### 12) Formal paper-to-live readiness gates
- Build tasks:
  - Define staged rollout criteria (shadow -> paper-forward -> limited notional live).
  - Add operational SLOs and incident rollback triggers.
  - Require sustained benchmark and paper-forward performance stability before each stage.
- Deliverables:
  - Live-readiness checklist and runbook.
  - Stage-gate dashboard of required metrics.
- Implementation assets:
  - `doc/LIVE_READINESS_RUNBOOK.md`
  - `scripts/run_live_readiness_stage_gate.py`

## Recommended execution sequence
1. Complete Priority 0 before any major model redesign.
2. Execute Priority 1 with strict ablation comparisons for every change.
3. Start Priority 2 feature work in parallel only after benchmark harness is frozen.
4. Promote Priority 3 gates before increasing real capital exposure.

## Acceptance criteria for “ready for limited live capital”
- No missing benchmark windows in canonical regime slices.
- Net return and drawdown improvements sustained over multiple rolling windows versus accepted baseline.
- Cost drag controlled within predefined threshold bands under stress scenarios.
- Stable contradiction/approval behavior with explainable reason-code distributions.
- Successful paper-forward run period with no critical operational incidents.
