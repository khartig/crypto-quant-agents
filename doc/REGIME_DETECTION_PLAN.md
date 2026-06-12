# Regime-Detection Module Research and Evaluation Plan
## Problem statement
The trading pipeline currently treats market regime as a lightweight heuristic field, but regime state is used downstream in confidence calibration, self-critique, and risk decisions. To improve real-trading readiness, we need a dedicated regime-detection module with measurable quality, stable transitions, and demonstrated net-performance impact.
## Current state
Regime is computed inline inside phase-1 feature context using fixed threshold logic and a single latest-bar classification (`src/quant_agents/agent_plane.py (805-934)`).
`StrategyProposalSignal` carries only a single `regime` string without confidence/transition metadata (`src/quant_agents/agent_contracts.py (45-69)`).
Calibration and risk already consume regime, but only as a coarse categorical adjustment/check (`src/quant_agents/agent_plane.py (1527-1662)`, `src/quant_agents/agent_plane.py (2624-3002)`).
Pipeline configuration surfaces risk and walk-forward controls but has no dedicated regime-detection controls (`src/quant_agents/config.py (17-80)`, `src/quant_agents/cli.py (61-341)`, `src/quant_agents/openclaw_native.py (66-286)`).
Validation coverage focuses on walk-forward contradiction policy and deterministic gating, not regime quality or regime-driven trading impact (`scripts/validate_walkforward_policy_suite.py (1-244)`).
The trigger model uses technical features but does not explicitly include regime-state features (`src/quant_agents/trigger_model.py (20-205)`).
## Research goals
Quantify whether improved regime detection reduces contradictory signals and improves out-of-sample net trading metrics versus the current heuristic baseline.
Define a deterministic regime interface that is explainable, replay-stable, and configurable via CLI/settings/OpenClaw.
Measure both regime-quality metrics and downstream trading outcomes before making regime logic gate-critical.
## Proposed changes
### 1) Create a standalone regime-detection module
Add `src/quant_agents/regime_detection.py` with a deterministic API (input frame + config -> regime label, confidence score, transition diagnostics, and reason codes).
Start with two candidate detectors behind the same interface: enhanced threshold/hysteresis baseline and a score-based multi-feature classifier (trend, volatility, momentum, breadth proxies already derivable from OHLCV).
Add persistence controls (minimum dwell bars, hysteresis margins) to reduce regime-flip noise.
### 2) Add regime configuration surface
Extend `Settings` with regime controls (enabled mode, lookback windows, volatility/trend thresholds, persistence bars, optional confidence floor).
Expose corresponding CLI flags on `quant-agents agent-plane` and thread fields through OpenClaw request mapping so local and orchestrated runs are consistent.
### 3) Integrate module into agent-plane and contracts
Refactor phase-1 feature context generation to call the new module instead of inline ad-hoc regime branching (`src/quant_agents/agent_plane.py (805-934)`).
Extend strategy and risk artifacts with regime diagnostics (for example `regime_confidence`, `regime_transition`, and regime evidence summary) while preserving backward-compatible fields.
Consume regime confidence in calibration and self-critique/risk checks with explicit reason codes when regime confidence is weak or contradictory.
### 4) Add regime-aware features to trigger modeling (research branch)
Add regime-derived features (state one-hot/ordinal, transition intensity, persistence age) to trigger model feature engineering behind a feature flag.
Run ablation against the existing trigger feature set to assess whether regime-aware features improve net expectancy and actionable precision/recall.
### 5) Build evaluation harness and experiment matrix
Add `scripts/evaluate_regime_module.py` to produce regime-quality metrics and comparative reports.
Evaluate candidates on rolling out-of-sample windows using the same cost-aware assumptions as agent-plane.
Run an ablation matrix: current inline regime baseline, new detector in shadow mode (observability only), new detector with calibration/risk integration enabled.
Track both model-level and trading-level outcomes: regime stability/churn, regime-forward return alignment, volatility separation, contradiction rate, net return, net Sharpe, max drawdown, turnover, cost drag, gate pass rate.
### 6) Validation, rollout, and documentation
Add deterministic tests for regime outputs under fixed inputs and extend policy-suite style tests to validate new reason-code paths and replay consistency.
Roll out in two phases: shadow mode first (no blocking behavior change), then controlled enforcement once acceptance criteria are met.
Document new settings, CLI flags, artifacts, and interpretation in `README.md` and `doc/MODEL_METRIC_DEFINITIONS_AND_RESULTS.md`.
## Evaluation methodology and acceptance criteria
Use walk-forward, cost-aware evaluation on the existing 6-month benchmark dataset plus at least one additional symbol/timeframe holdout for robustness.
Require deterministic replay parity for regime outputs and downstream risk decisions under identical inputs.
Require materially improved regime diagnostics (lower regime-churn noise and stronger volatility/trend separation) without degrading core net metrics.
Promotion target: non-negative change in net return and net Sharpe versus baseline, and measurable reduction in actionable contradiction-related blocks over the same evaluation windows.
## Deliverables
Regime module implementation with config/CLI/OpenClaw wiring, updated artifacts/contracts, evaluation script and test coverage, and documentation updates with benchmark results and rollout guidance.
