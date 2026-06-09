# Phase 2 Plan: Confidence Calibration and Walk-Forward Reconciliation
## Problem statement
The pipeline can still emit or preserve high confidence even when recent strategy performance is weak, which creates trust-breaking contradictions between recommendation and empirical evidence.
## Current state
`run_sma_backtest` executes a single backtest pass and outputs aggregate metrics only (`src/quant_agents/backtest.py:78`).
Risk gating compares static thresholds (`total_return`, `sharpe`, `max_drawdown`, `min_signal_confidence`) and does not explicitly model confidence-vs-performance reconciliation (`src/quant_agents/agent_plane.py:624`, `src/quant_agents/config.py:96`).
The current strategy contract has a scalar confidence field without calibrated components (`src/quant_agents/agent_contracts.py:48`).
## Proposed changes
Implement a walk-forward backtest engine that runs rolling train/validate windows and emits per-window returns, Sharpe, drawdown, hit-rate, and stability metrics, then aggregate these into a dedicated walk-forward artifact and contract extension.
Add a confidence-calibration module that computes posterior confidence from indicator agreement strength, detected regime, and walk-forward quality metrics rather than relying on model-native confidence alone.
Define an explicit contradiction policy: if actionable direction is `buy`/`sell` while walk-forward Sharpe (or equivalent quality score) is below threshold, auto-downgrade confidence, add reason codes (for example `walkforward_sharpe_below_threshold`), and annotate rationale with the conflict.
Refactor risk gating to evaluate both raw and penalized confidence, and to consume walk-forward metrics alongside aggregate backtest metrics (`src/quant_agents/agent_plane.py:624`).
Extend configuration/CLI to include walk-forward calibration controls (window sizes, minimum walk-forward Sharpe, maximum contradiction allowance, calibration floor/ceiling), exposed through `quant-agents agent-plane` (`src/quant_agents/cli.py:95`, `src/quant_agents/cli.py:395`).
Persist calibration diagnostics (reliability bins, confidence deciles vs realized return, contradiction counts) as run artifacts so ops can audit confidence behavior over time.
## Validation and exit criteria
When walk-forward quality is below threshold, actionable signals are either downgraded to low confidence or blocked, and reason codes clearly identify the downgrade path.
Confidence distributions become monotonic with realized quality in rolling evaluations (higher confidence deciles outperform lower deciles on average).
Risk decision artifacts expose walk-forward metrics and penalized confidence as first-class observed values.
Historical replays of the same input data produce identical calibration outputs and gating outcomes.
## Dependencies and sequencing
Phase 1 feature outputs and regime labels are required inputs for robust calibration logic; Phase 2 should begin after Phase 1 contracts and feature snapshot formats are stable.
## Completion verification status
Phase 2 commitments are implemented and now verified by `scripts/verify_phase2_completion.py`, which runs deterministic synthetic scenarios and validates contradiction behavior, calibration payload semantics, and replay determinism.
Walk-forward evaluation and aggregation artifacts are emitted from `src/quant_agents/agent_plane.py` (`walkforward_evaluation.v1`) and persisted in run manifests.
Confidence calibration with walk-forward quality reconciliation is implemented in `_calibrate_confidence(...)` with quality-band policies, contradiction flags/severity, and calibrated confidence output.
Risk gating consumes both aggregate backtest metrics and calibrated confidence / walk-forward diagnostics through `RiskDecision.observed` fields and reason-code mapping in `run_agent_plane(...)`.
Configuration and CLI controls for walk-forward calibration strictness are available in `src/quant_agents/config.py` and `src/quant_agents/cli.py` under the `quant-agents agent-plane` command.
Calibration diagnostics (`reliability_bins`, `confidence_deciles`, contradiction counters) are persisted in the `confidence_calibration` artifact and surfaced in run contracts/manifests.
