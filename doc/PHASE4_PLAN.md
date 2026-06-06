# Phase 4 Plan: Ensemble Strategy Lifecycle and Adaptive Weighting
## Problem statement
A single-strategy arm (SMA-centric) creates concentration risk; long-term robustness requires multiple signal arms with controlled, evidence-driven weight adaptation.
## Current state
Backtesting and run metadata are centered on one strategy name (`sma_crossover`) and one set of windows (`src/quant_agents/backtest.py:19`, `src/quant_agents/backtest.py:78`).
Agent-plane orchestration executes a single recommendation path through risk and paper execution (`src/quant_agents/agent_plane.py:392`).
Paper execution and metrics logging are run-level and do not yet capture arm-level attribution for adaptive weighting (`src/quant_agents/paper_trading.py:105`, `src/quant_agents/metrics.py:81`).
## Proposed changes
Define a strategy-arm interface and registry (for example SMA baseline arm, composite technical arm, optional LLM-context arm) with consistent output schema, deterministic scoring hooks, and per-arm confidence metadata.
Implement an ensemble combiner that fuses arm outputs using decaying weights driven by rolling paper-trading performance (PnL, drawdown contribution, stability), with hard safety caps/floors to prevent unstable weight oscillation.
Extend backtest/replay tooling to run arm-level and ensemble-level evaluations, emitting comparative metrics and attribution artifacts for each run.
Update paper-trading state and fills logging to store per-arm proposed action, selected final action, and realized outcome attribution so online weight updates are auditable.
Expand strategy/risk/ops contracts and manifests to include `arm_votes`, `arm_weights`, `selected_arms`, and `ensemble_reason_codes`, preserving deterministic verification in OpenClaw supervisor and gate checks.
Add operator controls for ensemble mode, arm enablement, decay horizon, and minimum exploration weight through CLI/config pathways (`src/quant_agents/cli.py:95`, `src/quant_agents/config.py:37`).
## Validation and exit criteria
Replay and paper runs produce stable arm-level attribution artifacts and reproducible ensemble decisions for fixed inputs.
Underperforming arms lose allocation over the configured decay horizon without violating safety bounds.
Ensemble output remains explainable via reason codes and arm vote traces in ops contracts.
Supervisor verification remains fail-closed and rejects runs missing arm-level decision evidence.
## Dependencies and sequencing
Phase 4 depends on Phase 1 feature richness, Phase 2 calibrated confidence, and Phase 3 decision-trace contracts; it should start only after those interfaces are stable.
