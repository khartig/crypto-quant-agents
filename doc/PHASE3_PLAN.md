# Phase 3 Plan: Ops Explainability and Self-Critique Hardening
## Problem statement
Even with deterministic gating, operator trust degrades when blocked/executed outcomes are not explained as an explicit chain of observed metrics, thresholds, and decision transitions.
## Current state
`RiskDecision` already stores `reason_codes`, `thresholds`, and `observed` values (`src/quant_agents/agent_contracts.py:93`), but default deterministic markdown output remains terse and does not present a structured observed-vs-threshold breakdown (`src/quant_agents/agent_plane.py:344`).
OpenClaw verification focuses on artifact presence and terminal gate outcomes, but not on report completeness/diagnostic richness (`src/quant_agents/openclaw_native.py:160`).
Async supervision is present (`submit/status/wait/run-sync`) with strict pass/fail semantics (`src/quant_agents/openclaw_native.py:390`, `src/quant_agents/openclaw_native.py:527`).
## Proposed changes
Upgrade ops reporting to include a deterministic gate-failure chain section that shows each evaluated condition, observed value, threshold, and pass/fail result in a stable schema consumed by both markdown and JSON contracts.
Add explicit contradiction and downgrade reason families (for example confidence contradiction, regime mismatch, data-freshness failure) so blocked states are never circular and always actionable.
Insert a self-critique stage between strategy proposal and final risk decision that checks for internal contradictions (rationale vs metrics vs regime), producing a structured critique artifact and reason codes consumed by risk gating.
Expand `OpsReportContract` and run-manifest payloads to include `decision_trace`, `reason_code_details`, and `gate_transition_sequence` so incident analysis does not require log spelunking (`src/quant_agents/agent_contracts.py:151`, `src/quant_agents/agent_plane.py:759`).
Extend `verify_orchestration_gate` to require these new diagnostic artifacts/fields for `passed=true`, preserving fail-closed behavior for incomplete reporting (`src/quant_agents/openclaw_native.py:160`).
Add CLI controls to tune self-critique strictness and deterministic report verbosity for operational environments (`src/quant_agents/cli.py:95`).
## Validation and exit criteria
Blocked runs produce non-circular reason codes with explicit observed-vs-threshold evidence for every failed gate.
Self-critique artifacts are generated for each run and materially reduce rationale/metric contradictions in downstream contracts.
OpenClaw verification fails when required diagnostic trace artifacts are missing or malformed.
Ops report markdown and JSON remain deterministic for identical inputs and configuration.
## Dependencies and sequencing
Phase 3 should start after Phase 2 confidence calibration fields exist, so contradiction checks can reference calibrated confidence and walk-forward evidence.
