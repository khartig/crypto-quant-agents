# Risk Budget Policy
## Purpose
This policy defines deterministic position-sizing controls used by the agent-plane execution gateway before paper intent emission.
The objective is to reduce avoidable drawdown while preserving participation in higher-quality opportunities.

## Scope
- Applies to `quant-agents agent-plane` paper intents.
- Uses calibrated recommendation confidence, realized volatility, and portfolio drawdown state.
- Produces auditable sizing diagnostics in:
  - `paper_trade_intent.json` (`sizing_policy`, `sizing_diagnostics`)
  - `run_manifest.json` (`outcome.paper_intent_sizing_*`)

## Sizing model
Base sizing starts from `PAPER_TRADE_NOTIONAL_USD`.
Resolved intent notional is computed from multiplicative controls:

- `volatility_scale`: targets configured annualized volatility (`PAPER_SIZING_TARGET_ANNUAL_VOLATILITY`) from rolling realized volatility.
- `confidence_scale`: maps calibrated confidence into `[PAPER_SIZING_MIN_FRACTION, PAPER_SIZING_MAX_FRACTION]` using:
  - `PAPER_SIZING_CONFIDENCE_FLOOR`
  - `PAPER_SIZING_CONFIDENCE_CEILING`
- `drawdown_throttle_scale`: reduces sizing as drawdown approaches the kill-switch threshold.

Combined scale is clipped to:
- lower bound: `PAPER_SIZING_MIN_FRACTION`
- upper bound: `PAPER_SIZING_MAX_FRACTION`

## Drawdown controls
- Drawdown throttle starts at `PAPER_SIZING_DRAWDOWN_THROTTLE_START`.
- Actionable intents are blocked when drawdown reaches `PAPER_SIZING_DRAWDOWN_KILL_SWITCH`.
- Kill-switch block reason is emitted in intent `reason` and run manifest outcome.

## Regime-independent fallback profile
Fallback is used when adaptive volatility sizing cannot be resolved (for example insufficient volatility signal) or adaptive sizing is disabled.
Fallback notional source:
- `PAPER_SIZING_FALLBACK_NOTIONAL_USD`

Fallback remains subject to confidence and drawdown scaling to preserve deterministic risk controls.

## Execution realism coupling
Sizing policy works with execution realism assumptions to avoid overstating fill quality:
- `EXECUTION_REALISM_SPREAD_BPS`
- `EXECUTION_REALISM_LATENCY_MS`
- `EXECUTION_REALISM_LATENCY_SLIPPAGE_BPS_PER_SECOND`
- `EXECUTION_REALISM_LIQUIDITY_SCORE`
- `EXECUTION_REALISM_MARKET_DEPTH_NOTIONAL_USD`
- `EXECUTION_REALISM_NOTIONAL_IMPACT_COEFF`

These values are persisted in `paper_trade_execution` and run manifests for post-run attribution.

## Validation workflow
Run the Priority 3 validation pack:

1. Execution realism stress suite:
   - `python scripts/run_execution_realism_stress_suite.py --exchange <exchange> --symbol <symbol> --timeframe <timeframe>`
2. Agent-plane regression run with sizing/execution realism controls enabled.
3. Readiness stage-gate dashboard:
   - `python scripts/run_live_readiness_stage_gate.py --exchange <exchange> --symbol <symbol> --timeframe <timeframe>`

## Acceptance expectations
- Intent-time diagnostics are present and schema-stable.
- Drawdown throttle and kill-switch behavior is observable in manifests.
- Stress scenarios show expected drag sensitivity without pathological failures.
- Readiness stage checks are used before promoting to higher deployment stages.
