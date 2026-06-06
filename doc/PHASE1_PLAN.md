# Phase 1 Plan: Signal Depth and Regime-Aware Feature Enrichment
## Problem statement
The current strategy signal is still too shallow for crypto market structure, so the recommendation rationale can be directionally plausible but not evidence-rich enough to justify confidence.
## Current state
`run_agent_plane` builds a minimal market snapshot (`bars`, `last_close`, `momentum_24h`, `volatility_48_bars`) before prompting the strategy model (`src/quant_agents/agent_plane.py:288`, `src/quant_agents/agent_plane.py:309`).
The strategy contract only carries recommendation/confidence/SMA windows/rationale, with no indicator-vote or regime context (`src/quant_agents/agent_contracts.py:48`).
Backtesting is a single SMA crossover path (`STRATEGY_NAME = "sma_crossover"`) and does not evaluate multi-indicator logic (`src/quant_agents/backtest.py:19`, `src/quant_agents/backtest.py:78`).
The storage tree already has `curated/features`, but feature datasets are not yet populated by the pipeline (`src/quant_agents/storage.py:7`).
## Proposed changes
Add a deterministic feature computation layer that produces RSI, MACD (line/signal/histogram), Bollinger Bands, and ADX-style trend/range regime labels from OHLCV data; persist aligned feature frames under `curated/features` keyed by exchange/symbol/timeframe/run.
Add a feature-enrichment ingestion path for high-ROI external context recommended by the source roadmap (funding rate first, then fear/greed and optional order-book imbalance) with explicit freshness metadata and fail-safe behavior when providers are unavailable.
Extend `StrategyProposalSignal` to include `indicator_votes`, `regime`, `feature_snapshot`, and `reason_codes`, while preserving current required fields for backward compatibility (`src/quant_agents/agent_contracts.py:48`).
Refactor strategy prompting so the model consumes the deterministic feature snapshot and must ground rationale in indicator/regime evidence rather than generic momentum/volatility language (`src/quant_agents/agent_plane.py:309`).
Add a deterministic composite voter (RSI/MACD/Bollinger/SMA + regime gate) that produces an evidence score and baseline recommendation, then let the LLM explain and optionally adjust within constrained policy bounds.
Expose CLI/config toggles for feature providers and feature strictness so operators can choose between fail-open and fail-closed behavior during provider outages (`src/quant_agents/cli.py:95`, `src/quant_agents/config.py:30`).
## Validation and exit criteria
Agent-plane outputs include non-empty indicator votes, regime labels, and reason codes in `strategy_proposal_signal.json` for successful runs.
Composite score and recommendation are reproducible for a fixed input parquet and provider snapshot.
When feature providers are unavailable, behavior matches configured strictness and is explicitly reflected in warnings/reason codes.
Run-manifest artifacts contain paths to generated curated feature datasets and checksums.
## Dependencies and sequencing
This phase is the foundation for confidence calibration and walk-forward penalties in Phase 2, so contract and feature shape decisions here must be finalized first.
