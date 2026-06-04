# Phase 1 Plan: Crypto Quant Agent Platform
## Project directory
- Root workspace: `/home/kevin/crypto-quant-agents`
- This document: `/home/kevin/crypto-quant-agents/PHASE1_PLAN.md`

## Phase 1 objectives
- Build a stable local research/trading pipeline for crypto quant workflows.
- Use Ollama for local LLM inference and experimentation.
- Add OpenClaw as an orchestration layer for specialized agents.
- Keep order execution and risk checks deterministic (non-LLM-gated).
- Prepare data/storage so large historical datasets live on external storage, not internal disk.

## Current storage access scope (observed)
- Internal system disk is mounted and healthy.
- External disk detected as `/dev/sdb` (~3.6T usable size).
- External partition layout currently appears as:
  - `/dev/sdb1` (~200M, vfat)
  - `/dev/sdb2` (~3.6T, `cs_fvault2`)
  - `/dev/sdb3` (~128M, hfsplus)
- No active mountpoint for `/dev/sdb` partitions was detected in current mounted filesystems.

## External storage plan (specific usage)
### Target usage split
- External storage should hold all large and growing artifacts:
  - Historical OHLCV/tick data
  - Feature stores and parquet datasets
  - Backtest outputs and simulation logs
  - Model cache/checkpoints and prompt/eval logs
  - Compressed archives/snapshots
- Internal disk should hold:
  - OS + core tools
  - Source code repositories
  - Active runtime temp files only

### Recommended external mount strategy
1. Choose a persistent mountpoint: `/mnt/quant-data`
2. Ensure external storage is readable/writable from Linux:
   - If current FileVault/CoreStorage layout is required for compatibility, keep it and mount via compatible tooling.
   - If this disk is dedicated to this laptop/workload, use a Linux-native filesystem (ext4 or xfs) for performance and reliability.
3. Add persistent mount in `/etc/fstab` (UUID-based) after final filesystem decision.

### Proposed external directory structure
- `/mnt/quant-data/raw/exchange=<exchange>/symbol=<pair>/interval=<tf>/year=<yyyy>/...`
- `/mnt/quant-data/curated/features/<strategy_name>/...`
- `/mnt/quant-data/backtests/<strategy_name>/<run_id>/...`
- `/mnt/quant-data/paper-trading/<date>/...`
- `/mnt/quant-data/models/ollama-cache/...`
- `/mnt/quant-data/logs/agents/<agent_name>/<date>/...`
- `/mnt/quant-data/archive/monthly/<yyyy-mm>/...`

## Phase 1 architecture
### Core principle
LLM agents can propose, analyze, and report. Only deterministic services can place orders after hard risk checks.

### Services
- `data-ingestor`: collects exchange market data (REST/websocket).
- `feature-builder`: computes indicators/features for strategies.
- `backtester`: runs historical simulation on feature sets.
- `risk-engine`: position sizing, max drawdown, circuit breakers.
- `execution-gateway`: paper trading in Phase 1 (live disabled).
- `reporting`: daily PnL, risk, drift, and data quality summaries.

### Agent roles
- `data-quality-agent`: detects gaps/spikes/missing bars.
- `strategy-agent`: proposes or tunes alpha signals.
- `backtest-agent`: queues and evaluates strategy test runs.
- `risk-review-agent`: checks violations and recommends limits.
- `ops-report-agent`: creates concise daily operational summaries.

## OpenClaw + orchestration plan
### Why OpenClaw here
- Coordinate multi-agent workflows while keeping each agent focused.
- Track tasks, dependencies, and handoffs.
- Enable controlled parallelization (e.g., strategy tuning + data checks in parallel).

### Orchestration pattern
1. **Trigger**
   - Scheduled (hourly/daily) or event-driven (new data batch complete).
2. **Fan-out**
   - Run `data-quality-agent`, `strategy-agent`, and `backtest-agent` on independent tasks.
3. **Gate**
   - Collect outputs and run deterministic validation rules.
4. **Fan-in**
   - `risk-review-agent` consolidates findings.
5. **Action**
   - In Phase 1: paper-trading actions only.
6. **Report**
   - `ops-report-agent` writes daily status + notable anomalies.

### Guardrails
- No agent can bypass `risk-engine`.
- `execution-gateway` accepts only validated strategy payloads.
- Global kill-switch and max loss limits are mandatory.

## Phase 1 implementation milestones
1. **Foundation**
   - Create repo scaffolding under project root.
   - Configure environment variables and secrets handling.
   - Add logging/metrics baseline.
2. **Data plane**
   - Ingest and store market data to external storage in partitioned format.
   - Add data integrity checks.
3. **Research plane**
   - Implement one baseline strategy.
   - Run reproducible backtests and archive outputs externally.
4. **Agent plane**
   - Integrate Ollama-backed agents for analysis/reporting.
   - Add OpenClaw workflow with explicit task boundaries and retries.
5. **Simulation plane**
   - Enable paper trading only.
   - Validate risk controls and incident handling.

## Phase 1 success criteria
- Historical data pipeline runs daily without manual intervention.
- At least one strategy has reproducible backtests and tracked metrics.
- OpenClaw can orchestrate end-to-end analysis and reporting flows.
- Paper trading loop runs with enforced risk constraints.
- Storage growth is handled primarily on external disk with clear retention rules.

## Immediate next actions
1. Confirm whether `/dev/sdb` can be mounted as-is or should be repurposed for Linux-native storage.
2. Finalize mountpoint (`/mnt/quant-data`) and create the directory tree.
3. Initialize code repository in `/home/kevin/crypto-quant-agents`.
4. Implement first ingestion + backtest + daily report workflow.
