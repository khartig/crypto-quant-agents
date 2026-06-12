# Crypto Quant Agents
Crypto Quant Agents is a deterministic trading-research and paper-execution system for crypto markets. It combines data ingestion, backtesting, model-assisted signal proposal, strict risk gating, paper-trade execution, and operational reporting.

Large outputs are written under `QUANT_DATA_ROOT` (default: `/mnt/quant-data`).

## What the system does
- Ingests OHLCV market data from exchanges via `ccxt`.
- Runs deterministic SMA-crossover and adaptive-ensemble backtests.
- Produces model-assisted strategy proposals (`buy` / `sell` / `hold`) with confidence and rationale.
- Supports Phase 4 adaptive strategy-arm ensembling with deterministic arm weighting and exploration controls.
- Trains a deterministic trigger model (`buy` / `sell` / `hold`) from historical OHLCV features.
- Produces explainable trigger predictions with class probabilities and top feature reasons.
- Monitors the market continuously and emits trigger notifications (local log + optional webhook).
- Applies deterministic risk thresholds before any execution.
- Emits and executes paper-trade intents in a local ledger.
- Generates ops reports and structured run artifacts.
- Ships a Next.js dashboard project for interactive prediction/alert review and model/trade performance analysis.
- Supports OpenClaw-native orchestration with async job supervision and strict verification gating.

## End-to-end process flow
1. **Ingest**  
   Fetch OHLCV candles and write parquet under `raw/...`.
2. **Data quality**  
   Validate bars, nulls, timestamp continuity, and duplicates.
3. **Strategy proposal**  
   Strategy model receives a market snapshot and returns strict JSON:
   - `recommendation` (`buy|sell|hold`)
   - `confidence`
   - `fast_window`, `slow_window`
   - `rationale`
4. **Backtest**  
   Deterministic SMA/ensemble backtests run using proposed windows with gross and net (cost-adjusted) metrics.
5. **Risk decision**  
   Deterministic gate checks data-quality status + backtest metrics + signal confidence.
6. **Execution gateway**  
   If approved, emit actionable paper intent (`buy`/`sell`); otherwise block.
7. **Paper execution**  
   Deterministic paper execution updates portfolio state and fills log (long-only; `sell` closes/reduces existing long inventory).
8. **Ops report + manifest**  
   Generate markdown summary and run manifest with artifact pointers.
9. **(OpenClaw path) strict verification gate**  
   Async supervisor validates artifacts and contract states; marks job `succeeded` only if all required checks pass.

## Phase 4 adaptive ensemble highlights
- Strategy proposals now carry ensemble evidence fields such as arm votes, selected arms, arm weights, and reason codes.
- Agent-plane supports adaptive per-arm performance weighting with decay + exploration and persists rolling state in paper-trading state storage.
- Backtest outputs include arm-level attribution (`arm_attribution.parquet`) and ensemble metrics for post-run diagnostics.
- Paper execution artifacts include per-arm attribution fields for executed intents.
- OpenClaw verification is fail-closed on missing/malformed ensemble evidence across strategy, risk, execution, and manifest contracts.
- Deterministic replay behavior is preserved by isolating ensemble weight state when explicit replay source data is used.

## How buy/sell signals are generated
- Signals are proposed by the configured **strategy model** from market context and constraints.
- The model output is schema-constrained JSON and normalized by code.
- If model output is invalid/unavailable, strategy falls back to deterministic `hold`.
- A proposed `buy`/`sell` is **not executable by itself**; it must pass deterministic risk checks first.
- `sell` in this system is long-only reduction/close behavior; execution does not open synthetic short positions.
- Risk checks include:
  - data quality validity,
  - backtest success,
  - threshold checks (return, sharpe, drawdown, cost drag),
  - minimum signal confidence,
  - minimum walk-forward quality score for actionable (`buy`/`sell`) recommendations.

## How models are used
Models are assistive, not authoritative:
- **Strategy model (`OLLAMA_STRATEGY_MODEL`)**: proposes action + parameters.
- **Ops model (`OLLAMA_OPS_MODEL`)**: drafts operational markdown summary.
- Core controls (risk gates, execution status, artifact validation, ledger math) are deterministic code paths.
- If models fail, the system falls back safely and records warnings in artifacts.

## How OpenClaw is used
OpenClaw integration is exposed via `quant-openclaw-entrypoint`.

It supports an async, job-based supervisor:
- `submit`: enqueue a job and return a `job_id`.
- `status`: inspect current snapshot.
- `wait`: block until terminal state.
- `run-sync`: run inline (still enforces strict verification gate).

Supervisor artifacts are written under:
- `logs/agents/openclaw-supervisor/<job_id>/request.json`
- `logs/agents/openclaw-supervisor/<job_id>/status.json`
- `logs/agents/openclaw-supervisor/<job_id>/result.json`
- `logs/agents/openclaw-supervisor/<job_id>/verification_gate.json`
- `logs/agents/openclaw-supervisor/<job_id>/worker.log`

## Strict orchestration verification gate
The OpenClaw supervisor is fail-closed:
- Required artifacts must exist and parse.
- Risk decision must be approved with deterministic gate pass.
- Intent must be emitted and actionable.
- Paper execution must be `executed` with positive executed notional.
- Run manifest outcome must match the above.

If any check fails, job status becomes `blocked` (not `succeeded`).

## Prerequisites
- Python 3.11+
- External storage mounted at `/mnt/quant-data` (or set `QUANT_DATA_ROOT`)

## Setup
```bash path=null start=null
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
cp .env.example .env
```

## Configuration
Key environment variables (see `.env.example`):
- Storage and safety:
  - `QUANT_DATA_ROOT`
  - `ALLOW_UNMOUNTED_DATA_ROOT`
  - `REQUIRE_EXCHANGE_SECRETS`
- Market defaults:
  - `EXCHANGE_ID`, `SYMBOL`, `TIMEFRAME`
- Models and orchestration:
  - `OLLAMA_BASE_URL`
  - `OLLAMA_STRATEGY_MODEL`
  - `OLLAMA_OPS_MODEL`
  - `AGENT_STEP_RETRIES`
  - `AGENT_MINIMUM_BARS`
  - `AGENT_ENSEMBLE_MODE`
  - `AGENT_ENSEMBLE_ARMS`
  - `AGENT_ENSEMBLE_DECAY_HORIZON`
  - `AGENT_ENSEMBLE_EXPLORATION_WEIGHT`
  - `AGENT_ENSEMBLE_TURNOVER_PENALTY_BPS`
- Backtest/walk-forward cost model:
  - `BACKTEST_FEE_BPS`
  - `BACKTEST_SLIPPAGE_BPS`
  - `WALK_FORWARD_FEE_BPS`
  - `WALK_FORWARD_SLIPPAGE_BPS`
- Trigger model:
  - `TRIGGER_MODEL_HORIZON_BARS`
  - `TRIGGER_MODEL_BUY_THRESHOLD`
  - `TRIGGER_MODEL_SELL_THRESHOLD`
  - `TRIGGER_MODEL_MIN_TRAIN_SAMPLES`
  - `TRIGGER_MODEL_COST_BPS`
  - `TRIGGER_MODEL_OPTIMIZE_THRESHOLDS`
- Trigger monitor:
  - `TRIGGER_MONITOR_POLL_SECONDS`
  - `TRIGGER_MONITOR_SIGNAL_CONFIDENCE`
  - `TRIGGER_MONITOR_WEBHOOK_URL`
  - `TRIGGER_MONITOR_NOTIFY_ON_HOLD`
- Risk thresholds:
  - `RISK_MIN_TOTAL_RETURN`
  - `RISK_MIN_SHARPE`
  - `RISK_MAX_DRAWDOWN`
  - `RISK_MAX_COST_RETURN_DRAG`
  - `RISK_MIN_SIGNAL_CONFIDENCE`
  - `RISK_MIN_WALKFORWARD_QUALITY_SCORE`
- Paper execution:
  - `PAPER_TRADE_NOTIONAL_USD`
  - `PAPER_TRADE_STARTING_CASH_USD`
  - `PAPER_TRADE_FEE_BPS`
  - `PAPER_TRADE_SLIPPAGE_BPS`
- Paper account connectivity probe:
  - `PAPER_ACCOUNT_PROVIDER`
  - `PAPER_ACCOUNT_EXCHANGE`
  - `PAPER_ACCOUNT_SANDBOX`
  - `PAPER_ACCOUNT_API_KEY`, `PAPER_ACCOUNT_API_SECRET`, `PAPER_ACCOUNT_API_PASSPHRASE`

### Recommended tuned defaults (current production profile: SMA 24/60)
- Strategy windows: `fast_window=24`, `slow_window=60`
- Risk thresholds:
  - `RISK_MAX_COST_RETURN_DRAG=0.06`
  - `RISK_MIN_WALKFORWARD_QUALITY_SCORE=0.43`
  - `RISK_MIN_SIGNAL_CONFIDENCE=0.55`
- Cost model:
  - `BACKTEST_FEE_BPS=5.0`, `BACKTEST_SLIPPAGE_BPS=2.5`
  - `WALK_FORWARD_FEE_BPS=5.0`, `WALK_FORWARD_SLIPPAGE_BPS=2.5`

## Commands
The main CLI command is `quant-agents`.

### Health and preflight
```bash path=null start=null
quant-agents doctor
quant-agents doctor --require-secrets
```

### Data and backtesting
```bash path=null start=null
quant-agents ingest --exchange kraken --symbol BTC/USDT --timeframe 1h --limit 1000
quant-agents backtest --exchange kraken --symbol BTC/USDT --timeframe 1h --fast-window 20 --slow-window 50
quant-agents backtest --exchange kraken --symbol BTC/USDT --timeframe 1h --fast-window 20 --slow-window 50 --fee-bps 5 --slippage-bps 2.5
quant-agents backtest --exchange kraken --symbol BTC/USDT --timeframe 1h --input-file /mnt/quant-data/raw/exchange=kraken/symbol=BTC-USDT/interval=1h/year=2026/month=06/ohlcv_YYYYMMDDTHHMMSSZ.parquet
quant-agents archive-backtest --strategy sma_crossover
```

### Reporting and daily workflow
```bash path=null start=null
quant-agents report --exchange kraken --symbol BTC/USDT --timeframe 1h
quant-agents run-daily --exchange kraken --symbol BTC/USDT --timeframe 1h --limit 1000
quant-agents run-daily --exchange kraken --symbol BTC/USDT --timeframe 1h --limit 1000 --fee-bps 5 --slippage-bps 2.5
quant-agents run-daily --exchange kraken --symbol BTC/USDT --timeframe 1h --limit 1000 --archive-backtest
```

### Full orchestration run
```bash path=null start=null
quant-agents agent-plane --exchange kraken --symbol BTC/USDT --timeframe 1h
quant-agents agent-plane --exchange kraken --symbol BTC/USDT --timeframe 1h --min-total-return 0.01 --min-sharpe 0.2 --max-drawdown -0.15 --min-signal-confidence 0.6 --step-retries 2
quant-agents agent-plane --exchange kraken --symbol BTC/USDT --timeframe 1h --max-cost-return-drag 0.06 --min-walkforward-quality-score 0.43 --backtest-fee-bps 5 --backtest-slippage-bps 2.5 --walkforward-fee-bps 5 --walkforward-slippage-bps 2.5
quant-agents agent-plane --exchange kraken --symbol BTC/USDT --timeframe 1h --paper-notional-usd 100 --paper-starting-cash-usd 10000 --paper-fee-bps 5 --paper-slippage-bps 1
quant-agents agent-plane --exchange kraken --symbol BTC/USDT --timeframe 1h --ensemble-mode adaptive --ensemble-arms sma_baseline,technical_composite,llm_context --ensemble-decay-horizon 48 --ensemble-exploration-weight 0.15 --ensemble-turnover-penalty-bps 8
```

### Paper-account connectivity checks
```bash path=null start=null
quant-agents paper-account-check
PAPER_ACCOUNT_PROVIDER=ccxt PAPER_ACCOUNT_EXCHANGE=binance PAPER_ACCOUNT_SANDBOX=1 PAPER_ACCOUNT_API_KEY={{PAPER_ACCOUNT_API_KEY}} PAPER_ACCOUNT_API_SECRET={{PAPER_ACCOUNT_API_SECRET}} quant-agents paper-account-check
```

### Visualization
```bash path=null start=null
quant-agents visualize-run
quant-agents visualize-run --run-dir /mnt/quant-data/logs/agents/openclaw-orchestrator/2026-06-05/20260605T024447Z --output-dir /mnt/quant-data/logs/agents/openclaw-orchestrator/2026-06-05/20260605T024447Z/visuals
```

### Trigger model and continuous monitoring
```bash path=null start=null
quant-agents train-trigger-model --exchange kraken --symbol BTC/USDT --timeframe 1h --cost-bps 7.5 --optimize-thresholds
quant-agents predict-trigger --exchange kraken --symbol BTC/USDT --timeframe 1h
quant-agents monitor-triggers --exchange kraken --symbol BTC/USDT --timeframe 1h --poll-seconds 3600 --confidence-threshold 0.60
quant-agents monitor-triggers --exchange kraken --symbol BTC/USDT --timeframe 1h --webhook-url {{TRIGGER_MONITOR_WEBHOOK_URL}} --max-cycles 3
python scripts/backfill_trigger_history.py --exchange kraken --symbol BTC/USDT --timeframe 1h --points 480 --alert-confidence-threshold 0.60
quant-agents train-trigger-model --exchange binanceus --symbol BTC/USDT --timeframe 1h --horizon-bars 2 --buy-threshold 0.005 --sell-threshold 0.005 --cost-bps 9 --no-optimize-thresholds --input-file /mnt/quant-data/curated/training/ohlcv_binanceus_BTC-USDT_1h_20260610T173442Z_train_preweek.parquet
```

### OpenClaw-native orchestration commands
```bash path=null start=null
quant-openclaw-entrypoint --job-mode submit --request-json '{"exchange":"kraken","symbol":"BTC/USDT","timeframe":"1h","strategy_model":"llama3.1:8b","ops_model":"llama3.1:8b"}' --print-json
quant-openclaw-entrypoint --job-mode status --job-id {{JOB_ID}} --print-json
quant-openclaw-entrypoint --job-mode wait --job-id {{JOB_ID}} --print-json
quant-openclaw-entrypoint --job-mode run-sync --request-json '{"exchange":"kraken","symbol":"BTC/USDT","timeframe":"1h","strategy_model":"llama3.1:8b","ops_model":"llama3.1:8b"}' --print-json
```

### Next.js dashboard project
The repository now includes a dashboard project under `apps/quant-dashboard` that reads trigger prediction and alert artifacts from `QUANT_DATA_ROOT`.
```bash path=null start=null
cd apps/quant-dashboard
npm install
QUANT_DATA_ROOT=/mnt/quant-data npm run dev
```
Then open `http://localhost:3000` and use the dashboard to:
- view stacked synchronized panels (price/SMA panel + oscillator panel),
- toggle BTC/SMA/MACD/RSI/volatility/MACD histogram overlays,
- adjust panel heights with slider controls,
- toggle prediction/alert markers directly on the chart,
- click markers to inspect confidence/probabilities/reason details below the chart,
- switch between `Signals & Markers` and `Model & Trade Performance` tabs,
- filter recommendations with multi-select controls (including `buy + sell` quick-select).

Model metric definitions and current performance snapshot:
- [doc/MODEL_METRIC_DEFINITIONS_AND_RESULTS.md](doc/MODEL_METRIC_DEFINITIONS_AND_RESULTS.md)

### Refreshing clustered development artifacts
If chart markers are clustered from old development data, use this refresh sequence:
```bash path=null start=null
systemctl --user stop quant-trigger-monitor.service
quant-agents ingest --exchange kraken --symbol BTC/USDT --timeframe 1h --limit 1500
quant-agents train-trigger-model --exchange kraken --symbol BTC/USDT --timeframe 1h
python scripts/backfill_trigger_history.py --exchange kraken --symbol BTC/USDT --timeframe 1h --points 480 --alert-confidence-threshold 0.60 --clear-existing
systemctl --user daemon-reload
systemctl --user restart quant-trigger-monitor.service
systemctl --user --no-pager status quant-trigger-monitor.service
```

## Output layout
All paths below are relative to `QUANT_DATA_ROOT`.

- Raw market data: `raw/exchange=<exchange>/symbol=<pair>/interval=<tf>/year=<yyyy>/month=<mm>/`
- Backtests: `backtests/<strategy_name>/<run_id>/`  
  Includes `metrics.json`, `equity_curve.parquet`, `run_manifest.json`; `metrics.json` carries gross+net return diagnostics (including turnover and cost drag), and adaptive ensemble runs also include `arm_attribution.parquet`.
- Orchestration runs: `logs/agents/openclaw-orchestrator/<yyyy-mm-dd>/<run_id>/`  
  Includes:
  - `data_quality_signal.json`
  - `strategy_proposal_signal.json`
  - `backtest_evaluation.json`
  - `risk_decision.json`
  - `paper_trade_intent.json`
  - `paper_trade_execution.json`
  - `ensemble_performance_update.json`
  - `ops_report.md`
  - `ops_report_contract.json`
  - `run_manifest.json`
  - `steps/<step-name>/attempt_*.json`
- Paper-trading state:
  - `paper-trading/state/portfolio_state.json`
  - `paper-trading/state/ensemble_weight_state.json`
  - `paper-trading/<yyyy-mm-dd>/fills.jsonl`
  - `paper-trading/<yyyy-mm-dd>/paper_trade_execution_<run_id>.json`
- Metrics:
  - `logs/metrics/<yyyy-mm-dd>/pipeline_metrics.jsonl`
  - `logs/metrics/summary.json`
- Trigger models:
  - `models/trigger-models/exchange=<exchange>/symbol=<pair>/interval=<tf>/<run_id>/model.json`
  - `models/trigger-models/exchange=<exchange>/symbol=<pair>/interval=<tf>/<run_id>/train_dataset.parquet`
  - `models/trigger-models/exchange=<exchange>/symbol=<pair>/interval=<tf>/<run_id>/test_dataset.parquet`
- Trigger predictions and alerts:
  - `logs/agents/model-predictor/<yyyy-mm-dd>/prediction_<timestamp>.json`
  - `logs/agents/trigger-monitor/<yyyy-mm-dd>/alerts.jsonl`
  - `logs/agents/trigger-monitor/state.json`
- Archives:
  - `archive/monthly/<yyyy-mm>/backtests/sma_crossover/<run_id>.tar.gz`
  - `archive/monthly/<yyyy-mm>/backtests/sma_crossover/<run_id>.tar.gz.sha256`

## Troubleshooting notes
- If models are unavailable, strategy/report steps retry and may fall back; inspect warnings in `strategy_proposal_signal.json` and `ops_report_contract.json`.
- If data quality fails due to low history, ingest more candles or lower `AGENT_MINIMUM_BARS` for controlled testing.
- If risk gate fails, intent stays blocked and paper execution is skipped by design.
- If a `sell` intent is emitted while no long inventory is open, execution is rejected with `reason=no_long_position_to_sell` by design.
- TradingView built-in paper trading is not exposed as a direct public trading API; use TradingView alerts/webhooks plus a broker/testnet API path.
