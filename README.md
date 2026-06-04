# Crypto Quant Agents — Phase 1 Implementation
This repository includes a runnable Phase 1 baseline pipeline:
1. Ingest OHLCV market data from an exchange (`ccxt`).
2. Run a deterministic SMA crossover backtest.
3. Generate a daily markdown operations report.

All large outputs are written under the external storage root (`QUANT_DATA_ROOT`), defaulting to `/mnt/quant-data`.

## Prerequisites
- Python 3.11+
- External storage mounted and available at `/mnt/quant-data` (or set `QUANT_DATA_ROOT`)

## Setup
```bash path=null start=null
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
cp .env.example .env
```

## Commands
### Preflight checks
```bash path=null start=null
quant-phase1 doctor
```

### Ingest market data
```bash path=null start=null
quant-phase1 ingest --exchange kraken --symbol BTC/USDT --timeframe 1h --limit 1000
```

### Run baseline backtest
```bash path=null start=null
quant-phase1 backtest --exchange kraken --symbol BTC/USDT --timeframe 1h --fast-window 20 --slow-window 50
```

### Run reproducible backtest from an explicit input dataset
```bash path=null start=null
quant-phase1 backtest --exchange kraken --symbol BTC/USDT --timeframe 1h --input-file /mnt/quant-data/raw/exchange=kraken/symbol=BTC-USDT/interval=1h/year=2026/month=06/ohlcv_YYYYMMDDTHHMMSSZ.parquet
```

### Archive a backtest run
```bash path=null start=null
quant-phase1 archive-backtest --strategy sma_crossover
```

### Generate daily report
```bash path=null start=null
quant-phase1 report --exchange kraken --symbol BTC/USDT --timeframe 1h
```

### Run end-to-end daily workflow
```bash path=null start=null
quant-phase1 run-daily --exchange kraken --symbol BTC/USDT --timeframe 1h --limit 1000
```

### Run daily workflow and archive generated backtest
```bash path=null start=null
quant-phase1 run-daily --exchange kraken --symbol BTC/USDT --timeframe 1h --limit 1000 --archive-backtest
```

### Enforce exchange-secret readiness
```bash path=null start=null
quant-phase1 doctor --require-secrets
quant-phase1 run-daily --require-secrets
```

## Secrets baseline
- Populate exchange credentials through `.env` or shell environment variables:
  - `EXCHANGE_API_KEY`
  - `EXCHANGE_API_SECRET`
  - optional: `EXCHANGE_API_PASSPHRASE`
- To make secret checks mandatory in daily workflow and doctor checks, set:
  - `REQUIRE_EXCHANGE_SECRETS=1`

## Output locations
- Raw market data: `raw/exchange=<exchange>/symbol=<pair>/interval=<tf>/year=<yyyy>/month=<mm>/`
- Backtests: `backtests/sma_crossover/<run_id>/`
  - includes `metrics.json`, `equity_curve.parquet`, `run_manifest.json`
- Ops reports: `logs/agents/ops-report-agent/<yyyy-mm-dd>/`
- Operational metrics baseline:
  - JSONL events: `logs/metrics/<yyyy-mm-dd>/pipeline_metrics.jsonl`
  - summary counters: `logs/metrics/summary.json`
- Backtest archives:
  - tarballs: `archive/monthly/<yyyy-mm>/backtests/sma_crossover/<run_id>.tar.gz`
  - checksums: `archive/monthly/<yyyy-mm>/backtests/sma_crossover/<run_id>.tar.gz.sha256`

All paths above are relative to `QUANT_DATA_ROOT`.
