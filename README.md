# Crypto Quant Agents — Phase 1 Implementation
This repository now includes a runnable Phase 1 baseline pipeline:
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
### Ingest market data
```bash path=null start=null
quant-phase1 ingest --exchange kraken --symbol BTC/USDT --timeframe 1h --limit 1000
```

### Run baseline backtest
```bash path=null start=null
quant-phase1 backtest --exchange kraken --symbol BTC/USDT --timeframe 1h --fast-window 20 --slow-window 50
```

### Generate daily report
```bash path=null start=null
quant-phase1 report --exchange kraken --symbol BTC/USDT --timeframe 1h
```

### Run end-to-end daily workflow
```bash path=null start=null
quant-phase1 run-daily --exchange kraken --symbol BTC/USDT --timeframe 1h --limit 1000
```

## Output locations
- Raw market data: `raw/exchange=<exchange>/symbol=<pair>/interval=<tf>/year=<yyyy>/month=<mm>/`
- Backtests: `backtests/sma_crossover/<run_id>/`
- Ops reports: `logs/agents/ops-report-agent/<yyyy-mm-dd>/`

All paths above are relative to `QUANT_DATA_ROOT`.

