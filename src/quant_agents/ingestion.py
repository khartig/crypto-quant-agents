from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

from quant_agents.config import Settings
from quant_agents.storage import raw_dataset_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    output_path: Path
    row_count: int
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp


def fetch_ohlcv_to_parquet(
    settings: Settings,
    exchange_id: str,
    symbol: str,
    timeframe: str,
    limit: int = 1000,
) -> IngestionResult:
    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"Unsupported exchange id: {exchange_id}")

    exchange = exchange_class({"enableRateLimit": True})
    logger.info("Loading markets for exchange=%s", exchange_id)
    exchange.load_markets()
    if symbol not in exchange.markets:
        raise ValueError(f"Symbol {symbol} is not available on exchange {exchange_id}")

    logger.info(
        "Fetching OHLCV exchange=%s symbol=%s timeframe=%s limit=%s",
        exchange_id,
        symbol,
        timeframe,
        limit,
    )
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not rows:
        raise RuntimeError("No OHLCV rows returned from exchange.")

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["exchange"] = exchange_id
    df["symbol"] = symbol
    df["timeframe"] = timeframe
    df["ingested_at"] = pd.Timestamp.now(tz="UTC")

    now_utc = datetime.now(timezone.utc)
    out_dir = raw_dataset_dir(
        settings.quant_data_root,
        exchange=exchange_id,
        symbol=symbol,
        timeframe=timeframe,
        ts=now_utc,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ohlcv_{now_utc:%Y%m%dT%H%M%SZ}.parquet"
    df.to_parquet(out_path, index=False)

    result = IngestionResult(
        output_path=out_path,
        row_count=len(df),
        start_timestamp=df["timestamp"].min(),
        end_timestamp=df["timestamp"].max(),
    )
    logger.info("Wrote %s OHLCV rows to %s", result.row_count, result.output_path)
    return result

