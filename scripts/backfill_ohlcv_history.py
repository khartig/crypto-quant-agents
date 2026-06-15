#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd

from quant_agents.config import load_settings
from quant_agents.storage import raw_dataset_dir

REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")


def _parse_timestamp(raw: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(str(raw).strip())
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill continuous OHLCV history over an explicit date range and write a "
            "single parquet snapshot under QUANT_DATA_ROOT/raw."
        )
    )
    parser.add_argument("--exchange", default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument(
        "--start-utc",
        required=True,
        help="Inclusive range start timestamp in UTC (for example 2025-01-01T00:00:00Z).",
    )
    parser.add_argument(
        "--end-utc",
        required=True,
        help="Exclusive range end timestamp in UTC (for example 2026-06-13T00:00:00Z).",
    )
    parser.add_argument(
        "--chunk-limit",
        type=int,
        default=1000,
        help="Per-request OHLCV row limit (default: 1000).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=5000,
        help="Maximum paginated fetch requests before abort (default: 5000).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow writes even when fetched range does not fully cover requested start/end.",
    )
    return parser


def _timeframe_milliseconds(exchange: ccxt.Exchange, timeframe: str) -> int:
    seconds = int(exchange.parse_timeframe(timeframe))
    if seconds <= 0:
        raise ValueError(f"Unable to resolve timeframe seconds for `{timeframe}`.")
    return seconds * 1000


def _fetch_backfill_rows(
    *,
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    chunk_limit: int,
    max_batches: int,
    timeframe_ms: int,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    cursor_ms = int(start_ms)
    for _batch_index in range(max(1, int(max_batches))):
        if cursor_ms >= end_ms:
            break
        batch = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=cursor_ms,
            limit=max(1, int(chunk_limit)),
        )
        if not batch:
            break

        filtered = [item for item in batch if int(item[0]) >= start_ms and int(item[0]) < end_ms]
        rows.extend(filtered)

        last_ts = int(batch[-1][0])
        next_cursor = max(cursor_ms + timeframe_ms, last_ts + timeframe_ms)
        if next_cursor <= cursor_ms:
            break
        cursor_ms = next_cursor
    return rows


def _coerce_rows(rows: list[list[Any]], *, exchange: str, symbol: str, timeframe: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    frame = frame.reset_index(drop=True)
    frame["exchange"] = exchange
    frame["symbol"] = symbol
    frame["timeframe"] = timeframe
    frame["ingested_at"] = pd.Timestamp.now(tz="UTC")
    return frame


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    settings = load_settings()
    exchange_id = args.exchange or settings.default_exchange
    symbol = args.symbol or settings.default_symbol
    timeframe = args.timeframe or settings.default_timeframe

    start_utc = _parse_timestamp(args.start_utc)
    end_utc = _parse_timestamp(args.end_utc)
    if not start_utc < end_utc:
        raise ValueError("Invalid range: --start-utc must be < --end-utc.")

    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"Unsupported exchange id: {exchange_id}")
    exchange = exchange_class({"enableRateLimit": True})
    exchange.load_markets()
    if symbol not in exchange.markets:
        raise ValueError(f"Symbol {symbol} is not available on exchange {exchange_id}")

    timeframe_ms = _timeframe_milliseconds(exchange, timeframe)
    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)

    rows = _fetch_backfill_rows(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_ms=start_ms,
        end_ms=end_ms,
        chunk_limit=max(1, int(args.chunk_limit)),
        max_batches=max(1, int(args.max_batches)),
        timeframe_ms=timeframe_ms,
    )
    if not rows:
        raise RuntimeError("No OHLCV rows were fetched for the requested range.")

    frame = _coerce_rows(rows, exchange=exchange_id, symbol=symbol, timeframe=timeframe)
    fetched_start = pd.Timestamp(frame["timestamp"].iloc[0]).tz_convert("UTC")
    fetched_end = pd.Timestamp(frame["timestamp"].iloc[-1]).tz_convert("UTC")
    required_end = end_utc - pd.Timedelta(milliseconds=timeframe_ms)
    if not args.allow_partial and (fetched_start > start_utc or fetched_end < required_end):
        raise RuntimeError(
            "Fetched range is partial. "
            + f"requested=[{start_utc.isoformat()}, {end_utc.isoformat()}) "
            + f"fetched=[{fetched_start.isoformat()}, {fetched_end.isoformat()}]. "
            + "Use --allow-partial to persist partial backfills."
        )

    now_utc = datetime.now(timezone.utc)
    output_dir = raw_dataset_dir(
        settings.quant_data_root,
        exchange=exchange_id,
        symbol=symbol,
        timeframe=timeframe,
        ts=now_utc,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    start_slug = start_utc.strftime("%Y%m%dT%H%M%SZ")
    end_slug = end_utc.strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"ohlcv_backfill_{start_slug}_{end_slug}_{now_utc:%Y%m%dT%H%M%SZ}.parquet"
    frame.to_parquet(output_path, index=False)

    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = {
        "contract": "ohlcv_backfill_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "exchange": exchange_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "requested_range": {
            "start_utc": start_utc.isoformat(),
            "end_utc": end_utc.isoformat(),
        },
        "fetched_range": {
            "start_utc": fetched_start.isoformat(),
            "end_utc": fetched_end.isoformat(),
        },
        "row_count": int(len(frame)),
        "output_path": str(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("OHLCV_BACKFILL_STATUS=PASS")
    print(f"OHLCV_BACKFILL_OUTPUT={output_path}")
    print(f"OHLCV_BACKFILL_MANIFEST={manifest_path}")
    print(f"OHLCV_BACKFILL_ROWS={len(frame)}")
    print(f"OHLCV_BACKFILL_FETCHED_START={fetched_start.isoformat()}")
    print(f"OHLCV_BACKFILL_FETCHED_END={fetched_end.isoformat()}")


if __name__ == "__main__":
    main()
