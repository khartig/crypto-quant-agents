from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd

from quant_agents.config import Settings
from quant_agents.storage import symbol_slug

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderBookCaptureResult:
    output_path: Path
    row_count: int
    start_timestamp: pd.Timestamp | None
    end_timestamp: pd.Timestamp | None
    depth_limit: int
    sample_interval_seconds: float


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not pd.notna(parsed):
        return None
    return parsed


def _sum_depth(levels: Any) -> tuple[float, float]:
    qty_total = 0.0
    notional_total = 0.0
    if not isinstance(levels, list):
        return qty_total, notional_total
    for level in levels:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            continue
        price = _safe_float(level[0])
        size = _safe_float(level[1])
        if price is None or size is None:
            continue
        price = max(0.0, float(price))
        size = max(0.0, float(size))
        qty_total += size
        notional_total += price * size
    return qty_total, notional_total


def _best_level(levels: Any) -> tuple[float | None, float | None]:
    if not isinstance(levels, list):
        return None, None
    for level in levels:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            continue
        price = _safe_float(level[0])
        size = _safe_float(level[1])
        if price is None or size is None:
            continue
        return max(0.0, float(price)), max(0.0, float(size))
    return None, None


def _raw_orderbook_snapshot_dir(root: Path, exchange: str, symbol: str, ts: datetime) -> Path:
    return (
        root
        / "raw"
        / "orderbook"
        / f"exchange={exchange}"
        / f"symbol={symbol_slug(symbol)}"
        / f"year={ts:%Y}"
        / f"month={ts:%m}"
    )


def latest_orderbook_snapshot_dataset(root: Path, exchange: str, symbol: str) -> Path:
    base = root / "raw" / "orderbook" / f"exchange={exchange}" / f"symbol={symbol_slug(symbol)}"
    if not base.exists():
        raise FileNotFoundError(f"No order book snapshot directory found: {base}")
    candidates = sorted(base.rglob("orderbook_snapshots_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No order book snapshot parquet files found under: {base}")
    return candidates[-1]


def _build_snapshot_row(
    *,
    exchange_id: str,
    symbol: str,
    depth_limit: int,
    snapshot: dict[str, Any],
    captured_at: datetime,
) -> dict[str, Any]:
    bids = snapshot.get("bids", [])
    asks = snapshot.get("asks", [])
    top_bids = bids[:depth_limit] if isinstance(bids, list) else []
    top_asks = asks[:depth_limit] if isinstance(asks, list) else []

    best_bid_price, best_bid_size = _best_level(top_bids)
    best_ask_price, best_ask_size = _best_level(top_asks)
    bid_qty_top, bid_notional_top = _sum_depth(top_bids)
    ask_qty_top, ask_notional_top = _sum_depth(top_asks)

    mid_price = None
    spread = None
    spread_bps = None
    microprice = None
    microprice_deviation_bps = None
    if (
        best_bid_price is not None
        and best_ask_price is not None
        and best_bid_price > 0.0
        and best_ask_price > 0.0
    ):
        mid_price = (best_bid_price + best_ask_price) / 2.0
        spread = max(0.0, best_ask_price - best_bid_price)
        spread_bps = (spread / mid_price) * 10_000.0 if mid_price > 0.0 else None
        if best_bid_size is not None and best_ask_size is not None:
            denom = max(0.0, best_bid_size) + max(0.0, best_ask_size)
            if denom > 0.0:
                microprice = (
                    (best_ask_price * max(0.0, best_bid_size))
                    + (best_bid_price * max(0.0, best_ask_size))
                ) / denom
                microprice_deviation_bps = (
                    ((microprice - mid_price) / mid_price) * 10_000.0
                    if mid_price > 0.0
                    else None
                )

    depth_notional_total = bid_notional_top + ask_notional_top
    top_level_imbalance = None
    if best_bid_size is not None and best_ask_size is not None:
        denom = max(0.0, best_bid_size) + max(0.0, best_ask_size)
        if denom > 0.0:
            top_level_imbalance = (max(0.0, best_bid_size) - max(0.0, best_ask_size)) / denom

    depth_imbalance_notional = None
    if depth_notional_total > 0.0:
        depth_imbalance_notional = (bid_notional_top - ask_notional_top) / depth_notional_total

    snapshot_timestamp_ms = snapshot.get("timestamp")
    if isinstance(snapshot_timestamp_ms, (int, float)):
        snapshot_timestamp = pd.to_datetime(snapshot_timestamp_ms, unit="ms", utc=True, errors="coerce")
        if pd.isna(snapshot_timestamp):
            snapshot_timestamp = pd.Timestamp(captured_at)
    else:
        snapshot_timestamp = pd.Timestamp(captured_at)

    return {
        "timestamp": snapshot_timestamp,
        "captured_at_utc": captured_at.isoformat(),
        "exchange": exchange_id,
        "symbol": symbol,
        "depth_limit": int(depth_limit),
        "raw_bid_levels": int(len(bids) if isinstance(bids, list) else 0),
        "raw_ask_levels": int(len(asks) if isinstance(asks, list) else 0),
        "best_bid_price": best_bid_price,
        "best_bid_size": best_bid_size,
        "best_ask_price": best_ask_price,
        "best_ask_size": best_ask_size,
        "mid_price": mid_price,
        "spread": spread,
        "spread_bps": spread_bps,
        "microprice": microprice,
        "microprice_deviation_bps": microprice_deviation_bps,
        "top_level_imbalance": top_level_imbalance,
        "bid_qty_top": bid_qty_top,
        "ask_qty_top": ask_qty_top,
        "bid_notional_top": bid_notional_top,
        "ask_notional_top": ask_notional_top,
        "depth_notional_total": depth_notional_total,
        "depth_imbalance_notional": depth_imbalance_notional,
    }


def capture_orderbook_snapshots_to_parquet(
    *,
    settings: Settings,
    exchange_id: str,
    symbol: str,
    sample_count: int = 120,
    sample_interval_seconds: float = 1.0,
    depth_limit: int = 50,
) -> OrderBookCaptureResult:
    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"Unsupported exchange id: {exchange_id}")

    exchange = exchange_class({"enableRateLimit": True})
    logger.info("Loading markets for exchange=%s", exchange_id)
    exchange.load_markets()
    if symbol not in exchange.markets:
        raise ValueError(f"Symbol {symbol} is not available on exchange {exchange_id}")

    resolved_count = max(1, int(sample_count))
    resolved_interval = max(0.0, float(sample_interval_seconds))
    resolved_depth = max(1, int(depth_limit))
    rows: list[dict[str, Any]] = []

    logger.info(
        "Capturing order book snapshots exchange=%s symbol=%s count=%s interval_seconds=%.3f depth_limit=%s",
        exchange_id,
        symbol,
        resolved_count,
        resolved_interval,
        resolved_depth,
    )

    for index in range(resolved_count):
        captured_at = datetime.now(timezone.utc)
        snapshot = exchange.fetch_order_book(symbol, limit=resolved_depth)
        rows.append(
            _build_snapshot_row(
                exchange_id=exchange_id,
                symbol=symbol,
                depth_limit=resolved_depth,
                snapshot=snapshot if isinstance(snapshot, dict) else {},
                captured_at=captured_at,
            )
        )
        if index < resolved_count - 1 and resolved_interval > 0.0:
            time.sleep(resolved_interval)

    frame = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    now_utc = datetime.now(timezone.utc)
    out_dir = _raw_orderbook_snapshot_dir(settings.quant_data_root, exchange_id, symbol, now_utc)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"orderbook_snapshots_{now_utc:%Y%m%dT%H%M%SZ}.parquet"
    frame.to_parquet(out_path, index=False)

    return OrderBookCaptureResult(
        output_path=out_path,
        row_count=int(len(frame)),
        start_timestamp=frame["timestamp"].min() if not frame.empty else None,
        end_timestamp=frame["timestamp"].max() if not frame.empty else None,
        depth_limit=resolved_depth,
        sample_interval_seconds=resolved_interval,
    )
