from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant_agents.config import Settings
from quant_agents.orderbook_features import (
    DEFAULT_STABLE_ORDERBOOK_FEATURE_COLUMNS,
    ORDERBOOK_FEATURE_COLUMNS,
    apply_orderbook_feature_column_selection,
    build_orderbook_feature_bundle,
    normalize_orderbook_feature_columns,
)
from quant_agents.orderbook_ingestion import latest_orderbook_snapshot_dataset
from quant_agents.storage import latest_raw_dataset, symbol_slug


@dataclass(frozen=True)
class OrderBookFeatureRetrievalResult:
    run_id: str
    source_data_path: Path
    source_data_sha256: str
    snapshot_source_path: Path
    parquet_path: Path
    contract_path: Path
    row_count: int
    coverage_ratio: float
    reason_codes: tuple[str, ...]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _base_output_dir(root: Path, exchange: str, symbol: str, timeframe: str) -> Path:
    return (
        root
        / "curated"
        / "features"
        / "orderbook"
        / f"exchange={exchange}"
        / f"symbol={symbol_slug(symbol)}"
        / f"interval={timeframe}"
    )


def latest_orderbook_features_path(
    *,
    quant_data_root: Path,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> Path | None:
    base_dir = _base_output_dir(quant_data_root, exchange, symbol, timeframe)
    pointer_path = base_dir / "latest_orderbook_features_path.txt"
    if pointer_path.exists():
        pointer_value = pointer_path.read_text(encoding="utf-8").strip()
        if pointer_value:
            candidate = Path(pointer_value).expanduser()
            if candidate.exists():
                return candidate.resolve()
    candidates = sorted(base_dir.glob("run_id=*/orderbook_features.parquet"))
    if not candidates:
        return None
    return candidates[-1].resolve()


def retrieve_orderbook_features(
    *,
    settings: Settings,
    exchange: str,
    symbol: str,
    timeframe: str,
    source_data_path: Path | None = None,
    snapshot_source_path: Path | None = None,
    orderbook_feature_columns: tuple[str, ...] | list[str] | None = None,
) -> OrderBookFeatureRetrievalResult:
    resolved_source = (
        source_data_path.expanduser().resolve()
        if source_data_path is not None
        else latest_raw_dataset(settings.quant_data_root, exchange, symbol, timeframe)
    )
    if not resolved_source.exists():
        raise FileNotFoundError(f"Source data file not found: {resolved_source}")
    resolved_snapshot_source = (
        snapshot_source_path.expanduser().resolve()
        if snapshot_source_path is not None
        else latest_orderbook_snapshot_dataset(settings.quant_data_root, exchange, symbol)
    )
    if not resolved_snapshot_source.exists():
        raise FileNotFoundError(f"Order book snapshot source file not found: {resolved_snapshot_source}")

    market_frame = pd.read_parquet(resolved_source)
    selected_feature_columns = normalize_orderbook_feature_columns(
        orderbook_feature_columns
        if orderbook_feature_columns is not None
        else DEFAULT_STABLE_ORDERBOOK_FEATURE_COLUMNS
    )
    bundle = build_orderbook_feature_bundle(
        market_frame=market_frame,
        features_enabled=True,
        orderbook_features_path=resolved_snapshot_source,
    )
    bundle = apply_orderbook_feature_column_selection(
        bundle=bundle,
        selected_feature_columns=selected_feature_columns,
    )
    feature_frame = bundle.feature_frame.copy()
    coverage_by_column = {
        column: float(pd.to_numeric(feature_frame[column], errors="coerce").replace(0.0, np.nan).notna().mean())
        for column in ORDERBOOK_FEATURE_COLUMNS
    }
    coverage_ratio = float(np.mean(list(coverage_by_column.values()))) if coverage_by_column else 0.0

    run_id = _new_run_id()
    base_dir = _base_output_dir(settings.quant_data_root, exchange, symbol, timeframe)
    output_dir = base_dir / f"run_id={run_id}"
    suffix = 0
    while output_dir.exists():
        suffix += 1
        output_dir = base_dir / f"run_id={run_id}_{suffix:02d}"
    output_dir.mkdir(parents=True, exist_ok=False)

    parquet_path = output_dir / "orderbook_features.parquet"
    contract_path = output_dir / "orderbook_feature_contract.json"
    latest_pointer_path = base_dir / "latest_orderbook_features_path.txt"
    latest_contract_pointer_path = base_dir / "latest_orderbook_feature_contract_path.txt"
    feature_frame.to_parquet(parquet_path, index=False)
    latest_pointer_path.parent.mkdir(parents=True, exist_ok=True)
    latest_pointer_path.write_text(str(parquet_path) + "\n", encoding="utf-8")
    latest_contract_pointer_path.write_text(str(contract_path) + "\n", encoding="utf-8")

    source_data_sha256 = _sha256_file(resolved_source)
    reason_codes = sorted(set([*bundle.reason_codes, "orderbook_features_retrieval_ready"]))
    payload = {
        "contract": "orderbook_feature_retrieval.v1",
        "created_at_utc": _utc_now_iso(),
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "source_data_path": str(resolved_source),
        "source_data_sha256": source_data_sha256,
        "snapshot_source_path": str(resolved_snapshot_source),
        "feature_columns": list(ORDERBOOK_FEATURE_COLUMNS),
        "selected_feature_columns": list(selected_feature_columns),
        "row_count": int(len(feature_frame)),
        "coverage_ratio": coverage_ratio,
        "column_coverage": coverage_by_column,
        "reason_codes": reason_codes,
        "diagnostics": dict(bundle.diagnostics),
        "artifacts": {
            "output_dir": str(output_dir),
            "orderbook_features_parquet": str(parquet_path),
            "latest_orderbook_features_pointer": str(latest_pointer_path),
            "latest_orderbook_contract_pointer": str(latest_contract_pointer_path),
        },
    }
    contract_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return OrderBookFeatureRetrievalResult(
        run_id=output_dir.name.replace("run_id=", ""),
        source_data_path=resolved_source,
        source_data_sha256=source_data_sha256,
        snapshot_source_path=resolved_snapshot_source,
        parquet_path=parquet_path,
        contract_path=contract_path,
        row_count=int(len(feature_frame)),
        coverage_ratio=coverage_ratio,
        reason_codes=tuple(reason_codes),
    )
