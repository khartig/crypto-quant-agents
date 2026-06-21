from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_agents.storage import symbol_slug

ORDERBOOK_FEATURE_COLUMNS: tuple[str, ...] = (
    "orderbook_spread_feature",
    "orderbook_top_level_imbalance_feature",
    "orderbook_depth_imbalance_feature",
    "orderbook_microprice_deviation_feature",
    "orderbook_depth_pressure_feature",
    "orderbook_depth_notional_zscore_feature",
)

DEFAULT_STABLE_ORDERBOOK_FEATURE_COLUMNS: tuple[str, ...] = (
    "orderbook_spread_feature",
    "orderbook_depth_imbalance_feature",
    "orderbook_microprice_deviation_feature",
)

_ORDERBOOK_COLUMN_ALIASES: dict[str, str] = {
    "best_bid": "best_bid_price",
    "bid": "best_bid_price",
    "best_bid_price": "best_bid_price",
    "bid_price": "best_bid_price",
    "best_bid_size": "best_bid_size",
    "bid_size": "best_bid_size",
    "best_ask": "best_ask_price",
    "ask": "best_ask_price",
    "best_ask_price": "best_ask_price",
    "ask_price": "best_ask_price",
    "best_ask_size": "best_ask_size",
    "ask_size": "best_ask_size",
    "bid_notional_top": "bid_notional_top",
    "ask_notional_top": "ask_notional_top",
    "depth_notional_total": "depth_notional_total",
    "top_level_imbalance": "top_level_imbalance",
    "depth_imbalance_notional": "depth_imbalance_notional",
    "spread_bps": "spread_bps",
    "microprice_deviation_bps": "microprice_deviation_bps",
    "orderbook_spread_feature": "orderbook_spread_feature",
    "orderbook_top_level_imbalance_feature": "orderbook_top_level_imbalance_feature",
    "orderbook_depth_imbalance_feature": "orderbook_depth_imbalance_feature",
    "orderbook_microprice_deviation_feature": "orderbook_microprice_deviation_feature",
    "orderbook_depth_pressure_feature": "orderbook_depth_pressure_feature",
    "orderbook_depth_notional_zscore_feature": "orderbook_depth_notional_zscore_feature",
}


@dataclass(frozen=True)
class OrderBookFeatureBundle:
    contract: str
    created_at_utc: str
    features_enabled: bool
    source_path: str | None
    feature_frame: pd.DataFrame
    feature_snapshot: dict[str, float]
    reason_codes: list[str]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class OrderBookFeatureArtifactPaths:
    parquet_path: Path
    contract_path: Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=max(4, window // 3)).mean()
    std = series.rolling(window=window, min_periods=max(4, window // 3)).std(ddof=0).replace(0.0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan)


def normalize_orderbook_feature_columns(
    selected_feature_columns: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if not selected_feature_columns:
        return tuple(ORDERBOOK_FEATURE_COLUMNS)
    normalized: list[str] = []
    for column in selected_feature_columns:
        value = str(column).strip()
        if not value:
            continue
        if value not in ORDERBOOK_FEATURE_COLUMNS:
            continue
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized) if normalized else tuple(ORDERBOOK_FEATURE_COLUMNS)


def apply_orderbook_feature_column_selection(
    *,
    bundle: OrderBookFeatureBundle,
    selected_feature_columns: tuple[str, ...] | list[str] | None,
) -> OrderBookFeatureBundle:
    selected_columns = normalize_orderbook_feature_columns(selected_feature_columns)
    disabled_columns = [column for column in ORDERBOOK_FEATURE_COLUMNS if column not in selected_columns]
    diagnostics = dict(bundle.diagnostics)
    diagnostics["selected_orderbook_feature_columns"] = list(selected_columns)
    diagnostics["disabled_orderbook_feature_columns"] = list(disabled_columns)
    adjusted_frame = bundle.feature_frame.copy()
    for column in ORDERBOOK_FEATURE_COLUMNS:
        if column not in adjusted_frame.columns:
            adjusted_frame[column] = 0.0
        if column in disabled_columns:
            adjusted_frame[column] = 0.0
    adjusted_frame = adjusted_frame.loc[:, ["timestamp", *ORDERBOOK_FEATURE_COLUMNS]].copy()
    adjusted_snapshot = {
        column: float(pd.to_numeric(adjusted_frame[column], errors="coerce").fillna(0.0).iloc[-1])
        if not adjusted_frame.empty
        else 0.0
        for column in ORDERBOOK_FEATURE_COLUMNS
    }
    source_selection = dict(diagnostics.get("source_selection", {}))
    for column in disabled_columns:
        source_selection[column] = "disabled_by_feature_selection"
    diagnostics["source_selection"] = source_selection

    raw_coverage = {
        str(column): float(value)
        for column, value in dict(diagnostics.get("orderbook_feature_raw_coverage", {})).items()
        if isinstance(column, str)
    }
    non_zero_coverage = {
        str(column): float(value)
        for column, value in dict(diagnostics.get("orderbook_feature_non_zero_coverage", {})).items()
        if isinstance(column, str)
    }
    fallback_rate = {
        str(column): float(value)
        for column, value in dict(diagnostics.get("orderbook_feature_fallback_rate", {})).items()
        if isinstance(column, str)
    }
    selected_raw_coverage = float(
        np.mean([raw_coverage.get(column, 0.0) for column in selected_columns])
    ) if selected_columns else 0.0
    selected_non_zero_coverage = float(
        np.mean([non_zero_coverage.get(column, 0.0) for column in selected_columns])
    ) if selected_columns else 0.0
    selected_fallback_rate = float(
        np.mean([fallback_rate.get(column, 1.0) for column in selected_columns])
    ) if selected_columns else 1.0
    selected_effective_signal_score = float(selected_raw_coverage * selected_non_zero_coverage)
    diagnostics["selected_external_raw_coverage"] = selected_raw_coverage
    diagnostics["selected_non_zero_coverage"] = selected_non_zero_coverage
    diagnostics["selected_fallback_rate"] = selected_fallback_rate
    diagnostics["selected_effective_signal_score"] = selected_effective_signal_score
    diagnostics["quality_score"] = selected_effective_signal_score

    reason_codes = list(bundle.reason_codes)
    if disabled_columns:
        reason_codes.append("orderbook_feature_column_selection_applied")
    return OrderBookFeatureBundle(
        contract=bundle.contract,
        created_at_utc=bundle.created_at_utc,
        features_enabled=bundle.features_enabled,
        source_path=bundle.source_path,
        feature_frame=adjusted_frame,
        feature_snapshot=adjusted_snapshot,
        reason_codes=sorted(set(reason_codes)),
        diagnostics=diagnostics,
    )


def _coerce_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Order book feature build missing required market columns: {missing}")
    output = frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    output = output.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    )
    for column in ("open", "high", "low", "close", "volume"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    if output.empty:
        raise RuntimeError("Order book feature build found no usable market rows.")
    return output


def _read_orderbook_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("rows", [])
            frame = pd.DataFrame(rows if isinstance(rows, list) else [])
        elif isinstance(payload, list):
            frame = pd.DataFrame(payload)
        else:
            frame = pd.DataFrame()
    else:
        raise RuntimeError(f"Unsupported order book feature format: {path}")
    if "timestamp" not in frame.columns:
        raise RuntimeError("Order book source file must include `timestamp` column.")
    output = frame.copy()
    rename_map: dict[str, str] = {}
    for column in output.columns:
        normalized = _ORDERBOOK_COLUMN_ALIASES.get(str(column).strip().lower())
        if normalized:
            rename_map[column] = normalized
    output = output.rename(columns=rename_map)
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    output = output.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    )
    if output.empty:
        raise RuntimeError("Order book source file has no usable timestamp rows.")
    return output.reset_index(drop=True)


def _engineer_orderbook_features(orderbook_input: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame({"timestamp": orderbook_input["timestamp"]})

    def numeric(name: str) -> pd.Series:
        if name in orderbook_input.columns:
            return pd.to_numeric(orderbook_input[name], errors="coerce")
        return pd.Series(np.nan, index=orderbook_input.index, dtype=float)

    best_bid_price = numeric("best_bid_price")
    best_bid_size = numeric("best_bid_size")
    best_ask_price = numeric("best_ask_price")
    best_ask_size = numeric("best_ask_size")
    bid_notional_top = numeric("bid_notional_top")
    ask_notional_top = numeric("ask_notional_top")
    depth_notional_total = numeric("depth_notional_total")

    mid_price = (best_bid_price + best_ask_price) / 2.0
    spread_bps = numeric("spread_bps")
    spread_bps = spread_bps.where(spread_bps.notna(), ((best_ask_price - best_bid_price) / mid_price) * 10_000.0)
    spread_feature = (spread_bps / 100.0).replace([np.inf, -np.inf], np.nan)

    top_level_imbalance = numeric("top_level_imbalance")
    top_level_imbalance = top_level_imbalance.where(
        top_level_imbalance.notna(),
        (best_bid_size - best_ask_size) / (best_bid_size + best_ask_size).replace(0.0, np.nan),
    )

    depth_imbalance = numeric("depth_imbalance_notional")
    depth_imbalance = depth_imbalance.where(
        depth_imbalance.notna(),
        (bid_notional_top - ask_notional_top) / (bid_notional_top + ask_notional_top).replace(0.0, np.nan),
    )

    microprice_deviation_bps = numeric("microprice_deviation_bps")
    if microprice_deviation_bps.notna().sum() <= 0:
        denom = (best_bid_size + best_ask_size).replace(0.0, np.nan)
        microprice = ((best_ask_price * best_bid_size) + (best_bid_price * best_ask_size)) / denom
        microprice_deviation_bps = ((microprice - mid_price) / mid_price) * 10_000.0
    microprice_feature = (microprice_deviation_bps / 100.0).replace([np.inf, -np.inf], np.nan)

    depth_pressure = np.log1p(bid_notional_top.clip(lower=0.0)) - np.log1p(ask_notional_top.clip(lower=0.0))
    depth_notional_total = depth_notional_total.where(
        depth_notional_total.notna(),
        bid_notional_top.clip(lower=0.0) + ask_notional_top.clip(lower=0.0),
    )
    depth_notional_zscore = _rolling_zscore(depth_notional_total.fillna(0.0), window=48)

    output["orderbook_spread_feature"] = spread_feature
    output["orderbook_top_level_imbalance_feature"] = top_level_imbalance
    output["orderbook_depth_imbalance_feature"] = depth_imbalance
    output["orderbook_microprice_deviation_feature"] = microprice_feature
    output["orderbook_depth_pressure_feature"] = depth_pressure
    output["orderbook_depth_notional_zscore_feature"] = depth_notional_zscore

    for column in ORDERBOOK_FEATURE_COLUMNS:
        output[column] = (
            pd.to_numeric(output[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .clip(lower=-8.0, upper=8.0)
        )
    return output


def _align_orderbook_features(market_timestamps: pd.Series, orderbook_features: pd.DataFrame) -> pd.DataFrame:
    market = pd.DataFrame({"timestamp": pd.to_datetime(market_timestamps, utc=True, errors="coerce")})
    market = market.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if not market.empty:
        market["timestamp"] = market["timestamp"].dt.as_unit("ns")
    if orderbook_features.empty:
        output = market.copy()
        output["external_timestamp"] = pd.NaT
        for column in ORDERBOOK_FEATURE_COLUMNS:
            output[column] = np.nan
        return output
    orderbook = orderbook_features.copy()
    orderbook["timestamp"] = pd.to_datetime(orderbook["timestamp"], utc=True, errors="coerce")
    orderbook = (
        orderbook
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )
    if orderbook.empty:
        output = market.copy()
        output["external_timestamp"] = pd.NaT
        for column in ORDERBOOK_FEATURE_COLUMNS:
            output[column] = np.nan
        return output
    orderbook["timestamp"] = orderbook["timestamp"].dt.as_unit("ns")
    aligned = pd.merge_asof(
        market,
        orderbook.rename(columns={"timestamp": "external_timestamp"}),
        left_on="timestamp",
        right_on="external_timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    for column in ORDERBOOK_FEATURE_COLUMNS:
        if column not in aligned.columns:
            aligned[column] = np.nan
    return aligned


def build_orderbook_feature_bundle(
    market_frame: pd.DataFrame,
    *,
    features_enabled: bool,
    orderbook_features_path: Path | None,
) -> OrderBookFeatureBundle:
    created_at_utc = _utc_now_iso()
    normalized_market = _coerce_market_frame(market_frame)
    zero_frame = pd.DataFrame({"timestamp": normalized_market["timestamp"]})
    for column in ORDERBOOK_FEATURE_COLUMNS:
        zero_frame[column] = 0.0
    raw_coverage_map: dict[str, float] = {column: 0.0 for column in ORDERBOOK_FEATURE_COLUMNS}
    non_zero_coverage_map: dict[str, float] = {column: 0.0 for column in ORDERBOOK_FEATURE_COLUMNS}
    fallback_rate_map: dict[str, float] = {column: 1.0 for column in ORDERBOOK_FEATURE_COLUMNS}

    diagnostics: dict[str, Any] = {
        "feature_version": "orderbook_features.v1",
        "rows": int(len(normalized_market)),
        "source_path": str(orderbook_features_path) if orderbook_features_path else None,
        "external_features_loaded": False,
        "orderbook_feature_raw_coverage": dict(raw_coverage_map),
        "orderbook_feature_non_zero_coverage": dict(non_zero_coverage_map),
        "orderbook_feature_fallback_rate": dict(fallback_rate_map),
        "selected_external_raw_coverage": 0.0,
        "selected_non_zero_coverage": 0.0,
        "selected_fallback_rate": 1.0,
        "selected_effective_signal_score": 0.0,
        "quality_score": 0.0,
    }
    reason_codes: list[str] = []
    if not features_enabled:
        diagnostics["source_selection"] = {column: "disabled" for column in ORDERBOOK_FEATURE_COLUMNS}
        reason_codes.append("orderbook_features_disabled")
        snapshot = {
            column: float(zero_frame[column].iloc[-1]) if not zero_frame.empty else 0.0
            for column in ORDERBOOK_FEATURE_COLUMNS
        }
        return OrderBookFeatureBundle(
            contract="orderbook_feature_bundle.v1",
            created_at_utc=created_at_utc,
            features_enabled=False,
            source_path=str(orderbook_features_path) if orderbook_features_path else None,
            feature_frame=zero_frame,
            feature_snapshot=snapshot,
            reason_codes=reason_codes,
            diagnostics=diagnostics,
        )

    if orderbook_features_path is None:
        diagnostics["source_selection"] = {column: "zero_fallback" for column in ORDERBOOK_FEATURE_COLUMNS}
        reason_codes.extend(["orderbook_features_path_unset", "orderbook_zero_fallback"])
        snapshot = {
            column: float(zero_frame[column].iloc[-1]) if not zero_frame.empty else 0.0
            for column in ORDERBOOK_FEATURE_COLUMNS
        }
        return OrderBookFeatureBundle(
            contract="orderbook_feature_bundle.v1",
            created_at_utc=created_at_utc,
            features_enabled=True,
            source_path=None,
            feature_frame=zero_frame,
            feature_snapshot=snapshot,
            reason_codes=reason_codes,
            diagnostics=diagnostics,
        )

    resolved_path = orderbook_features_path.expanduser().resolve()
    if not resolved_path.exists():
        diagnostics["source_selection"] = {column: "missing_source_zero_fallback" for column in ORDERBOOK_FEATURE_COLUMNS}
        diagnostics["missing_source_path"] = str(resolved_path)
        reason_codes.extend(["orderbook_source_missing", "orderbook_zero_fallback"])
        snapshot = {
            column: float(zero_frame[column].iloc[-1]) if not zero_frame.empty else 0.0
            for column in ORDERBOOK_FEATURE_COLUMNS
        }
        return OrderBookFeatureBundle(
            contract="orderbook_feature_bundle.v1",
            created_at_utc=created_at_utc,
            features_enabled=True,
            source_path=str(resolved_path),
            feature_frame=zero_frame,
            feature_snapshot=snapshot,
            reason_codes=reason_codes,
            diagnostics=diagnostics,
        )

    source_table = _read_orderbook_table(resolved_path)
    engineered = _engineer_orderbook_features(source_table)
    output_table = engineered.copy()
    direct_feature_coverage: dict[str, float] = {}
    for column in ORDERBOOK_FEATURE_COLUMNS:
        if column in source_table.columns:
            direct = pd.to_numeric(source_table[column], errors="coerce")
            direct_feature_coverage[column] = float(direct.notna().mean())
            output_table[column] = np.where(direct.notna(), direct, output_table[column])
        else:
            direct_feature_coverage[column] = 0.0

    aligned = _align_orderbook_features(normalized_market["timestamp"], output_table)
    raw_coverage_by_column = {
        column: float(pd.to_numeric(aligned[column], errors="coerce").notna().mean())
        for column in ORDERBOOK_FEATURE_COLUMNS
    }
    fallback_rate_by_column = {
        column: float(max(0.0, min(1.0, 1.0 - raw_coverage_by_column[column])))
        for column in ORDERBOOK_FEATURE_COLUMNS
    }
    for column in ORDERBOOK_FEATURE_COLUMNS:
        aligned[column] = (
            pd.to_numeric(aligned[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=-8.0, upper=8.0)
        )
    non_zero_coverage_by_column = {
        column: float(pd.to_numeric(aligned[column], errors="coerce").fillna(0.0).abs().gt(1e-12).mean())
        for column in ORDERBOOK_FEATURE_COLUMNS
    }
    source_selection: dict[str, str] = {}
    for column in ORDERBOOK_FEATURE_COLUMNS:
        raw_cov = raw_coverage_by_column[column]
        non_zero_cov = non_zero_coverage_by_column[column]
        if raw_cov <= 0.0:
            source_selection[column] = "zero_fallback"
        elif non_zero_cov <= 0.0:
            source_selection[column] = "zero_only_external"
        elif raw_cov < 1.0:
            source_selection[column] = "external+zero_fallback"
        else:
            source_selection[column] = "external_only"

    quality_components = [
        float(raw_coverage_by_column[column] * non_zero_coverage_by_column[column])
        for column in ORDERBOOK_FEATURE_COLUMNS
    ]
    quality_score = float(np.mean(quality_components)) if quality_components else 0.0
    latency_seconds = (
        pd.to_datetime(aligned["timestamp"], utc=True, errors="coerce")
        - pd.to_datetime(aligned.get("external_timestamp"), utc=True, errors="coerce")
    ).dt.total_seconds()
    diagnostics["external_features_loaded"] = True
    diagnostics["source_rows"] = int(len(source_table))
    diagnostics["source_selection"] = source_selection
    diagnostics["direct_feature_coverage"] = direct_feature_coverage
    diagnostics["orderbook_feature_coverage"] = raw_coverage_by_column
    diagnostics["orderbook_feature_raw_coverage"] = raw_coverage_by_column
    diagnostics["orderbook_feature_non_zero_coverage"] = non_zero_coverage_by_column
    diagnostics["orderbook_feature_fallback_rate"] = fallback_rate_by_column
    diagnostics["selected_external_raw_coverage"] = float(
        np.mean(list(raw_coverage_by_column.values()))
    ) if raw_coverage_by_column else 0.0
    diagnostics["selected_non_zero_coverage"] = float(
        np.mean(list(non_zero_coverage_by_column.values()))
    ) if non_zero_coverage_by_column else 0.0
    diagnostics["selected_fallback_rate"] = float(
        np.mean(list(fallback_rate_by_column.values()))
    ) if fallback_rate_by_column else 1.0
    diagnostics["selected_effective_signal_score"] = float(
        diagnostics["selected_external_raw_coverage"] * diagnostics["selected_non_zero_coverage"]
    )
    diagnostics["orderbook_alignment_latency_seconds"] = {
        "median": _safe_float(latency_seconds.median()),
        "p95": _safe_float(latency_seconds.quantile(0.95)),
        "max": _safe_float(latency_seconds.max()),
    }
    diagnostics["quality_score"] = quality_score
    reason_codes.extend(
        [
            "orderbook_source_ingested",
            "orderbook_features_engineered",
            "orderbook_features_ready",
        ]
    )

    snapshot = {
        column: float(aligned[column].iloc[-1]) if not aligned.empty else 0.0
        for column in ORDERBOOK_FEATURE_COLUMNS
    }
    return OrderBookFeatureBundle(
        contract="orderbook_feature_bundle.v1",
        created_at_utc=created_at_utc,
        features_enabled=True,
        source_path=str(resolved_path),
        feature_frame=aligned.loc[:, ["timestamp", *ORDERBOOK_FEATURE_COLUMNS]].copy(),
        feature_snapshot=snapshot,
        reason_codes=sorted(set(reason_codes)),
        diagnostics=diagnostics,
    )


def write_orderbook_feature_artifacts(
    *,
    quant_data_root: Path,
    exchange: str,
    symbol: str,
    timeframe: str,
    run_id: str,
    bundle: OrderBookFeatureBundle,
) -> OrderBookFeatureArtifactPaths:
    base_dir = (
        quant_data_root
        / "curated"
        / "features"
        / "orderbook-model"
        / f"exchange={exchange}"
        / f"symbol={symbol_slug(symbol)}"
        / f"interval={timeframe}"
        / f"run_id={run_id}"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = base_dir / "orderbook_model_features.parquet"
    contract_path = base_dir / "orderbook_model_feature_contract.json"
    bundle.feature_frame.to_parquet(parquet_path, index=False)
    contract_payload = {
        "contract": "orderbook_model_feature_contract.v1",
        "created_at_utc": _utc_now_iso(),
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "run_id": run_id,
        "features_enabled": bool(bundle.features_enabled),
        "source_path": bundle.source_path,
        "feature_columns": list(ORDERBOOK_FEATURE_COLUMNS),
        "feature_snapshot": dict(bundle.feature_snapshot),
        "reason_codes": list(bundle.reason_codes),
        "diagnostics": dict(bundle.diagnostics),
        "artifacts": {
            "feature_parquet_path": str(parquet_path),
            "feature_contract_path": str(contract_path),
        },
    }
    contract_path.write_text(json.dumps(contract_payload, indent=2), encoding="utf-8")
    return OrderBookFeatureArtifactPaths(parquet_path=parquet_path, contract_path=contract_path)
