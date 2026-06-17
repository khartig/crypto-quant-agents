from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_agents.storage import symbol_slug

PRIORITY2_FEATURE_COLUMNS: tuple[str, ...] = (
    "funding_rate_feature",
    "open_interest_feature",
    "basis_feature",
    "liquidation_intensity_feature",
    "vol_term_structure_feature",
    "volume_imbalance_4",
    "volume_imbalance_24",
    "momentum_persistence_6",
    "momentum_persistence_24",
    "whale_flow_imbalance_feature",
    "whale_transfer_spike_feature",
    "participant_positioning_feature",
    "concentration_spike_feature",
)

_EXTERNAL_FEATURE_COLUMN_MAP: dict[str, str] = {
    "funding_rate": "funding_rate_feature",
    "funding_rate_feature": "funding_rate_feature",
    "open_interest": "open_interest_feature",
    "open_interest_feature": "open_interest_feature",
    "basis": "basis_feature",
    "basis_feature": "basis_feature",
    "liquidation_intensity": "liquidation_intensity_feature",
    "liquidation_intensity_feature": "liquidation_intensity_feature",
    "vol_term_structure": "vol_term_structure_feature",
    "vol_term_structure_feature": "vol_term_structure_feature",
    "volume_imbalance_4": "volume_imbalance_4",
    "volume_imbalance_24": "volume_imbalance_24",
    "momentum_persistence_6": "momentum_persistence_6",
    "momentum_persistence_24": "momentum_persistence_24",
    "whale_flow_imbalance": "whale_flow_imbalance_feature",
    "whale_flow_imbalance_feature": "whale_flow_imbalance_feature",
    "whale_transfer_spike": "whale_transfer_spike_feature",
    "whale_transfer_spike_feature": "whale_transfer_spike_feature",
    "participant_positioning": "participant_positioning_feature",
    "participant_positioning_feature": "participant_positioning_feature",
    "concentration_spike": "concentration_spike_feature",
    "concentration_spike_feature": "concentration_spike_feature",
}


@dataclass(frozen=True)
class Priority2FeatureBundle:
    contract: str
    created_at_utc: str
    features_enabled: bool
    external_features_path: str | None
    feature_frame: pd.DataFrame
    feature_snapshot: dict[str, float]
    reason_codes: list[str]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class Priority2FeatureArtifactPaths:
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


def _coerce_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Priority 2 feature build missing required market columns: {missing}")
    output = frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    output = output.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    )
    for column in ("open", "high", "low", "close", "volume"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    if output.empty:
        raise RuntimeError("Priority 2 feature build found no usable market rows.")
    return output


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=max(4, window // 3)).mean()
    std = series.rolling(window=window, min_periods=max(4, window // 3)).std(ddof=0).replace(0.0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _compute_proxy_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)

    ret_1 = close.pct_change(periods=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_8 = close.pct_change(periods=8).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_24 = close.pct_change(periods=24).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_96 = close.pct_change(periods=96).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol_24 = ret_1.rolling(window=24, min_periods=8).std(ddof=0).fillna(0.0)
    vol_96 = ret_1.rolling(window=96, min_periods=24).std(ddof=0).fillna(0.0)
    volume_z_24 = _rolling_zscore(volume, window=24)
    volume_z_48 = _rolling_zscore(volume, window=48)

    funding_proxy = (ret_8 - ret_1).ewm(span=12, adjust=False).mean()
    open_interest_proxy = (
        volume.rolling(window=24, min_periods=8).mean()
        / volume.rolling(window=96, min_periods=24).mean().replace(0.0, np.nan)
        - 1.0
    ).replace([np.inf, -np.inf], np.nan)
    basis_proxy = (
        close / close.rolling(window=24, min_periods=8).mean().replace(0.0, np.nan) - 1.0
    ).replace([np.inf, -np.inf], np.nan)
    liquidation_proxy = (
        ret_1.abs() * volume_z_24.abs()
    ).replace([np.inf, -np.inf], np.nan)
    vol_term_structure_proxy = vol_24 - vol_96

    up_volume = volume.where(ret_1 > 0.0, 0.0)
    down_volume = volume.where(ret_1 < 0.0, 0.0)
    up_4 = up_volume.rolling(window=4, min_periods=2).sum()
    down_4 = down_volume.rolling(window=4, min_periods=2).sum()
    up_24 = up_volume.rolling(window=24, min_periods=8).sum()
    down_24 = down_volume.rolling(window=24, min_periods=8).sum()
    volume_imbalance_4 = (
        (up_4 - down_4)
        / (up_4 + down_4).replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    volume_imbalance_24 = (
        (up_24 - down_24)
        / (up_24 + down_24).replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    direction = np.sign(ret_1).replace(0.0, np.nan)
    same_direction = (direction == direction.shift(1)).astype(float)
    momentum_persistence_6 = same_direction.rolling(window=6, min_periods=3).mean()
    momentum_persistence_24 = same_direction.rolling(window=24, min_periods=8).mean()

    whale_flow_proxy = volume_imbalance_24.fillna(0.0) * volume_z_48.abs()
    whale_transfer_proxy = np.maximum(0.0, volume_z_24 - 2.0)
    participant_positioning_proxy = ret_24 - ret_96
    concentration_spike_proxy = (
        ret_1.abs().rolling(window=24, min_periods=8).max() * volume_z_48.abs()
    )

    output = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "funding_rate_feature": funding_proxy,
            "open_interest_feature": open_interest_proxy,
            "basis_feature": basis_proxy,
            "liquidation_intensity_feature": liquidation_proxy,
            "vol_term_structure_feature": vol_term_structure_proxy,
            "volume_imbalance_4": volume_imbalance_4,
            "volume_imbalance_24": volume_imbalance_24,
            "momentum_persistence_6": momentum_persistence_6,
            "momentum_persistence_24": momentum_persistence_24,
            "whale_flow_imbalance_feature": whale_flow_proxy,
            "whale_transfer_spike_feature": whale_transfer_proxy,
            "participant_positioning_feature": participant_positioning_proxy,
            "concentration_spike_feature": concentration_spike_proxy,
        }
    )
    for column in PRIORITY2_FEATURE_COLUMNS:
        output[column] = (
            pd.to_numeric(output[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=-8.0, upper=8.0)
        )
    output["realized_volatility_proxy_24"] = vol_24.fillna(0.0).clip(lower=0.0, upper=2.0)
    output["high_low_range_proxy_14"] = (
        ((high - low) / close.replace(0.0, np.nan))
        .rolling(window=14, min_periods=5)
        .mean()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0, upper=2.0)
    )
    return output


def _read_external_feature_table(path: Path) -> pd.DataFrame:
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
        raise RuntimeError(f"Unsupported external Priority 2 feature format: {path}")
    if "timestamp" not in frame.columns:
        raise RuntimeError("External Priority 2 feature file must include `timestamp` column.")
    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    output = output.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    )
    if output.empty:
        raise RuntimeError("External Priority 2 feature file has no usable timestamp rows.")
    return output.reset_index(drop=True)


def _normalize_external_features(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for column in frame.columns:
        normalized = _EXTERNAL_FEATURE_COLUMN_MAP.get(str(column).strip().lower())
        if normalized:
            rename_map[column] = normalized
    output = frame.rename(columns=rename_map).copy()
    output = output.loc[:, ["timestamp", *[col for col in PRIORITY2_FEATURE_COLUMNS if col in output.columns]]]
    for column in PRIORITY2_FEATURE_COLUMNS:
        if column not in output.columns:
            output[column] = np.nan
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output.sort_values("timestamp").reset_index(drop=True)


def _align_external_features(market_timestamps: pd.Series, external: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame({"timestamp": pd.to_datetime(market_timestamps, utc=True, errors="coerce")})
    base = base.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    aligned = pd.merge_asof(
        base,
        external.rename(columns={"timestamp": "external_timestamp"}),
        left_on="timestamp",
        right_on="external_timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    for column in PRIORITY2_FEATURE_COLUMNS:
        if column not in aligned.columns:
            aligned[column] = np.nan
    return aligned


def build_priority2_feature_bundle(
    market_frame: pd.DataFrame,
    *,
    features_enabled: bool,
    external_features_path: Path | None,
) -> Priority2FeatureBundle:
    created_at_utc = _utc_now_iso()
    normalized_market = _coerce_market_frame(market_frame)
    proxy_features = _compute_proxy_features(normalized_market)
    proxy_feature_map = proxy_features.set_index("timestamp")

    reason_codes: list[str] = []
    diagnostics: dict[str, Any] = {
        "feature_version": "priority2_features.v1",
        "rows": int(len(proxy_features)),
        "external_features_path": str(external_features_path) if external_features_path else None,
        "external_features_loaded": False,
    }

    if not features_enabled:
        feature_frame = proxy_features.loc[:, ["timestamp", *PRIORITY2_FEATURE_COLUMNS]].copy()
        feature_snapshot = {
            column: float(feature_frame[column].iloc[-1]) if not feature_frame.empty else 0.0
            for column in PRIORITY2_FEATURE_COLUMNS
        }
        reason_codes.append("priority2_features_disabled")
        diagnostics["source_selection"] = {column: "proxy" for column in PRIORITY2_FEATURE_COLUMNS}
        diagnostics["quality_score"] = 0.0
        return Priority2FeatureBundle(
            contract="priority2_feature_bundle.v1",
            created_at_utc=created_at_utc,
            features_enabled=False,
            external_features_path=str(external_features_path) if external_features_path else None,
            feature_frame=feature_frame,
            feature_snapshot=feature_snapshot,
            reason_codes=reason_codes,
            diagnostics=diagnostics,
        )

    external_aligned: pd.DataFrame | None = None
    if external_features_path is not None:
        resolved_path = external_features_path.expanduser().resolve()
        if resolved_path.exists():
            external_raw = _read_external_feature_table(resolved_path)
            external_normalized = _normalize_external_features(external_raw)
            external_aligned = _align_external_features(proxy_features["timestamp"], external_normalized)
            diagnostics["external_features_loaded"] = True
            diagnostics["external_rows"] = int(len(external_normalized))
            latency_seconds = (
                proxy_features["timestamp"] - pd.to_datetime(external_aligned["external_timestamp"], utc=True, errors="coerce")
            ).dt.total_seconds()
            diagnostics["external_alignment_latency_seconds"] = {
                "median": _safe_float(latency_seconds.median()),
                "p95": _safe_float(latency_seconds.quantile(0.95)),
                "max": _safe_float(latency_seconds.max()),
            }
            reason_codes.append("priority2_external_features_ingested")
        else:
            diagnostics["external_features_missing"] = str(resolved_path)
            reason_codes.append("priority2_external_features_missing")

    merged = proxy_features.loc[:, ["timestamp", *PRIORITY2_FEATURE_COLUMNS]].copy()
    source_selection: dict[str, str] = {column: "proxy" for column in PRIORITY2_FEATURE_COLUMNS}
    external_coverage: dict[str, float] = {}
    if external_aligned is not None:
        for column in PRIORITY2_FEATURE_COLUMNS:
            external_values = pd.to_numeric(external_aligned[column], errors="coerce")
            external_available = external_values.notna()
            external_coverage[column] = float(external_available.mean())
            merged[column] = np.where(external_available, external_values.to_numpy(dtype=float), merged[column])
            if bool(external_available.any()):
                source_selection[column] = "external+proxy_fallback"
        diagnostics["external_feature_coverage"] = external_coverage
    diagnostics["source_selection"] = source_selection

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    for column in PRIORITY2_FEATURE_COLUMNS:
        fallback_series = (
            proxy_feature_map[column]
            .reindex(merged["timestamp"])
            .fillna(0.0)
            .reset_index(drop=True)
        )
        series = pd.to_numeric(merged[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        merged[column] = (
            series.where(series.notna(), fallback_series)
            .fillna(0.0)
            .clip(lower=-8.0, upper=8.0)
        )
    feature_snapshot = {
        column: float(merged[column].iloc[-1]) if not merged.empty else 0.0
        for column in PRIORITY2_FEATURE_COLUMNS
    }
    quality_components = [
        float((merged[column].notna()).mean())
        for column in PRIORITY2_FEATURE_COLUMNS
    ]
    diagnostics["quality_score"] = float(np.clip(np.mean(quality_components), 0.0, 1.0))
    reason_codes.append("priority2_features_ready")
    if diagnostics.get("external_features_loaded"):
        reason_codes.append("priority2_deterministic_timestamp_alignment_applied")
    else:
        reason_codes.append("priority2_proxy_mode_active")
    return Priority2FeatureBundle(
        contract="priority2_feature_bundle.v1",
        created_at_utc=created_at_utc,
        features_enabled=True,
        external_features_path=str(external_features_path) if external_features_path else None,
        feature_frame=merged,
        feature_snapshot=feature_snapshot,
        reason_codes=sorted(set(reason_codes)),
        diagnostics=diagnostics,
    )


def write_priority2_feature_artifacts(
    *,
    quant_data_root: Path,
    exchange: str,
    symbol: str,
    timeframe: str,
    run_id: str,
    bundle: Priority2FeatureBundle,
) -> Priority2FeatureArtifactPaths:
    base = (
        quant_data_root
        / "curated"
        / "features"
        / f"exchange={exchange}"
        / f"symbol={symbol_slug(symbol)}"
        / f"interval={timeframe}"
        / f"run_id={run_id}"
    )
    base.mkdir(parents=True, exist_ok=True)
    parquet_path = base / "priority2_features.parquet"
    contract_path = base / "priority2_feature_contract.json"
    bundle.feature_frame.to_parquet(parquet_path, index=False)
    payload = {
        "contract": bundle.contract,
        "created_at_utc": bundle.created_at_utc,
        "features_enabled": bundle.features_enabled,
        "external_features_path": bundle.external_features_path,
        "feature_columns": list(PRIORITY2_FEATURE_COLUMNS),
        "row_count": int(len(bundle.feature_frame)),
        "feature_snapshot": bundle.feature_snapshot,
        "reason_codes": list(bundle.reason_codes),
        "diagnostics": bundle.diagnostics,
        "parquet_path": str(parquet_path),
    }
    contract_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return Priority2FeatureArtifactPaths(
        parquet_path=parquet_path,
        contract_path=contract_path,
    )
