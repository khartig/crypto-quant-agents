from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_agents.storage import symbol_slug

RANKED_FEATURE_COLUMNS: tuple[str, ...] = (
    "flow_taker_buy_share_6",
    "flow_taker_buy_share_24",
    "flow_signed_volume_imbalance_24",
    "flow_vwap_dislocation_12",
    "derivatives_open_interest_delta_24",
    "derivatives_funding_rate_z_24",
    "derivatives_basis_z_24",
    "derivatives_long_short_ratio_z_24",
    "onchain_exchange_netflow_z_24",
    "onchain_stablecoin_inflow_ratio_24",
    "onchain_exchange_reserve_delta_24",
    "options_put_call_oi_ratio_z_24",
    "options_iv_term_slope_7_30",
    "options_skew_25d_z_24",
    "regime_trend_strength_24",
    "regime_volatility_ratio_24_96",
    "regime_momentum_vol_adj_24",
)

DEFAULT_STABLE_RANKED_FEATURE_COLUMNS: tuple[str, ...] = (
    "flow_signed_volume_imbalance_24",
    "derivatives_open_interest_delta_24",
    "derivatives_basis_z_24",
    "onchain_exchange_netflow_z_24",
    "options_put_call_oi_ratio_z_24",
    "regime_trend_strength_24",
    "regime_momentum_vol_adj_24",
)

_EXTERNAL_FEATURE_COLUMN_MAP: dict[str, str] = {
    "flow_taker_buy_share_6": "flow_taker_buy_share_6",
    "taker_buy_share_6": "flow_taker_buy_share_6",
    "flow_taker_buy_share_24": "flow_taker_buy_share_24",
    "taker_buy_share_24": "flow_taker_buy_share_24",
    "flow_signed_volume_imbalance_24": "flow_signed_volume_imbalance_24",
    "signed_volume_imbalance_24": "flow_signed_volume_imbalance_24",
    "flow_vwap_dislocation_12": "flow_vwap_dislocation_12",
    "vwap_dislocation_12": "flow_vwap_dislocation_12",
    "derivatives_open_interest_delta_24": "derivatives_open_interest_delta_24",
    "open_interest_delta_24": "derivatives_open_interest_delta_24",
    "derivatives_funding_rate_z_24": "derivatives_funding_rate_z_24",
    "funding_rate_z_24": "derivatives_funding_rate_z_24",
    "derivatives_basis_z_24": "derivatives_basis_z_24",
    "basis_z_24": "derivatives_basis_z_24",
    "derivatives_long_short_ratio_z_24": "derivatives_long_short_ratio_z_24",
    "long_short_ratio_z_24": "derivatives_long_short_ratio_z_24",
    "onchain_exchange_netflow_z_24": "onchain_exchange_netflow_z_24",
    "exchange_netflow_z_24": "onchain_exchange_netflow_z_24",
    "onchain_stablecoin_inflow_ratio_24": "onchain_stablecoin_inflow_ratio_24",
    "stablecoin_inflow_ratio_24": "onchain_stablecoin_inflow_ratio_24",
    "onchain_exchange_reserve_delta_24": "onchain_exchange_reserve_delta_24",
    "exchange_reserve_delta_24": "onchain_exchange_reserve_delta_24",
    "options_put_call_oi_ratio_z_24": "options_put_call_oi_ratio_z_24",
    "put_call_oi_ratio_z_24": "options_put_call_oi_ratio_z_24",
    "options_iv_term_slope_7_30": "options_iv_term_slope_7_30",
    "iv_term_slope_7_30": "options_iv_term_slope_7_30",
    "options_skew_25d_z_24": "options_skew_25d_z_24",
    "skew_25d_z_24": "options_skew_25d_z_24",
    "regime_trend_strength_24": "regime_trend_strength_24",
    "trend_strength_24": "regime_trend_strength_24",
    "regime_volatility_ratio_24_96": "regime_volatility_ratio_24_96",
    "volatility_ratio_24_96": "regime_volatility_ratio_24_96",
    "regime_momentum_vol_adj_24": "regime_momentum_vol_adj_24",
    "momentum_vol_adj_24": "regime_momentum_vol_adj_24",
}


@dataclass(frozen=True)
class RankedFeatureBundle:
    contract: str
    created_at_utc: str
    features_enabled: bool
    external_features_path: str | None
    feature_frame: pd.DataFrame
    feature_snapshot: dict[str, float]
    reason_codes: list[str]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class RankedFeatureArtifactPaths:
    parquet_path: Path
    contract_path: Path


def normalize_ranked_feature_columns(
    selected_feature_columns: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if not selected_feature_columns:
        return tuple(RANKED_FEATURE_COLUMNS)
    normalized: list[str] = []
    for column in selected_feature_columns:
        value = str(column).strip()
        if not value:
            continue
        if value not in RANKED_FEATURE_COLUMNS:
            continue
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized) if normalized else tuple(RANKED_FEATURE_COLUMNS)


def apply_ranked_feature_column_selection(
    *,
    bundle: RankedFeatureBundle,
    selected_feature_columns: tuple[str, ...] | list[str] | None,
) -> RankedFeatureBundle:
    selected_columns = normalize_ranked_feature_columns(selected_feature_columns)
    disabled_columns = [column for column in RANKED_FEATURE_COLUMNS if column not in selected_columns]
    diagnostics = dict(bundle.diagnostics)
    diagnostics["selected_ranked_feature_columns"] = list(selected_columns)
    diagnostics["disabled_ranked_feature_columns"] = list(disabled_columns)
    adjusted_frame = bundle.feature_frame.copy()
    for column in RANKED_FEATURE_COLUMNS:
        if column not in adjusted_frame.columns:
            adjusted_frame[column] = 0.0
        if column in disabled_columns:
            adjusted_frame[column] = 0.0
    adjusted_frame = adjusted_frame.loc[:, ["timestamp", *RANKED_FEATURE_COLUMNS]].copy()
    adjusted_snapshot = {
        column: float(pd.to_numeric(adjusted_frame[column], errors="coerce").fillna(0.0).iloc[-1])
        if not adjusted_frame.empty
        else 0.0
        for column in RANKED_FEATURE_COLUMNS
    }
    source_selection = dict(diagnostics.get("source_selection", {}))
    for column in disabled_columns:
        source_selection[column] = "disabled_by_feature_selection"
    diagnostics["source_selection"] = source_selection

    external_raw_coverage = {
        str(column): float(value)
        for column, value in dict(diagnostics.get("external_raw_coverage", {})).items()
        if isinstance(column, str)
    }
    effective_non_zero_coverage = {
        str(column): float(value)
        for column, value in dict(diagnostics.get("effective_non_zero_coverage", {})).items()
        if isinstance(column, str)
    }
    proxy_fallback_rate = {
        str(column): float(value)
        for column, value in dict(diagnostics.get("proxy_fallback_rate", {})).items()
        if isinstance(column, str)
    }
    selected_external_raw_coverage = float(
        np.mean([external_raw_coverage.get(column, 0.0) for column in selected_columns])
    ) if selected_columns else 0.0
    selected_non_zero_coverage = float(
        np.mean([effective_non_zero_coverage.get(column, 0.0) for column in selected_columns])
    ) if selected_columns else 0.0
    selected_proxy_fallback_rate = float(
        np.mean([proxy_fallback_rate.get(column, 1.0) for column in selected_columns])
    ) if selected_columns else 1.0
    selected_effective_signal_score = float(
        selected_external_raw_coverage * selected_non_zero_coverage
    )
    diagnostics["selected_external_raw_coverage"] = selected_external_raw_coverage
    diagnostics["selected_non_zero_coverage"] = selected_non_zero_coverage
    diagnostics["selected_proxy_fallback_rate"] = selected_proxy_fallback_rate
    diagnostics["selected_effective_signal_score"] = selected_effective_signal_score
    diagnostics["quality_score"] = selected_effective_signal_score

    reason_codes = list(bundle.reason_codes)
    if disabled_columns:
        reason_codes.append("ranked_feature_column_selection_applied")
    return RankedFeatureBundle(
        contract=bundle.contract,
        created_at_utc=bundle.created_at_utc,
        features_enabled=bundle.features_enabled,
        external_features_path=bundle.external_features_path,
        feature_frame=adjusted_frame,
        feature_snapshot=adjusted_snapshot,
        reason_codes=sorted(set(reason_codes)),
        diagnostics=diagnostics,
    )


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
        raise RuntimeError(f"Ranked feature build missing required market columns: {missing}")
    output = frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    output = output.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    )
    for column in ("open", "high", "low", "close", "volume"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    if output.empty:
        raise RuntimeError("Ranked feature build found no usable market rows.")
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
    ret_4 = close.pct_change(periods=4).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_12 = close.pct_change(periods=12).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_24 = close.pct_change(periods=24).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_96 = close.pct_change(periods=96).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    up_volume = volume.where(ret_1 > 0.0, 0.0)
    down_volume = volume.where(ret_1 < 0.0, 0.0)
    up_6 = up_volume.rolling(window=6, min_periods=2).sum()
    down_6 = down_volume.rolling(window=6, min_periods=2).sum()
    up_24 = up_volume.rolling(window=24, min_periods=8).sum()
    down_24 = down_volume.rolling(window=24, min_periods=8).sum()
    volume_24 = volume.rolling(window=24, min_periods=8).sum()
    flow_taker_buy_share_6 = (up_6 / (up_6 + down_6).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    flow_taker_buy_share_24 = (up_24 / (up_24 + down_24).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    flow_signed_volume_imbalance_24 = (
        (up_24 - down_24) / (up_24 + down_24).replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    vwap_12 = (
        (close * volume).rolling(window=12, min_periods=4).sum()
        / volume.rolling(window=12, min_periods=4).sum().replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    flow_vwap_dislocation_12 = (close / vwap_12.replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)

    volume_ratio_24_96 = (
        volume.rolling(window=24, min_periods=8).mean()
        / volume.rolling(window=96, min_periods=24).mean().replace(0.0, np.nan)
        - 1.0
    ).replace([np.inf, -np.inf], np.nan)
    funding_proxy = (ret_4 - ret_1).ewm(span=12, adjust=False).mean()
    basis_proxy = (
        close / close.rolling(window=24, min_periods=8).mean().replace(0.0, np.nan) - 1.0
    ).replace([np.inf, -np.inf], np.nan)
    long_short_proxy = flow_signed_volume_imbalance_24.ewm(span=12, adjust=False).mean()

    exchange_netflow_proxy = (
        (down_24 - up_24) / volume_24.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    stablecoin_inflow_ratio = (
        (up_24 + 1.0) / (down_24 + 1.0)
    ).replace([np.inf, -np.inf], np.nan)
    stablecoin_inflow_ratio = np.log(stablecoin_inflow_ratio.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    reserve_delta_proxy = (-(ret_24) * (1.0 + _rolling_zscore(volume, window=24).abs())).replace(
        [np.inf, -np.inf], np.nan
    )

    downside_move = (-ret_1.clip(upper=0.0)).rolling(window=24, min_periods=8).mean()
    upside_move = ret_1.clip(lower=0.0).rolling(window=24, min_periods=8).mean()
    put_call_proxy = (
        (downside_move + 1e-6) / (upside_move + 1e-6)
    ).replace([np.inf, -np.inf], np.nan)
    vol_7 = ret_1.rolling(window=7, min_periods=3).std(ddof=0).fillna(0.0)
    vol_24 = ret_1.rolling(window=24, min_periods=8).std(ddof=0).fillna(0.0)
    vol_30 = ret_1.rolling(window=30, min_periods=10).std(ddof=0).fillna(0.0)
    vol_96 = ret_1.rolling(window=96, min_periods=24).std(ddof=0).fillna(0.0)
    tail_25 = ret_1.rolling(window=24, min_periods=8).quantile(0.25).abs()
    tail_75 = ret_1.rolling(window=24, min_periods=8).quantile(0.75).abs()
    skew_proxy = (tail_25 - tail_75).replace([np.inf, -np.inf], np.nan)

    trend_strength_24 = (
        close.rolling(window=24, min_periods=8).mean()
        / close.rolling(window=96, min_periods=24).mean().replace(0.0, np.nan)
        - 1.0
    ).replace([np.inf, -np.inf], np.nan)
    volatility_ratio_24_96 = (vol_24 / vol_96.replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)
    momentum_vol_adj_24 = (ret_24 / (vol_24 + 1e-6)).replace([np.inf, -np.inf], np.nan)

    output = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "flow_taker_buy_share_6": flow_taker_buy_share_6,
            "flow_taker_buy_share_24": flow_taker_buy_share_24,
            "flow_signed_volume_imbalance_24": flow_signed_volume_imbalance_24,
            "flow_vwap_dislocation_12": flow_vwap_dislocation_12,
            "derivatives_open_interest_delta_24": volume_ratio_24_96,
            "derivatives_funding_rate_z_24": _rolling_zscore(funding_proxy, window=24),
            "derivatives_basis_z_24": _rolling_zscore(basis_proxy.fillna(0.0), window=24),
            "derivatives_long_short_ratio_z_24": _rolling_zscore(long_short_proxy.fillna(0.0), window=24),
            "onchain_exchange_netflow_z_24": _rolling_zscore(exchange_netflow_proxy.fillna(0.0), window=24),
            "onchain_stablecoin_inflow_ratio_24": stablecoin_inflow_ratio,
            "onchain_exchange_reserve_delta_24": reserve_delta_proxy,
            "options_put_call_oi_ratio_z_24": _rolling_zscore(np.log(put_call_proxy + 1e-6).fillna(0.0), window=24),
            "options_iv_term_slope_7_30": (vol_7 - vol_30).replace([np.inf, -np.inf], np.nan),
            "options_skew_25d_z_24": _rolling_zscore(skew_proxy.fillna(0.0), window=24),
            "regime_trend_strength_24": trend_strength_24,
            "regime_volatility_ratio_24_96": volatility_ratio_24_96,
            "regime_momentum_vol_adj_24": momentum_vol_adj_24,
        }
    )
    for column in RANKED_FEATURE_COLUMNS:
        output[column] = (
            pd.to_numeric(output[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=-8.0, upper=8.0)
        )
    output["high_low_range_proxy_14"] = (
        ((high - low) / close.replace(0.0, np.nan))
        .rolling(window=14, min_periods=5)
        .mean()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0, upper=2.0)
    )
    output["realized_volatility_proxy_24"] = vol_24.fillna(0.0).clip(lower=0.0, upper=2.0)
    output["ret_12_proxy"] = ret_12.clip(lower=-2.0, upper=2.0)
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
        raise RuntimeError(f"Unsupported external ranked feature format: {path}")
    if "timestamp" not in frame.columns:
        raise RuntimeError("External ranked feature file must include `timestamp` column.")
    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    output = output.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    )
    if output.empty:
        raise RuntimeError("External ranked feature file has no usable timestamp rows.")
    return output.reset_index(drop=True)


def _normalize_external_features(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for column in frame.columns:
        normalized = _EXTERNAL_FEATURE_COLUMN_MAP.get(str(column).strip().lower())
        if normalized:
            rename_map[column] = normalized
    output = frame.rename(columns=rename_map).copy()
    output = output.loc[:, ["timestamp", *[col for col in RANKED_FEATURE_COLUMNS if col in output.columns]]]
    for column in RANKED_FEATURE_COLUMNS:
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
    for column in RANKED_FEATURE_COLUMNS:
        if column not in aligned.columns:
            aligned[column] = np.nan
    return aligned


def build_ranked_feature_bundle(
    market_frame: pd.DataFrame,
    *,
    features_enabled: bool,
    external_features_path: Path | None,
) -> RankedFeatureBundle:
    created_at_utc = _utc_now_iso()
    normalized_market = _coerce_market_frame(market_frame)
    proxy_features = _compute_proxy_features(normalized_market)
    proxy_feature_map = proxy_features.set_index("timestamp")
    zero_frame = pd.DataFrame({"timestamp": normalized_market["timestamp"]})
    for column in RANKED_FEATURE_COLUMNS:
        zero_frame[column] = 0.0

    reason_codes: list[str] = []
    external_raw_coverage: dict[str, float] = {column: 0.0 for column in RANKED_FEATURE_COLUMNS}
    effective_non_zero_coverage: dict[str, float] = {column: 0.0 for column in RANKED_FEATURE_COLUMNS}
    proxy_fallback_rate: dict[str, float] = {column: 1.0 for column in RANKED_FEATURE_COLUMNS}
    diagnostics: dict[str, Any] = {
        "feature_version": "ranked_features.v1",
        "rows": int(len(proxy_features)),
        "external_features_path": str(external_features_path) if external_features_path else None,
        "external_features_loaded": False,
        "external_raw_coverage": dict(external_raw_coverage),
        "effective_non_zero_coverage": dict(effective_non_zero_coverage),
        "proxy_fallback_rate": dict(proxy_fallback_rate),
        "quality_score": 0.0,
    }

    if not features_enabled:
        feature_frame = zero_frame.loc[:, ["timestamp", *RANKED_FEATURE_COLUMNS]].copy()
        feature_snapshot = {
            column: float(feature_frame[column].iloc[-1]) if not feature_frame.empty else 0.0
            for column in RANKED_FEATURE_COLUMNS
        }
        reason_codes.append("ranked_features_disabled")
        diagnostics["source_selection"] = {column: "disabled_zero_fallback" for column in RANKED_FEATURE_COLUMNS}
        diagnostics["selected_external_raw_coverage"] = 0.0
        diagnostics["selected_non_zero_coverage"] = 0.0
        diagnostics["selected_proxy_fallback_rate"] = 1.0
        diagnostics["selected_effective_signal_score"] = 0.0
        return RankedFeatureBundle(
            contract="ranked_feature_bundle.v1",
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
            reason_codes.append("ranked_external_features_ingested")
        else:
            diagnostics["external_features_missing"] = str(resolved_path)
            reason_codes.append("ranked_external_features_missing")

    merged = proxy_features.loc[:, ["timestamp", *RANKED_FEATURE_COLUMNS]].copy()
    source_selection: dict[str, str] = {column: "proxy" for column in RANKED_FEATURE_COLUMNS}
    external_coverage: dict[str, float] = {}
    selected_external_timestamp = pd.Series(pd.NaT, index=merged.index, dtype="datetime64[ns, UTC]")
    if external_aligned is not None:
        selected_external_timestamp = pd.to_datetime(
            external_aligned.get("external_timestamp"),
            utc=True,
            errors="coerce",
        )
        for column in RANKED_FEATURE_COLUMNS:
            external_values = pd.to_numeric(external_aligned[column], errors="coerce")
            external_available = external_values.notna()
            external_coverage[column] = float(external_available.mean())
            merged[column] = np.where(external_available, external_values.to_numpy(dtype=float), merged[column])
            if bool(external_available.any()):
                source_selection[column] = "external+proxy_fallback"
        diagnostics["external_feature_coverage"] = dict(external_coverage)
    diagnostics["source_selection"] = source_selection

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    for column in RANKED_FEATURE_COLUMNS:
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
        non_zero_mask = pd.to_numeric(merged[column], errors="coerce").fillna(0.0).abs() > 1e-12
        effective_non_zero_coverage[column] = float(non_zero_mask.mean())

    if external_aligned is not None:
        for column in RANKED_FEATURE_COLUMNS:
            raw_cov = float(external_coverage.get(column, 0.0))
            external_raw_coverage[column] = raw_cov
            proxy_fallback_rate[column] = float(max(0.0, min(1.0, 1.0 - raw_cov)))
    else:
        for column in RANKED_FEATURE_COLUMNS:
            external_raw_coverage[column] = 0.0
            proxy_fallback_rate[column] = 1.0
    feature_snapshot = {
        column: float(merged[column].iloc[-1]) if not merged.empty else 0.0
        for column in RANKED_FEATURE_COLUMNS
    }
    quality_components = [
        float(external_raw_coverage[column] * effective_non_zero_coverage[column])
        for column in RANKED_FEATURE_COLUMNS
    ]
    diagnostics["external_raw_coverage"] = dict(external_raw_coverage)
    diagnostics["effective_non_zero_coverage"] = dict(effective_non_zero_coverage)
    diagnostics["proxy_fallback_rate"] = dict(proxy_fallback_rate)
    diagnostics["selected_external_raw_coverage"] = float(
        np.mean(list(external_raw_coverage.values()))
    ) if external_raw_coverage else 0.0
    diagnostics["selected_non_zero_coverage"] = float(
        np.mean(list(effective_non_zero_coverage.values()))
    ) if effective_non_zero_coverage else 0.0
    diagnostics["selected_proxy_fallback_rate"] = float(
        np.mean(list(proxy_fallback_rate.values()))
    ) if proxy_fallback_rate else 1.0
    diagnostics["selected_effective_signal_score"] = float(
        diagnostics["selected_external_raw_coverage"] * diagnostics["selected_non_zero_coverage"]
    )
    diagnostics["quality_score"] = float(np.clip(np.mean(quality_components), 0.0, 1.0))
    if external_aligned is not None:
        latency_seconds = (
            pd.to_datetime(merged["timestamp"], utc=True, errors="coerce") - selected_external_timestamp
        ).dt.total_seconds()
        diagnostics["external_staleness_seconds"] = {
            "median": _safe_float(latency_seconds.median()),
            "p95": _safe_float(latency_seconds.quantile(0.95)),
            "max": _safe_float(latency_seconds.max()),
        }
    if diagnostics.get("external_features_loaded"):
        reason_codes.append("ranked_deterministic_timestamp_alignment_applied")
    else:
        reason_codes.append("ranked_proxy_mode_active")
    return RankedFeatureBundle(
        contract="ranked_feature_bundle.v1",
        created_at_utc=created_at_utc,
        features_enabled=True,
        external_features_path=str(external_features_path) if external_features_path else None,
        feature_frame=merged,
        feature_snapshot=feature_snapshot,
        reason_codes=sorted(set(reason_codes)),
        diagnostics=diagnostics,
    )


def write_ranked_feature_artifacts(
    *,
    quant_data_root: Path,
    exchange: str,
    symbol: str,
    timeframe: str,
    run_id: str,
    bundle: RankedFeatureBundle,
) -> RankedFeatureArtifactPaths:
    base = (
        quant_data_root
        / "curated"
        / "features"
        / "ranked-model"
        / f"exchange={exchange}"
        / f"symbol={symbol_slug(symbol)}"
        / f"interval={timeframe}"
        / f"run_id={run_id}"
    )
    base.mkdir(parents=True, exist_ok=True)
    parquet_path = base / "ranked_model_features.parquet"
    contract_path = base / "ranked_model_feature_contract.json"
    bundle.feature_frame.to_parquet(parquet_path, index=False)
    payload = {
        "contract": "ranked_model_feature_contract.v1",
        "created_at_utc": bundle.created_at_utc,
        "features_enabled": bundle.features_enabled,
        "external_features_path": bundle.external_features_path,
        "feature_columns": list(RANKED_FEATURE_COLUMNS),
        "row_count": int(len(bundle.feature_frame)),
        "feature_snapshot": bundle.feature_snapshot,
        "reason_codes": list(bundle.reason_codes),
        "diagnostics": bundle.diagnostics,
        "parquet_path": str(parquet_path),
    }
    contract_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return RankedFeatureArtifactPaths(
        parquet_path=parquet_path,
        contract_path=contract_path,
    )
