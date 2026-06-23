from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

ALTERNATIVE_DATA_FEATURE_MODULE_CONTRACT = "alternative_data_proxy_features.v1"
ALTERNATIVE_DATA_FEATURE_SCHEMA_VERSION = "alternative_data_feature_schema.v1"

ALTERNATIVE_DATA_FEATURE_COLUMNS: tuple[str, ...] = (
    "whale_flow_imbalance_feature",
    "whale_transfer_spike_feature",
    "participant_positioning_feature",
    "concentration_spike_feature",
)

ALTERNATIVE_DATA_RAW_COLUMNS: tuple[str, ...] = (
    "onchain_exchange_inflow_raw",
    "onchain_exchange_outflow_raw",
    "onchain_exchange_netflow_raw",
    "onchain_large_transfer_volume_raw",
    "onchain_large_transfer_count_raw",
    "onchain_holder_concentration_raw",
    "known_trader_long_short_ratio_raw",
    "known_trader_net_position_raw",
)

ALTERNATIVE_DATA_RAW_COLUMN_ALIASES: dict[str, str] = {
    "onchain_exchange_inflow_raw": "onchain_exchange_inflow_raw",
    "exchange_inflow_usd": "onchain_exchange_inflow_raw",
    "exchange_inflow": "onchain_exchange_inflow_raw",
    "onchain_exchange_outflow_raw": "onchain_exchange_outflow_raw",
    "exchange_outflow_usd": "onchain_exchange_outflow_raw",
    "exchange_outflow": "onchain_exchange_outflow_raw",
    "onchain_exchange_netflow_raw": "onchain_exchange_netflow_raw",
    "exchange_netflow_usd": "onchain_exchange_netflow_raw",
    "exchange_netflow": "onchain_exchange_netflow_raw",
    "onchain_large_transfer_volume_raw": "onchain_large_transfer_volume_raw",
    "large_transfer_volume_usd": "onchain_large_transfer_volume_raw",
    "large_transfer_notional_usd": "onchain_large_transfer_volume_raw",
    "whale_transfer_volume_usd": "onchain_large_transfer_volume_raw",
    "onchain_large_transfer_count_raw": "onchain_large_transfer_count_raw",
    "large_transfer_count": "onchain_large_transfer_count_raw",
    "whale_transfer_count": "onchain_large_transfer_count_raw",
    "onchain_holder_concentration_raw": "onchain_holder_concentration_raw",
    "holder_concentration_ratio": "onchain_holder_concentration_raw",
    "whale_wallet_concentration_ratio": "onchain_holder_concentration_raw",
    "top10_holder_share": "onchain_holder_concentration_raw",
    "top_10_holder_share": "onchain_holder_concentration_raw",
    "known_trader_long_short_ratio_raw": "known_trader_long_short_ratio_raw",
    "known_trader_long_short_ratio": "known_trader_long_short_ratio_raw",
    "top_trader_long_short_ratio": "known_trader_long_short_ratio_raw",
    "participant_long_short_ratio": "known_trader_long_short_ratio_raw",
    "known_trader_net_position_raw": "known_trader_net_position_raw",
    "known_trader_net_position": "known_trader_net_position_raw",
    "participant_net_position": "known_trader_net_position_raw",
}


@dataclass(frozen=True)
class AlternativeDataFeatureBundle:
    contract: str
    created_at_utc: str
    schema_version: str
    feature_frame: pd.DataFrame
    diagnostics: dict[str, Any]
    reason_codes: list[str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=max(4, window // 3)).mean()
    std = series.rolling(window=window, min_periods=max(4, window // 3)).std(ddof=0).replace(0.0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan)


def _resolve_aliases(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    renamed = frame.copy()
    source_map: dict[str, str] = {}
    for canonical in ALTERNATIVE_DATA_RAW_COLUMNS:
        if canonical in renamed.columns:
            source_map[canonical] = canonical
            continue
        for alias, mapped in ALTERNATIVE_DATA_RAW_COLUMN_ALIASES.items():
            if mapped != canonical:
                continue
            if alias in renamed.columns:
                renamed[canonical] = renamed[alias]
                source_map[canonical] = alias
                break
    return renamed, source_map


def _coerce_timestamp_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in frame.columns:
        raise RuntimeError("Alternative data feature build requires `timestamp` column.")
    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    output = (
        output.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )
    if output.empty:
        raise RuntimeError("Alternative data feature build found no usable timestamp rows.")
    return output


def build_alternative_data_feature_bundle(raw_frame: pd.DataFrame) -> AlternativeDataFeatureBundle:
    normalized = _coerce_timestamp_frame(raw_frame)
    aliased, source_columns = _resolve_aliases(normalized)

    for column in ALTERNATIVE_DATA_RAW_COLUMNS:
        if column not in aliased.columns:
            aliased[column] = np.nan
        aliased[column] = pd.to_numeric(aliased[column], errors="coerce")

    inflow = aliased["onchain_exchange_inflow_raw"]
    outflow = aliased["onchain_exchange_outflow_raw"]
    netflow = aliased["onchain_exchange_netflow_raw"]
    transfer_volume = aliased["onchain_large_transfer_volume_raw"]
    transfer_count = aliased["onchain_large_transfer_count_raw"]
    holder_concentration = aliased["onchain_holder_concentration_raw"]
    known_trader_ratio = aliased["known_trader_long_short_ratio_raw"]
    known_trader_net_position = aliased["known_trader_net_position_raw"]

    derived_netflow = netflow.copy()
    netflow_source = "onchain_exchange_netflow_raw"
    if derived_netflow.notna().sum() <= 0 and (inflow.notna().sum() > 0 or outflow.notna().sum() > 0):
        derived_netflow = inflow.fillna(0.0) - outflow.fillna(0.0)
        netflow_source = "derived_inflow_minus_outflow"

    gross_flow = (inflow.abs() + outflow.abs()).replace(0.0, np.nan)
    flow_imbalance = ((inflow - outflow) / gross_flow).replace([np.inf, -np.inf], np.nan)
    netflow_z = _rolling_zscore(derived_netflow.fillna(0.0), window=24)
    whale_flow_imbalance = flow_imbalance.where(flow_imbalance.notna(), np.tanh(netflow_z / 3.0))
    whale_flow_imbalance = whale_flow_imbalance.where(
        whale_flow_imbalance.notna(),
        np.tanh(derived_netflow / (derived_netflow.abs().rolling(window=24, min_periods=6).mean().replace(0.0, np.nan))),
    )

    transfer_signal = transfer_volume.where(transfer_volume.notna(), transfer_count)
    transfer_spike = np.maximum(0.0, _rolling_zscore(transfer_signal.fillna(0.0), window=48))

    participant_positioning = np.log(known_trader_ratio.clip(lower=1e-6)).replace([np.inf, -np.inf], np.nan)
    participant_positioning = participant_positioning.where(
        participant_positioning.notna(),
        _rolling_zscore(known_trader_net_position.fillna(0.0), window=24),
    )

    concentration_signal = holder_concentration.copy()
    concentration_signal = concentration_signal.where(
        concentration_signal.notna(),
        _rolling_zscore(transfer_count.fillna(0.0), window=24),
    )
    concentration_signal = concentration_signal.where(
        concentration_signal.notna(),
        _rolling_zscore(derived_netflow.abs().fillna(0.0), window=24),
    )
    concentration_spike = np.maximum(0.0, _rolling_zscore(concentration_signal.fillna(0.0), window=24))
    concentration_spike = concentration_spike * (1.0 + transfer_spike.fillna(0.0))

    output = pd.DataFrame(
        {
            "timestamp": aliased["timestamp"],
            "whale_flow_imbalance_feature": whale_flow_imbalance,
            "whale_transfer_spike_feature": transfer_spike,
            "participant_positioning_feature": participant_positioning,
            "concentration_spike_feature": concentration_spike,
        }
    )
    for column in ALTERNATIVE_DATA_FEATURE_COLUMNS:
        output[column] = (
            pd.to_numeric(output[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=-8.0, upper=8.0)
        )

    diagnostics = {
        "schema_version": ALTERNATIVE_DATA_FEATURE_SCHEMA_VERSION,
        "source_columns": source_columns,
        "netflow_source": netflow_source,
        "input_coverage": {
            column: float(pd.to_numeric(aliased[column], errors="coerce").notna().mean())
            for column in ALTERNATIVE_DATA_RAW_COLUMNS
        },
        "feature_coverage": {
            column: float(output[column].notna().mean())
            for column in ALTERNATIVE_DATA_FEATURE_COLUMNS
        },
        "rows": int(len(output)),
    }

    reason_codes: list[str] = ["alternative_data_feature_module_applied"]
    if derived_netflow.notna().sum() > 0:
        reason_codes.append("alternative_data_netflow_available")
    if transfer_signal.notna().sum() > 0:
        reason_codes.append("alternative_data_transfer_signal_available")
    if known_trader_ratio.notna().sum() > 0 or known_trader_net_position.notna().sum() > 0:
        reason_codes.append("alternative_data_known_trader_proxy_available")
    if holder_concentration.notna().sum() > 0:
        reason_codes.append("alternative_data_concentration_proxy_available")
    if netflow_source == "derived_inflow_minus_outflow":
        reason_codes.append("alternative_data_netflow_derived_from_inflow_outflow")

    return AlternativeDataFeatureBundle(
        contract=ALTERNATIVE_DATA_FEATURE_MODULE_CONTRACT,
        created_at_utc=_utc_now_iso(),
        schema_version=ALTERNATIVE_DATA_FEATURE_SCHEMA_VERSION,
        feature_frame=output,
        diagnostics=diagnostics,
        reason_codes=sorted(set(reason_codes)),
    )
