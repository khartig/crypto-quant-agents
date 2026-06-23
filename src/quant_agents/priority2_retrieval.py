from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from quant_agents.alternative_data_features import (
    ALTERNATIVE_DATA_FEATURE_COLUMNS,
    ALTERNATIVE_DATA_FEATURE_SCHEMA_VERSION,
    ALTERNATIVE_DATA_RAW_COLUMN_ALIASES,
    build_alternative_data_feature_bundle,
)
from quant_agents.config import Settings
from quant_agents.priority2_features import PRIORITY2_FEATURE_COLUMNS, build_priority2_feature_bundle
from quant_agents.storage import latest_raw_dataset, symbol_slug

logger = logging.getLogger(__name__)

_SUPPORTED_BINANCE_PERIODS: frozenset[str] = frozenset(
    {
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "12h",
        "1d",
    }
)

_SUPPORTED_OKX_BARS: dict[str, str] = {
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "12h": "12H",
    "1d": "1D",
}

_SUPPORTED_PRIORITY2_RETRIEVAL_PROVIDERS: tuple[str, ...] = (
    "binance_futures_public",
    "okx_public",
)

_PROVIDER_DEFAULT_BASE_URLS: dict[str, str] = {
    "binance_futures_public": "https://fapi.binance.com",
    "okx_public": "https://www.okx.com",
}

_LOCAL_FEATURE_COLUMN_ALIASES: dict[str, str] = {
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
class Priority2ExternalRetrievalResult:
    run_id: str
    source_data_path: Path
    source_data_sha256: str
    parquet_path: Path
    contract_path: Path
    provider: str
    row_count: int
    coverage_ratio: float
    reason_codes: tuple[str, ...]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=max(4, window // 3)).mean()
    std = series.rolling(window=window, min_periods=max(4, window // 3)).std(ddof=0).replace(0.0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan)


def _http_get_json(*, url: str, timeout_seconds: float) -> Any:
    request = urllib.request.Request(
        url=url,
        headers={
            "Accept": "application/json",
            "User-Agent": "crypto-quant-agents/0.1 priority2-retrieval",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=max(1.0, float(timeout_seconds))) as response:
        payload = response.read()
    return json.loads(payload.decode("utf-8"))


def _is_http_451(exc: Exception) -> bool:
    return isinstance(exc, urllib.error.HTTPError) and int(getattr(exc, "code", 0)) == 451


def _coerce_market_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "timestamp" not in frame.columns:
        raise RuntimeError(f"Market source file missing `timestamp` column: {path}")
    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    output = output.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )
    for column in ("close", "volume"):
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.reset_index(drop=True)
    if output.empty:
        raise RuntimeError(f"No usable timestamp rows in market source file: {path}")
    return output


def _normalize_binance_symbol(symbol: str) -> str:
    return symbol.upper().replace("/", "").replace(":", "")


def _normalize_binance_period(timeframe: str) -> str:
    normalized = str(timeframe).strip().lower()
    if normalized in _SUPPORTED_BINANCE_PERIODS:
        return normalized
    if normalized.endswith("m") and normalized in _SUPPORTED_BINANCE_PERIODS:
        return normalized
    if normalized.endswith("h") and normalized in _SUPPORTED_BINANCE_PERIODS:
        return normalized
    if normalized.endswith("d") and normalized in _SUPPORTED_BINANCE_PERIODS:
        return normalized
    return "1h"


def _normalize_okx_bar(timeframe: str) -> str:
    normalized = str(timeframe).strip().lower()
    return _SUPPORTED_OKX_BARS.get(normalized, "1H")


def _normalize_okx_period(timeframe: str) -> str:
    return _normalize_okx_bar(timeframe)


def _normalize_okx_identifiers(symbol: str) -> dict[str, str]:
    raw = str(symbol).strip().upper()
    if not raw:
        return {
            "inst_swap": "BTC-USDT-SWAP",
            "inst_index": "BTC-USDT",
            "uly": "BTC-USDT",
            "ccy": "BTC",
        }
    delimiters = ["/", ":", "-", "_"]
    parts: list[str] = [raw]
    for delimiter in delimiters:
        if delimiter in raw:
            parts = [part for part in raw.replace(":", "/").replace("-", "/").replace("_", "/").split("/") if part]
            break
    base = parts[0] if parts else raw
    quote = parts[1] if len(parts) > 1 else "USDT"
    if len(parts) == 1:
        if raw.endswith("USDT") and len(raw) > 4:
            base = raw[:-4]
            quote = "USDT"
        elif raw.endswith("USD") and len(raw) > 3:
            base = raw[:-3]
            quote = "USD"
    pair = f"{base}-{quote}"
    return {
        "inst_swap": f"{pair}-SWAP",
        "inst_index": pair,
        "uly": pair,
        "ccy": base,
    }


def _resolve_provider_base_url(*, provider_name: str, base_url: str | None) -> str:
    normalized = str(base_url or "").strip()
    provider_default = _PROVIDER_DEFAULT_BASE_URLS.get(provider_name, "")
    if not normalized:
        return provider_default or _PROVIDER_DEFAULT_BASE_URLS["binance_futures_public"]
    known_defaults = set(_PROVIDER_DEFAULT_BASE_URLS.values())
    if normalized in known_defaults and normalized != provider_default and provider_default:
        return provider_default or normalized
    return normalized


def _build_url(base_url: str, endpoint: str, params: dict[str, Any]) -> str:
    filtered = {
        key: value
        for key, value in params.items()
        if value is not None
    }
    query = urllib.parse.urlencode(filtered)
    return f"{base_url.rstrip('/')}{endpoint}?{query}"


def _parse_timestamp_ms(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _fetch_binance_paginated_objects(
    *,
    base_url: str,
    endpoint: str,
    params: dict[str, Any],
    timestamp_field: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    timeout_seconds: float,
    max_pages: int = 64,
) -> list[dict[str, Any]]:
    cursor = max(0, int(start_ms))
    rows: list[dict[str, Any]] = []
    for _ in range(max_pages):
        url = _build_url(
            base_url,
            endpoint,
            {
                **params,
                "startTime": cursor,
                "endTime": int(end_ms),
                "limit": int(limit),
            },
        )
        payload = _http_get_json(url=url, timeout_seconds=timeout_seconds)
        if not isinstance(payload, list) or not payload:
            break
        valid_items = [item for item in payload if isinstance(item, dict)]
        if not valid_items:
            break
        rows.extend(valid_items)
        last_ts = _parse_timestamp_ms(valid_items[-1].get(timestamp_field))
        if last_ts is None or last_ts >= end_ms:
            break
        if len(valid_items) < int(limit):
            break
        cursor = last_ts + 1
    return rows


def _fetch_okx_payload(
    *,
    base_url: str,
    endpoint: str,
    params: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    url = _build_url(base_url, endpoint, params)
    payload: Any = None
    for attempt in range(4):
        try:
            payload = _http_get_json(
                url=url,
                timeout_seconds=timeout_seconds,
            )
            break
        except urllib.error.HTTPError as exc:
            if int(getattr(exc, "code", 0)) == 429 and attempt < 3:
                time.sleep(min(8.0, 1.5 * float(attempt + 1)))
                continue
            raise
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected OKX payload type for endpoint={endpoint}: {type(payload).__name__}")
    code = str(payload.get("code", ""))
    if code not in {"0", ""}:
        raise RuntimeError(
            f"OKX endpoint error endpoint={endpoint} code={code} message={payload.get('msg')}"
        )
    return payload


def _fetch_okx_paginated_rows(
    *,
    base_url: str,
    endpoint: str,
    params: dict[str, Any],
    cursor_param: str,
    limit: int,
    timeout_seconds: float,
    timestamp_getter: Callable[[Any], int | None],
    row_filter: Callable[[Any], bool],
    start_ms: int,
    max_pages: int,
) -> list[Any]:
    request_limit = max(5, min(100, int(limit)))
    rows: list[Any] = []
    cursor: str | None = None
    previous_oldest: int | None = None
    for _ in range(max_pages):
        request_params = {**params, "limit": request_limit}
        if cursor is not None:
            request_params[cursor_param] = cursor
        payload = _fetch_okx_payload(
            base_url=base_url,
            endpoint=endpoint,
            params=request_params,
            timeout_seconds=timeout_seconds,
        )
        data = payload.get("data", [])
        if not isinstance(data, list) or not data:
            break
        valid_rows = [row for row in data if row_filter(row)]
        if not valid_rows:
            break
        rows.extend(valid_rows)
        timestamps = [timestamp_getter(row) for row in valid_rows]
        timestamps = [value for value in timestamps if value is not None]
        if not timestamps:
            break
        oldest = min(timestamps)
        if previous_oldest is not None and oldest >= previous_oldest:
            break
        previous_oldest = oldest
        if oldest <= int(start_ms):
            break
        if len(valid_rows) < request_limit:
            break
        cursor = str(oldest)
    return rows


def _fetch_binance_paginated_klines(
    *,
    base_url: str,
    endpoint: str,
    params: dict[str, Any],
    start_ms: int,
    end_ms: int,
    limit: int,
    timeout_seconds: float,
    max_pages: int = 64,
) -> list[list[Any]]:
    cursor = max(0, int(start_ms))
    rows: list[list[Any]] = []
    for _ in range(max_pages):
        url = _build_url(
            base_url,
            endpoint,
            {
                **params,
                "startTime": cursor,
                "endTime": int(end_ms),
                "limit": int(limit),
            },
        )
        payload = _http_get_json(url=url, timeout_seconds=timeout_seconds)
        if not isinstance(payload, list) or not payload:
            break
        valid_items = [item for item in payload if isinstance(item, list) and len(item) >= 5]
        if not valid_items:
            break
        rows.extend(valid_items)
        last_ts = _parse_timestamp_ms(valid_items[-1][0] if len(valid_items[-1]) > 0 else None)
        if last_ts is None or last_ts >= end_ms:
            break
        if len(valid_items) < int(limit):
            break
        cursor = last_ts + 1
    return rows


def _frame_from_funding_rates(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamp", "funding_rate_raw"])
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(
        pd.to_numeric(frame.get("fundingTime"), errors="coerce"),
        unit="ms",
        utc=True,
        errors="coerce",
    )
    frame["funding_rate_raw"] = pd.to_numeric(frame.get("fundingRate"), errors="coerce")
    return (
        frame.loc[:, ["timestamp", "funding_rate_raw"]]
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _frame_from_okx_open_interest(rows: list[Any]) -> pd.DataFrame:
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 4:
            continue
        parsed_rows.append(
            {
                "timestamp": pd.to_datetime(
                    pd.to_numeric(row[0], errors="coerce"),
                    unit="ms",
                    utc=True,
                    errors="coerce",
                ),
                "open_interest_contracts_raw": pd.to_numeric(row[1], errors="coerce"),
                "open_interest_coin_raw": pd.to_numeric(row[2], errors="coerce"),
                "open_interest_value_raw": pd.to_numeric(row[3], errors="coerce"),
            }
        )
    if not parsed_rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "open_interest_contracts_raw",
                "open_interest_coin_raw",
                "open_interest_value_raw",
            ]
        )
    frame = pd.DataFrame(parsed_rows)
    return (
        frame.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _frame_from_okx_open_interest_volume(rows: list[Any]) -> pd.DataFrame:
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 3:
            continue
        parsed_rows.append(
            {
                "timestamp": pd.to_datetime(
                    pd.to_numeric(row[0], errors="coerce"),
                    unit="ms",
                    utc=True,
                    errors="coerce",
                ),
                "open_interest_volume_raw": pd.to_numeric(row[1], errors="coerce"),
                "open_interest_volume_quote_raw": pd.to_numeric(row[2], errors="coerce"),
            }
        )
    if not parsed_rows:
        return pd.DataFrame(
            columns=["timestamp", "open_interest_volume_raw", "open_interest_volume_quote_raw"]
        )
    frame = pd.DataFrame(parsed_rows)
    return (
        frame.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _frame_from_okx_mark_or_index_candles(
    rows: list[Any],
    *,
    close_column: str,
) -> pd.DataFrame:
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 5:
            continue
        parsed_rows.append(
            {
                "timestamp": pd.to_datetime(
                    pd.to_numeric(row[0], errors="coerce"),
                    unit="ms",
                    utc=True,
                    errors="coerce",
                ),
                close_column: pd.to_numeric(row[4], errors="coerce"),
            }
        )
    if not parsed_rows:
        return pd.DataFrame(columns=["timestamp", close_column])
    frame = pd.DataFrame(parsed_rows)
    return (
        frame.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _frame_from_okx_basis(mark_rows: list[Any], index_rows: list[Any]) -> pd.DataFrame:
    mark_frame = _frame_from_okx_mark_or_index_candles(mark_rows, close_column="mark_price_raw")
    index_frame = _frame_from_okx_mark_or_index_candles(index_rows, close_column="index_price_raw")
    if mark_frame.empty and index_frame.empty:
        return pd.DataFrame(columns=["timestamp", "basis_premium_raw"])
    merged = _merge_timestamp_frames([mark_frame, index_frame])
    index_price = pd.to_numeric(merged.get("index_price_raw"), errors="coerce").replace(0.0, np.nan)
    mark_price = pd.to_numeric(merged.get("mark_price_raw"), errors="coerce")
    merged["basis_premium_raw"] = ((mark_price - index_price) / index_price).replace([np.inf, -np.inf], np.nan)
    return (
        merged.loc[:, ["timestamp", "basis_premium_raw", "mark_price_raw", "index_price_raw"]]
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _frame_from_okx_ratio(rows: list[Any], *, column_name: str) -> pd.DataFrame:
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 2:
            continue
        parsed_rows.append(
            {
                "timestamp": pd.to_datetime(
                    pd.to_numeric(row[0], errors="coerce"),
                    unit="ms",
                    utc=True,
                    errors="coerce",
                ),
                column_name: pd.to_numeric(row[1], errors="coerce"),
            }
        )
    if not parsed_rows:
        return pd.DataFrame(columns=["timestamp", column_name])
    frame = pd.DataFrame(parsed_rows)
    return (
        frame.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _frame_from_okx_liquidation_orders(
    rows: list[Any],
    *,
    timeframe: str,
) -> pd.DataFrame:
    normalized_timeframe = str(timeframe).strip().lower() or "1h"
    bucket_alias = normalized_timeframe if normalized_timeframe[-1:] in {"m", "h", "d"} else "1h"
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        details = row.get("details")
        if not isinstance(details, list):
            continue
        for detail in details:
            if not isinstance(detail, dict):
                continue
            timestamp_ms = _parse_timestamp_ms(detail.get("time") or detail.get("ts"))
            if timestamp_ms is None:
                continue
            price = pd.to_numeric(detail.get("bkPx"), errors="coerce")
            size = pd.to_numeric(detail.get("sz"), errors="coerce")
            if pd.isna(size):
                continue
            loss = pd.to_numeric(detail.get("bkLoss"), errors="coerce")
            if pd.notna(price):
                notional = float(abs(size * price))
            elif pd.notna(loss):
                notional = float(abs(loss))
            else:
                notional = float(abs(size))
            parsed_rows.append(
                {
                    "timestamp": pd.to_datetime(timestamp_ms, unit="ms", utc=True, errors="coerce"),
                    "liquidation_notional_raw": notional,
                }
            )
    if not parsed_rows:
        return pd.DataFrame(columns=["timestamp", "liquidation_notional_raw", "liquidation_event_count_raw"])
    frame = pd.DataFrame(parsed_rows)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "liquidation_notional_raw", "liquidation_event_count_raw"])
    frame["timestamp_bucket"] = frame["timestamp"].dt.floor(bucket_alias)
    grouped = (
        frame.groupby("timestamp_bucket", as_index=False)
        .agg(
            liquidation_notional_raw=("liquidation_notional_raw", "sum"),
            liquidation_event_count_raw=("liquidation_notional_raw", "count"),
        )
        .rename(columns={"timestamp_bucket": "timestamp"})
    )
    return (
        grouped.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _frame_from_open_interest(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open_interest_value_raw"])
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(
        pd.to_numeric(frame.get("timestamp"), errors="coerce"),
        unit="ms",
        utc=True,
        errors="coerce",
    )
    frame["open_interest_value_raw"] = pd.to_numeric(
        frame.get("sumOpenInterestValue"),
        errors="coerce",
    )
    return (
        frame.loc[:, ["timestamp", "open_interest_value_raw"]]
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _frame_from_premium_klines(rows: list[list[Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamp", "basis_premium_raw"])
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(
        pd.to_numeric(frame.iloc[:, 0], errors="coerce"),
        unit="ms",
        utc=True,
        errors="coerce",
    )
    frame["basis_premium_raw"] = pd.to_numeric(frame.iloc[:, 4], errors="coerce")
    return (
        frame.loc[:, ["timestamp", "basis_premium_raw"]]
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _frame_from_taker_ratio(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamp", "taker_buy_volume_raw", "taker_sell_volume_raw"])
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(
        pd.to_numeric(frame.get("timestamp"), errors="coerce"),
        unit="ms",
        utc=True,
        errors="coerce",
    )
    frame["taker_buy_volume_raw"] = pd.to_numeric(frame.get("buyVol"), errors="coerce")
    frame["taker_sell_volume_raw"] = pd.to_numeric(frame.get("sellVol"), errors="coerce")
    return (
        frame.loc[:, ["timestamp", "taker_buy_volume_raw", "taker_sell_volume_raw"]]
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _frame_from_top_ratio(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamp", "top_long_short_ratio_raw"])
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(
        pd.to_numeric(frame.get("timestamp"), errors="coerce"),
        unit="ms",
        utc=True,
        errors="coerce",
    )
    frame["top_long_short_ratio_raw"] = pd.to_numeric(frame.get("longShortRatio"), errors="coerce")
    return (
        frame.loc[:, ["timestamp", "top_long_short_ratio_raw"]]
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _merge_timestamp_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid_frames = [frame.copy() for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not valid_frames:
        return pd.DataFrame(columns=["timestamp"])
    merged = valid_frames[0]
    for frame in valid_frames[1:]:
        merged = pd.merge(
            merged,
            frame,
            how="outer",
            on="timestamp",
        )
    return (
        merged.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _load_local_feature_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            frame = pd.DataFrame(payload)
        elif isinstance(payload, dict):
            rows = payload.get("rows", [])
            frame = pd.DataFrame(rows if isinstance(rows, list) else [])
        else:
            frame = pd.DataFrame()
    else:
        raise RuntimeError(f"Unsupported local feature format: {path}")

    if "timestamp" not in frame.columns:
        raise RuntimeError(f"Local feature file missing `timestamp` column: {path}")

    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    rename_map: dict[str, str] = {}
    combined_aliases = {
        **ALTERNATIVE_DATA_RAW_COLUMN_ALIASES,
        **_LOCAL_FEATURE_COLUMN_ALIASES,
    }
    for column in output.columns:
        normalized = combined_aliases.get(str(column).strip().lower())
        if normalized:
            rename_map[column] = normalized
    output = output.rename(columns=rename_map)

    raw_local_columns = sorted(set(ALTERNATIVE_DATA_RAW_COLUMN_ALIASES.values()))
    keep_columns = [
        "timestamp",
        *[col for col in PRIORITY2_FEATURE_COLUMNS if col in output.columns],
        *[col for col in raw_local_columns if col in output.columns],
    ]
    output = output.loc[:, list(dict.fromkeys(keep_columns))]
    for column in [col for col in output.columns if col != "timestamp"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return (
        output.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _build_external_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        output = pd.DataFrame(columns=["timestamp", *PRIORITY2_FEATURE_COLUMNS])
        return output
    def _numeric_column(column: str) -> pd.Series:
        if column in raw.columns:
            return pd.to_numeric(raw[column], errors="coerce")
        return pd.Series(np.nan, index=raw.index, dtype=float)
    def _ratio_to_imbalance(series: pd.Series) -> pd.Series:
        clipped = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(lower=1e-6)
        return ((clipped - 1.0) / (clipped + 1.0)).replace([np.inf, -np.inf], np.nan)

    output = pd.DataFrame({"timestamp": raw["timestamp"]})
    funding_rate = _numeric_column("funding_rate_raw")
    open_interest_value = _numeric_column("open_interest_value_raw")
    open_interest_volume = _numeric_column("open_interest_volume_raw")
    basis_premium = _numeric_column("basis_premium_raw")
    taker_buy = _numeric_column("taker_buy_volume_raw")
    taker_sell = _numeric_column("taker_sell_volume_raw")
    top_ratio = _numeric_column("top_long_short_ratio_raw")
    account_ratio = _numeric_column("account_long_short_ratio_raw")
    liquidation_notional = _numeric_column("liquidation_notional_raw")
    liquidation_events = _numeric_column("liquidation_event_count_raw")
    mark_price = _numeric_column("mark_price_raw")
    index_price = _numeric_column("index_price_raw")
    if basis_premium.notna().sum() <= 0 and mark_price.notna().sum() > 0 and index_price.notna().sum() > 0:
        basis_premium = ((mark_price - index_price) / index_price.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    if open_interest_value.notna().sum() <= 0 and open_interest_volume.notna().sum() > 0:
        open_interest_value = open_interest_volume
    if account_ratio.notna().sum() <= 0 and top_ratio.notna().sum() > 0:
        account_ratio = top_ratio

    output["funding_rate_feature"] = funding_rate
    oi_baseline = open_interest_value.rolling(window=48, min_periods=8).mean().replace(0.0, np.nan)
    output["open_interest_feature"] = ((open_interest_value / oi_baseline) - 1.0).replace([np.inf, -np.inf], np.nan)
    output["basis_feature"] = basis_premium

    total_taker = (taker_buy + taker_sell).replace(0.0, np.nan)
    taker_imbalance = ((taker_buy - taker_sell) / total_taker).replace([np.inf, -np.inf], np.nan)
    top_imbalance = _ratio_to_imbalance(top_ratio)
    account_imbalance = _ratio_to_imbalance(account_ratio)
    ratio_imbalance = top_imbalance.where(top_imbalance.notna(), account_imbalance)
    combined_imbalance = taker_imbalance.where(taker_imbalance.notna(), ratio_imbalance)
    whale_flow_imbalance = (top_imbalance - account_imbalance).where(
        (top_imbalance - account_imbalance).notna(),
        combined_imbalance,
    )
    output["whale_flow_imbalance_feature"] = whale_flow_imbalance
    output["volume_imbalance_4"] = combined_imbalance.rolling(window=4, min_periods=2).mean()
    output["volume_imbalance_24"] = combined_imbalance.rolling(window=24, min_periods=8).mean()

    direction = np.sign(combined_imbalance).replace(0.0, np.nan)
    same_direction = (direction == direction.shift(1)).astype(float)
    output["momentum_persistence_6"] = same_direction.rolling(window=6, min_periods=3).mean()
    output["momentum_persistence_24"] = same_direction.rolling(window=24, min_periods=8).mean()
    total_flow_scale = total_taker
    if total_flow_scale.notna().sum() <= 0 and open_interest_volume.notna().sum() > 0:
        total_flow_scale = open_interest_volume.replace(0.0, np.nan)
    if total_flow_scale.notna().sum() <= 0 and open_interest_value.notna().sum() > 0:
        total_flow_scale = open_interest_value.replace(0.0, np.nan)
    flow_scale_z = _rolling_zscore(total_flow_scale.fillna(0.0), window=24)
    liquidation_scale = liquidation_notional.where(liquidation_notional.notna(), liquidation_events)
    liquidation_z = _rolling_zscore(liquidation_scale.fillna(0.0), window=24)
    liquidation_intensity = pd.Series(
        np.where(
            liquidation_scale.notna(),
            liquidation_z.abs(),
            combined_imbalance.abs() * flow_scale_z.abs(),
        ),
        index=raw.index,
        dtype=float,
    )
    output["liquidation_intensity_feature"] = liquidation_intensity.replace([np.inf, -np.inf], np.nan)
    transfer_scale = liquidation_scale.where(liquidation_scale.notna(), total_flow_scale)
    output["whale_transfer_spike_feature"] = np.maximum(0.0, _rolling_zscore(transfer_scale.fillna(0.0), window=48))

    positioning_ratio = top_ratio.where(top_ratio.notna(), account_ratio)
    top_ratio_clipped = positioning_ratio.clip(lower=1e-6)
    output["participant_positioning_feature"] = np.log(top_ratio_clipped).replace([np.inf, -np.inf], np.nan)

    oi_change = open_interest_value.pct_change().abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ratio_spread = (top_imbalance - account_imbalance).abs().replace([np.inf, -np.inf], np.nan)
    participant_abs = output["participant_positioning_feature"].abs()
    participant_abs = participant_abs.where(participant_abs.notna(), ratio_spread)
    output["concentration_spike_feature"] = np.maximum(
        0.0,
        _rolling_zscore(oi_change, window=24),
    ) * participant_abs.fillna(0.0)

    basis_vol_fast = basis_premium.rolling(window=24, min_periods=8).std(ddof=0)
    basis_vol_slow = basis_premium.rolling(window=96, min_periods=24).std(ddof=0)
    output["vol_term_structure_feature"] = (basis_vol_fast - basis_vol_slow).replace([np.inf, -np.inf], np.nan)

    for column in PRIORITY2_FEATURE_COLUMNS:
        if column not in output.columns:
            output[column] = np.nan
        output[column] = pd.to_numeric(output[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return (
        output.loc[:, ["timestamp", *PRIORITY2_FEATURE_COLUMNS]]
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _align_features_to_market(
    market_timestamps: pd.Series,
    external_frame: pd.DataFrame,
) -> pd.DataFrame:
    market = pd.DataFrame({"timestamp": pd.to_datetime(market_timestamps, utc=True, errors="coerce")})
    market = market.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if external_frame.empty:
        output = market.copy()
        output["external_timestamp"] = pd.NaT
        for column in PRIORITY2_FEATURE_COLUMNS:
            output[column] = np.nan
        return output

    aligned = pd.merge_asof(
        market,
        external_frame.rename(columns={"timestamp": "external_timestamp"}),
        left_on="timestamp",
        right_on="external_timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    for column in PRIORITY2_FEATURE_COLUMNS:
        if column not in aligned.columns:
            aligned[column] = np.nan
    return aligned


def _with_local_feature_overrides(
    *,
    aligned: pd.DataFrame,
    market_timestamps: pd.Series,
    local_features_path: Path,
) -> tuple[pd.DataFrame, bool, dict[str, Any], list[str]]:
    local_frame = _load_local_feature_table(local_features_path)
    if local_frame.empty:
        return aligned, False, {"local_rows": 0}, ["local_priority2_overrides_empty"]
    alternative_bundle = build_alternative_data_feature_bundle(local_frame)
    input_coverage = dict(alternative_bundle.diagnostics.get("input_coverage", {}))
    alternative_inputs_available = any(float(value) > 0.0 for value in input_coverage.values())
    local_with_alternatives = local_frame.copy()
    if alternative_inputs_available:
        local_with_alternatives = pd.merge(
            local_with_alternatives,
            alternative_bundle.feature_frame,
            how="left",
            on="timestamp",
            suffixes=("", "__alternative"),
        )
    for column in ALTERNATIVE_DATA_FEATURE_COLUMNS:
        derived_column = f"{column}__alternative"
        if column not in local_with_alternatives.columns:
            local_with_alternatives[column] = np.nan
        if alternative_inputs_available and derived_column in local_with_alternatives.columns:
            existing = pd.to_numeric(local_with_alternatives[column], errors="coerce")
            derived = pd.to_numeric(local_with_alternatives[derived_column], errors="coerce")
            local_with_alternatives[column] = np.where(existing.notna(), existing, derived)
            local_with_alternatives = local_with_alternatives.drop(columns=[derived_column])
        local_with_alternatives[column] = pd.to_numeric(local_with_alternatives[column], errors="coerce")
    market = pd.DataFrame({"timestamp": pd.to_datetime(market_timestamps, utc=True, errors="coerce")})
    market = market.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    local_aligned = pd.merge_asof(
        market,
        local_with_alternatives.rename(columns={"timestamp": "local_timestamp"}),
        left_on="timestamp",
        right_on="local_timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    output = aligned.copy()
    override_non_null_by_column: dict[str, int] = {}
    for column in PRIORITY2_FEATURE_COLUMNS:
        if column in local_aligned.columns:
            local_series = pd.to_numeric(local_aligned[column], errors="coerce")
            current = pd.to_numeric(output[column], errors="coerce")
            output[column] = np.where(local_series.notna(), local_series, current)
            override_non_null_by_column[column] = int(local_series.notna().sum())
    override_non_null_total = int(sum(override_non_null_by_column.values()))
    alternative_non_null_by_feature = {
        column: int(pd.to_numeric(local_with_alternatives[column], errors="coerce").notna().sum())
        for column in ALTERNATIVE_DATA_FEATURE_COLUMNS
        if column in local_with_alternatives.columns
    }
    diagnostics = {
        "local_rows": int(len(local_with_alternatives)),
        "override_non_null_total": override_non_null_total,
        "override_non_null_by_column": override_non_null_by_column,
        "alternative_data_feature_module": {
            "contract": alternative_bundle.contract,
            "schema_version": alternative_bundle.schema_version,
            "applied_from_raw_inputs": bool(alternative_inputs_available),
            "diagnostics": alternative_bundle.diagnostics,
            "reason_codes": list(alternative_bundle.reason_codes),
            "non_null_by_feature": alternative_non_null_by_feature,
        },
    }
    extra_reason_codes: list[str] = []
    if any(count > 0 for count in alternative_non_null_by_feature.values()):
        extra_reason_codes.append("local_alternative_data_proxy_features_derived")
    if override_non_null_total > 0:
        extra_reason_codes.append("local_priority2_overrides_non_null")
    return output, override_non_null_total > 0, diagnostics, extra_reason_codes


def _sanitize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.loc[:, ["timestamp", *PRIORITY2_FEATURE_COLUMNS]].copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    output = output.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )
    for column in PRIORITY2_FEATURE_COLUMNS:
        output[column] = (
            pd.to_numeric(output[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .clip(lower=-8.0, upper=8.0)
        )
    return output.reset_index(drop=True)


def _feature_non_null_count(frame: pd.DataFrame) -> int:
    total = 0
    for column in PRIORITY2_FEATURE_COLUMNS:
        if column not in frame.columns:
            continue
        total += int(pd.to_numeric(frame[column], errors="coerce").notna().sum())
    return total


def _base_output_dir(root: Path, exchange: str, symbol: str, timeframe: str) -> Path:
    return (
        root
        / "curated"
        / "features"
        / "external"
        / f"exchange={exchange}"
        / f"symbol={symbol_slug(symbol)}"
        / f"interval={timeframe}"
    )


def latest_priority2_external_features_path(
    *,
    quant_data_root: Path,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> Path | None:
    base_dir = _base_output_dir(quant_data_root, exchange, symbol, timeframe)
    pointer_path = base_dir / "latest_external_features_path.txt"
    if pointer_path.exists():
        pointer_value = pointer_path.read_text(encoding="utf-8").strip()
        if pointer_value:
            candidate = Path(pointer_value).expanduser()
            if candidate.exists():
                return candidate.resolve()
    candidates = sorted(base_dir.glob("run_id=*/priority2_external_features.parquet"))
    if not candidates:
        return None
    return candidates[-1].resolve()


def retrieve_priority2_external_features(
    *,
    settings: Settings,
    exchange: str,
    symbol: str,
    timeframe: str,
    source_data_path: Path | None = None,
    provider: str = "binance_futures_public",
    timeout_seconds: float = 20.0,
    max_points_per_request: int = 500,
    base_url: str = "https://fapi.binance.com",
    local_feature_overrides_path: Path | None = None,
) -> Priority2ExternalRetrievalResult:
    resolved_source = (
        source_data_path.expanduser().resolve()
        if source_data_path is not None
        else latest_raw_dataset(settings.quant_data_root, exchange, symbol, timeframe)
    )
    if not resolved_source.exists():
        raise FileNotFoundError(f"Source data file not found: {resolved_source}")

    provider_name = str(provider).strip().lower()
    if provider_name not in _SUPPORTED_PRIORITY2_RETRIEVAL_PROVIDERS:
        supported = ", ".join(_SUPPORTED_PRIORITY2_RETRIEVAL_PROVIDERS)
        raise ValueError(
            f"Unsupported Priority 2 retrieval provider: {provider_name}. "
            f"Supported providers: {supported}"
        )

    market_frame = _coerce_market_frame(resolved_source)
    market_timestamps = market_frame["timestamp"]
    start_ms = int(pd.Timestamp(market_timestamps.min()).timestamp() * 1000)
    end_ms = int(pd.Timestamp(market_timestamps.max()).timestamp() * 1000)
    limit = max(50, min(1500, int(max_points_per_request)))
    resolved_timeout = max(1.0, float(timeout_seconds))
    resolved_base_url = _resolve_provider_base_url(
        provider_name=provider_name,
        base_url=base_url,
    )

    reason_codes: list[str] = ["priority2_external_retrieval_started"]
    endpoint_diagnostics: dict[str, Any] = {}
    endpoint_frames: list[pd.DataFrame] = []
    geo_restricted_endpoints: list[str] = []
    fallback_mode = "none"
    fallback_diagnostics: dict[str, Any] = {}

    def capture_endpoint(
        *,
        name: str,
        loader: Callable[[], pd.DataFrame],
    ) -> None:
        started_at = datetime.now(timezone.utc)
        try:
            frame = loader()
            elapsed_ms = (datetime.now(timezone.utc) - started_at).total_seconds() * 1000.0
            endpoint_diagnostics[name] = {
                "status": "ok",
                "rows": int(len(frame)),
                "duration_ms": round(float(elapsed_ms), 3),
                "start_timestamp": (
                    frame["timestamp"].min().isoformat() if not frame.empty and "timestamp" in frame.columns else None
                ),
                "end_timestamp": (
                    frame["timestamp"].max().isoformat() if not frame.empty and "timestamp" in frame.columns else None
                ),
            }
            endpoint_frames.append(frame)
            if frame.empty:
                reason_codes.append(f"{name}_empty")
            else:
                reason_codes.append(f"{name}_loaded")
        except Exception as exc:
            elapsed_ms = (datetime.now(timezone.utc) - started_at).total_seconds() * 1000.0
            http_status_code = (
                int(getattr(exc, "code"))
                if isinstance(exc, urllib.error.HTTPError)
                else None
            )
            endpoint_diagnostics[name] = {
                "status": "error",
                "duration_ms": round(float(elapsed_ms), 3),
                "error": str(exc),
                "http_status_code": http_status_code,
            }
            reason_codes.append(f"{name}_error")
            if _is_http_451(exc):
                geo_restricted_endpoints.append(name)
                reason_codes.append(f"{name}_geo_restricted_451")
            logger.warning("Priority 2 endpoint failure endpoint=%s error=%s", name, exc)

    if provider_name == "binance_futures_public":
        binance_symbol = _normalize_binance_symbol(symbol)
        binance_period = _normalize_binance_period(timeframe)
        capture_endpoint(
            name="funding_rate",
            loader=lambda: _frame_from_funding_rates(
                _fetch_binance_paginated_objects(
                    base_url=resolved_base_url,
                    endpoint="/fapi/v1/fundingRate",
                    params={"symbol": binance_symbol},
                    timestamp_field="fundingTime",
                    start_ms=start_ms,
                    end_ms=end_ms,
                    limit=limit,
                    timeout_seconds=resolved_timeout,
                )
            ),
        )
        capture_endpoint(
            name="open_interest",
            loader=lambda: _frame_from_open_interest(
                _fetch_binance_paginated_objects(
                    base_url=resolved_base_url,
                    endpoint="/futures/data/openInterestHist",
                    params={"symbol": binance_symbol, "period": binance_period},
                    timestamp_field="timestamp",
                    start_ms=start_ms,
                    end_ms=end_ms,
                    limit=min(limit, 500),
                    timeout_seconds=resolved_timeout,
                )
            ),
        )
        capture_endpoint(
            name="basis_premium",
            loader=lambda: _frame_from_premium_klines(
                _fetch_binance_paginated_klines(
                    base_url=resolved_base_url,
                    endpoint="/fapi/v1/premiumIndexKlines",
                    params={"symbol": binance_symbol, "interval": timeframe},
                    start_ms=start_ms,
                    end_ms=end_ms,
                    limit=limit,
                    timeout_seconds=resolved_timeout,
                )
            ),
        )
        capture_endpoint(
            name="taker_ratio",
            loader=lambda: _frame_from_taker_ratio(
                _fetch_binance_paginated_objects(
                    base_url=resolved_base_url,
                    endpoint="/futures/data/takerlongshortRatio",
                    params={"symbol": binance_symbol, "period": binance_period},
                    timestamp_field="timestamp",
                    start_ms=start_ms,
                    end_ms=end_ms,
                    limit=min(limit, 500),
                    timeout_seconds=resolved_timeout,
                )
            ),
        )
        capture_endpoint(
            name="top_position_ratio",
            loader=lambda: _frame_from_top_ratio(
                _fetch_binance_paginated_objects(
                    base_url=resolved_base_url,
                    endpoint="/futures/data/topLongShortPositionRatio",
                    params={"symbol": binance_symbol, "period": binance_period},
                    timestamp_field="timestamp",
                    start_ms=start_ms,
                    end_ms=end_ms,
                    limit=min(limit, 500),
                    timeout_seconds=resolved_timeout,
                )
            ),
        )
    elif provider_name == "okx_public":
        okx_ids = _normalize_okx_identifiers(symbol)
        okx_bar = _normalize_okx_bar(timeframe)
        okx_period = _normalize_okx_period(timeframe)
        okx_limit = max(5, min(100, int(max_points_per_request)))
        estimated_pages = max(6, min(48, int(np.ceil(max(1, len(market_timestamps)) / max(okx_limit, 1))) + 4))
        ratio_pages = max(2, min(10, estimated_pages))
        capture_endpoint(
            name="funding_rate",
            loader=lambda: _frame_from_funding_rates(
                _fetch_okx_paginated_rows(
                    base_url=resolved_base_url,
                    endpoint="/api/v5/public/funding-rate-history",
                    params={"instId": okx_ids["inst_swap"]},
                    cursor_param="after",
                    limit=okx_limit,
                    timeout_seconds=resolved_timeout,
                    timestamp_getter=lambda row: (
                        _parse_timestamp_ms(row.get("fundingTime"))
                        if isinstance(row, dict)
                        else None
                    ),
                    row_filter=lambda row: (
                        isinstance(row, dict)
                        and _parse_timestamp_ms(row.get("fundingTime")) is not None
                    ),
                    start_ms=start_ms,
                    max_pages=estimated_pages,
                )
            ),
        )
        capture_endpoint(
            name="open_interest",
            loader=lambda: _frame_from_okx_open_interest(
                _fetch_okx_paginated_rows(
                    base_url=resolved_base_url,
                    endpoint="/api/v5/rubik/stat/contracts/open-interest-history",
                    params={"instId": okx_ids["inst_swap"], "period": okx_period},
                    cursor_param="end",
                    limit=okx_limit,
                    timeout_seconds=resolved_timeout,
                    timestamp_getter=lambda row: (
                        _parse_timestamp_ms(row[0])
                        if isinstance(row, list) and len(row) >= 1
                        else None
                    ),
                    row_filter=lambda row: (
                        isinstance(row, list)
                        and len(row) >= 4
                        and _parse_timestamp_ms(row[0]) is not None
                    ),
                    start_ms=start_ms,
                    max_pages=estimated_pages,
                )
            ),
        )
        capture_endpoint(
            name="open_interest_volume",
            loader=lambda: _frame_from_okx_open_interest_volume(
                _fetch_okx_payload(
                    base_url=resolved_base_url,
                    endpoint="/api/v5/rubik/stat/contracts/open-interest-volume",
                    params={"ccy": okx_ids["ccy"], "period": okx_period},
                    timeout_seconds=resolved_timeout,
                ).get("data", [])
            ),
        )
        capture_endpoint(
            name="basis_premium",
            loader=lambda: _frame_from_okx_basis(
                _fetch_okx_paginated_rows(
                    base_url=resolved_base_url,
                    endpoint="/api/v5/market/history-mark-price-candles",
                    params={"instId": okx_ids["inst_swap"], "bar": okx_bar},
                    cursor_param="after",
                    limit=okx_limit,
                    timeout_seconds=resolved_timeout,
                    timestamp_getter=lambda row: (
                        _parse_timestamp_ms(row[0])
                        if isinstance(row, list) and len(row) >= 1
                        else None
                    ),
                    row_filter=lambda row: (
                        isinstance(row, list)
                        and len(row) >= 5
                        and _parse_timestamp_ms(row[0]) is not None
                    ),
                    start_ms=start_ms,
                    max_pages=estimated_pages,
                ),
                _fetch_okx_paginated_rows(
                    base_url=resolved_base_url,
                    endpoint="/api/v5/market/history-index-candles",
                    params={"instId": okx_ids["inst_index"], "bar": okx_bar},
                    cursor_param="after",
                    limit=okx_limit,
                    timeout_seconds=resolved_timeout,
                    timestamp_getter=lambda row: (
                        _parse_timestamp_ms(row[0])
                        if isinstance(row, list) and len(row) >= 1
                        else None
                    ),
                    row_filter=lambda row: (
                        isinstance(row, list)
                        and len(row) >= 5
                        and _parse_timestamp_ms(row[0]) is not None
                    ),
                    start_ms=start_ms,
                    max_pages=estimated_pages,
                ),
            ),
        )
        capture_endpoint(
            name="top_position_ratio",
            loader=lambda: _frame_from_okx_ratio(
                _fetch_okx_paginated_rows(
                    base_url=resolved_base_url,
                    endpoint="/api/v5/rubik/stat/contracts/long-short-account-ratio-contract-top-trader",
                    params={"instId": okx_ids["inst_swap"], "period": okx_period},
                    cursor_param="end",
                    limit=okx_limit,
                    timeout_seconds=resolved_timeout,
                    timestamp_getter=lambda row: (
                        _parse_timestamp_ms(row[0])
                        if isinstance(row, list) and len(row) >= 1
                        else None
                    ),
                    row_filter=lambda row: (
                        isinstance(row, list)
                        and len(row) >= 2
                        and _parse_timestamp_ms(row[0]) is not None
                    ),
                    start_ms=start_ms,
                    max_pages=ratio_pages,
                ),
                column_name="top_long_short_ratio_raw",
            ),
        )
        capture_endpoint(
            name="account_position_ratio",
            loader=lambda: _frame_from_okx_ratio(
                _fetch_okx_paginated_rows(
                    base_url=resolved_base_url,
                    endpoint="/api/v5/rubik/stat/contracts/long-short-account-ratio-contract",
                    params={"instId": okx_ids["inst_swap"], "period": okx_period},
                    cursor_param="end",
                    limit=okx_limit,
                    timeout_seconds=resolved_timeout,
                    timestamp_getter=lambda row: (
                        _parse_timestamp_ms(row[0])
                        if isinstance(row, list) and len(row) >= 1
                        else None
                    ),
                    row_filter=lambda row: (
                        isinstance(row, list)
                        and len(row) >= 2
                        and _parse_timestamp_ms(row[0]) is not None
                    ),
                    start_ms=start_ms,
                    max_pages=ratio_pages,
                ),
                column_name="account_long_short_ratio_raw",
            ),
        )
        capture_endpoint(
            name="liquidation_orders",
            loader=lambda: _frame_from_okx_liquidation_orders(
                _fetch_okx_payload(
                    base_url=resolved_base_url,
                    endpoint="/api/v5/public/liquidation-orders",
                    params={
                        "instType": "SWAP",
                        "state": "filled",
                        "uly": okx_ids["uly"],
                    },
                    timeout_seconds=resolved_timeout,
                ).get("data", []),
                timeframe=timeframe,
            ),
        )

    raw_external = _merge_timestamp_frames(endpoint_frames)
    derived_external = _build_external_feature_frame(raw_external)
    if not derived_external.empty:
        market_start = pd.Timestamp(market_timestamps.min())
        market_end = pd.Timestamp(market_timestamps.max())
        derived_external = (
            derived_external[
                (pd.to_datetime(derived_external["timestamp"], utc=True, errors="coerce") >= market_start)
                & (pd.to_datetime(derived_external["timestamp"], utc=True, errors="coerce") <= market_end)
            ]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
    aligned = _align_features_to_market(market_timestamps, derived_external)
    external_non_null_count_before_fallback = _feature_non_null_count(aligned)
    if geo_restricted_endpoints:
        reason_codes.append("priority2_provider_geo_restriction_detected")
    if external_non_null_count_before_fallback <= 0:
        proxy_bundle = build_priority2_feature_bundle(
            market_frame,
            features_enabled=True,
            external_features_path=None,
        )
        proxy_frame = proxy_bundle.feature_frame.loc[:, ["timestamp", *PRIORITY2_FEATURE_COLUMNS]].copy()
        aligned = _align_features_to_market(market_timestamps, proxy_frame)
        fallback_mode = "market_proxy_bundle"
        fallback_diagnostics = {
            "source": "build_priority2_feature_bundle",
            "contract": proxy_bundle.contract,
            "reason_codes": list(proxy_bundle.reason_codes),
            "diagnostics": dict(proxy_bundle.diagnostics),
            "external_non_null_count_before_fallback": int(external_non_null_count_before_fallback),
        }
        reason_codes.append("priority2_external_unavailable_proxy_fallback_applied")
        reason_codes.append("priority2_proxy_fallback_from_market_frame")

    overrides_applied = False
    local_override_diagnostics: dict[str, Any] = {"local_rows": 0}
    if local_feature_overrides_path is not None:
        resolved_override = local_feature_overrides_path.expanduser().resolve()
        if resolved_override.exists():
            (
                aligned,
                overrides_applied,
                local_override_diagnostics,
                local_override_reason_codes,
            ) = _with_local_feature_overrides(
                aligned=aligned,
                market_timestamps=market_timestamps,
                local_features_path=resolved_override,
            )
            reason_codes.append("local_priority2_overrides_applied" if overrides_applied else "local_priority2_overrides_empty")
            reason_codes.extend(local_override_reason_codes)
        else:
            reason_codes.append("local_priority2_overrides_missing")

    sanitized = _sanitize_feature_frame(aligned)
    if sanitized.empty:
        raise RuntimeError("Priority 2 retrieval produced no aligned feature rows.")

    coverage_by_column = {
        column: float(sanitized[column].notna().mean())
        for column in PRIORITY2_FEATURE_COLUMNS
    }
    null_rate_by_column = {
        column: float(1.0 - coverage)
        for column, coverage in coverage_by_column.items()
    }
    coverage_ratio = float(np.mean(list(coverage_by_column.values()))) if coverage_by_column else 0.0
    quality_band = (
        "high"
        if coverage_ratio >= 0.80
        else "medium"
        if coverage_ratio >= 0.50
        else "low"
    )

    alignment_latency_stats = {"median": None, "p95": None, "max": None}
    if "external_timestamp" in aligned.columns:
        latency_seconds = (
            pd.to_datetime(aligned["timestamp"], utc=True, errors="coerce")
            - pd.to_datetime(aligned["external_timestamp"], utc=True, errors="coerce")
        ).dt.total_seconds()
        alignment_latency_stats = {
            "median": _safe_float(latency_seconds.median()),
            "p95": _safe_float(latency_seconds.quantile(0.95)),
            "max": _safe_float(latency_seconds.max()),
        }

    run_id = _new_run_id()
    base_dir = _base_output_dir(settings.quant_data_root, exchange, symbol, timeframe)
    output_dir = base_dir / f"run_id={run_id}"
    suffix = 0
    while output_dir.exists():
        suffix += 1
        output_dir = base_dir / f"run_id={run_id}_{suffix:02d}"
    output_dir.mkdir(parents=True, exist_ok=False)
    parquet_path = output_dir / "priority2_external_features.parquet"
    contract_path = output_dir / "priority2_external_feature_contract.json"
    latest_pointer_path = base_dir / "latest_external_features_path.txt"
    latest_contract_pointer_path = base_dir / "latest_external_feature_contract_path.txt"

    sanitized.to_parquet(parquet_path, index=False)
    latest_pointer_path.parent.mkdir(parents=True, exist_ok=True)
    latest_pointer_path.write_text(str(parquet_path) + "\n", encoding="utf-8")
    latest_contract_pointer_path.write_text(str(contract_path) + "\n", encoding="utf-8")

    source_data_sha256 = _sha256_file(resolved_source)
    reason_codes.append("priority2_external_features_ready")
    if coverage_ratio < 0.50:
        reason_codes.append("priority2_external_coverage_low")
    if overrides_applied:
        reason_codes.append("priority2_external_local_overrides_contributed")

    payload = {
        "contract": "priority2_external_feature_retrieval.v1",
        "created_at_utc": _utc_now_iso(),
        "provider": provider_name,
        "provider_base_url": resolved_base_url,
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "source_data_path": str(resolved_source),
        "source_data_sha256": source_data_sha256,
        "window": {
            "start_utc": pd.Timestamp(market_timestamps.min()).isoformat(),
            "end_utc": pd.Timestamp(market_timestamps.max()).isoformat(),
            "market_row_count": int(len(market_timestamps)),
        },
        "feature_columns": list(PRIORITY2_FEATURE_COLUMNS),
        "alternative_data_feature_columns": list(ALTERNATIVE_DATA_FEATURE_COLUMNS),
        "alternative_data_feature_schema_version": ALTERNATIVE_DATA_FEATURE_SCHEMA_VERSION,
        "row_count": int(len(sanitized)),
        "coverage_ratio": coverage_ratio,
        "coverage_quality_band": quality_band,
        "column_coverage": coverage_by_column,
        "column_null_rate": null_rate_by_column,
        "alignment_latency_seconds": alignment_latency_stats,
        "endpoint_diagnostics": endpoint_diagnostics,
        "geo_restricted_endpoints": sorted(set(geo_restricted_endpoints)),
        "fallback_mode": fallback_mode,
        "fallback_diagnostics": fallback_diagnostics,
        "local_feature_overrides_path": (
            str(local_feature_overrides_path.expanduser().resolve())
            if local_feature_overrides_path is not None
            else None
        ),
        "local_override_diagnostics": local_override_diagnostics,
        "reason_codes": sorted(set(reason_codes)),
        "artifacts": {
            "output_dir": str(output_dir),
            "priority2_external_features_parquet": str(parquet_path),
            "latest_external_features_path_pointer": str(latest_pointer_path),
            "latest_external_feature_contract_pointer": str(latest_contract_pointer_path),
        },
    }
    contract_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return Priority2ExternalRetrievalResult(
        run_id=output_dir.name.replace("run_id=", ""),
        source_data_path=resolved_source,
        source_data_sha256=source_data_sha256,
        parquet_path=parquet_path,
        contract_path=contract_path,
        provider=provider_name,
        row_count=int(len(sanitized)),
        coverage_ratio=coverage_ratio,
        reason_codes=tuple(sorted(set(reason_codes))),
    )
