from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from quant_agents.agent_contracts import PaperTradeIntent, Recommendation, write_contract

from quant_agents.config import Settings
from quant_agents.ingestion import fetch_ohlcv_to_parquet
from quant_agents.paper_trading import (
    execute_paper_trade_intent,
    simulate_paper_trade_execution_step,
)
from quant_agents.priority2_features import (
    DEFAULT_STABLE_PRIORITY2_FEATURE_COLUMNS,
    PRIORITY2_FEATURE_COLUMNS,
    Priority2FeatureBundle,
    apply_priority2_feature_column_selection,
    build_priority2_feature_bundle,
    normalize_priority2_feature_columns,
    write_priority2_feature_artifacts,
)
from quant_agents.priority2_retrieval import latest_priority2_external_features_path
from quant_agents.storage import latest_raw_dataset, symbol_slug

logger = logging.getLogger(__name__)

TRIGGER_LABELS: tuple[str, str, str] = ("buy", "hold", "sell")
BASE_FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1",
    "ret_4",
    "ret_24",
    "volatility_24",
    "sma_fast_spread",
    "sma_slow_spread",
    "macd",
    "macd_hist",
    "rsi_14",
    "volume_zscore_24",
    "hl_range_14",
)
FEATURE_COLUMNS: tuple[str, ...] = BASE_FEATURE_COLUMNS + PRIORITY2_FEATURE_COLUMNS
EXECUTION_THRESHOLD_SELECTION_OBJECTIVE = "execution_backtest_realized_pnl_delta_usd"


@dataclass(frozen=True)
class TriggerModelTrainingResult:
    model_path: Path
    run_dir: Path
    source_data_path: Path
    source_data_sha256: str
    sample_count: int
    train_count: int
    test_count: int
    accuracy: float
    label_distribution: dict[str, int]
    confusion_matrix: dict[str, dict[str, int]]
    selected_buy_threshold: float
    selected_sell_threshold: float
    selected_trade_quality_threshold: float
    selected_action_confidence_threshold: float
    net_expectancy_per_actionable: float
    execution_backtest_equity_return: float
    execution_backtest_realized_pnl_delta_usd: float


@dataclass(frozen=True)
class TriggerPredictionResult:
    model_path: Path
    source_data_path: Path
    source_data_sha256: str
    timestamp_utc: str
    recommendation: str
    confidence: float
    probabilities: dict[str, float]
    close_price: float
    sma_fast: float
    sma_slow: float
    macd: float
    macd_hist: float
    rsi_14: float
    volatility_24: float
    feature_values: dict[str, float]
    top_reasons: list[str]
    reason_details: list[dict[str, Any]]
    action_confidence_threshold: float
    prediction_path: Path | None


@dataclass(frozen=True)
class TriggerMonitorResult:
    cycles_completed: int
    alerts_emitted: int
    paper_trades_attempted: int
    paper_trades_executed: int
    latest_alert_path: Path | None
    latest_paper_execution_path: Path | None
    state_path: Path


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _coerce_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    required_columns = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Input parquet is missing required columns: {missing}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].isna().all():
        raise RuntimeError("Input parquet has no parseable timestamps")

    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).reset_index(
        drop=True
    )
    if frame.empty:
        raise RuntimeError("Input parquet has no usable rows after numeric/timestamp coercion")
    return frame


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _build_feature_frame(
    frame: pd.DataFrame,
    *,
    priority2_features_enabled: bool = True,
    priority2_external_features_path: Path | None = None,
    priority2_feature_columns: tuple[str, ...] | list[str] | None = None,
) -> tuple[pd.DataFrame, Priority2FeatureBundle]:
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)

    returns = close.pct_change()
    sma_fast = close.rolling(window=12, min_periods=12).mean()
    sma_slow = close.rolling(window=48, min_periods=48).mean()
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    volume_mean = volume.rolling(window=24, min_periods=24).mean()
    volume_std = volume.rolling(window=24, min_periods=24).std(ddof=0).replace(0.0, np.nan)
    hl_range = ((high - low) / close.replace(0.0, np.nan)).rolling(window=14, min_periods=14).mean()

    feature_frame = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "close": close,
            "high": high,
            "low": low,
            "ret_1": close.pct_change(periods=1),
            "ret_4": close.pct_change(periods=4),
            "ret_24": close.pct_change(periods=24),
            "volatility_24": returns.rolling(window=24, min_periods=24).std(ddof=0),
            "sma_fast_spread": (close / sma_fast) - 1.0,
            "sma_slow_spread": (close / sma_slow) - 1.0,
            "macd": macd,
            "macd_hist": macd_hist,
            "rsi_14": _compute_rsi(close, period=14),
            "volume_zscore_24": (volume - volume_mean) / volume_std,
            "hl_range_14": hl_range,
        }
    )
    priority2_bundle = build_priority2_feature_bundle(
        frame,
        features_enabled=bool(priority2_features_enabled),
        external_features_path=priority2_external_features_path,
    )
    priority2_bundle = apply_priority2_feature_column_selection(
        bundle=priority2_bundle,
        selected_feature_columns=priority2_feature_columns,
    )
    priority2_frame = priority2_bundle.feature_frame.copy()
    priority2_frame["timestamp"] = pd.to_datetime(
        priority2_frame["timestamp"],
        utc=True,
        errors="coerce",
    )
    merged = pd.merge_asof(
        feature_frame.sort_values("timestamp").reset_index(drop=True),
        priority2_frame.sort_values("timestamp").reset_index(drop=True),
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    for column in PRIORITY2_FEATURE_COLUMNS:
        if column not in merged.columns:
            merged[column] = 0.0
        merged[column] = (
            pd.to_numeric(merged[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
    return merged, priority2_bundle


def _resolve_priority2_external_features_path(
    *,
    settings: Settings,
    exchange: str,
    symbol: str,
    timeframe: str,
    priority2_features_enabled: bool,
    requested_path: Path | None,
) -> tuple[Path | None, str]:
    if not priority2_features_enabled:
        return None, "priority2_disabled"
    if requested_path is not None:
        return requested_path, "explicit"
    latest_path = latest_priority2_external_features_path(
        quant_data_root=settings.quant_data_root,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
    )
    if latest_path is not None:
        return latest_path, "latest_retrieval_artifact"
    return None, "none_found"

def _resolve_labeling_mode(labeling_mode: str) -> str:
    normalized = str(labeling_mode or "directional_v1").strip().lower()
    if normalized in {"directional_v1", "triple_barrier_v2"}:
        return normalized
    return "directional_v1"


def _compute_forward_path_statistics(
    feature_frame: pd.DataFrame,
    *,
    horizon_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizon = max(1, int(horizon_bars))
    close = feature_frame["close"].to_numpy(dtype=float)
    high = feature_frame["high"].to_numpy(dtype=float)
    low = feature_frame["low"].to_numpy(dtype=float)
    count = len(feature_frame)

    forward_return = np.full(count, np.nan, dtype=float)
    max_up_return = np.full(count, np.nan, dtype=float)
    min_down_return = np.full(count, np.nan, dtype=float)

    for index in range(count):
        end = min(count - 1, index + horizon)
        if end <= index:
            continue
        entry = close[index]
        if not np.isfinite(entry) or abs(entry) < 1e-12:
            continue
        terminal = close[end]
        if np.isfinite(terminal):
            forward_return[index] = float(terminal / entry - 1.0)

        high_window = high[index + 1 : end + 1]
        if high_window.size > 0:
            high_returns = (high_window / entry) - 1.0
            finite_high = high_returns[np.isfinite(high_returns)]
            if finite_high.size > 0:
                max_up_return[index] = float(np.max(finite_high))

        low_window = low[index + 1 : end + 1]
        if low_window.size > 0:
            low_returns = (low_window / entry) - 1.0
            finite_low = low_returns[np.isfinite(low_returns)]
            if finite_low.size > 0:
                min_down_return[index] = float(np.min(finite_low))

    return forward_return, max_up_return, min_down_return


def _label_triple_barrier(
    feature_frame: pd.DataFrame,
    *,
    horizon_bars: int,
    buy_threshold: float,
    sell_threshold: float,
) -> tuple[list[str], list[str]]:
    horizon = max(1, int(horizon_bars))
    buy_cutoff = float(max(0.0005, buy_threshold))
    sell_cutoff = float(max(0.0005, abs(sell_threshold)))
    close = feature_frame["close"].to_numpy(dtype=float)
    high = feature_frame["high"].to_numpy(dtype=float)
    low = feature_frame["low"].to_numpy(dtype=float)
    count = len(feature_frame)
    labels: list[str] = []
    barrier_events: list[str] = []

    for index in range(count):
        end = min(count - 1, index + horizon)
        entry = close[index]
        if end <= index or (not np.isfinite(entry)) or abs(entry) < 1e-12:
            labels.append("hold")
            barrier_events.append("time_expiry")
            continue

        assigned_label = "hold"
        event = "time_expiry"
        for offset in range(1, (end - index) + 1):
            high_return = (high[index + offset] / entry) - 1.0
            low_return = (low[index + offset] / entry) - 1.0
            upper_hit = np.isfinite(high_return) and high_return >= buy_cutoff
            lower_hit = np.isfinite(low_return) and low_return <= -sell_cutoff
            if upper_hit and lower_hit:
                upper_strength = abs(high_return) / max(buy_cutoff, 1e-9)
                lower_strength = abs(low_return) / max(sell_cutoff, 1e-9)
                if upper_strength >= lower_strength:
                    assigned_label = "buy"
                    event = "double_hit_upper_dominant"
                else:
                    assigned_label = "sell"
                    event = "double_hit_lower_dominant"
                break
            if upper_hit:
                assigned_label = "buy"
                event = "upper_barrier_hit"
                break
            if lower_hit:
                assigned_label = "sell"
                event = "lower_barrier_hit"
                break
        labels.append(assigned_label)
        barrier_events.append(event)
    return labels, barrier_events


def _compute_trade_quality_scores(
    labeled: pd.DataFrame,
    *,
    buy_threshold: float,
    sell_threshold: float,
    one_way_cost_bps: float,
) -> np.ndarray:
    buy_cutoff = float(max(0.0005, buy_threshold))
    sell_cutoff = float(max(0.0005, abs(sell_threshold)))
    cost_rate = max(0.0, float(one_way_cost_bps)) / 10_000.0
    target_scale = max(buy_cutoff, sell_cutoff, cost_rate, 1e-6)

    scores = np.zeros(len(labeled), dtype=float)
    raw_labels = labeled["raw_label"].astype(str).to_numpy()
    forward_returns = labeled["forward_return"].to_numpy(dtype=float)
    max_up = labeled["max_up_return"].to_numpy(dtype=float)
    min_down = labeled["min_down_return"].to_numpy(dtype=float)
    volatility = (
        pd.to_numeric(labeled.get("volatility_24", pd.Series(np.zeros(len(labeled)))), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    for index, label in enumerate(raw_labels):
        if label not in {"buy", "sell"}:
            scores[index] = 0.0
            continue
        forward_return = float(forward_returns[index]) if np.isfinite(forward_returns[index]) else 0.0
        up_move = float(max_up[index]) if np.isfinite(max_up[index]) else 0.0
        down_move = float(min_down[index]) if np.isfinite(min_down[index]) else 0.0

        if label == "buy":
            realized_directional_return = forward_return
            favorable_move = max(up_move, forward_return, 0.0)
            adverse_move = max(abs(down_move), 0.0)
        else:
            realized_directional_return = -forward_return
            favorable_move = max(abs(down_move), -forward_return, 0.0)
            adverse_move = max(up_move, 0.0)

        edge_component = float(
            np.clip((realized_directional_return - cost_rate) / max(target_scale, 1e-9), 0.0, 1.0)
        )
        reward_risk_ratio = favorable_move / max(adverse_move + cost_rate, 1e-9)
        reward_risk_component = float(np.clip(reward_risk_ratio / 2.0, 0.0, 1.0))
        confidence_component = float(
            np.clip(abs(realized_directional_return) / max(target_scale, 1e-9), 0.0, 1.0)
        )
        stability_component = float(np.clip(1.0 - (max(0.0, volatility[index]) * 12.0), 0.0, 1.0))
        score = (
            (0.40 * edge_component)
            + (0.25 * reward_risk_component)
            + (0.20 * confidence_component)
            + (0.15 * stability_component)
        )
        scores[index] = float(np.clip(score, 0.0, 1.0))
    return scores


def _label_training_frame(
    feature_frame: pd.DataFrame,
    *,
    horizon_bars: int,
    buy_threshold: float,
    sell_threshold: float,
    labeling_mode: str,
    trade_quality_min_score: float,
    one_way_cost_bps: float,
) -> pd.DataFrame:
    horizon = max(1, int(horizon_bars))
    buy_cutoff = float(max(0.0005, buy_threshold))
    sell_cutoff = float(max(0.0005, abs(sell_threshold)))
    mode = _resolve_labeling_mode(labeling_mode)
    quality_threshold = float(np.clip(trade_quality_min_score, 0.0, 1.0))

    forward_return, max_up_return, min_down_return = _compute_forward_path_statistics(
        feature_frame,
        horizon_bars=horizon,
    )

    labeled = feature_frame.copy()
    labeled["forward_return"] = forward_return
    labeled["max_up_return"] = max_up_return
    labeled["min_down_return"] = min_down_return

    if mode == "triple_barrier_v2":
        raw_labels, barrier_events = _label_triple_barrier(
            feature_frame,
            horizon_bars=horizon,
            buy_threshold=buy_cutoff,
            sell_threshold=sell_cutoff,
        )
    else:
        raw_labels = ["hold"] * len(labeled)
        barrier_events = ["forward_return_threshold"] * len(labeled)
        forward_series = pd.to_numeric(labeled["forward_return"], errors="coerce")
        for index, value in enumerate(forward_series.to_numpy(dtype=float)):
            if not np.isfinite(value):
                continue
            if value >= buy_cutoff:
                raw_labels[index] = "buy"
            elif value <= -sell_cutoff:
                raw_labels[index] = "sell"

    labeled["raw_label"] = raw_labels
    labeled["barrier_event"] = barrier_events
    trade_quality_score = _compute_trade_quality_scores(
        labeled,
        buy_threshold=buy_cutoff,
        sell_threshold=sell_cutoff,
        one_way_cost_bps=one_way_cost_bps,
    )
    labeled["trade_quality_score"] = trade_quality_score
    actionable_raw = labeled["raw_label"].isin({"buy", "sell"})
    labeled["trade_quality_pass"] = actionable_raw & (labeled["trade_quality_score"] >= quality_threshold)
    labeled["meta_label"] = labeled["trade_quality_pass"].astype(int)
    labeled["label"] = labeled["raw_label"].where(labeled["trade_quality_pass"], "hold")
    labeled["labeling_mode"] = mode
    labeled["trade_quality_threshold"] = quality_threshold
    labeled["trade_quality_demoted_to_hold"] = actionable_raw & (~labeled["trade_quality_pass"])
    labeled = labeled.dropna(
        subset=[*FEATURE_COLUMNS, "forward_return", "max_up_return", "min_down_return"]
    ).reset_index(drop=True)
    return labeled


def _trigger_model_base_dir(root: Path, exchange: str, symbol: str, timeframe: str) -> Path:
    return (
        root
        / "models"
        / "trigger-models"
        / f"exchange={exchange}"
        / f"symbol={symbol_slug(symbol)}"
        / f"interval={timeframe}"
    )


def _new_trigger_model_dir(root: Path, exchange: str, symbol: str, timeframe: str) -> Path:
    base = _trigger_model_base_dir(root, exchange, symbol, timeframe)
    base.mkdir(parents=True, exist_ok=True)
    base_run_id = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    for suffix in range(100):
        run_id = base_run_id if suffix == 0 else f"{base_run_id}_{suffix:02d}"
        run_dir = base / run_id
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
    raise RuntimeError(f"Unable to allocate trigger-model run directory under {base}")


def latest_trigger_model_path(root: Path, exchange: str, symbol: str, timeframe: str) -> Path:
    base = _trigger_model_base_dir(root, exchange, symbol, timeframe)
    if not base.exists():
        raise FileNotFoundError(f"No trigger model directory found: {base}")
    candidates = sorted(base.rglob("model.json"))
    if not candidates:
        raise FileNotFoundError(f"No trigger model artifacts found under: {base}")
    return candidates[-1]


def _fit_gaussian_model(train_frame: pd.DataFrame) -> dict[str, Any]:
    x_train = train_frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y_train = train_frame["label"].astype(str).to_numpy()
    if x_train.size == 0:
        raise RuntimeError("Training frame is empty")

    global_mean = np.nanmean(x_train, axis=0)
    global_std = np.nanstd(x_train, axis=0)
    global_std = np.where(global_std < 1e-6, 1e-6, global_std)
    total_samples = len(train_frame)

    class_stats: dict[str, Any] = {}
    prior_total = 0.0
    for label in TRIGGER_LABELS:
        mask = y_train == label
        class_samples = x_train[mask]
        count = int(class_samples.shape[0])
        if count >= 2:
            mean = np.nanmean(class_samples, axis=0)
            std = np.nanstd(class_samples, axis=0)
            std = np.where(std < 1e-6, 1e-6, std)
        else:
            mean = global_mean
            std = global_std
        prior = max(1e-9, count / total_samples)
        prior_total += prior
        class_stats[label] = {
            "count": count,
            "prior": prior,
            "mean": [float(v) for v in mean.tolist()],
            "std": [float(v) for v in std.tolist()],
        }

    for label in TRIGGER_LABELS:
        class_stats[label]["prior"] = float(class_stats[label]["prior"] / prior_total)

    return {"feature_columns": list(FEATURE_COLUMNS), "class_stats": class_stats}


def _predict_probabilities(model_payload: dict[str, Any], feature_vector: np.ndarray) -> tuple[str, dict[str, float]]:
    class_stats = model_payload.get("class_stats", {})
    if not isinstance(class_stats, dict):
        raise RuntimeError("Model payload missing class_stats")

    log_probs: dict[str, float] = {}
    for label in TRIGGER_LABELS:
        stats = class_stats.get(label)
        if not isinstance(stats, dict):
            raise RuntimeError(f"Model payload missing class stats for label: {label}")
        mean = np.asarray(stats.get("mean", []), dtype=float)
        std = np.asarray(stats.get("std", []), dtype=float)
        prior = float(stats.get("prior", 0.0))
        if mean.shape[0] != feature_vector.shape[0] or std.shape[0] != feature_vector.shape[0]:
            raise RuntimeError("Model feature size mismatch")
        std = np.where(std < 1e-6, 1e-6, std)
        z = (feature_vector - mean) / std
        gaussian_log_likelihood = -0.5 * np.sum(
            (z**2) + np.log(2.0 * math.pi * (std**2)),
            dtype=float,
        )
        log_probs[label] = float(math.log(max(prior, 1e-12)) + gaussian_log_likelihood)

    max_log = max(log_probs.values())
    exp_probs = {label: math.exp(value - max_log) for label, value in log_probs.items()}
    denominator = sum(exp_probs.values())
    probabilities = {label: float(exp_probs[label] / denominator) for label in TRIGGER_LABELS}
    recommendation = max(probabilities, key=probabilities.get)
    return recommendation, probabilities


def _predict_labels(model_payload: dict[str, Any], feature_matrix: np.ndarray) -> list[str]:
    predictions: list[str] = []
    for row in feature_matrix:
        label, _ = _predict_probabilities(model_payload, row)
        predictions.append(label)
    return predictions


def _classification_metrics(expected: list[str], predicted: list[str]) -> tuple[float, dict[str, dict[str, int]]]:
    if not expected:
        matrix = {label: {inner: 0 for inner in TRIGGER_LABELS} for label in TRIGGER_LABELS}
        return 0.0, matrix

    confusion: dict[str, dict[str, int]] = {
        label: {inner: 0 for inner in TRIGGER_LABELS} for label in TRIGGER_LABELS
    }
    for truth, pred in zip(expected, predicted):
        if truth not in confusion:
            continue
        if pred not in confusion[truth]:
            continue
        confusion[truth][pred] += 1
    matches = sum(1 for truth, pred in zip(expected, predicted) if truth == pred)
    accuracy = float(matches / len(expected))
    return accuracy, confusion


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _derive_per_class_metrics(confusion: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    per_class: dict[str, dict[str, float | int]] = {}
    for label in TRIGGER_LABELS:
        tp = int(confusion.get(label, {}).get(label, 0))
        fp = int(sum(confusion.get(other, {}).get(label, 0) for other in TRIGGER_LABELS if other != label))
        fn = int(sum(confusion.get(label, {}).get(other, 0) for other in TRIGGER_LABELS if other != label))
        support = int(sum(confusion.get(label, {}).values()))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    macro_precision = float(np.mean([float(item["precision"]) for item in per_class.values()])) if per_class else 0.0
    macro_recall = float(np.mean([float(item["recall"]) for item in per_class.values()])) if per_class else 0.0
    macro_f1 = float(np.mean([float(item["f1"]) for item in per_class.values()])) if per_class else 0.0
    total_support = float(sum(int(item["support"]) for item in per_class.values()))
    weighted_f1 = (
        float(
            sum(float(item["f1"]) * int(item["support"]) for item in per_class.values()) / total_support
        )
        if total_support > 0
        else 0.0
    )
    return {
        "classes": per_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def _derive_actionable_metrics(expected: list[str], predicted: list[str]) -> dict[str, float | int]:
    actionable_truth = [label in {"buy", "sell"} for label in expected]
    actionable_pred = [label in {"buy", "sell"} for label in predicted]
    tp = sum(1 for truth, pred in zip(actionable_truth, actionable_pred) if truth and pred)
    fp = sum(1 for truth, pred in zip(actionable_truth, actionable_pred) if (not truth) and pred)
    fn = sum(1 for truth, pred in zip(actionable_truth, actionable_pred) if truth and (not pred))
    actionable_count = int(sum(1 for pred in actionable_pred if pred))
    actionable_rate = _safe_div(float(actionable_count), float(len(predicted)))
    actionable_hits = sum(
        1
        for truth, pred in zip(expected, predicted)
        if pred in {"buy", "sell"} and truth == pred
    )
    actionable_accuracy = _safe_div(float(actionable_hits), float(actionable_count))
    directional_hit_rate = actionable_accuracy
    return {
        "actionable_count": actionable_count,
        "actionable_rate": actionable_rate,
        "binary_actionable_precision": _safe_div(float(tp), float(tp + fp)),
        "binary_actionable_recall": _safe_div(float(tp), float(tp + fn)),
        "actionable_accuracy": actionable_accuracy,
        "directional_hit_rate": directional_hit_rate,
    }


def _derive_calibration_metrics(
    *,
    expected: list[str],
    predicted: list[str],
    probabilities: list[dict[str, float]],
) -> dict[str, Any]:
    if not expected or not probabilities:
        return {
            "avg_confidence": 0.0,
            "accuracy": 0.0,
            "brier_score": 0.0,
            "log_loss": 0.0,
            "expected_calibration_error": 0.0,
            "confidence_bins": [],
        }
    confidence_values = [float(max(prob.values())) for prob in probabilities]
    correctness = [1.0 if truth == pred else 0.0 for truth, pred in zip(expected, predicted)]
    brier_rows: list[float] = []
    log_loss_rows: list[float] = []
    epsilon = 1e-12
    for index, truth in enumerate(expected):
        prob = probabilities[index]
        row_brier = 0.0
        for label in TRIGGER_LABELS:
            p = float(prob.get(label, 0.0))
            y = 1.0 if truth == label else 0.0
            row_brier += (p - y) ** 2
        brier_rows.append(row_brier)
        truth_prob = float(prob.get(truth, 0.0))
        log_loss_rows.append(-math.log(max(epsilon, truth_prob)))
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(10):
        lower = index / 10.0
        upper = (index + 1) / 10.0
        selected_indices = [
            row_index
            for row_index, conf in enumerate(confidence_values)
            if ((lower <= conf <= upper) if index == 9 else (lower <= conf < upper))
        ]
        if selected_indices:
            avg_conf = float(np.mean([confidence_values[row_index] for row_index in selected_indices]))
            avg_acc = float(np.mean([correctness[row_index] for row_index in selected_indices]))
            frac = float(len(selected_indices) / len(confidence_values))
            ece += abs(avg_conf - avg_acc) * frac
        else:
            avg_conf = 0.0
            avg_acc = 0.0
        bins.append(
            {
                "bin": f"{lower:.1f}-{upper:.1f}",
                "count": len(selected_indices),
                "avg_confidence": avg_conf,
                "avg_accuracy": avg_acc,
            }
        )
    return {
        "avg_confidence": float(np.mean(confidence_values)),
        "accuracy": float(np.mean(correctness)),
        "brier_score": float(np.mean(brier_rows)),
        "log_loss": float(np.mean(log_loss_rows)),
        "expected_calibration_error": float(ece),
        "confidence_bins": bins,
    }


def _derive_expectancy_metrics(
    *,
    predicted: list[str],
    forward_returns: np.ndarray,
    one_way_cost_bps: float,
) -> dict[str, float | int]:
    cost_rate = max(0.0, float(one_way_cost_bps)) / 10_000.0
    gross_returns: list[float] = []
    net_returns: list[float] = []
    actionable_gross_returns: list[float] = []
    actionable_net_returns: list[float] = []
    for label, forward_return in zip(predicted, forward_returns):
        gross = 0.0
        if label == "buy":
            gross = float(forward_return)
        elif label == "sell":
            gross = float(-forward_return)
        is_actionable = label in {"buy", "sell"}
        net = gross - (cost_rate if is_actionable else 0.0)
        gross_returns.append(gross)
        net_returns.append(net)
        if is_actionable:
            actionable_gross_returns.append(gross)
            actionable_net_returns.append(net)
    actionable_count = len(actionable_net_returns)
    actionable_rate = _safe_div(float(actionable_count), float(len(predicted)))
    actionable_gross_expectancy = (
        float(np.mean(actionable_gross_returns)) if actionable_gross_returns else 0.0
    )
    actionable_net_expectancy = (
        float(np.mean(actionable_net_returns)) if actionable_net_returns else 0.0
    )
    break_even_one_way_cost_bps = max(0.0, actionable_gross_expectancy * 10_000.0)
    return {
        "gross_expectancy_per_bar": float(np.mean(gross_returns)) if gross_returns else 0.0,
        "net_expectancy_per_bar": float(np.mean(net_returns)) if net_returns else 0.0,
        "gross_expectancy_per_actionable": actionable_gross_expectancy,
        "net_expectancy_per_actionable": actionable_net_expectancy,
        "cumulative_gross_return": float(np.prod(np.asarray(gross_returns) + 1.0) - 1.0)
        if gross_returns
        else 0.0,
        "cumulative_net_return": float(np.prod(np.asarray(net_returns) + 1.0) - 1.0)
        if net_returns
        else 0.0,
        "actionable_count": int(actionable_count),
        "actionable_rate": actionable_rate,
        "one_way_cost_bps": float(max(0.0, one_way_cost_bps)),
        "break_even_one_way_cost_bps": break_even_one_way_cost_bps,
    }

def _run_execution_aligned_backtest(
    *,
    predicted: list[str],
    close_prices: np.ndarray,
    symbol: str,
    paper_notional_usd: float,
    paper_starting_cash_usd: float,
    paper_fee_bps: float,
    paper_slippage_bps: float,
) -> dict[str, Any]:
    starting_cash = max(0.0, float(paper_starting_cash_usd))
    state: dict[str, Any] = {
        "contract": "paper_portfolio_state.v1",
        "updated_at_utc": _utc_now_iso(),
        "starting_cash_usd": starting_cash,
        "cash_usd": starting_cash,
        "fee_bps": max(0.0, float(paper_fee_bps)),
        "positions": {},
    }

    status_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    attempted = 0
    executed = 0

    first_mark_price = 0.0
    last_mark_price = 0.0
    for label, mark_price_raw in zip(predicted, close_prices):
        mark_price = (
            float(mark_price_raw)
            if np.isfinite(mark_price_raw)
            else None
        )
        if mark_price is not None and mark_price > 0:
            if first_mark_price <= 0.0:
                first_mark_price = mark_price
            last_mark_price = mark_price
        if label not in {"buy", "sell"}:
            continue
        attempted += 1
        execution = simulate_paper_trade_execution_step(
            state=state,
            symbol=symbol,
            intent_status="emitted",
            intent_action=label,
            requested_notional_usd=max(0.0, float(paper_notional_usd)),
            mark_price=mark_price,
            fee_bps=max(0.0, float(paper_fee_bps)),
            slippage_bps=max(0.0, float(paper_slippage_bps)),
        )
        execution_status = str(execution.get("execution_status", "skipped"))
        executed_action = str(execution.get("executed_action", "hold"))
        reason = str(execution.get("reason", "unknown"))
        status_counts.update([execution_status])
        action_counts.update([executed_action])
        reason_counts.update([reason])
        if execution_status == "executed":
            executed += 1

    position = dict(state.get("positions", {})).get(symbol, {})
    final_cash = float(state.get("cash_usd", starting_cash))
    final_quantity = float(position.get("quantity", 0.0))
    realized_pnl_usd = float(position.get("realized_pnl_usd", 0.0))
    if last_mark_price <= 0.0:
        last_mark_price = first_mark_price if first_mark_price > 0.0 else 0.0
    final_equity = final_cash + (final_quantity * last_mark_price)
    equity_delta = final_equity - starting_cash
    rejection_count = int(status_counts.get("rejected", 0))
    return {
        "paper_trades_attempted": int(attempted),
        "paper_trades_executed": int(executed),
        "paper_trades_rejected": rejection_count,
        "fill_rate": _safe_div(float(executed), float(attempted)),
        "rejection_rate": _safe_div(float(rejection_count), float(attempted)),
        "status_counts": dict(status_counts),
        "action_counts": dict(action_counts),
        "reason_counts": dict(reason_counts),
        "starting_cash_usd": float(starting_cash),
        "final_cash_usd": float(final_cash),
        "final_quantity": float(final_quantity),
        "realized_pnl_delta_usd": float(realized_pnl_usd),
        "equity_before_usd": float(starting_cash),
        "equity_after_usd": float(final_equity),
        "equity_delta_usd": float(equity_delta),
        "equity_return": _safe_div(float(equity_delta), float(starting_cash)),
        "first_mark_price": float(first_mark_price),
        "last_mark_price": float(last_mark_price),
        "paper_notional_usd": float(max(0.0, paper_notional_usd)),
        "paper_fee_bps": float(max(0.0, paper_fee_bps)),
        "paper_slippage_bps": float(max(0.0, paper_slippage_bps)),
    }

def _apply_action_confidence_gate(
    *,
    predicted: list[str],
    probabilities: list[dict[str, float]],
    action_confidence_threshold: float,
) -> tuple[list[str], int]:
    threshold = float(np.clip(action_confidence_threshold, 0.0, 1.0))
    gated: list[str] = []
    demoted_count = 0
    for label, probs in zip(predicted, probabilities):
        confidence = float(probs.get(label, 0.0))
        if label in {"buy", "sell"} and confidence < threshold:
            gated.append("hold")
            demoted_count += 1
        else:
            gated.append(label)
    return gated, demoted_count



def _evaluate_model(
    *,
    model_payload: dict[str, Any],
    test_frame: pd.DataFrame,
    symbol: str,
    one_way_cost_bps: float,
    action_confidence_threshold: float,
    paper_notional_usd: float,
    paper_starting_cash_usd: float,
    paper_fee_bps: float,
    paper_slippage_bps: float,
) -> dict[str, Any]:
    expected = test_frame["label"].astype(str).tolist()
    forward_returns = test_frame["forward_return"].to_numpy(dtype=float)
    feature_matrix = test_frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    raw_predicted: list[str] = []
    probability_rows: list[dict[str, float]] = []
    for row in feature_matrix:
        label, probabilities = _predict_probabilities(model_payload, row)
        raw_predicted.append(label)
        probability_rows.append(probabilities)
    predicted, demoted_action_count = _apply_action_confidence_gate(
        predicted=raw_predicted,
        probabilities=probability_rows,
        action_confidence_threshold=action_confidence_threshold,
    )
    accuracy, confusion = _classification_metrics(expected, predicted)
    per_class = _derive_per_class_metrics(confusion)
    actionable = _derive_actionable_metrics(expected, predicted)
    calibration = _derive_calibration_metrics(
        expected=expected,
        predicted=predicted,
        probabilities=probability_rows,
    )
    expectancy = _derive_expectancy_metrics(
        predicted=predicted,
        forward_returns=forward_returns,
        one_way_cost_bps=one_way_cost_bps,
    )
    execution_backtest = _run_execution_aligned_backtest(
        predicted=predicted,
        close_prices=test_frame["close"].to_numpy(dtype=float),
        symbol=symbol,
        paper_notional_usd=paper_notional_usd,
        paper_starting_cash_usd=paper_starting_cash_usd,
        paper_fee_bps=paper_fee_bps,
        paper_slippage_bps=paper_slippage_bps,
    )
    return {
        "accuracy": accuracy,
        "confusion_matrix": confusion,
        "per_class_metrics": per_class,
        "actionable_metrics": actionable,
        "calibration_metrics": calibration,
        "expectancy_metrics": expectancy,
        "execution_backtest_metrics": execution_backtest,
        "action_confidence_threshold": float(np.clip(action_confidence_threshold, 0.0, 1.0)),
        "demoted_action_count": int(demoted_action_count),
        "raw_actionable_count": int(sum(1 for label in raw_predicted if label in {"buy", "sell"})),
        "raw_actionable_rate": _safe_div(
            float(sum(1 for label in raw_predicted if label in {"buy", "sell"})),
            float(len(raw_predicted)),
        ),
    }


def _split_train_test(
    labeled: pd.DataFrame,
    *,
    min_train_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    min_train = max(20, int(min_train_samples))
    test_count = max(10, int(round(len(labeled) * 0.2)))
    if len(labeled) - test_count < min_train:
        test_count = max(1, len(labeled) - min_train)
    train_count = len(labeled) - test_count
    if train_count < min_train:
        raise RuntimeError(
            f"Unable to satisfy min_train_samples={min_train}; available labeled samples={len(labeled)}"
        )
    train_frame = labeled.iloc[:train_count].reset_index(drop=True)
    test_frame = labeled.iloc[train_count:].reset_index(drop=True)
    return train_frame, test_frame, train_count, test_count


def _candidate_threshold_pairs(
    *,
    buy_threshold: float,
    sell_threshold: float,
) -> list[tuple[float, float]]:
    min_threshold = 0.004
    max_threshold = 0.012
    base_values = {0.004, 0.005, 0.006, 0.008, 0.010, 0.012}
    base_values.add(float(np.clip(max(0.0005, buy_threshold), min_threshold, max_threshold)))
    base_values.add(float(np.clip(max(0.0005, abs(sell_threshold)), min_threshold, max_threshold)))
    ordered = sorted(
        value
        for value in base_values
        if min_threshold <= float(value) <= max_threshold
    )
    pairs: list[tuple[float, float]] = []
    for buy_value in ordered:
        for sell_value in ordered:
            pairs.append((float(buy_value), float(sell_value)))
    return pairs


def _candidate_action_confidence_thresholds(minimum_threshold: float) -> list[float]:
    floor = float(np.clip(minimum_threshold, 0.0, 1.0))
    base = {0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, floor}
    return [
        value
        for value in sorted(float(np.clip(item, 0.0, 1.0)) for item in base)
        if value >= floor
    ]


def _execution_selection_rank(metrics: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    execution_realized_pnl_delta_usd = float(
        metrics.get(
            "execution_realized_pnl_delta_usd",
            metrics.get("realized_pnl_delta_usd", 0.0),
        )
    )
    execution_equity_return = float(
        metrics.get("execution_equity_return", metrics.get("equity_return", 0.0))
    )
    execution_equity_delta_usd = float(
        metrics.get("execution_equity_delta_usd", metrics.get("equity_delta_usd", 0.0))
    )
    binary_actionable_precision = float(metrics.get("binary_actionable_precision", 0.0))
    actionable_rate = float(metrics.get("actionable_rate", 0.0))
    execution_fill_rate = float(metrics.get("execution_fill_rate", metrics.get("fill_rate", 0.0)))
    execution_rejection_rate = float(
        metrics.get("execution_rejection_rate", metrics.get("rejection_rate", 0.0))
    )
    return (
        execution_realized_pnl_delta_usd,
        execution_equity_return,
        execution_equity_delta_usd,
        binary_actionable_precision,
        actionable_rate,
        execution_fill_rate,
        -execution_rejection_rate,
    )


def _select_action_confidence_frontier(
    *,
    model_payload: dict[str, Any],
    test_frame: pd.DataFrame,
    symbol: str,
    one_way_cost_bps: float,
    paper_notional_usd: float,
    paper_starting_cash_usd: float,
    paper_fee_bps: float,
    paper_slippage_bps: float,
    minimum_threshold: float,
    optimize_thresholds: bool,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    candidate_thresholds = (
        _candidate_action_confidence_thresholds(minimum_threshold)
        if optimize_thresholds
        else [float(np.clip(minimum_threshold, 0.0, 1.0))]
    )
    selected_threshold = float(np.clip(minimum_threshold, 0.0, 1.0))
    selected_evaluation: dict[str, Any] | None = None
    frontier_rows: list[dict[str, Any]] = []
    best_rank: tuple[float, float, float, float, float, float, float] | None = None

    for threshold in candidate_thresholds:
        evaluation = _evaluate_model(
            model_payload=model_payload,
            test_frame=test_frame,
            symbol=symbol,
            one_way_cost_bps=one_way_cost_bps,
            action_confidence_threshold=threshold,
            paper_notional_usd=paper_notional_usd,
            paper_starting_cash_usd=paper_starting_cash_usd,
            paper_fee_bps=paper_fee_bps,
            paper_slippage_bps=paper_slippage_bps,
        )
        expectancy = dict(evaluation.get("expectancy_metrics", {}))
        actionable = dict(evaluation.get("actionable_metrics", {}))
        execution_backtest = dict(evaluation.get("execution_backtest_metrics", {}))
        net_expectancy_per_bar = float(expectancy.get("net_expectancy_per_bar", 0.0))
        net_expectancy_per_actionable = float(expectancy.get("net_expectancy_per_actionable", 0.0))
        execution_equity_return = float(execution_backtest.get("equity_return", 0.0))
        execution_equity_delta_usd = float(execution_backtest.get("equity_delta_usd", 0.0))
        execution_realized_pnl_delta_usd = float(
            execution_backtest.get("realized_pnl_delta_usd", 0.0)
        )
        execution_fill_rate = float(execution_backtest.get("fill_rate", 0.0))
        execution_rejection_rate = float(execution_backtest.get("rejection_rate", 0.0))
        row = {
            "threshold": float(threshold),
            "accuracy": float(evaluation.get("accuracy", 0.0)),
            "demoted_action_count": int(evaluation.get("demoted_action_count", 0)),
            "raw_actionable_rate": float(evaluation.get("raw_actionable_rate", 0.0)),
            "actionable_rate": float(actionable.get("actionable_rate", 0.0)),
            "binary_actionable_precision": float(
                actionable.get("binary_actionable_precision", 0.0)
            ),
            "binary_actionable_recall": float(actionable.get("binary_actionable_recall", 0.0)),
            "net_expectancy_per_bar": net_expectancy_per_bar,
            "net_expectancy_per_actionable": net_expectancy_per_actionable,
            "execution_equity_return": execution_equity_return,
            "execution_equity_delta_usd": execution_equity_delta_usd,
            "execution_realized_pnl_delta_usd": execution_realized_pnl_delta_usd,
            "execution_fill_rate": execution_fill_rate,
            "execution_rejection_rate": execution_rejection_rate,
        }
        frontier_rows.append(row)
        rank = _execution_selection_rank(row)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            selected_threshold = float(threshold)
            selected_evaluation = evaluation

    if selected_evaluation is None:
        selected_evaluation = _evaluate_model(
            model_payload=model_payload,
            test_frame=test_frame,
            symbol=symbol,
            one_way_cost_bps=one_way_cost_bps,
            action_confidence_threshold=selected_threshold,
            paper_notional_usd=paper_notional_usd,
            paper_starting_cash_usd=paper_starting_cash_usd,
            paper_fee_bps=paper_fee_bps,
            paper_slippage_bps=paper_slippage_bps,
        )
        expectancy = dict(selected_evaluation.get("expectancy_metrics", {}))
        actionable = dict(selected_evaluation.get("actionable_metrics", {}))
        execution_backtest = dict(selected_evaluation.get("execution_backtest_metrics", {}))
        frontier_rows.append(
            {
                "threshold": float(selected_threshold),
                "accuracy": float(selected_evaluation.get("accuracy", 0.0)),
                "demoted_action_count": int(selected_evaluation.get("demoted_action_count", 0)),
                "raw_actionable_rate": float(selected_evaluation.get("raw_actionable_rate", 0.0)),
                "actionable_rate": float(actionable.get("actionable_rate", 0.0)),
                "binary_actionable_precision": float(
                    actionable.get("binary_actionable_precision", 0.0)
                ),
                "binary_actionable_recall": float(
                    actionable.get("binary_actionable_recall", 0.0)
                ),
                "net_expectancy_per_bar": float(expectancy.get("net_expectancy_per_bar", 0.0)),
                "net_expectancy_per_actionable": float(
                    expectancy.get("net_expectancy_per_actionable", 0.0)
                ),
                "execution_equity_return": float(execution_backtest.get("equity_return", 0.0)),
                "execution_equity_delta_usd": float(execution_backtest.get("equity_delta_usd", 0.0)),
                "execution_realized_pnl_delta_usd": float(
                    execution_backtest.get("realized_pnl_delta_usd", 0.0)
                ),
                "execution_fill_rate": float(execution_backtest.get("fill_rate", 0.0)),
                "execution_rejection_rate": float(execution_backtest.get("rejection_rate", 0.0)),
            }
        )
    return selected_threshold, selected_evaluation, frontier_rows


def _feature_reasons(
    model_payload: dict[str, Any],
    feature_vector: np.ndarray,
    recommendation: str,
    probabilities: dict[str, float],
) -> tuple[list[str], list[dict[str, Any]]]:
    ordered_labels = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    if len(ordered_labels) < 2:
        return [], []
    alternative_label = ordered_labels[1][0]

    stats_best = model_payload["class_stats"][recommendation]
    stats_alt = model_payload["class_stats"][alternative_label]
    mean_best = np.asarray(stats_best["mean"], dtype=float)
    std_best = np.where(np.asarray(stats_best["std"], dtype=float) < 1e-6, 1e-6, np.asarray(stats_best["std"], dtype=float))
    mean_alt = np.asarray(stats_alt["mean"], dtype=float)
    std_alt = np.where(np.asarray(stats_alt["std"], dtype=float) < 1e-6, 1e-6, np.asarray(stats_alt["std"], dtype=float))

    reasons: list[dict[str, Any]] = []
    for index, feature_name in enumerate(FEATURE_COLUMNS):
        best_term = -0.5 * (
            ((feature_vector[index] - mean_best[index]) / std_best[index]) ** 2
            + math.log(std_best[index] ** 2)
        )
        alt_term = -0.5 * (
            ((feature_vector[index] - mean_alt[index]) / std_alt[index]) ** 2
            + math.log(std_alt[index] ** 2)
        )
        delta = float(best_term - alt_term)
        supports = recommendation if delta >= 0 else alternative_label
        reasons.append(
            {
                "feature": feature_name,
                "value": float(feature_vector[index]),
                "impact": delta,
                "supports": supports,
                "vs_alternative": alternative_label,
            }
        )

    reasons.sort(key=lambda item: abs(float(item["impact"])), reverse=True)
    top = reasons[:5]
    rendered = [
        (
            f"{item['feature']}={item['value']:.6f} "
            f"{'supports' if item['supports'] == recommendation else 'leans'} "
            f"{item['supports']} (impact={item['impact']:+.3f})"
        )
        for item in top
    ]
    return rendered, top


def train_trigger_model(
    *,
    settings: Settings,
    exchange: str,
    symbol: str,
    timeframe: str,
    input_file: Path | None,
    horizon_bars: int,
    buy_threshold: float,
    sell_threshold: float,
    min_train_samples: int,
    cost_bps: float = 0.0,
    optimize_thresholds: bool = True,
    labeling_mode: str = "triple_barrier_v2",
    trade_quality_min_score: float = 0.55,
    action_confidence_threshold: float = 0.55,
    priority2_features_enabled: bool = True,
    priority2_external_features_path: Path | None = None,
    priority2_feature_columns: tuple[str, ...] | list[str] | None = None,
) -> TriggerModelTrainingResult:
    source_data_path = (
        input_file.expanduser().resolve()
        if input_file is not None
        else latest_raw_dataset(settings.quant_data_root, exchange, symbol, timeframe)
    )
    if not source_data_path.exists():
        raise FileNotFoundError(f"Training input file not found: {source_data_path}")
    source_data_sha256 = _sha256_file(source_data_path)
    resolved_priority2_external_features_path, priority2_external_resolution = (
        _resolve_priority2_external_features_path(
            settings=settings,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            priority2_features_enabled=bool(priority2_features_enabled),
            requested_path=priority2_external_features_path,
        )
    )
    resolved_priority2_feature_columns = normalize_priority2_feature_columns(
        priority2_feature_columns
        if priority2_feature_columns is not None
        else DEFAULT_STABLE_PRIORITY2_FEATURE_COLUMNS
    )
    resolved_priority2_feature_columns = normalize_priority2_feature_columns(
        priority2_feature_columns
        if priority2_feature_columns is not None
        else DEFAULT_STABLE_PRIORITY2_FEATURE_COLUMNS
    )

    frame = _coerce_frame(source_data_path)
    feature_frame, priority2_bundle = _build_feature_frame(
        frame,
        priority2_features_enabled=bool(priority2_features_enabled),
        priority2_external_features_path=resolved_priority2_external_features_path,
        priority2_feature_columns=resolved_priority2_feature_columns,
    )
    resolved_labeling_mode = _resolve_labeling_mode(labeling_mode)
    resolved_horizon = int(max(1, horizon_bars))
    resolved_buy_threshold = float(max(0.0005, buy_threshold))
    resolved_sell_threshold = float(max(0.0005, abs(sell_threshold)))
    resolved_cost_bps = float(max(0.0, cost_bps))
    resolved_trade_quality_min_score = float(np.clip(trade_quality_min_score, 0.0, 1.0))
    resolved_action_confidence_threshold = float(np.clip(action_confidence_threshold, 0.0, 1.0))
    resolved_paper_notional_usd = float(max(0.0, settings.paper_trade_notional_usd))
    resolved_paper_starting_cash_usd = float(max(0.0, settings.paper_trade_starting_cash_usd))
    resolved_paper_fee_bps = float(max(0.0, settings.paper_trade_fee_bps))
    resolved_paper_slippage_bps = float(max(0.0, settings.paper_trade_slippage_bps))
    required_rows = max(30, int(min_train_samples) + 5)

    threshold_candidates: list[dict[str, Any]] = []
    selected_buy_threshold = resolved_buy_threshold
    selected_sell_threshold = resolved_sell_threshold
    if optimize_thresholds:
        for candidate_buy, candidate_sell in _candidate_threshold_pairs(
            buy_threshold=resolved_buy_threshold,
            sell_threshold=resolved_sell_threshold,
        ):
            candidate_labeled = _label_training_frame(
                feature_frame,
                horizon_bars=resolved_horizon,
                buy_threshold=candidate_buy,
                sell_threshold=candidate_sell,
                labeling_mode=resolved_labeling_mode,
                trade_quality_min_score=resolved_trade_quality_min_score,
                one_way_cost_bps=resolved_cost_bps,
            )
            if len(candidate_labeled) < required_rows:
                continue
            try:
                candidate_train, candidate_test, candidate_train_count, candidate_test_count = _split_train_test(
                    candidate_labeled,
                    min_train_samples=min_train_samples,
                )
            except RuntimeError:
                continue
            candidate_model = _fit_gaussian_model(candidate_train)
            candidate_eval = _evaluate_model(
                model_payload=candidate_model,
                test_frame=candidate_test,
                symbol=symbol,
                one_way_cost_bps=resolved_cost_bps,
                action_confidence_threshold=resolved_action_confidence_threshold,
                paper_notional_usd=resolved_paper_notional_usd,
                paper_starting_cash_usd=resolved_paper_starting_cash_usd,
                paper_fee_bps=resolved_paper_fee_bps,
                paper_slippage_bps=resolved_paper_slippage_bps,
            )
            expectancy = dict(candidate_eval.get("expectancy_metrics", {}))
            actionable = dict(candidate_eval.get("actionable_metrics", {}))
            execution_backtest = dict(candidate_eval.get("execution_backtest_metrics", {}))
            quality_pass_rate = (
                float(candidate_labeled["trade_quality_pass"].mean())
                if "trade_quality_pass" in candidate_labeled and len(candidate_labeled) > 0
                else 0.0
            )
            threshold_candidates.append(
                {
                    "buy_threshold": candidate_buy,
                    "sell_threshold": candidate_sell,
                    "trade_quality_min_score": resolved_trade_quality_min_score,
                    "sample_count": int(len(candidate_labeled)),
                    "train_count": int(candidate_train_count),
                    "test_count": int(candidate_test_count),
                    "accuracy": float(candidate_eval.get("accuracy", 0.0)),
                    "net_expectancy_per_bar": float(expectancy.get("net_expectancy_per_bar", 0.0)),
                    "net_expectancy_per_actionable": float(
                        expectancy.get("net_expectancy_per_actionable", 0.0)
                    ),
                    "execution_equity_return": float(execution_backtest.get("equity_return", 0.0)),
                    "execution_equity_delta_usd": float(execution_backtest.get("equity_delta_usd", 0.0)),
                    "execution_realized_pnl_delta_usd": float(
                        execution_backtest.get("realized_pnl_delta_usd", 0.0)
                    ),
                    "execution_fill_rate": float(execution_backtest.get("fill_rate", 0.0)),
                    "execution_rejection_rate": float(execution_backtest.get("rejection_rate", 0.0)),
                    "actionable_rate": float(actionable.get("actionable_rate", 0.0)),
                    "binary_actionable_precision": float(
                        actionable.get("binary_actionable_precision", 0.0)
                    ),
                    "trade_quality_pass_rate": quality_pass_rate,
                }
            )
        if threshold_candidates:
            threshold_candidates.sort(
                key=_execution_selection_rank,
                reverse=True,
            )
            selected_buy_threshold = float(threshold_candidates[0]["buy_threshold"])
            selected_sell_threshold = float(threshold_candidates[0]["sell_threshold"])

    labeled = _label_training_frame(
        feature_frame,
        horizon_bars=resolved_horizon,
        buy_threshold=selected_buy_threshold,
        sell_threshold=selected_sell_threshold,
        labeling_mode=resolved_labeling_mode,
        trade_quality_min_score=resolved_trade_quality_min_score,
        one_way_cost_bps=resolved_cost_bps,
    )
    if len(labeled) < required_rows:
        raise RuntimeError(
            f"Insufficient labeled rows for training: {len(labeled)} "
            f"(need at least {required_rows})"
        )

    train_frame, test_frame, train_count, test_count = _split_train_test(
        labeled,
        min_train_samples=min_train_samples,
    )
    model = _fit_gaussian_model(train_frame)
    selected_action_confidence_threshold, evaluation, action_confidence_frontier = _select_action_confidence_frontier(
        model_payload=model,
        test_frame=test_frame,
        symbol=symbol,
        one_way_cost_bps=resolved_cost_bps,
        paper_notional_usd=resolved_paper_notional_usd,
        paper_starting_cash_usd=resolved_paper_starting_cash_usd,
        paper_fee_bps=resolved_paper_fee_bps,
        paper_slippage_bps=resolved_paper_slippage_bps,
        minimum_threshold=resolved_action_confidence_threshold,
        optimize_thresholds=bool(optimize_thresholds),
    )
    accuracy = float(evaluation.get("accuracy", 0.0))
    confusion = dict(evaluation.get("confusion_matrix", {}))
    per_class_metrics = dict(evaluation.get("per_class_metrics", {}))
    actionable_metrics = dict(evaluation.get("actionable_metrics", {}))
    calibration_metrics = dict(evaluation.get("calibration_metrics", {}))
    expectancy_metrics = dict(evaluation.get("expectancy_metrics", {}))
    execution_backtest_metrics = dict(evaluation.get("execution_backtest_metrics", {}))

    distribution = {
        label: int((labeled["label"] == label).sum())
        for label in TRIGGER_LABELS
    }
    raw_distribution = {
        label: int((labeled["raw_label"] == label).sum())
        for label in TRIGGER_LABELS
    }
    trade_quality_pass_count = int(labeled["trade_quality_pass"].sum())
    trade_quality_demoted_count = int(labeled["trade_quality_demoted_to_hold"].sum())
    meta_label_positive_count = int(labeled["meta_label"].sum())

    run_dir = _new_trigger_model_dir(settings.quant_data_root, exchange, symbol, timeframe)
    model_path = run_dir / "model.json"
    train_frame_path = run_dir / "train_dataset.parquet"
    test_frame_path = run_dir / "test_dataset.parquet"
    train_frame.to_parquet(train_frame_path, index=False)
    test_frame.to_parquet(test_frame_path, index=False)
    priority2_artifacts = write_priority2_feature_artifacts(
        quant_data_root=settings.quant_data_root,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        run_id=run_dir.name,
        bundle=priority2_bundle,
    )

    model_payload = {
        "contract": "trigger_model.gaussian_nb.v2",
        "created_at_utc": _utc_now_iso(),
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "source_data_path": str(source_data_path),
        "source_data_sha256": source_data_sha256,
        "labeling_mode": resolved_labeling_mode,
        "horizon_bars": resolved_horizon,
        "buy_threshold": float(selected_buy_threshold),
        "sell_threshold": float(selected_sell_threshold),
        "trade_quality_min_score": float(resolved_trade_quality_min_score),
        "selected_action_confidence_threshold": float(selected_action_confidence_threshold),
        "feature_columns": list(FEATURE_COLUMNS),
        "labels": list(TRIGGER_LABELS),
        "class_stats": model["class_stats"],
        "priority2_features_enabled": bool(priority2_features_enabled),
        "priority2_external_features_path": (
            str(resolved_priority2_external_features_path)
            if resolved_priority2_external_features_path is not None
            else None
        ),
        "priority2_feature_columns": list(resolved_priority2_feature_columns),
        "priority2_external_features_path_resolution": priority2_external_resolution,
        "priority2_reason_codes": list(priority2_bundle.reason_codes),
        "priority2_diagnostics": dict(priority2_bundle.diagnostics),
        "training_metrics": {
            "sample_count": int(len(labeled)),
            "train_count": int(train_count),
            "test_count": int(test_count),
            "accuracy": accuracy,
            "label_distribution": distribution,
            "raw_label_distribution": raw_distribution,
            "trade_quality_stats": {
                "trade_quality_min_score": float(resolved_trade_quality_min_score),
                "trade_quality_pass_count": trade_quality_pass_count,
                "trade_quality_demoted_to_hold_count": trade_quality_demoted_count,
                "trade_quality_pass_rate": _safe_div(
                    float(trade_quality_pass_count),
                    float(len(labeled)),
                ),
                "meta_label_positive_count": meta_label_positive_count,
                "meta_label_positive_rate": _safe_div(
                    float(meta_label_positive_count),
                    float(len(labeled)),
                ),
            },
            "confusion_matrix": confusion,
            "per_class_metrics": per_class_metrics,
            "actionable_metrics": actionable_metrics,
            "calibration_metrics": calibration_metrics,
            "expectancy_metrics": expectancy_metrics,
            "execution_backtest_metrics": execution_backtest_metrics,
            "execution_backtest_config": {
                "paper_notional_usd": resolved_paper_notional_usd,
                "paper_starting_cash_usd": resolved_paper_starting_cash_usd,
                "paper_fee_bps": resolved_paper_fee_bps,
                "paper_slippage_bps": resolved_paper_slippage_bps,
            },
            "one_way_cost_bps": resolved_cost_bps,
        },
        "threshold_optimization": {
            "enabled": bool(optimize_thresholds),
            "objective": EXECUTION_THRESHOLD_SELECTION_OBJECTIVE,
            "candidate_count": int(len(threshold_candidates)),
            "selected": {
                "buy_threshold": float(selected_buy_threshold),
                "sell_threshold": float(selected_sell_threshold),
                "trade_quality_min_score": float(resolved_trade_quality_min_score),
                "action_confidence_threshold": float(selected_action_confidence_threshold),
            },
            "top_candidates": threshold_candidates[:10],
            "action_confidence_frontier": {
                "objective": EXECUTION_THRESHOLD_SELECTION_OBJECTIVE,
                "candidate_count": int(len(action_confidence_frontier)),
                "selected_threshold": float(selected_action_confidence_threshold),
                "rows": action_confidence_frontier[:20],
            },
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "train_dataset_path": str(train_frame_path),
            "test_dataset_path": str(test_frame_path),
            "priority2_feature_parquet": str(priority2_artifacts.parquet_path),
            "priority2_feature_contract": str(priority2_artifacts.contract_path),
        },
    }
    _write_json(model_path, model_payload)
    (_trigger_model_base_dir(settings.quant_data_root, exchange, symbol, timeframe) / "latest_model_path.txt").write_text(
        str(model_path) + "\n",
        encoding="utf-8",
    )

    return TriggerModelTrainingResult(
        model_path=model_path,
        run_dir=run_dir,
        source_data_path=source_data_path,
        source_data_sha256=source_data_sha256,
        sample_count=int(len(labeled)),
        train_count=int(train_count),
        test_count=int(test_count),
        accuracy=accuracy,
        label_distribution=distribution,
        confusion_matrix=confusion,
        selected_buy_threshold=float(selected_buy_threshold),
        selected_sell_threshold=float(selected_sell_threshold),
        selected_trade_quality_threshold=float(resolved_trade_quality_min_score),
        selected_action_confidence_threshold=float(selected_action_confidence_threshold),
        net_expectancy_per_actionable=float(expectancy_metrics.get("net_expectancy_per_actionable", 0.0)),
        execution_backtest_equity_return=float(execution_backtest_metrics.get("equity_return", 0.0)),
        execution_backtest_realized_pnl_delta_usd=float(
            execution_backtest_metrics.get("realized_pnl_delta_usd", 0.0)
        ),
    )

def _resolve_prediction_action_confidence_threshold(
    *,
    model_payload: dict[str, Any],
    override_threshold: float | None,
) -> float:
    if override_threshold is not None:
        return float(np.clip(override_threshold, 0.0, 1.0))
    model_value = model_payload.get("selected_action_confidence_threshold")
    if isinstance(model_value, (int, float)):
        return float(np.clip(float(model_value), 0.0, 1.0))
    threshold_optimization = model_payload.get("threshold_optimization")
    if isinstance(threshold_optimization, dict):
        selected = threshold_optimization.get("selected")
        if isinstance(selected, dict):
            selected_value = selected.get("action_confidence_threshold")
            if isinstance(selected_value, (int, float)):
                return float(np.clip(float(selected_value), 0.0, 1.0))
        frontier = threshold_optimization.get("action_confidence_frontier")
        if isinstance(frontier, dict):
            frontier_value = frontier.get("selected_threshold")
            if isinstance(frontier_value, (int, float)):
                return float(np.clip(float(frontier_value), 0.0, 1.0))
    return 0.0


def predict_trigger_signal(
    *,
    settings: Settings,
    exchange: str,
    symbol: str,
    timeframe: str,
    model_path: Path | None,
    input_file: Path | None,
    write_artifact: bool = True,
    action_confidence_threshold: float | None = None,
    priority2_features_enabled: bool = True,
    priority2_external_features_path: Path | None = None,
    priority2_feature_columns: tuple[str, ...] | list[str] | None = None,
) -> TriggerPredictionResult:
    resolved_model_path = (
        model_path.expanduser().resolve()
        if model_path is not None
        else latest_trigger_model_path(settings.quant_data_root, exchange, symbol, timeframe)
    )
    model_payload = json.loads(resolved_model_path.read_text(encoding="utf-8"))
    source_data_path = (
        input_file.expanduser().resolve()
        if input_file is not None
        else latest_raw_dataset(settings.quant_data_root, exchange, symbol, timeframe)
    )
    source_data_sha256 = _sha256_file(source_data_path)
    resolved_priority2_external_features_path, priority2_external_resolution = (
        _resolve_priority2_external_features_path(
            settings=settings,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            priority2_features_enabled=bool(priority2_features_enabled),
            requested_path=priority2_external_features_path,
        )
    )
    resolved_priority2_feature_columns = normalize_priority2_feature_columns(
        priority2_feature_columns
        if priority2_feature_columns is not None
        else DEFAULT_STABLE_PRIORITY2_FEATURE_COLUMNS
    )

    frame = _coerce_frame(source_data_path)
    feature_frame, priority2_bundle = _build_feature_frame(
        frame,
        priority2_features_enabled=bool(priority2_features_enabled),
        priority2_external_features_path=resolved_priority2_external_features_path,
        priority2_feature_columns=resolved_priority2_feature_columns,
    )
    feature_frame = feature_frame.dropna(subset=list(FEATURE_COLUMNS)).reset_index(drop=True)
    if feature_frame.empty:
        raise RuntimeError("No usable feature rows for prediction")

    latest_row = feature_frame.iloc[-1]
    vector = np.asarray([float(latest_row[feature]) for feature in FEATURE_COLUMNS], dtype=float)
    raw_recommendation, probabilities = _predict_probabilities(model_payload, vector)
    confidence = float(probabilities[raw_recommendation])
    resolved_action_confidence_threshold = _resolve_prediction_action_confidence_threshold(
        model_payload=model_payload,
        override_threshold=action_confidence_threshold,
    )
    recommendation = raw_recommendation
    confidence_gate_applied = False
    if recommendation in {"buy", "sell"} and confidence < resolved_action_confidence_threshold:
        recommendation = "hold"
        confidence_gate_applied = True
    training_metrics_payload = (
        model_payload.get("training_metrics", {})
        if isinstance(model_payload.get("training_metrics", {}), dict)
        else {}
    )
    expectancy_metrics = (
        training_metrics_payload.get("expectancy_metrics", {})
        if isinstance(training_metrics_payload.get("expectancy_metrics", {}), dict)
        else {}
    )
    execution_backtest_metrics = (
        training_metrics_payload.get("execution_backtest_metrics", {})
        if isinstance(training_metrics_payload.get("execution_backtest_metrics", {}), dict)
        else {}
    )
    model_net_expectancy_per_actionable = float(
        expectancy_metrics.get("net_expectancy_per_actionable", 0.0)
    ) if isinstance(expectancy_metrics, dict) else 0.0
    model_break_even_one_way_cost_bps = float(
        expectancy_metrics.get("break_even_one_way_cost_bps", 0.0)
    ) if isinstance(expectancy_metrics, dict) else 0.0
    model_one_way_cost_bps = float(
        expectancy_metrics.get("one_way_cost_bps", 0.0)
    ) if isinstance(expectancy_metrics, dict) else 0.0
    model_execution_equity_return = float(
        execution_backtest_metrics.get("equity_return", 0.0)
    ) if isinstance(execution_backtest_metrics, dict) else 0.0
    model_execution_realized_pnl_delta_usd = float(
        execution_backtest_metrics.get("realized_pnl_delta_usd", 0.0)
    ) if isinstance(execution_backtest_metrics, dict) else 0.0
    model_execution_fill_rate = float(
        execution_backtest_metrics.get("fill_rate", 0.0)
    ) if isinstance(execution_backtest_metrics, dict) else 0.0
    model_execution_rejection_rate = float(
        execution_backtest_metrics.get("rejection_rate", 0.0)
    ) if isinstance(execution_backtest_metrics, dict) else 0.0
    execution_metric_available = isinstance(execution_backtest_metrics, dict) and bool(execution_backtest_metrics)
    action_gate_basis = "execution_backtest" if execution_metric_available else "expectancy"
    cost_gate_applied = False
    cost_gate_fail = False
    if action_gate_basis == "execution_backtest":
        cost_gate_fail = (
            model_execution_equity_return <= 0.0
            or model_execution_realized_pnl_delta_usd <= 0.0
        )
    else:
        cost_gate_fail = (
            model_net_expectancy_per_actionable <= 0.0
            or model_break_even_one_way_cost_bps < model_one_way_cost_bps
        )
    if (
        recommendation in {"buy", "sell"}
        and cost_gate_fail
    ):
        recommendation = "hold"
        cost_gate_applied = True
    top_reasons, reason_details = _feature_reasons(
        model_payload,
        vector,
        raw_recommendation,
        probabilities,
    )
    if confidence_gate_applied:
        gate_reason = (
            f"confidence_gate_demoted_to_hold raw={raw_recommendation} "
            f"confidence={confidence:.3f} threshold={resolved_action_confidence_threshold:.3f}"
        )
        top_reasons = [gate_reason, *top_reasons]
        reason_details = [
            {
                "feature": "action_confidence_threshold",
                "value": float(confidence),
                "impact": float(confidence - resolved_action_confidence_threshold),
                "supports": "hold",
                "vs_alternative": raw_recommendation,
                "reason": "confidence_gate_demoted_to_hold",
            },
            *reason_details,
        ]
    if cost_gate_applied:
        if action_gate_basis == "execution_backtest":
            gate_reason = (
                "cost_gate_demoted_to_hold execution_equity_return="
                f"{model_execution_equity_return:.6f} execution_realized_pnl_delta_usd="
                f"{model_execution_realized_pnl_delta_usd:.6f} fill_rate="
                f"{model_execution_fill_rate:.6f}"
            )
        else:
            gate_reason = (
                f"cost_gate_demoted_to_hold net_expectancy_per_actionable="
                f"{model_net_expectancy_per_actionable:.6f} break_even_bps="
                f"{model_break_even_one_way_cost_bps:.3f} one_way_cost_bps="
                f"{model_one_way_cost_bps:.3f}"
            )
        top_reasons = [gate_reason, *top_reasons]
        reason_details = [
            {
                "feature": (
                    "execution_backtest_action_gate"
                    if action_gate_basis == "execution_backtest"
                    else "cost_aware_action_gate"
                ),
                "value": (
                    float(model_execution_equity_return)
                    if action_gate_basis == "execution_backtest"
                    else float(model_net_expectancy_per_actionable)
                ),
                "impact": (
                    float(model_execution_realized_pnl_delta_usd)
                    if action_gate_basis == "execution_backtest"
                    else float(model_break_even_one_way_cost_bps - model_one_way_cost_bps)
                ),
                "supports": "hold",
                "vs_alternative": raw_recommendation,
                "reason": "cost_gate_demoted_to_hold",
                "basis": action_gate_basis,
            },
            *reason_details,
        ]
    feature_values = {feature: float(latest_row[feature]) for feature in FEATURE_COLUMNS}
    close_price = float(latest_row["close"])
    sma_fast = close_price / (1.0 + float(latest_row["sma_fast_spread"]))
    sma_slow = close_price / (1.0 + float(latest_row["sma_slow_spread"]))
    macd = float(latest_row["macd"])
    macd_hist = float(latest_row["macd_hist"])
    rsi_14 = float(latest_row["rsi_14"])
    volatility_24 = float(latest_row["volatility_24"])
    timestamp_utc = pd.Timestamp(latest_row["timestamp"]).tz_convert("UTC").isoformat()

    prediction_path: Path | None = None
    if write_artifact:
        now = _utc_now()
        prediction_dir = settings.quant_data_root / "logs" / "agents" / "model-predictor" / f"{now:%Y-%m-%d}"
        prediction_path = prediction_dir / f"prediction_{now:%Y%m%dT%H%M%SZ}.json"
        _write_json(
            prediction_path,
            {
                "contract": "trigger_prediction.v2",
                "created_at_utc": now.isoformat(),
                "exchange": exchange,
                "symbol": symbol,
                "timeframe": timeframe,
                "prediction_timestamp_utc": timestamp_utc,
                "raw_recommendation": raw_recommendation,
                "recommendation": recommendation,
                "confidence": confidence,
                "action_confidence_threshold": float(resolved_action_confidence_threshold),
                "confidence_gate_applied": bool(confidence_gate_applied),
                "cost_gate_applied": bool(cost_gate_applied),
                "cost_gate_inputs": {
                    "basis": action_gate_basis,
                    "model_net_expectancy_per_actionable": model_net_expectancy_per_actionable,
                    "model_break_even_one_way_cost_bps": model_break_even_one_way_cost_bps,
                    "model_one_way_cost_bps": model_one_way_cost_bps,
                    "model_execution_equity_return": model_execution_equity_return,
                    "model_execution_realized_pnl_delta_usd": model_execution_realized_pnl_delta_usd,
                    "model_execution_fill_rate": model_execution_fill_rate,
                    "model_execution_rejection_rate": model_execution_rejection_rate,
                },
                "probabilities": probabilities,
                "close_price": close_price,
                "sma_fast": sma_fast,
                "sma_slow": sma_slow,
                "macd": macd,
                "macd_hist": macd_hist,
                "rsi_14": rsi_14,
                "volatility_24": volatility_24,
                "feature_values": feature_values,
                "top_reasons": top_reasons,
                "reason_details": reason_details,
                "source_data_path": str(source_data_path),
                "source_data_sha256": source_data_sha256,
                "model_path": str(resolved_model_path),
                "model_contract": str(model_payload.get("contract", "")),
                "model_created_at_utc": model_payload.get("created_at_utc"),
                "priority2_features_enabled": bool(priority2_features_enabled),
                "priority2_external_features_path": (
                    str(resolved_priority2_external_features_path)
                    if resolved_priority2_external_features_path is not None
                    else None
                ),
                "priority2_feature_columns": list(resolved_priority2_feature_columns),
                "priority2_external_features_path_resolution": priority2_external_resolution,
                "priority2_reason_codes": list(priority2_bundle.reason_codes),
                "priority2_diagnostics": dict(priority2_bundle.diagnostics),
                "priority2_feature_snapshot": dict(priority2_bundle.feature_snapshot),
            },
        )

    return TriggerPredictionResult(
        model_path=resolved_model_path,
        source_data_path=source_data_path,
        source_data_sha256=source_data_sha256,
        timestamp_utc=timestamp_utc,
        recommendation=recommendation,
        confidence=confidence,
        probabilities=probabilities,
        close_price=close_price,
        sma_fast=sma_fast,
        sma_slow=sma_slow,
        macd=macd,
        macd_hist=macd_hist,
        rsi_14=rsi_14,
        volatility_24=volatility_24,
        feature_values=feature_values,
        top_reasons=top_reasons,
        reason_details=reason_details,
        action_confidence_threshold=float(resolved_action_confidence_threshold),
        prediction_path=prediction_path,
    )


def _load_monitor_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return {}
    return {}


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(payload), sort_keys=True) + "\n")


def _send_webhook_notification(url: str, payload: dict[str, Any], timeout_seconds: float = 10.0) -> None:
    body = json.dumps(_json_safe(payload)).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(getattr(response, "status", 200))
            logger.info("Webhook notification delivered status=%s url=%s", status, url)
    except urllib.error.URLError as exc:
        logger.warning("Webhook notification failed url=%s error=%s", url, exc)


def _execute_trigger_paper_trade(
    *,
    settings: Settings,
    exchange: str,
    symbol: str,
    timeframe: str,
    cycle_count: int,
    prediction: TriggerPredictionResult,
    notional_usd: float,
    starting_cash_usd: float,
    fee_bps: float,
    slippage_bps: float,
) -> tuple[PaperTradeIntent, Any]:
    now = _utc_now()
    run_id = f"trigger-monitor-{now:%Y%m%dT%H%M%S%fZ}-c{cycle_count:04d}"
    intent_destination_path = (
        settings.quant_data_root
        / "paper-trading"
        / f"{now:%Y-%m-%d}"
        / f"paper_trade_intent_{run_id}.json"
    )
    action: Recommendation = (
        prediction.recommendation if prediction.recommendation in {"buy", "sell"} else "hold"
    )
    intent = PaperTradeIntent(
        contract="paper_trade_intent.v1",
        run_id=run_id,
        created_at_utc=now.isoformat(),
        mode="paper",
        status="emitted" if action in {"buy", "sell"} else "blocked",
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        action=action,
        notional_usd=max(0.0, float(notional_usd)),
        risk_approved=action in {"buy", "sell"},
        reason="trigger_monitor_signal",
        destination_path=str(intent_destination_path),
    )
    write_contract(intent_destination_path, intent)
    execution = execute_paper_trade_intent(
        quant_data_root=settings.quant_data_root,
        run_id=run_id,
        created_at_utc=now.isoformat(),
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        intent=intent,
        mark_price=float(prediction.close_price),
        starting_cash_usd=max(0.0, float(starting_cash_usd)),
        fee_bps=max(0.0, float(fee_bps)),
        slippage_bps=max(0.0, float(slippage_bps)),
    )
    return intent, execution


def monitor_trigger_signals(
    *,
    settings: Settings,
    exchange: str,
    symbol: str,
    timeframe: str,
    model_path: Path | None,
    limit: int,
    poll_seconds: float,
    confidence_threshold: float,
    webhook_url: str | None,
    notify_on_hold: bool,
    max_cycles: int | None,
    priority2_features_enabled: bool = True,
    priority2_external_features_path: Path | None = None,
    priority2_feature_columns: tuple[str, ...] | list[str] | None = None,
    paper_trading_enabled: bool = False,
    paper_notional_usd: float = 100.0,
    paper_starting_cash_usd: float = 10000.0,
    paper_fee_bps: float = 5.0,
    paper_slippage_bps: float = 1.0,
) -> TriggerMonitorResult:
    cycle_count = 0
    alert_count = 0
    paper_trades_attempted = 0
    paper_trades_executed = 0
    latest_alert_path: Path | None = None
    latest_paper_execution_path: Path | None = None
    state_path = settings.quant_data_root / "logs" / "agents" / "trigger-monitor" / "state.json"
    state = _load_monitor_state(state_path)

    poll_interval = max(5.0, float(poll_seconds))
    required_confidence = min(1.0, max(0.0, float(confidence_threshold)))
    hard_limit = max(50, int(limit))
    hard_max_cycles = int(max_cycles) if max_cycles is not None and int(max_cycles) > 0 else None

    logger.info(
        "Starting trigger monitor exchange=%s symbol=%s timeframe=%s poll_seconds=%.2f confidence_threshold=%.3f",
        exchange,
        symbol,
        timeframe,
        poll_interval,
        required_confidence,
    )

    while True:
        cycle_count += 1
        cycle_started = _utc_now()
        ingested_file: Path | None = None

        try:
            ingest_result = fetch_ohlcv_to_parquet(
                settings=settings,
                exchange_id=exchange,
                symbol=symbol,
                timeframe=timeframe,
                limit=hard_limit,
            )
            ingested_file = ingest_result.output_path
            logger.info("Monitor cycle=%s ingest rows=%s output=%s", cycle_count, ingest_result.row_count, ingested_file)
        except Exception as exc:
            logger.warning("Monitor cycle=%s ingest failed: %s", cycle_count, exc)

        prediction = predict_trigger_signal(
            settings=settings,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            model_path=model_path,
            input_file=ingested_file,
            write_artifact=True,
            priority2_features_enabled=bool(priority2_features_enabled),
            priority2_external_features_path=priority2_external_features_path,
            priority2_feature_columns=priority2_feature_columns,
        )

        is_actionable = prediction.recommendation in {"buy", "sell"}
        meets_confidence = prediction.confidence >= required_confidence
        should_alert = (is_actionable or notify_on_hold) and meets_confidence
        alert_signature = (
            f"{prediction.timestamp_utc}|{prediction.recommendation}|"
            f"{round(prediction.confidence, 6)}"
        )
        is_duplicate = state.get("last_alert_signature") == alert_signature
        if should_alert and not is_duplicate:
            now = _utc_now()
            alert_path = (
                settings.quant_data_root
                / "logs"
                / "agents"
                / "trigger-monitor"
                / f"{now:%Y-%m-%d}"
                / "alerts.jsonl"
            )
            alert_payload = {
                "contract": "trigger_alert.v1",
                "created_at_utc": now.isoformat(),
                "cycle": cycle_count,
                "exchange": exchange,
                "symbol": symbol,
                "timeframe": timeframe,
                "recommendation": prediction.recommendation,
                "confidence": prediction.confidence,
                "probabilities": prediction.probabilities,
                "prediction_timestamp_utc": prediction.timestamp_utc,
                "close_price": prediction.close_price,
                "sma_fast": prediction.sma_fast,
                "sma_slow": prediction.sma_slow,
                "macd": prediction.macd,
                "macd_hist": prediction.macd_hist,
                "rsi_14": prediction.rsi_14,
                "volatility_24": prediction.volatility_24,
                "top_reasons": prediction.top_reasons,
                "reason_details": prediction.reason_details,
                "prediction_path": str(prediction.prediction_path) if prediction.prediction_path else None,
                "model_path": str(prediction.model_path),
                "source_data_path": str(prediction.source_data_path),
                "source_data_sha256": prediction.source_data_sha256,
                "paper_trading_enabled": bool(paper_trading_enabled),
            }
            if paper_trading_enabled and is_actionable:
                try:
                    paper_intent, paper_execution = _execute_trigger_paper_trade(
                        settings=settings,
                        exchange=exchange,
                        symbol=symbol,
                        timeframe=timeframe,
                        cycle_count=cycle_count,
                        prediction=prediction,
                        notional_usd=paper_notional_usd,
                        starting_cash_usd=paper_starting_cash_usd,
                        fee_bps=paper_fee_bps,
                        slippage_bps=paper_slippage_bps,
                    )
                    paper_trades_attempted += 1
                    if str(paper_execution.execution_status) == "executed":
                        paper_trades_executed += 1
                    if paper_execution.execution_record_path:
                        latest_paper_execution_path = Path(str(paper_execution.execution_record_path))
                    alert_payload["paper_trade_intent_path"] = paper_intent.destination_path
                    alert_payload["paper_trade_execution_status"] = paper_execution.execution_status
                    alert_payload["paper_trade_executed_action"] = paper_execution.executed_action
                    alert_payload["paper_trade_executed_notional_usd"] = paper_execution.executed_notional_usd
                    alert_payload["paper_trade_execution_reason"] = paper_execution.reason
                    alert_payload["paper_trade_execution_record_path"] = (
                        str(paper_execution.execution_record_path)
                        if paper_execution.execution_record_path
                        else None
                    )
                    print(
                        f"[paper] cycle={cycle_count} "
                        f"status={paper_execution.execution_status} "
                        f"action={str(paper_execution.executed_action).upper()} "
                        f"notional={paper_execution.executed_notional_usd:.2f}"
                    )
                    state["last_paper_trade_status"] = str(paper_execution.execution_status)
                    state["last_paper_trade_action"] = str(paper_execution.executed_action)
                    state["last_paper_trade_notional_usd"] = float(paper_execution.executed_notional_usd)
                    state["last_paper_trade_reason"] = str(paper_execution.reason)
                    state["last_paper_trade_execution_record_path"] = (
                        str(paper_execution.execution_record_path)
                        if paper_execution.execution_record_path
                        else None
                    )
                except Exception as exc:
                    logger.warning("Monitor cycle=%s paper trade execution failed: %s", cycle_count, exc)
                    alert_payload["paper_trade_execution_status"] = "error"
                    alert_payload["paper_trade_execution_error"] = str(exc)
                    state["last_paper_trade_status"] = "error"
                    state["last_paper_trade_reason"] = str(exc)
            _append_jsonl(alert_path, alert_payload)
            latest_alert_path = alert_path
            alert_count += 1

            if webhook_url:
                _send_webhook_notification(webhook_url, alert_payload)

            print(
                f"[alert] cycle={cycle_count} "
                f"{prediction.recommendation.upper()} "
                f"confidence={prediction.confidence:.3f} "
                f"ts={prediction.timestamp_utc}"
            )
            state["last_alert_signature"] = alert_signature
            state["last_alert_at_utc"] = now.isoformat()
            state["last_alert_recommendation"] = prediction.recommendation
            state["last_alert_confidence"] = prediction.confidence
        else:
            logger.info(
                "Monitor cycle=%s no alert recommendation=%s confidence=%.3f threshold=%.3f actionable=%s duplicate=%s",
                cycle_count,
                prediction.recommendation,
                prediction.confidence,
                required_confidence,
                is_actionable,
                is_duplicate,
            )

        state["last_cycle_at_utc"] = cycle_started.isoformat()
        state["last_cycle_recommendation"] = prediction.recommendation
        state["last_cycle_confidence"] = prediction.confidence
        state["last_cycle_prediction_path"] = (
            str(prediction.prediction_path) if prediction.prediction_path else None
        )
        _write_json(state_path, state)

        if hard_max_cycles is not None and cycle_count >= hard_max_cycles:
            break
        time.sleep(poll_interval)

    return TriggerMonitorResult(
        cycles_completed=cycle_count,
        alerts_emitted=alert_count,
        paper_trades_attempted=paper_trades_attempted,
        paper_trades_executed=paper_trades_executed,
        latest_alert_path=latest_alert_path,
        latest_paper_execution_path=latest_paper_execution_path,
        state_path=state_path,
    )
