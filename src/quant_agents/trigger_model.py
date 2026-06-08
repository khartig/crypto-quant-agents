from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_agents.config import Settings
from quant_agents.ingestion import fetch_ohlcv_to_parquet
from quant_agents.storage import latest_raw_dataset, symbol_slug

logger = logging.getLogger(__name__)

TRIGGER_LABELS: tuple[str, str, str] = ("buy", "hold", "sell")
FEATURE_COLUMNS: tuple[str, ...] = (
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
    prediction_path: Path | None


@dataclass(frozen=True)
class TriggerMonitorResult:
    cycles_completed: int
    alerts_emitted: int
    latest_alert_path: Path | None
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


def _build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
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
    return feature_frame


def _label_training_frame(
    feature_frame: pd.DataFrame,
    *,
    horizon_bars: int,
    buy_threshold: float,
    sell_threshold: float,
) -> pd.DataFrame:
    horizon = max(1, int(horizon_bars))
    buy_cutoff = float(buy_threshold)
    sell_cutoff = abs(float(sell_threshold))

    labeled = feature_frame.copy()
    labeled["forward_return"] = labeled["close"].shift(-horizon) / labeled["close"] - 1.0
    labeled["label"] = "hold"
    labeled.loc[labeled["forward_return"] >= buy_cutoff, "label"] = "buy"
    labeled.loc[labeled["forward_return"] <= -sell_cutoff, "label"] = "sell"
    labeled = labeled.dropna(subset=[*FEATURE_COLUMNS, "forward_return"]).reset_index(drop=True)
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
) -> TriggerModelTrainingResult:
    source_data_path = (
        input_file.expanduser().resolve()
        if input_file is not None
        else latest_raw_dataset(settings.quant_data_root, exchange, symbol, timeframe)
    )
    if not source_data_path.exists():
        raise FileNotFoundError(f"Training input file not found: {source_data_path}")
    source_data_sha256 = _sha256_file(source_data_path)

    frame = _coerce_frame(source_data_path)
    feature_frame = _build_feature_frame(frame)
    labeled = _label_training_frame(
        feature_frame,
        horizon_bars=max(1, int(horizon_bars)),
        buy_threshold=float(buy_threshold),
        sell_threshold=float(sell_threshold),
    )
    if len(labeled) < max(30, int(min_train_samples) + 5):
        raise RuntimeError(
            f"Insufficient labeled rows for training: {len(labeled)} "
            f"(need at least {max(30, int(min_train_samples) + 5)})"
        )

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
    model = _fit_gaussian_model(train_frame)

    expected = test_frame["label"].astype(str).tolist()
    predicted = _predict_labels(model, test_frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float))
    accuracy, confusion = _classification_metrics(expected, predicted)

    distribution = {
        label: int((labeled["label"] == label).sum())
        for label in TRIGGER_LABELS
    }
    run_dir = _new_trigger_model_dir(settings.quant_data_root, exchange, symbol, timeframe)
    model_path = run_dir / "model.json"
    train_frame_path = run_dir / "train_dataset.parquet"
    test_frame_path = run_dir / "test_dataset.parquet"
    train_frame.to_parquet(train_frame_path, index=False)
    test_frame.to_parquet(test_frame_path, index=False)

    model_payload = {
        "contract": "trigger_model.gaussian_nb.v1",
        "created_at_utc": _utc_now_iso(),
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "source_data_path": str(source_data_path),
        "source_data_sha256": source_data_sha256,
        "horizon_bars": int(max(1, horizon_bars)),
        "buy_threshold": float(buy_threshold),
        "sell_threshold": float(abs(sell_threshold)),
        "feature_columns": list(FEATURE_COLUMNS),
        "labels": list(TRIGGER_LABELS),
        "class_stats": model["class_stats"],
        "training_metrics": {
            "sample_count": int(len(labeled)),
            "train_count": int(train_count),
            "test_count": int(test_count),
            "accuracy": accuracy,
            "label_distribution": distribution,
            "confusion_matrix": confusion,
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "train_dataset_path": str(train_frame_path),
            "test_dataset_path": str(test_frame_path),
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
    )


def predict_trigger_signal(
    *,
    settings: Settings,
    exchange: str,
    symbol: str,
    timeframe: str,
    model_path: Path | None,
    input_file: Path | None,
    write_artifact: bool = True,
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

    frame = _coerce_frame(source_data_path)
    feature_frame = _build_feature_frame(frame).dropna(subset=list(FEATURE_COLUMNS)).reset_index(drop=True)
    if feature_frame.empty:
        raise RuntimeError("No usable feature rows for prediction")

    latest_row = feature_frame.iloc[-1]
    vector = np.asarray([float(latest_row[feature]) for feature in FEATURE_COLUMNS], dtype=float)
    recommendation, probabilities = _predict_probabilities(model_payload, vector)
    confidence = float(probabilities[recommendation])
    top_reasons, reason_details = _feature_reasons(model_payload, vector, recommendation, probabilities)
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
                "contract": "trigger_prediction.v1",
                "created_at_utc": now.isoformat(),
                "exchange": exchange,
                "symbol": symbol,
                "timeframe": timeframe,
                "prediction_timestamp_utc": timestamp_utc,
                "recommendation": recommendation,
                "confidence": confidence,
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
) -> TriggerMonitorResult:
    cycle_count = 0
    alert_count = 0
    latest_alert_path: Path | None = None
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
            }
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
        latest_alert_path=latest_alert_path,
        state_path=state_path,
    )
