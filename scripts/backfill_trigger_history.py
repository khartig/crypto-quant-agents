#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_agents.config import load_settings
from quant_agents.storage import latest_raw_dataset
from quant_agents.trigger_model import (
    FEATURE_COLUMNS,
    _build_feature_frame,
    _coerce_frame,
    _feature_reasons,
    _predict_probabilities,
    _resolve_prediction_action_confidence_threshold,
    latest_trigger_model_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill historical trigger prediction/alert artifacts from latest raw dataset."
    )
    parser.add_argument("--exchange", default="kraken")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument(
        "--points",
        type=int,
        default=480,
        help="How many latest feature rows to backfill (default: 480).",
    )
    parser.add_argument(
        "--alert-confidence-threshold",
        type=float,
        default=0.6,
        help="Minimum confidence for backfilled buy/sell alerts (default: 0.6).",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Delete existing model-predictor and trigger-monitor logs before backfill.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()

    if args.clear_existing:
        predictor_root = settings.quant_data_root / "logs" / "agents" / "model-predictor"
        monitor_root = settings.quant_data_root / "logs" / "agents" / "trigger-monitor"
        if predictor_root.exists():
            for path in predictor_root.rglob("*"):
                if path.is_file():
                    path.unlink()
            for path in sorted((p for p in predictor_root.rglob("*") if p.is_dir()), reverse=True):
                path.rmdir()
            predictor_root.rmdir()
        if monitor_root.exists():
            for path in monitor_root.rglob("*"):
                if path.is_file():
                    path.unlink()
            for path in sorted((p for p in monitor_root.rglob("*") if p.is_dir()), reverse=True):
                path.rmdir()
            monitor_root.rmdir()

    model_path = latest_trigger_model_path(
        settings.quant_data_root,
        args.exchange,
        args.symbol,
        args.timeframe,
    )
    model_payload = json.loads(model_path.read_text(encoding="utf-8"))
    source_data_path = latest_raw_dataset(
        settings.quant_data_root,
        args.exchange,
        args.symbol,
        args.timeframe,
    )
    source_data_sha256 = hashlib.sha256(source_data_path.read_bytes()).hexdigest()

    frame = _coerce_frame(source_data_path)
    priority2_external_features_path = (
        Path(settings.priority2_external_features_path).expanduser().resolve()
        if settings.priority2_external_features_path
        else None
    )
    feature_frame, _ = _build_feature_frame(
        frame,
        priority2_features_enabled=bool(settings.priority2_features_enabled),
        priority2_external_features_path=priority2_external_features_path,
        priority2_feature_columns=tuple(settings.priority2_feature_columns),
    )
    feature_frame = feature_frame.dropna(subset=list(FEATURE_COLUMNS)).reset_index(drop=True)
    if feature_frame.empty:
        raise RuntimeError("No usable feature rows for backfill")

    points = max(1, int(args.points))
    backfill_frame = feature_frame.tail(points).reset_index(drop=True)
    threshold = max(0.0, min(1.0, float(args.alert_confidence_threshold)))
    action_confidence_threshold = _resolve_prediction_action_confidence_threshold(
        model_payload=model_payload,
        override_threshold=None,
    )

    prediction_count = 0
    alert_count = 0

    for index, row in backfill_frame.iterrows():
        timestamp = pd.Timestamp(row["timestamp"]).tz_convert("UTC")
        vector = np.asarray([float(row[feature]) for feature in FEATURE_COLUMNS], dtype=float)
        raw_recommendation, probabilities = _predict_probabilities(model_payload, vector)
        confidence = float(probabilities[raw_recommendation])
        recommendation = raw_recommendation
        confidence_gate_applied = False
        if recommendation in {"buy", "sell"} and confidence < action_confidence_threshold:
            recommendation = "hold"
            confidence_gate_applied = True
        top_reasons, reason_details = _feature_reasons(
            model_payload,
            vector,
            raw_recommendation,
            probabilities,
        )
        if confidence_gate_applied:
            gate_reason = (
                f"confidence_gate_demoted_to_hold raw={raw_recommendation} "
                f"confidence={confidence:.3f} threshold={action_confidence_threshold:.3f}"
            )
            top_reasons = [gate_reason, *top_reasons]
            reason_details = [
                {
                    "feature": "action_confidence_threshold",
                    "value": float(confidence),
                    "impact": float(confidence - action_confidence_threshold),
                    "supports": "hold",
                    "vs_alternative": raw_recommendation,
                    "reason": "confidence_gate_demoted_to_hold",
                },
                *reason_details,
            ]

        close_price = float(row["close"])
        sma_fast = close_price / (1.0 + float(row["sma_fast_spread"]))
        sma_slow = close_price / (1.0 + float(row["sma_slow_spread"]))
        macd = float(row["macd"])
        macd_hist = float(row["macd_hist"])
        rsi_14 = float(row["rsi_14"])
        volatility_24 = float(row["volatility_24"])

        prediction_dir = (
            settings.quant_data_root
            / "logs"
            / "agents"
            / "model-predictor"
            / f"{timestamp:%Y-%m-%d}"
        )
        prediction_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = prediction_dir / f"prediction_{timestamp:%Y%m%dT%H%M%SZ}_{index:04d}.json"
        prediction_payload = {
            "contract": "trigger_prediction.v2",
            "created_at_utc": timestamp.isoformat(),
            "exchange": args.exchange,
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "prediction_timestamp_utc": timestamp.isoformat(),
            "raw_recommendation": raw_recommendation,
            "recommendation": recommendation,
            "confidence": confidence,
            "action_confidence_threshold": float(action_confidence_threshold),
            "confidence_gate_applied": bool(confidence_gate_applied),
            "probabilities": probabilities,
            "close_price": close_price,
            "sma_fast": sma_fast,
            "sma_slow": sma_slow,
            "macd": macd,
            "macd_hist": macd_hist,
            "rsi_14": rsi_14,
            "volatility_24": volatility_24,
            "feature_values": {feature: float(row[feature]) for feature in FEATURE_COLUMNS},
            "top_reasons": top_reasons,
            "reason_details": reason_details,
            "source_data_path": str(source_data_path),
            "source_data_sha256": source_data_sha256,
            "model_path": str(model_path),
            "model_contract": str(model_payload.get("contract", "")),
            "model_created_at_utc": model_payload.get("created_at_utc"),
        }
        prediction_path.write_text(json.dumps(prediction_payload, indent=2), encoding="utf-8")
        prediction_count += 1

        if recommendation in {"buy", "sell"} and confidence >= threshold:
            alert_dir = (
                settings.quant_data_root
                / "logs"
                / "agents"
                / "trigger-monitor"
                / f"{timestamp:%Y-%m-%d}"
            )
            alert_dir.mkdir(parents=True, exist_ok=True)
            alert_path = alert_dir / "alerts.jsonl"
            alert_payload = {
                "contract": "trigger_alert.v1",
                "created_at_utc": timestamp.isoformat(),
                "cycle": index + 1,
                "exchange": args.exchange,
                "symbol": args.symbol,
                "timeframe": args.timeframe,
                "recommendation": recommendation,
                "confidence": confidence,
                "action_confidence_threshold": float(action_confidence_threshold),
                "probabilities": probabilities,
                "prediction_timestamp_utc": timestamp.isoformat(),
                "close_price": close_price,
                "sma_fast": sma_fast,
                "sma_slow": sma_slow,
                "macd": macd,
                "macd_hist": macd_hist,
                "rsi_14": rsi_14,
                "volatility_24": volatility_24,
                "top_reasons": top_reasons,
                "reason_details": reason_details,
                "prediction_path": str(prediction_path),
                "model_path": str(model_path),
                "source_data_path": str(source_data_path),
                "source_data_sha256": source_data_sha256,
            }
            with alert_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(alert_payload) + "\n")
            alert_count += 1

    print(f"backfilled_predictions={prediction_count}")
    print(f"backfilled_alerts={alert_count}")
    print(f"model_path={model_path}")
    print(f"source_data_path={source_data_path}")


if __name__ == "__main__":
    main()
