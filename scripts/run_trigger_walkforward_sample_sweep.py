#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_agents.config import ensure_data_root_ready, load_settings
from quant_agents.ingestion import fetch_ohlcv_to_parquet
from quant_agents.orderbook_features import (
    DEFAULT_STABLE_ORDERBOOK_FEATURE_COLUMNS,
    normalize_orderbook_feature_columns,
)
from quant_agents.priority2_features import (
    DEFAULT_STABLE_PRIORITY2_FEATURE_COLUMNS,
    normalize_priority2_feature_columns,
)
from quant_agents.ranked_features import (
    DEFAULT_STABLE_RANKED_FEATURE_COLUMNS,
    normalize_ranked_feature_columns,
)
from quant_agents.storage import latest_raw_dataset, symbol_slug
from quant_agents.trigger_model import (
    _apply_orderbook_features,
    _apply_orderbook_quality_gate,
    _apply_priority2_quality_gate,
    _apply_ranked_features,
    _apply_ranked_quality_gate,
    _build_feature_frame,
    _coerce_frame,
    _evaluate_model,
    _evaluation_selection_rank,
    _fit_gaussian_model,
    _fit_model_family_candidates,
    _fit_regularized_lda_model,
    _label_training_frame,
    _resolve_orderbook_features_path,
    _resolve_priority2_external_features_path,
    _resolve_ranked_external_features_path,
    _select_action_confidence_frontier,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_sample_sizes(raw: str) -> list[int]:
    values: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        parsed = int(token)
        if parsed <= 0:
            continue
        values.append(parsed)
    deduped = sorted(set(values))
    if not deduped:
        raise ValueError("No valid sample sizes were provided.")
    return deduped


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


def _build_recency_weights(row_count: int, half_life_bars: int) -> np.ndarray:
    if row_count <= 0:
        return np.zeros(0, dtype=float)
    if half_life_bars <= 0:
        return np.ones(row_count, dtype=float)
    ages = np.arange(row_count - 1, -1, -1, dtype=float)
    weights = np.power(0.5, ages / float(max(1, half_life_bars)))
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.clip(weights, 0.0, None)
    if float(np.sum(weights)) <= 0.0:
        return np.ones(row_count, dtype=float)
    return weights


def _apply_rolling_window(frame: pd.DataFrame, rolling_train_window_bars: int) -> pd.DataFrame:
    if rolling_train_window_bars <= 0:
        return frame
    if len(frame) <= rolling_train_window_bars:
        return frame
    return frame.iloc[-rolling_train_window_bars:].reset_index(drop=True)


def _extract_eval_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    per_class = (
        dict(evaluation.get("per_class_metrics", {}).get("classes", {}))
        if isinstance(evaluation.get("per_class_metrics", {}), dict)
        else {}
    )
    actionable = (
        dict(evaluation.get("actionable_metrics", {}))
        if isinstance(evaluation.get("actionable_metrics", {}), dict)
        else {}
    )
    calibration = (
        dict(evaluation.get("calibration_metrics", {}))
        if isinstance(evaluation.get("calibration_metrics", {}), dict)
        else {}
    )
    expectancy = (
        dict(evaluation.get("expectancy_metrics", {}))
        if isinstance(evaluation.get("expectancy_metrics", {}), dict)
        else {}
    )
    execution = (
        dict(evaluation.get("execution_backtest_metrics", {}))
        if isinstance(evaluation.get("execution_backtest_metrics", {}), dict)
        else {}
    )
    buy_metrics = dict(per_class.get("buy", {})) if isinstance(per_class.get("buy", {}), dict) else {}
    sell_metrics = (
        dict(per_class.get("sell", {})) if isinstance(per_class.get("sell", {}), dict) else {}
    )
    hold_metrics = (
        dict(per_class.get("hold", {})) if isinstance(per_class.get("hold", {}), dict) else {}
    )
    return {
        "accuracy": _safe_float(evaluation.get("accuracy"), 0.0),
        "buy_precision": _safe_float(buy_metrics.get("precision"), 0.0),
        "buy_recall": _safe_float(buy_metrics.get("recall"), 0.0),
        "sell_precision": _safe_float(sell_metrics.get("precision"), 0.0),
        "sell_recall": _safe_float(sell_metrics.get("recall"), 0.0),
        "hold_precision": _safe_float(hold_metrics.get("precision"), 0.0),
        "hold_recall": _safe_float(hold_metrics.get("recall"), 0.0),
        "macro_f1": _safe_float(evaluation.get("per_class_metrics", {}).get("macro_f1"), 0.0)
        if isinstance(evaluation.get("per_class_metrics", {}), dict)
        else 0.0,
        "binary_actionable_precision": _safe_float(
            actionable.get("binary_actionable_precision"),
            0.0,
        ),
        "binary_actionable_recall": _safe_float(
            actionable.get("binary_actionable_recall"),
            0.0,
        ),
        "actionable_rate": _safe_float(actionable.get("actionable_rate"), 0.0),
        "expected_calibration_error": _safe_float(
            calibration.get("expected_calibration_error"),
            0.0,
        ),
        "brier_score": _safe_float(calibration.get("brier_score"), 0.0),
        "log_loss": _safe_float(calibration.get("log_loss"), 0.0),
        "net_expectancy_per_actionable": _safe_float(
            expectancy.get("net_expectancy_per_actionable"),
            0.0,
        ),
        "net_expectancy_per_bar": _safe_float(expectancy.get("net_expectancy_per_bar"), 0.0),
        "execution_equity_return": _safe_float(execution.get("equity_return"), 0.0),
        "execution_realized_pnl_delta_usd": _safe_float(
            execution.get("realized_pnl_delta_usd"),
            0.0,
        ),
        "execution_fill_rate": _safe_float(execution.get("fill_rate"), 0.0),
        "execution_rejection_rate": _safe_float(execution.get("rejection_rate"), 0.0),
        "execution_max_drawdown": _safe_float(execution.get("max_drawdown"), 0.0),
    }


def _aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    numeric_keys: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                numeric_keys.add(key)
    aggregated: dict[str, float] = {}
    for key in sorted(numeric_keys):
        values = [_safe_float(row.get(key), 0.0) for row in rows]
        aggregated[f"mean_{key}"] = float(np.mean(values))
        aggregated[f"std_{key}"] = float(np.std(values))
    return aggregated


def _build_walkforward_windows(
    total_rows: int,
    *,
    train_bars: int,
    validate_bars: int,
    step_bars: int,
) -> list[tuple[int, int, int, int]]:
    windows: list[tuple[int, int, int, int]] = []
    train_end = max(1, int(train_bars))
    while train_end + validate_bars <= total_rows:
        validate_start = train_end
        validate_end = validate_start + validate_bars
        windows.append((0, train_end, validate_start, validate_end))
        train_end += step_bars
    return windows


def _select_family_for_final_fit(fold_rows: list[dict[str, Any]]) -> tuple[str, float]:
    family_counter: Counter[str] = Counter(
        str(row.get("selected_model_family", "gaussian_nb")) for row in fold_rows
    )
    if not family_counter:
        return "gaussian_nb", 0.55
    candidate_families = sorted(family_counter.keys())
    if len(candidate_families) == 1:
        chosen_family = candidate_families[0]
    else:
        family_scores: dict[str, float] = defaultdict(float)
        for family in candidate_families:
            family_rows = [row for row in fold_rows if str(row.get("selected_model_family")) == family]
            family_scores[family] = float(
                np.mean(
                    [
                        _safe_float(row.get("validation_rank_primary"), 0.0)
                        for row in family_rows
                    ]
                )
            )
        chosen_family = max(candidate_families, key=lambda family: (family_counter[family], family_scores[family]))
    thresholds = [
        _safe_float(row.get("selected_action_confidence_threshold"), 0.0)
        for row in fold_rows
        if np.isfinite(_safe_float(row.get("selected_action_confidence_threshold"), np.nan))
    ]
    if not thresholds:
        return chosen_family, 0.55
    return chosen_family, float(np.median(thresholds))


def _fit_family_model(
    *,
    family: str,
    train_frame: pd.DataFrame,
    sample_weights: np.ndarray | None,
) -> dict[str, Any]:
    normalized_family = str(family).strip().lower()
    if normalized_family == "regularized_lda":
        return _fit_regularized_lda_model(train_frame, sample_weights=sample_weights)
    return _fit_gaussian_model(train_frame, sample_weights=sample_weights)


def _estimate_storage_for_pull(
    *,
    quant_data_root: Path,
    source_data_path: Path,
    available_rows: int,
    required_rows: int,
) -> dict[str, Any]:
    disk_usage = shutil.disk_usage(quant_data_root)
    free_bytes = int(disk_usage.free)
    current_file_bytes = int(source_data_path.stat().st_size) if source_data_path.exists() else 0
    bytes_per_row = (
        float(current_file_bytes / max(1, available_rows))
        if available_rows > 0 and current_file_bytes > 0
        else 512.0
    )
    additional_rows_needed = int(max(0, required_rows - available_rows))
    estimated_additional_bytes = int(additional_rows_needed * bytes_per_row * 1.5)
    return {
        "quant_data_root": str(quant_data_root),
        "source_data_path": str(source_data_path),
        "free_bytes": free_bytes,
        "source_file_bytes": current_file_bytes,
        "available_rows": int(available_rows),
        "required_rows": int(required_rows),
        "additional_rows_needed": additional_rows_needed,
        "estimated_bytes_per_row": float(bytes_per_row),
        "estimated_additional_bytes_needed": estimated_additional_bytes,
        "has_enough_free_space_for_estimated_pull": bool(free_bytes >= estimated_additional_bytes),
    }


def _build_markdown_report(payload: dict[str, Any]) -> str:
    rows = list(payload.get("sample_size_results", []))
    storage = dict(payload.get("storage_precheck", {}))
    lines = [
        "# Trigger walk-forward sample-size sweep",
        f"- created_at_utc: `{payload.get('created_at_utc')}`",
        f"- exchange: `{payload.get('exchange')}`",
        f"- symbol: `{payload.get('symbol')}`",
        f"- timeframe: `{payload.get('timeframe')}`",
        f"- source_data_path: `{payload.get('source_data_path')}`",
        "",
        "## Storage precheck for max sample size",
        f"- required_rows: `{storage.get('required_rows')}`",
        f"- available_rows: `{storage.get('available_rows')}`",
        f"- additional_rows_needed: `{storage.get('additional_rows_needed')}`",
        f"- free_bytes: `{storage.get('free_bytes')}`",
        f"- estimated_additional_bytes_needed: `{storage.get('estimated_additional_bytes_needed')}`",
        (
            "- has_enough_free_space_for_estimated_pull: "
            f"`{storage.get('has_enough_free_space_for_estimated_pull')}`"
        ),
        "",
        "## Sample-size results",
    ]
    if not rows:
        lines.append("- none")
    for row in rows:
        lines.append(f"### sample_size={row.get('sample_size')}")
        lines.append(f"- status: `{row.get('status')}`")
        if str(row.get("status")) != "ok":
            lines.append(f"- reason: `{row.get('reason')}`")
            continue
        lines.append(f"- pretest_rows: `{row.get('pretest_rows')}`")
        lines.append(f"- oot_rows: `{row.get('oot_rows')}`")
        lines.append(f"- walkforward_window_count: `{row.get('walkforward_window_count')}`")
        validation = dict(row.get("walkforward_validation_summary", {}))
        oot = dict(row.get("oot_metrics", {}))
        lines.append(
            "- walkforward_mean_accuracy / oot_accuracy: "
            f"`{validation.get('mean_accuracy')}` / `{oot.get('accuracy')}`"
        )
        lines.append(
            "- walkforward_mean_buy_precision / oot_buy_precision: "
            f"`{validation.get('mean_buy_precision')}` / `{oot.get('buy_precision')}`"
        )
        lines.append(
            "- walkforward_mean_sell_precision / oot_sell_precision: "
            f"`{validation.get('mean_sell_precision')}` / `{oot.get('sell_precision')}`"
        )
        lines.append(
            "- walkforward_mean_execution_realized_pnl_delta_usd / oot_execution_realized_pnl_delta_usd: "
            f"`{validation.get('mean_execution_realized_pnl_delta_usd')}` / "
            f"`{oot.get('execution_realized_pnl_delta_usd')}`"
        )
        lines.append(
            "- walkforward_mean_execution_max_drawdown / oot_execution_max_drawdown: "
            f"`{validation.get('mean_execution_max_drawdown')}` / "
            f"`{oot.get('execution_max_drawdown')}`"
        )
    return "\n".join(lines) + "\n"


def _prepare_labeled_dataset(
    *,
    settings: Any,
    exchange: str,
    symbol: str,
    timeframe: str,
    source_data_path: Path,
    horizon_bars: int,
    buy_threshold: float,
    sell_threshold: float,
    labeling_mode: str,
    trade_quality_min_score: float,
    cost_bps: float,
    priority2_features_enabled: bool,
    priority2_external_features_path: Path | None,
    priority2_feature_columns: tuple[str, ...],
    ranked_features_enabled: bool,
    ranked_external_features_path: Path | None,
    ranked_feature_columns: tuple[str, ...],
    orderbook_features_enabled: bool,
    orderbook_features_path: Path | None,
    orderbook_feature_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved_priority2_external_features_path, priority2_external_resolution = (
        _resolve_priority2_external_features_path(
            settings=settings,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            priority2_features_enabled=priority2_features_enabled,
            requested_path=priority2_external_features_path,
        )
    )
    resolved_ranked_external_features_path, ranked_external_resolution = (
        _resolve_ranked_external_features_path(
            settings=settings,
            ranked_features_enabled=ranked_features_enabled,
            requested_path=ranked_external_features_path,
        )
    )
    resolved_orderbook_features_path, orderbook_features_resolution = (
        _resolve_orderbook_features_path(
            settings=settings,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            orderbook_features_enabled=orderbook_features_enabled,
            requested_path=orderbook_features_path,
        )
    )

    frame = _coerce_frame(source_data_path)
    feature_frame, priority2_bundle = _build_feature_frame(
        frame,
        priority2_features_enabled=priority2_features_enabled,
        priority2_external_features_path=resolved_priority2_external_features_path,
        priority2_feature_columns=priority2_feature_columns,
    )
    feature_frame, priority2_bundle = _apply_priority2_quality_gate(
        feature_frame=feature_frame,
        bundle=priority2_bundle,
        selected_feature_columns=priority2_feature_columns,
        settings=settings,
    )
    feature_frame, ranked_bundle = _apply_ranked_features(
        market_frame=frame,
        feature_frame=feature_frame,
        ranked_features_enabled=ranked_features_enabled,
        ranked_external_features_path=resolved_ranked_external_features_path,
        ranked_feature_columns=ranked_feature_columns,
    )
    feature_frame, ranked_bundle = _apply_ranked_quality_gate(
        feature_frame=feature_frame,
        bundle=ranked_bundle,
        selected_feature_columns=ranked_feature_columns,
        settings=settings,
    )
    feature_frame, orderbook_bundle = _apply_orderbook_features(
        market_frame=frame,
        feature_frame=feature_frame,
        orderbook_features_enabled=orderbook_features_enabled,
        orderbook_features_path=resolved_orderbook_features_path,
        orderbook_feature_columns=orderbook_feature_columns,
    )
    feature_frame, orderbook_bundle = _apply_orderbook_quality_gate(
        feature_frame=feature_frame,
        bundle=orderbook_bundle,
        selected_feature_columns=orderbook_feature_columns,
        settings=settings,
    )
    labeled = _label_training_frame(
        feature_frame,
        horizon_bars=max(1, int(horizon_bars)),
        buy_threshold=max(0.0005, float(buy_threshold)),
        sell_threshold=max(0.0005, abs(float(sell_threshold))),
        labeling_mode=str(labeling_mode),
        trade_quality_min_score=float(np.clip(trade_quality_min_score, 0.0, 1.0)),
        one_way_cost_bps=max(0.0, float(cost_bps)),
    )
    metadata = {
        "priority2_external_features_path_resolution": priority2_external_resolution,
        "priority2_external_features_path": (
            str(resolved_priority2_external_features_path)
            if resolved_priority2_external_features_path is not None
            else None
        ),
        "priority2_reason_codes": list(priority2_bundle.reason_codes),
        "ranked_external_features_path_resolution": ranked_external_resolution,
        "ranked_external_features_path": (
            str(resolved_ranked_external_features_path)
            if resolved_ranked_external_features_path is not None
            else None
        ),
        "ranked_reason_codes": list(ranked_bundle.reason_codes),
        "orderbook_features_path_resolution": orderbook_features_resolution,
        "orderbook_features_path": (
            str(resolved_orderbook_features_path)
            if resolved_orderbook_features_path is not None
            else None
        ),
        "orderbook_reason_codes": list(orderbook_bundle.reason_codes),
    }
    return labeled, metadata


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Run overfit-safe trigger-model sample-size sweeps using walk-forward tuning and "
            "a final untouched out-of-time test window."
        )
    )
    parser.add_argument("--exchange", default=settings.default_exchange)
    parser.add_argument("--symbol", default=settings.default_symbol)
    parser.add_argument("--timeframe", default=settings.default_timeframe)
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument(
        "--sample-sizes",
        default="11232,25000,50000,100000",
        help="Comma-separated sample sizes to evaluate.",
    )
    parser.add_argument(
        "--oot-test-fraction",
        type=float,
        default=0.20,
        help="Fraction of each sample held out as untouched final OOT test.",
    )
    parser.add_argument(
        "--walkforward-train-fraction",
        type=float,
        default=0.60,
        help="Initial train fraction (of pretest data) used for walk-forward windows.",
    )
    parser.add_argument(
        "--walkforward-validate-fraction",
        type=float,
        default=0.10,
        help="Validation fraction (of pretest data) used per walk-forward window.",
    )
    parser.add_argument(
        "--walkforward-step-fraction",
        type=float,
        default=0.10,
        help="Step fraction (of pretest data) used to advance walk-forward windows.",
    )
    parser.add_argument(
        "--walkforward-min-windows",
        type=int,
        default=3,
        help="Minimum required walk-forward windows.",
    )
    parser.add_argument(
        "--rolling-train-window-bars",
        type=int,
        default=0,
        help="If >0, cap each fold's train set to most recent N rows (bounded rolling window).",
    )
    parser.add_argument(
        "--recency-half-life-bars",
        type=int,
        default=0,
        help="If >0, apply exponential recency weights with this half-life in bars.",
    )
    parser.add_argument(
        "--auto-pull-missing",
        action="store_true",
        help="Attempt to pull additional rows if max sample size exceeds available rows.",
    )
    parser.add_argument("--horizon-bars", type=int, default=int(settings.trigger_model_horizon_bars))
    parser.add_argument("--buy-threshold", type=float, default=float(settings.trigger_model_buy_threshold))
    parser.add_argument("--sell-threshold", type=float, default=float(settings.trigger_model_sell_threshold))
    parser.add_argument(
        "--min-train-samples",
        type=int,
        default=int(settings.trigger_model_min_train_samples),
    )
    parser.add_argument("--cost-bps", type=float, default=float(settings.trigger_model_cost_bps))
    parser.add_argument(
        "--action-confidence-threshold",
        type=float,
        default=float(settings.trigger_model_action_confidence_threshold),
    )
    parser.add_argument(
        "--labeling-mode",
        choices=("directional_v1", "triple_barrier_v2"),
        default=str(settings.trigger_model_labeling_mode),
    )
    parser.add_argument(
        "--trade-quality-min-score",
        type=float,
        default=float(settings.trigger_model_trade_quality_min_score),
    )
    parser.add_argument(
        "--disable-priority2-features",
        action="store_true",
        help="Disable Priority2 features for this sweep.",
    )
    parser.add_argument(
        "--disable-ranked-features",
        action="store_true",
        help="Disable ranked features for this sweep.",
    )
    parser.add_argument(
        "--disable-orderbook-features",
        action="store_true",
        help="Disable orderbook features for this sweep.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    ensure_data_root_ready(
        settings.quant_data_root,
        allow_unmounted=settings.allow_unmounted_data_root,
    )

    sample_sizes = _parse_sample_sizes(args.sample_sizes)
    exchange = str(args.exchange)
    symbol = str(args.symbol)
    timeframe = str(args.timeframe)
    source_data_path = (
        args.input_file.expanduser().resolve()
        if args.input_file is not None
        else latest_raw_dataset(settings.quant_data_root, exchange, symbol, timeframe)
    )
    if not source_data_path.exists():
        raise FileNotFoundError(f"Input file not found: {source_data_path}")

    raw_frame = _coerce_frame(source_data_path)
    max_sample_size = max(sample_sizes)
    storage_precheck = _estimate_storage_for_pull(
        quant_data_root=settings.quant_data_root,
        source_data_path=source_data_path,
        available_rows=len(raw_frame),
        required_rows=max_sample_size,
    )
    if storage_precheck["additional_rows_needed"] > 0 and not storage_precheck[
        "has_enough_free_space_for_estimated_pull"
    ]:
        raise RuntimeError(
            "Insufficient free disk for estimated 100k-row pull expansion: "
            f"required~{storage_precheck['estimated_additional_bytes_needed']} bytes, "
            f"free={storage_precheck['free_bytes']} bytes."
        )
    if args.auto_pull_missing and storage_precheck["additional_rows_needed"] > 0:
        ingestion_result = fetch_ohlcv_to_parquet(
            settings=settings,
            exchange_id=exchange,
            symbol=symbol,
            timeframe=timeframe,
            limit=max_sample_size,
        )
        source_data_path = ingestion_result.output_path
        raw_frame = _coerce_frame(source_data_path)
        storage_precheck = _estimate_storage_for_pull(
            quant_data_root=settings.quant_data_root,
            source_data_path=source_data_path,
            available_rows=len(raw_frame),
            required_rows=max_sample_size,
        )

    priority2_features_enabled = bool(settings.priority2_features_enabled) and not bool(
        args.disable_priority2_features
    )
    ranked_features_enabled = bool(settings.ranked_features_enabled) and not bool(
        args.disable_ranked_features
    )
    orderbook_features_enabled = bool(settings.orderbook_features_enabled) and not bool(
        args.disable_orderbook_features
    )

    labeled, feature_metadata = _prepare_labeled_dataset(
        settings=settings,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        source_data_path=source_data_path,
        horizon_bars=max(1, int(args.horizon_bars)),
        buy_threshold=float(args.buy_threshold),
        sell_threshold=abs(float(args.sell_threshold)),
        labeling_mode=str(args.labeling_mode),
        trade_quality_min_score=float(args.trade_quality_min_score),
        cost_bps=float(args.cost_bps),
        priority2_features_enabled=priority2_features_enabled,
        priority2_external_features_path=(
            Path(settings.priority2_external_features_path).expanduser().resolve()
            if settings.priority2_external_features_path
            else None
        ),
        priority2_feature_columns=normalize_priority2_feature_columns(
            tuple(settings.priority2_feature_columns)
            if settings.priority2_feature_columns
            else DEFAULT_STABLE_PRIORITY2_FEATURE_COLUMNS
        ),
        ranked_features_enabled=ranked_features_enabled,
        ranked_external_features_path=(
            Path(settings.ranked_external_features_path).expanduser().resolve()
            if settings.ranked_external_features_path
            else None
        ),
        ranked_feature_columns=normalize_ranked_feature_columns(
            tuple(settings.ranked_feature_columns)
            if settings.ranked_feature_columns
            else DEFAULT_STABLE_RANKED_FEATURE_COLUMNS
        ),
        orderbook_features_enabled=orderbook_features_enabled,
        orderbook_features_path=(
            Path(settings.orderbook_features_path).expanduser().resolve()
            if settings.orderbook_features_path
            else None
        ),
        orderbook_feature_columns=normalize_orderbook_feature_columns(
            tuple(settings.orderbook_feature_columns)
            if settings.orderbook_feature_columns
            else DEFAULT_STABLE_ORDERBOOK_FEATURE_COLUMNS
        ),
    )

    min_train_samples = max(20, int(args.min_train_samples))
    oot_test_fraction = float(np.clip(args.oot_test_fraction, 0.05, 0.50))
    walkforward_train_fraction = float(np.clip(args.walkforward_train_fraction, 0.20, 0.90))
    walkforward_validate_fraction = float(np.clip(args.walkforward_validate_fraction, 0.05, 0.40))
    walkforward_step_fraction = float(np.clip(args.walkforward_step_fraction, 0.05, 0.40))
    walkforward_min_windows = max(1, int(args.walkforward_min_windows))
    rolling_train_window_bars = max(0, int(args.rolling_train_window_bars))
    recency_half_life_bars = max(0, int(args.recency_half_life_bars))

    sample_size_results: list[dict[str, Any]] = []
    for sample_size in sample_sizes:
        if len(labeled) < sample_size:
            sample_size_results.append(
                {
                    "sample_size": int(sample_size),
                    "status": "skipped_insufficient_rows",
                    "reason": (
                        f"requested {sample_size} rows but labeled dataset has {len(labeled)} rows"
                    ),
                }
            )
            continue

        sample_frame = labeled.tail(sample_size).reset_index(drop=True)
        oot_rows = max(20, int(round(sample_size * oot_test_fraction)))
        if sample_size - oot_rows < min_train_samples:
            oot_rows = max(1, sample_size - min_train_samples)
        if oot_rows <= 0 or sample_size - oot_rows < min_train_samples:
            sample_size_results.append(
                {
                    "sample_size": int(sample_size),
                    "status": "skipped_invalid_split",
                    "reason": "unable to allocate OOT rows while satisfying min train samples",
                }
            )
            continue

        pretest_frame = sample_frame.iloc[:-oot_rows].reset_index(drop=True)
        oot_frame = sample_frame.iloc[-oot_rows:].reset_index(drop=True)
        if pretest_frame.empty or oot_frame.empty:
            sample_size_results.append(
                {
                    "sample_size": int(sample_size),
                    "status": "skipped_invalid_split",
                    "reason": "pretest or OOT segment is empty",
                }
            )
            continue

        train_bars = max(min_train_samples, int(round(len(pretest_frame) * walkforward_train_fraction)))
        validate_bars = max(20, int(round(len(pretest_frame) * walkforward_validate_fraction)))
        step_bars = max(20, int(round(len(pretest_frame) * walkforward_step_fraction)))
        if train_bars + validate_bars > len(pretest_frame):
            validate_bars = max(1, len(pretest_frame) - train_bars)
        if validate_bars <= 0:
            sample_size_results.append(
                {
                    "sample_size": int(sample_size),
                    "status": "skipped_invalid_walkforward",
                    "reason": "validation bars resolved to zero",
                }
            )
            continue

        windows = _build_walkforward_windows(
            len(pretest_frame),
            train_bars=train_bars,
            validate_bars=validate_bars,
            step_bars=step_bars,
        )
        if len(windows) < walkforward_min_windows:
            sample_size_results.append(
                {
                    "sample_size": int(sample_size),
                    "status": "skipped_insufficient_walkforward_windows",
                    "reason": (
                        f"constructed {len(windows)} windows, need at least {walkforward_min_windows}"
                    ),
                    "pretest_rows": int(len(pretest_frame)),
                    "oot_rows": int(len(oot_frame)),
                    "train_bars": int(train_bars),
                    "validate_bars": int(validate_bars),
                    "step_bars": int(step_bars),
                }
            )
            continue

        fold_rows: list[dict[str, Any]] = []
        for window_index, (_, train_end, validate_start, validate_end) in enumerate(windows, start=1):
            train_frame_raw = pretest_frame.iloc[:train_end].reset_index(drop=True)
            train_frame = _apply_rolling_window(train_frame_raw, rolling_train_window_bars)
            validate_frame = pretest_frame.iloc[validate_start:validate_end].reset_index(drop=True)
            if len(train_frame) < min_train_samples or validate_frame.empty:
                continue
            recency_weights = _build_recency_weights(len(train_frame), recency_half_life_bars)
            model_candidates = _fit_model_family_candidates(
                train_frame,
                sample_weights=recency_weights,
            )
            best_family = "gaussian_nb"
            best_threshold = float(np.clip(args.action_confidence_threshold, 0.0, 1.0))
            best_evaluation: dict[str, Any] | None = None
            best_rank: tuple[float, ...] | None = None
            for family, model_payload in model_candidates.items():
                selected_threshold, evaluation, _ = _select_action_confidence_frontier(
                    model_payload=model_payload,
                    test_frame=validate_frame,
                    symbol=symbol,
                    one_way_cost_bps=max(0.0, float(args.cost_bps)),
                    paper_notional_usd=max(0.0, float(settings.paper_trade_notional_usd)),
                    paper_starting_cash_usd=max(0.0, float(settings.paper_trade_starting_cash_usd)),
                    paper_fee_bps=max(0.0, float(settings.paper_trade_fee_bps)),
                    paper_slippage_bps=max(0.0, float(settings.paper_trade_slippage_bps)),
                    regime_hint=None,
                    minimum_threshold=float(np.clip(args.action_confidence_threshold, 0.0, 1.0)),
                    optimize_thresholds=True,
                )
                rank = _evaluation_selection_rank(evaluation)
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_family = str(family)
                    best_threshold = float(selected_threshold)
                    best_evaluation = evaluation
            if best_evaluation is None:
                continue
            eval_metrics = _extract_eval_metrics(best_evaluation)
            fold_rows.append(
                {
                    "window_index": int(window_index),
                    "train_start_utc": str(train_frame.iloc[0]["timestamp"]),
                    "train_end_utc": str(train_frame.iloc[-1]["timestamp"]),
                    "validate_start_utc": str(validate_frame.iloc[0]["timestamp"]),
                    "validate_end_utc": str(validate_frame.iloc[-1]["timestamp"]),
                    "train_rows": int(len(train_frame)),
                    "validate_rows": int(len(validate_frame)),
                    "selected_model_family": best_family,
                    "selected_action_confidence_threshold": float(best_threshold),
                    "validation_rank_primary": float(best_rank[0]) if best_rank else 0.0,
                    **eval_metrics,
                }
            )

        if len(fold_rows) < walkforward_min_windows:
            sample_size_results.append(
                {
                    "sample_size": int(sample_size),
                    "status": "failed_walkforward",
                    "reason": (
                        f"only {len(fold_rows)} successful windows after model fitting; "
                        f"need at least {walkforward_min_windows}"
                    ),
                    "pretest_rows": int(len(pretest_frame)),
                    "oot_rows": int(len(oot_frame)),
                }
            )
            continue

        selected_family, selected_threshold = _select_family_for_final_fit(fold_rows)
        final_train_frame_raw = pretest_frame.copy().reset_index(drop=True)
        final_train_frame = _apply_rolling_window(final_train_frame_raw, rolling_train_window_bars)
        final_recency_weights = _build_recency_weights(len(final_train_frame), recency_half_life_bars)
        final_model = _fit_family_model(
            family=selected_family,
            train_frame=final_train_frame,
            sample_weights=final_recency_weights,
        )
        oot_evaluation = _evaluate_model(
            model_payload=final_model,
            test_frame=oot_frame,
            symbol=symbol,
            one_way_cost_bps=max(0.0, float(args.cost_bps)),
            action_confidence_threshold=float(np.clip(selected_threshold, 0.0, 1.0)),
            paper_notional_usd=max(0.0, float(settings.paper_trade_notional_usd)),
            paper_starting_cash_usd=max(0.0, float(settings.paper_trade_starting_cash_usd)),
            paper_fee_bps=max(0.0, float(settings.paper_trade_fee_bps)),
            paper_slippage_bps=max(0.0, float(settings.paper_trade_slippage_bps)),
            regime_hint=None,
        )
        sample_size_results.append(
            {
                "sample_size": int(sample_size),
                "status": "ok",
                "pretest_rows": int(len(pretest_frame)),
                "oot_rows": int(len(oot_frame)),
                "train_bars": int(train_bars),
                "validate_bars": int(validate_bars),
                "step_bars": int(step_bars),
                "walkforward_window_count": int(len(fold_rows)),
                "selected_model_family": selected_family,
                "selected_action_confidence_threshold": float(selected_threshold),
                "walkforward_validation_summary": _aggregate_metric_rows(fold_rows),
                "walkforward_windows": fold_rows,
                "oot_metrics": _extract_eval_metrics(oot_evaluation),
            }
        )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            settings.quant_data_root
            / "curated"
            / "evaluations"
            / "trigger_walkforward_sample_sweep"
            / f"exchange={exchange}"
            / f"symbol={symbol_slug(symbol)}"
            / f"interval={timeframe}"
            / f"run_id={_run_id()}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    payload = {
        "contract": "trigger_walkforward_sample_sweep.v1",
        "created_at_utc": _utc_now_iso(),
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "source_data_path": str(source_data_path),
        "sample_sizes_requested": [int(value) for value in sample_sizes],
        "labeled_row_count": int(len(labeled)),
        "storage_precheck": storage_precheck,
        "protocol": {
            "oot_test_fraction": float(oot_test_fraction),
            "walkforward_train_fraction": float(walkforward_train_fraction),
            "walkforward_validate_fraction": float(walkforward_validate_fraction),
            "walkforward_step_fraction": float(walkforward_step_fraction),
            "walkforward_min_windows": int(walkforward_min_windows),
            "rolling_train_window_bars": int(rolling_train_window_bars),
            "recency_half_life_bars": int(recency_half_life_bars),
            "min_train_samples": int(min_train_samples),
            "tuning_policy": "model family + action confidence thresholds tuned on walk-forward validation only",
            "final_test_policy": "final OOT segment untouched until final evaluation",
        },
        "training_parameters": {
            "horizon_bars": int(max(1, int(args.horizon_bars))),
            "buy_threshold": float(max(0.0005, float(args.buy_threshold))),
            "sell_threshold": float(max(0.0005, abs(float(args.sell_threshold)))),
            "labeling_mode": str(args.labeling_mode),
            "trade_quality_min_score": float(np.clip(args.trade_quality_min_score, 0.0, 1.0)),
            "cost_bps": float(max(0.0, float(args.cost_bps))),
            "action_confidence_threshold_floor": float(
                np.clip(args.action_confidence_threshold, 0.0, 1.0)
            ),
            "priority2_features_enabled": bool(priority2_features_enabled),
            "ranked_features_enabled": bool(ranked_features_enabled),
            "orderbook_features_enabled": bool(orderbook_features_enabled),
        },
        "feature_resolution": feature_metadata,
        "sample_size_results": sample_size_results,
    }

    summary_json_path = output_dir / "trigger_walkforward_sample_sweep.json"
    summary_md_path = output_dir / "trigger_walkforward_sample_sweep.md"
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_md_path.write_text(_build_markdown_report(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "summary_json_path": str(summary_json_path),
                "summary_markdown_path": str(summary_md_path),
                "sample_sizes_requested": [int(value) for value in sample_sizes],
                "sample_sizes_succeeded": int(
                    sum(1 for row in sample_size_results if str(row.get("status")) == "ok")
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
