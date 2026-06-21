#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_agents.config import load_settings
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
from quant_agents.storage import symbol_slug
from quant_agents.trigger_model import train_trigger_model

REQUIRED_MARKET_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")
DEFAULT_CONFIG_PATH = Path(__file__).with_name("ranked_feature_ablation_plan.json")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not pd.notna(parsed):
        return float(default)
    return float(parsed)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_timestamp(raw: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(raw)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _collect_scope_parquet_files(
    *,
    quant_data_root: Path,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> list[Path]:
    base = (
        quant_data_root
        / "raw"
        / f"exchange={exchange}"
        / f"symbol={symbol_slug(symbol)}"
        / f"interval={timeframe}"
    )
    if not base.exists():
        raise FileNotFoundError(f"Raw dataset scope not found: {base}")
    files = sorted(base.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in scope: {base}")
    return files


def _load_market_frame(source_files: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in source_files:
        frame = pd.read_parquet(path, columns=list(REQUIRED_MARKET_COLUMNS))
        frame = frame.loc[:, list(REQUIRED_MARKET_COLUMNS)].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"])
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No usable market rows found in raw dataset scope.")
    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return merged.reset_index(drop=True)


def _load_plan_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Ablation config not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Ablation config must be a JSON object.")
    if str(payload.get("contract", "")).strip() != "trigger_ranked_ablation_plan.v1":
        raise ValueError("Ablation config contract must be trigger_ranked_ablation_plan.v1.")
    scenarios = payload.get("ablation_scenarios")
    splits = payload.get("validation_splits")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Ablation config requires a non-empty ablation_scenarios array.")
    if not isinstance(splits, list) or not splits:
        raise ValueError("Ablation config requires a non-empty validation_splits array.")
    training_defaults = payload.get("training_defaults")
    if not isinstance(training_defaults, dict):
        raise ValueError("Ablation config requires a training_defaults object.")
    return payload


def _slice_window(
    frame: pd.DataFrame,
    *,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> pd.DataFrame:
    scoped = frame.loc[(frame["timestamp"] >= start_utc) & (frame["timestamp"] < end_utc)].copy()
    scoped = scoped.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return scoped.reset_index(drop=True)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _path_or_none(raw: Any) -> Path | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ranked-feature trigger-model ablation matrix across configured validation splits."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to ablation-plan JSON file.",
    )
    parser.add_argument(
        "--exchange",
        default=None,
        help="Optional override for exchange scope.",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Optional override for symbol scope.",
    )
    parser.add_argument(
        "--timeframe",
        default=None,
        help="Optional override for timeframe scope.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional run tag appended to the output directory name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    config_path = Path(args.config).expanduser().resolve()
    config = _load_plan_config(config_path)

    scope = config.get("scope", {}) if isinstance(config.get("scope"), dict) else {}
    exchange = str(args.exchange or scope.get("exchange") or settings.default_exchange)
    symbol = str(args.symbol or scope.get("symbol") or settings.default_symbol)
    timeframe = str(args.timeframe or scope.get("timeframe") or settings.default_timeframe)
    training_defaults = dict(config.get("training_defaults", {}))

    source_files = _collect_scope_parquet_files(
        quant_data_root=settings.quant_data_root,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
    )
    market_frame = _load_market_frame(source_files)
    source_dataset_sha256 = hashlib.sha256(
        json.dumps(
            {
                "files": [
                    {
                        "path": str(path),
                        "sha256": _sha256_file(path),
                    }
                    for path in source_files
                ],
                "row_count": int(len(market_frame)),
                "min_timestamp_utc": pd.Timestamp(market_frame["timestamp"].iloc[0]).isoformat(),
                "max_timestamp_utc": pd.Timestamp(market_frame["timestamp"].iloc[-1]).isoformat(),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    now = datetime.now(timezone.utc)
    timestamp_id = f"{now:%Y%m%dT%H%M%SZ}"
    tag = f"_{str(args.tag).strip()}" if args.tag else ""
    run_dir = settings.quant_data_root / "logs" / "analysis" / f"ranked_feature_ablation_{timestamp_id}{tag}"
    split_dir = run_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    split_rows: list[dict[str, Any]] = []
    split_paths: dict[str, Path] = {}
    for split in config.get("validation_splits", []):
        if not isinstance(split, dict):
            continue
        split_name = str(split.get("name", "")).strip()
        if not split_name:
            raise ValueError("validation_splits entries must include a non-empty name.")
        start = _parse_timestamp(split.get("start_utc"))
        end = _parse_timestamp(split.get("end_utc"))
        if not start < end:
            raise ValueError(f"Split `{split_name}` has invalid time range.")
        scoped = _slice_window(market_frame, start_utc=start, end_utc=end)
        minimum_rows = max(20, int(split.get("minimum_rows", 1200)))
        expected_bars = int(split.get("expected_bars", 0)) if split.get("expected_bars") is not None else None
        split_record: dict[str, Any] = {
            "name": split_name,
            "regime": str(split.get("regime", "unknown")),
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "row_count": int(len(scoped)),
            "minimum_rows": int(minimum_rows),
            "meets_minimum_rows": bool(len(scoped) >= minimum_rows),
            "expected_bars": expected_bars,
            "coverage_vs_expected": (
                float(len(scoped) / expected_bars)
                if expected_bars is not None and expected_bars > 0
                else None
            ),
            "dataset_path": None,
        }
        if len(scoped) >= minimum_rows:
            dataset_path = split_dir / f"{split_name}.parquet"
            scoped.to_parquet(dataset_path, index=False)
            split_paths[split_name] = dataset_path
            split_record["dataset_path"] = str(dataset_path)
        split_rows.append(split_record)

    scenario_results: list[dict[str, Any]] = []
    for scenario in config.get("ablation_scenarios", []):
        if not isinstance(scenario, dict):
            continue
        scenario_name = str(scenario.get("name", "")).strip()
        if not scenario_name:
            raise ValueError("ablation_scenarios entries must include a non-empty name.")
        priority2_columns = normalize_priority2_feature_columns(
            scenario.get("priority2_feature_columns") or DEFAULT_STABLE_PRIORITY2_FEATURE_COLUMNS
        )
        ranked_columns = normalize_ranked_feature_columns(
            scenario.get("ranked_feature_columns") or DEFAULT_STABLE_RANKED_FEATURE_COLUMNS
        )
        orderbook_columns = normalize_orderbook_feature_columns(
            scenario.get("orderbook_feature_columns") or DEFAULT_STABLE_ORDERBOOK_FEATURE_COLUMNS
        )
        for split in split_rows:
            split_name = str(split["name"])
            dataset_path = split_paths.get(split_name)
            if dataset_path is None:
                scenario_results.append(
                    {
                        "scenario": scenario_name,
                        "split": split_name,
                        "regime": split.get("regime"),
                        "status": "skipped_insufficient_split_rows",
                        "error": None,
                    }
                )
                continue
            try:
                result = train_trigger_model(
                    settings=settings,
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    input_file=dataset_path,
                    horizon_bars=max(1, int(training_defaults.get("horizon_bars", 2))),
                    buy_threshold=float(training_defaults.get("buy_threshold", 0.005)),
                    sell_threshold=float(training_defaults.get("sell_threshold", 0.005)),
                    min_train_samples=max(20, int(training_defaults.get("min_train_samples", 300))),
                    cost_bps=max(0.0, float(training_defaults.get("cost_bps", 9.0))),
                    optimize_thresholds=bool(training_defaults.get("optimize_thresholds", True)),
                    labeling_mode=str(training_defaults.get("labeling_mode", "triple_barrier_v2")),
                    trade_quality_min_score=float(
                        training_defaults.get("trade_quality_min_score", 0.55)
                    ),
                    action_confidence_threshold=float(
                        training_defaults.get("action_confidence_threshold", 0.6)
                    ),
                    priority2_features_enabled=bool(scenario.get("priority2_features_enabled", False)),
                    priority2_external_features_path=_path_or_none(
                        scenario.get("priority2_external_features_path")
                    ),
                    priority2_feature_columns=priority2_columns,
                    ranked_features_enabled=bool(scenario.get("ranked_features_enabled", False)),
                    ranked_external_features_path=_path_or_none(
                        scenario.get("ranked_external_features_path")
                    ),
                    ranked_feature_columns=ranked_columns,
                    orderbook_features_enabled=bool(scenario.get("orderbook_features_enabled", False)),
                    orderbook_features_path=_path_or_none(scenario.get("orderbook_features_path")),
                    orderbook_feature_columns=orderbook_columns,
                )
                model_payload = json.loads(result.model_path.read_text(encoding="utf-8"))
                training_metrics = (
                    model_payload.get("training_metrics", {})
                    if isinstance(model_payload.get("training_metrics", {}), dict)
                    else {}
                )
                expectancy = (
                    training_metrics.get("expectancy_metrics", {})
                    if isinstance(training_metrics.get("expectancy_metrics", {}), dict)
                    else {}
                )
                execution = (
                    training_metrics.get("execution_backtest_metrics", {})
                    if isinstance(training_metrics.get("execution_backtest_metrics", {}), dict)
                    else {}
                )
                scenario_results.append(
                    {
                        "scenario": scenario_name,
                        "split": split_name,
                        "regime": split.get("regime"),
                        "status": "ok",
                        "error": None,
                        "dataset_path": str(dataset_path),
                        "model_path": str(result.model_path),
                        "run_dir": str(result.run_dir),
                        "sample_count": int(result.sample_count),
                        "train_count": int(result.train_count),
                        "test_count": int(result.test_count),
                        "accuracy": float(result.accuracy),
                        "selected_buy_threshold": float(result.selected_buy_threshold),
                        "selected_sell_threshold": float(result.selected_sell_threshold),
                        "selected_trade_quality_threshold": float(
                            result.selected_trade_quality_threshold
                        ),
                        "selected_action_confidence_threshold": float(
                            result.selected_action_confidence_threshold
                        ),
                        "net_expectancy_per_actionable": float(
                            expectancy.get("net_expectancy_per_actionable", result.net_expectancy_per_actionable)
                        ),
                        "execution_backtest_equity_return": float(
                            execution.get("equity_return", result.execution_backtest_equity_return)
                        ),
                        "execution_backtest_realized_pnl_delta_usd": float(
                            execution.get(
                                "realized_pnl_delta_usd",
                                result.execution_backtest_realized_pnl_delta_usd,
                            )
                        ),
                        "execution_backtest_fill_rate": _safe_float(execution.get("fill_rate"), 0.0),
                        "execution_backtest_rejection_rate": _safe_float(
                            execution.get("rejection_rate"),
                            0.0,
                        ),
                        "priority2_quality_score": _safe_float(
                            dict(model_payload.get("priority2_diagnostics", {})).get("quality_score"),
                            0.0,
                        ),
                        "ranked_quality_score": _safe_float(
                            dict(model_payload.get("ranked_diagnostics", {})).get("quality_score"),
                            0.0,
                        ),
                        "orderbook_quality_score": _safe_float(
                            dict(model_payload.get("orderbook_diagnostics", {})).get("quality_score"),
                            0.0,
                        ),
                        "priority2_reason_codes": list(model_payload.get("priority2_reason_codes", [])),
                        "ranked_reason_codes": list(model_payload.get("ranked_reason_codes", [])),
                        "orderbook_reason_codes": list(model_payload.get("orderbook_reason_codes", [])),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                scenario_results.append(
                    {
                        "scenario": scenario_name,
                        "split": split_name,
                        "regime": split.get("regime"),
                        "status": "failed",
                        "error": str(exc),
                    }
                )

    scenario_summary: list[dict[str, Any]] = []
    scenario_names = sorted({str(row.get("scenario")) for row in scenario_results})
    for scenario_name in scenario_names:
        rows = [
            row
            for row in scenario_results
            if str(row.get("scenario")) == scenario_name and str(row.get("status")) == "ok"
        ]
        scenario_summary.append(
            {
                "scenario": scenario_name,
                "successful_splits": int(len(rows)),
                "total_splits": int(len(split_rows)),
                "mean_accuracy": _mean_or_none([float(row["accuracy"]) for row in rows]),
                "mean_net_expectancy_per_actionable": _mean_or_none(
                    [float(row["net_expectancy_per_actionable"]) for row in rows]
                ),
                "mean_execution_backtest_equity_return": _mean_or_none(
                    [float(row["execution_backtest_equity_return"]) for row in rows]
                ),
                "mean_execution_backtest_realized_pnl_delta_usd": _mean_or_none(
                    [float(row["execution_backtest_realized_pnl_delta_usd"]) for row in rows]
                ),
                "mean_ranked_quality_score": _mean_or_none(
                    [float(row["ranked_quality_score"]) for row in rows]
                ),
            }
        )

    baseline_name = str(config.get("ablation_scenarios", [{}])[0].get("name", "baseline_ohlcv_only"))
    baseline_summary = next(
        (row for row in scenario_summary if str(row.get("scenario")) == baseline_name),
        None,
    )
    baseline_deltas: list[dict[str, Any]] = []
    if baseline_summary is not None:
        for row in scenario_summary:
            baseline_deltas.append(
                {
                    "scenario": row["scenario"],
                    "vs_baseline_execution_backtest_realized_pnl_delta_usd": (
                        (
                            float(row["mean_execution_backtest_realized_pnl_delta_usd"])
                            - float(baseline_summary["mean_execution_backtest_realized_pnl_delta_usd"])
                        )
                        if row["mean_execution_backtest_realized_pnl_delta_usd"] is not None
                        and baseline_summary["mean_execution_backtest_realized_pnl_delta_usd"] is not None
                        else None
                    ),
                    "vs_baseline_execution_backtest_equity_return": (
                        (
                            float(row["mean_execution_backtest_equity_return"])
                            - float(baseline_summary["mean_execution_backtest_equity_return"])
                        )
                        if row["mean_execution_backtest_equity_return"] is not None
                        and baseline_summary["mean_execution_backtest_equity_return"] is not None
                        else None
                    ),
                    "vs_baseline_net_expectancy_per_actionable": (
                        (
                            float(row["mean_net_expectancy_per_actionable"])
                            - float(baseline_summary["mean_net_expectancy_per_actionable"])
                        )
                        if row["mean_net_expectancy_per_actionable"] is not None
                        and baseline_summary["mean_net_expectancy_per_actionable"] is not None
                        else None
                    ),
                    "vs_baseline_accuracy": (
                        float(row["mean_accuracy"]) - float(baseline_summary["mean_accuracy"])
                        if row["mean_accuracy"] is not None
                        and baseline_summary["mean_accuracy"] is not None
                        else None
                    ),
                }
            )

    ranking = sorted(
        scenario_summary,
        key=lambda item: (
            float(item["mean_execution_backtest_realized_pnl_delta_usd"])
            if item["mean_execution_backtest_realized_pnl_delta_usd"] is not None
            else float("-inf"),
            float(item["mean_execution_backtest_equity_return"])
            if item["mean_execution_backtest_equity_return"] is not None
            else float("-inf"),
            float(item["mean_net_expectancy_per_actionable"])
            if item["mean_net_expectancy_per_actionable"] is not None
            else float("-inf"),
            float(item["mean_accuracy"]) if item["mean_accuracy"] is not None else float("-inf"),
        ),
        reverse=True,
    )

    summary_payload = {
        "contract": "trigger_ranked_ablation_results.v1",
        "created_at_utc": now.isoformat(),
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "run_dir": str(run_dir),
        "scope": {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
        },
        "source_dataset": {
            "source_file_count": int(len(source_files)),
            "source_files": [str(path) for path in source_files],
            "source_dataset_sha256": source_dataset_sha256,
            "row_count": int(len(market_frame)),
            "min_timestamp_utc": pd.Timestamp(market_frame["timestamp"].iloc[0]).isoformat(),
            "max_timestamp_utc": pd.Timestamp(market_frame["timestamp"].iloc[-1]).isoformat(),
        },
        "split_coverage": split_rows,
        "scenario_runs": scenario_results,
        "scenario_summary": scenario_summary,
        "baseline_name": baseline_name,
        "baseline_deltas": baseline_deltas,
        "ranked_scenarios_by_priority_metrics": ranking,
    }

    summary_json_path = run_dir / "ranked_feature_ablation_results.json"
    summary_markdown_path = run_dir / "ranked_feature_ablation_results.md"
    _write_json(summary_json_path, summary_payload)

    md_lines: list[str] = [
        "# Ranked Feature Ablation Results",
        f"- run_dir: `{run_dir}`",
        f"- config: `{config_path}`",
        f"- scope: `{exchange} {symbol} {timeframe}`",
        f"- source_dataset_sha256: `{source_dataset_sha256}`",
        "",
        "## Split coverage",
    ]
    for split in split_rows:
        md_lines.append(
            "- "
            + f"`{split['name']}` ({split['regime']}) "
            + f"rows={split['row_count']} minimum={split['minimum_rows']} "
            + f"covered={split['meets_minimum_rows']}"
        )
    md_lines.extend(["", "## Scenario summary"])
    for row in ranking:
        md_lines.append(f"### {row['scenario']}")
        md_lines.append(f"- successful_splits: `{row['successful_splits']}/{row['total_splits']}`")
        md_lines.append(f"- mean_execution_backtest_realized_pnl_delta_usd: `{row['mean_execution_backtest_realized_pnl_delta_usd']}`")
        md_lines.append(f"- mean_execution_backtest_equity_return: `{row['mean_execution_backtest_equity_return']}`")
        md_lines.append(f"- mean_net_expectancy_per_actionable: `{row['mean_net_expectancy_per_actionable']}`")
        md_lines.append(f"- mean_accuracy: `{row['mean_accuracy']}`")
    md_lines.extend(["", "## Baseline deltas"])
    for row in baseline_deltas:
        md_lines.append(f"### {row['scenario']}")
        md_lines.append(
            "- "
            + f"realized_pnl_delta_usd_vs_baseline: "
            + f"`{row['vs_baseline_execution_backtest_realized_pnl_delta_usd']}`"
        )
        md_lines.append(
            "- "
            + f"equity_return_vs_baseline: "
            + f"`{row['vs_baseline_execution_backtest_equity_return']}`"
        )
        md_lines.append(
            "- "
            + f"net_expectancy_vs_baseline: "
            + f"`{row['vs_baseline_net_expectancy_per_actionable']}`"
        )
        md_lines.append("- " + f"accuracy_vs_baseline: `{row['vs_baseline_accuracy']}`")
    summary_markdown_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    top_scenario = ranking[0]["scenario"] if ranking else "none"
    print(f"ablation_results_json={summary_json_path}")
    print(f"ablation_results_markdown={summary_markdown_path}")
    print(f"top_scenario={top_scenario}")


if __name__ == "__main__":
    main()
