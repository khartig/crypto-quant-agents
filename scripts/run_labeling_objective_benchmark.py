#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_agents.config import ensure_data_root_ready, load_settings
from quant_agents.trigger_model import train_trigger_model
from quant_agents.storage import symbol_slug


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _optional_path(value: str | None) -> Path | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return Path(normalized).expanduser().resolve()


def _default_scenarios(default_trade_quality_threshold: float) -> list[dict[str, Any]]:
    baseline_quality = float(max(0.0, min(1.0, default_trade_quality_threshold)))
    strict_quality = float(min(0.90, max(baseline_quality + 0.15, 0.70)))
    scenarios = [
        {
            "name": "directional_no_quality_gate",
            "labeling_mode": "directional_v1",
            "trade_quality_min_score": 0.0,
        },
        {
            "name": "directional_quality_gate",
            "labeling_mode": "directional_v1",
            "trade_quality_min_score": baseline_quality,
        },
        {
            "name": "triple_barrier_quality_gate",
            "labeling_mode": "triple_barrier_v2",
            "trade_quality_min_score": baseline_quality,
        },
        {
            "name": "triple_barrier_strict_quality_gate",
            "labeling_mode": "triple_barrier_v2",
            "trade_quality_min_score": strict_quality,
        },
    ]
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for scenario in scenarios:
        key = (str(scenario["labeling_mode"]), round(float(scenario["trade_quality_min_score"]), 6))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(scenario)
    return deduped


def _extract_scenario_metrics(
    *,
    scenario: dict[str, Any],
    model_payload: dict[str, Any],
    model_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    training_metrics = dict(model_payload.get("training_metrics", {}))
    actionable = dict(training_metrics.get("actionable_metrics", {}))
    expectancy = dict(training_metrics.get("expectancy_metrics", {}))
    execution = dict(training_metrics.get("execution_backtest_metrics", {}))
    quality_stats = dict(training_metrics.get("trade_quality_stats", {}))
    frontier = (
        dict(model_payload.get("threshold_optimization", {}))
        .get("action_confidence_frontier", {})
    )
    frontier_rows = list(frontier.get("rows", [])) if isinstance(frontier, dict) else []
    threshold_selection = dict(model_payload.get("threshold_optimization", {})).get("selected", {})
    return {
        "name": scenario["name"],
        "labeling_mode": scenario["labeling_mode"],
        "trade_quality_min_score": float(scenario["trade_quality_min_score"]),
        "selected_buy_threshold": float(model_payload.get("buy_threshold", 0.0)),
        "selected_sell_threshold": float(model_payload.get("sell_threshold", 0.0)),
        "selected_action_confidence_threshold": float(
            threshold_selection.get(
                "action_confidence_threshold",
                model_payload.get("selected_action_confidence_threshold", 0.0),
            )
        ),
        "accuracy": float(training_metrics.get("accuracy", 0.0)),
        "actionable_rate": float(actionable.get("actionable_rate", 0.0)),
        "binary_actionable_precision": float(actionable.get("binary_actionable_precision", 0.0)),
        "binary_actionable_recall": float(actionable.get("binary_actionable_recall", 0.0)),
        "net_expectancy_per_bar": float(expectancy.get("net_expectancy_per_bar", 0.0)),
        "net_expectancy_per_actionable": float(expectancy.get("net_expectancy_per_actionable", 0.0)),
        "execution_equity_return": float(execution.get("equity_return", 0.0)),
        "execution_realized_pnl_delta_usd": float(execution.get("realized_pnl_delta_usd", 0.0)),
        "execution_fill_rate": float(execution.get("fill_rate", 0.0)),
        "trade_quality_pass_rate": float(quality_stats.get("trade_quality_pass_rate", 0.0)),
        "meta_label_positive_rate": float(quality_stats.get("meta_label_positive_rate", 0.0)),
        "frontier_rows": frontier_rows,
        "artifacts": {
            "model_path": str(model_path),
            "run_dir": str(run_dir),
        },
    }


def _plot_coverage_precision_frontier(
    *,
    scenario_metrics: list[dict[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(11, 6.5))
    plotted = 0
    for scenario in scenario_metrics:
        rows = list(scenario.get("frontier_rows", []))
        if not rows:
            continue
        ordered = sorted(
            rows,
            key=lambda item: float(item.get("actionable_rate", 0.0)),
        )
        x_vals = [float(item.get("actionable_rate", 0.0)) for item in ordered]
        y_vals = [float(item.get("binary_actionable_precision", 0.0)) for item in ordered]
        if not x_vals:
            continue
        label = (
            f"{scenario['name']} | {scenario['labeling_mode']} | "
            f"q>={float(scenario['trade_quality_min_score']):.2f}"
        )
        plt.plot(x_vals, y_vals, marker="o", linewidth=1.6, label=label)
        selected_threshold = float(scenario.get("selected_action_confidence_threshold", 0.0))
        selected_row = min(
            ordered,
            key=lambda item: abs(float(item.get("threshold", 0.0)) - selected_threshold),
        )
        plt.scatter(
            [float(selected_row.get("actionable_rate", 0.0))],
            [float(selected_row.get("binary_actionable_precision", 0.0))],
            marker="*",
            s=220,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )
        plotted += 1

    plt.title("Actionable Coverage vs Precision Frontier")
    plt.xlabel("Actionable coverage rate")
    plt.ylabel("Actionable precision")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.30, linestyle="--")
    if plotted > 0:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=170)
    plt.close()


def _build_markdown_summary(
    *,
    payload: dict[str, Any],
) -> str:
    metrics = list(payload.get("scenario_metrics", []))
    failures = list(payload.get("failed_scenarios", []))
    lines = [
        "# Trigger labeling/objective benchmark comparison",
        f"- Created at: `{payload.get('created_at_utc')}`",
        f"- Exchange: `{payload.get('exchange')}`",
        f"- Symbol: `{payload.get('symbol')}`",
        f"- Timeframe: `{payload.get('timeframe')}`",
        f"- Scenario count (successful): `{len(metrics)}`",
        f"- Scenario count (failed): `{len(failures)}`",
        "## Best scenario by actionable precision",
    ]

    if metrics:
        best_by_precision = max(metrics, key=lambda item: float(item.get("binary_actionable_precision", 0.0)))
        lines.extend(
            [
                f"- Name: `{best_by_precision.get('name')}`",
                f"- Labeling mode: `{best_by_precision.get('labeling_mode')}`",
                f"- Trade quality min score: `{float(best_by_precision.get('trade_quality_min_score', 0.0)):.4f}`",
                f"- Actionable precision: `{float(best_by_precision.get('binary_actionable_precision', 0.0)):.6f}`",
                f"- Actionable coverage: `{float(best_by_precision.get('actionable_rate', 0.0)):.6f}`",
            ]
        )
    else:
        lines.append("- No successful scenarios.")

    lines.append("## Scenario summaries")
    if not metrics:
        lines.append("- None")
    for scenario in metrics:
        lines.extend(
            [
                f"- `{scenario.get('name')}`",
                f"  - labeling_mode: `{scenario.get('labeling_mode')}`",
                f"  - trade_quality_min_score: `{float(scenario.get('trade_quality_min_score', 0.0)):.4f}`",
                f"  - accuracy: `{float(scenario.get('accuracy', 0.0)):.6f}`",
                f"  - actionable_rate: `{float(scenario.get('actionable_rate', 0.0)):.6f}`",
                f"  - binary_actionable_precision: `{float(scenario.get('binary_actionable_precision', 0.0)):.6f}`",
                f"  - binary_actionable_recall: `{float(scenario.get('binary_actionable_recall', 0.0)):.6f}`",
                f"  - net_expectancy_per_actionable: `{float(scenario.get('net_expectancy_per_actionable', 0.0)):.6f}`",
                f"  - execution_realized_pnl_delta_usd: `{float(scenario.get('execution_realized_pnl_delta_usd', 0.0)):.6f}`",
            ]
        )

    lines.append("## Failed scenarios")
    if failures:
        for failure in failures:
            lines.append(f"- {failure}")
    else:
        lines.append("- None")

    artifacts = dict(payload.get("artifacts", {}))
    lines.extend(
        [
            "## Artifacts",
            f"- Benchmark JSON: `{artifacts.get('benchmark_json_path')}`",
            f"- Benchmark Markdown: `{artifacts.get('benchmark_markdown_path')}`",
            f"- Coverage/precision frontier plot: `{artifacts.get('coverage_precision_frontier_plot_path')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark trigger labeling/objective variants and generate actionable coverage/precision frontier artifacts."
        )
    )
    parser.add_argument("--exchange", default=settings.default_exchange)
    parser.add_argument("--symbol", default=settings.default_symbol)
    parser.add_argument("--timeframe", default=settings.default_timeframe)
    parser.add_argument("--input-file", type=Path, default=None)
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
        "--scenario-set",
        choices=("default", "extended"),
        default="default",
        help="default=4 scenarios, extended=adds triple_barrier with no quality gate",
    )
    parser.add_argument(
        "--priority2-external-features-path",
        type=Path,
        default=_optional_path(settings.priority2_external_features_path),
    )
    parser.add_argument(
        "--ranked-external-features-path",
        type=Path,
        default=_optional_path(settings.ranked_external_features_path),
    )
    parser.add_argument(
        "--orderbook-features-path",
        type=Path,
        default=_optional_path(settings.orderbook_features_path),
    )
    parser.add_argument(
        "--disable-priority2-features",
        action="store_true",
        help="Disable Priority2 feature usage during benchmark runs.",
    )
    parser.add_argument(
        "--disable-ranked-features",
        action="store_true",
        help="Disable ranked feature usage during benchmark runs.",
    )
    parser.add_argument(
        "--disable-orderbook-features",
        action="store_true",
        help="Disable orderbook feature usage during benchmark runs.",
    )
    parser.add_argument(
        "--disable-threshold-optimization",
        action="store_true",
        help="Disable threshold optimization and use provided confidence threshold directly.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory for benchmark artifacts.",
    )
    args = parser.parse_args()

    ensure_data_root_ready(
        settings.quant_data_root,
        allow_unmounted=settings.allow_unmounted_data_root,
    )

    scenario_definitions = _default_scenarios(float(settings.trigger_model_trade_quality_min_score))
    if args.scenario_set == "extended":
        scenario_definitions.insert(
            2,
            {
                "name": "triple_barrier_no_quality_gate",
                "labeling_mode": "triple_barrier_v2",
                "trade_quality_min_score": 0.0,
            },
        )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            settings.quant_data_root
            / "curated"
            / "evaluations"
            / "trigger_labeling_objective_benchmark"
            / f"exchange={args.exchange}"
            / f"symbol={symbol_slug(str(args.symbol))}"
            / f"interval={args.timeframe}"
            / f"run_id={_run_id()}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    scenario_metrics: list[dict[str, Any]] = []
    failed_scenarios: list[str] = []
    for scenario in scenario_definitions:
        try:
            result = train_trigger_model(
                settings=settings,
                exchange=str(args.exchange),
                symbol=str(args.symbol),
                timeframe=str(args.timeframe),
                input_file=args.input_file.expanduser().resolve() if args.input_file is not None else None,
                horizon_bars=max(1, int(args.horizon_bars)),
                buy_threshold=float(args.buy_threshold),
                sell_threshold=abs(float(args.sell_threshold)),
                min_train_samples=max(20, int(args.min_train_samples)),
                cost_bps=max(0.0, float(args.cost_bps)),
                optimize_thresholds=not bool(args.disable_threshold_optimization),
                labeling_mode=str(scenario["labeling_mode"]),
                trade_quality_min_score=float(scenario["trade_quality_min_score"]),
                action_confidence_threshold=float(args.action_confidence_threshold),
                priority2_features_enabled=(
                    bool(settings.priority2_features_enabled) and not bool(args.disable_priority2_features)
                ),
                priority2_external_features_path=(
                    args.priority2_external_features_path.expanduser().resolve()
                    if args.priority2_external_features_path is not None
                    else None
                ),
                priority2_feature_columns=settings.priority2_feature_columns,
                ranked_features_enabled=(
                    bool(settings.ranked_features_enabled) and not bool(args.disable_ranked_features)
                ),
                ranked_external_features_path=(
                    args.ranked_external_features_path.expanduser().resolve()
                    if args.ranked_external_features_path is not None
                    else None
                ),
                ranked_feature_columns=settings.ranked_feature_columns,
                orderbook_features_enabled=(
                    bool(settings.orderbook_features_enabled) and not bool(args.disable_orderbook_features)
                ),
                orderbook_features_path=(
                    args.orderbook_features_path.expanduser().resolve()
                    if args.orderbook_features_path is not None
                    else None
                ),
                orderbook_feature_columns=settings.orderbook_feature_columns,
            )
            model_payload = json.loads(result.model_path.read_text(encoding="utf-8"))
            scenario_metrics.append(
                _extract_scenario_metrics(
                    scenario=scenario,
                    model_payload=model_payload,
                    model_path=result.model_path,
                    run_dir=result.run_dir,
                )
            )
        except Exception as exc:  # noqa: BLE001
            failed_scenarios.append(f"{scenario['name']}: {exc}")

    if not scenario_metrics:
        raise RuntimeError(
            "All labeling benchmark scenarios failed. "
            f"Failed scenarios: {failed_scenarios}"
        )

    baseline = scenario_metrics[0]
    delta_keys = (
        "accuracy",
        "actionable_rate",
        "binary_actionable_precision",
        "binary_actionable_recall",
        "net_expectancy_per_actionable",
        "execution_realized_pnl_delta_usd",
    )
    for row in scenario_metrics:
        row["delta_vs_baseline"] = {
            key: float(row.get(key, 0.0)) - float(baseline.get(key, 0.0))
            for key in delta_keys
        }

    frontier_plot_path = output_dir / "actionable_coverage_vs_precision_frontier.png"
    _plot_coverage_precision_frontier(
        scenario_metrics=scenario_metrics,
        output_path=frontier_plot_path,
    )

    payload = {
        "contract": "trigger_labeling_objective_benchmark.v1",
        "created_at_utc": _utc_now_iso(),
        "exchange": str(args.exchange),
        "symbol": str(args.symbol),
        "timeframe": str(args.timeframe),
        "input_file": str(args.input_file.expanduser().resolve()) if args.input_file is not None else None,
        "scenario_set": str(args.scenario_set),
        "scenario_count_requested": len(scenario_definitions),
        "scenario_count_successful": len(scenario_metrics),
        "scenario_count_failed": len(failed_scenarios),
        "baseline_scenario": {
            "name": baseline.get("name"),
            "labeling_mode": baseline.get("labeling_mode"),
            "trade_quality_min_score": baseline.get("trade_quality_min_score"),
        },
        "scenario_metrics": scenario_metrics,
        "failed_scenarios": failed_scenarios,
        "artifacts": {
            "output_dir": str(output_dir),
            "coverage_precision_frontier_plot_path": str(frontier_plot_path),
        },
    }

    benchmark_json_path = output_dir / "trigger_labeling_objective_benchmark.json"
    benchmark_md_path = output_dir / "trigger_labeling_objective_benchmark.md"
    payload["artifacts"]["benchmark_json_path"] = str(benchmark_json_path)
    payload["artifacts"]["benchmark_markdown_path"] = str(benchmark_md_path)
    benchmark_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    benchmark_md_path.write_text(_build_markdown_summary(payload=payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(output_dir),
                "benchmark_json_path": str(benchmark_json_path),
                "benchmark_markdown_path": str(benchmark_md_path),
                "coverage_precision_frontier_plot_path": str(frontier_plot_path),
                "scenario_count_successful": len(scenario_metrics),
                "scenario_count_failed": len(failed_scenarios),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
