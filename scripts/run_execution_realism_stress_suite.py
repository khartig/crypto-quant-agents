#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_agents.config import ensure_data_root_ready, load_settings
from quant_agents.paper_trading import simulate_paper_trade_execution_step
from quant_agents.storage import symbol_slug


@dataclass(frozen=True)
class ExecutionScenario:
    name: str
    spread_bps: float
    latency_ms: float
    latency_slippage_bps_per_second: float
    liquidity_score: float
    market_depth_notional_usd: float
    notional_impact_coeff: float


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_source_data_path(
    *,
    quant_data_root: Path,
    exchange: str,
    symbol: str,
    timeframe: str,
    explicit_path: Path | None,
) -> Path:
    if explicit_path is not None:
        candidate = explicit_path.expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Input file does not exist: {candidate}")
        return candidate
    base = (
        quant_data_root
        / "raw"
        / f"exchange={exchange}"
        / f"symbol={symbol_slug(symbol)}"
        / f"interval={timeframe}"
    )
    if not base.exists():
        raise FileNotFoundError(f"No raw data scope found for execution stress suite: {base}")
    candidates = sorted(base.rglob("*.parquet"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No parquet data found for execution stress suite under: {base}")
    return candidates[-1].resolve()


def _load_mark_price(source_data_path: Path) -> tuple[float, str]:
    frame = pd.read_parquet(source_data_path, columns=["timestamp", "close"])
    closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
    if closes.empty:
        raise RuntimeError(f"Unable to derive mark price from close column: {source_data_path}")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
    latest_timestamp = (
        pd.Timestamp(timestamps.iloc[-1]).isoformat()
        if not timestamps.empty
        else datetime.now(timezone.utc).isoformat()
    )
    return float(closes.iloc[-1]), latest_timestamp


def _build_scenarios(
    *,
    scenario_set: str,
    spread_bps: float,
    latency_ms: float,
    latency_slippage_bps_per_second: float,
    liquidity_score: float,
    market_depth_notional_usd: float,
    notional_impact_coeff: float,
) -> list[ExecutionScenario]:
    baseline = ExecutionScenario(
        name="baseline",
        spread_bps=max(0.0, spread_bps),
        latency_ms=max(0.0, latency_ms),
        latency_slippage_bps_per_second=max(0.0, latency_slippage_bps_per_second),
        liquidity_score=min(1.0, max(0.0, liquidity_score)),
        market_depth_notional_usd=max(1.0, market_depth_notional_usd),
        notional_impact_coeff=max(0.0, notional_impact_coeff),
    )
    scenarios: list[ExecutionScenario] = [
        baseline,
        ExecutionScenario(
            name="spread_shock",
            spread_bps=max(1.0, baseline.spread_bps * 3.0),
            latency_ms=baseline.latency_ms,
            latency_slippage_bps_per_second=baseline.latency_slippage_bps_per_second,
            liquidity_score=baseline.liquidity_score,
            market_depth_notional_usd=baseline.market_depth_notional_usd,
            notional_impact_coeff=baseline.notional_impact_coeff,
        ),
        ExecutionScenario(
            name="latency_shock",
            spread_bps=baseline.spread_bps,
            latency_ms=max(50.0, baseline.latency_ms * 4.0),
            latency_slippage_bps_per_second=max(
                baseline.latency_slippage_bps_per_second,
                baseline.latency_slippage_bps_per_second * 2.0,
            ),
            liquidity_score=baseline.liquidity_score,
            market_depth_notional_usd=baseline.market_depth_notional_usd,
            notional_impact_coeff=baseline.notional_impact_coeff,
        ),
        ExecutionScenario(
            name="liquidity_stress",
            spread_bps=max(1.0, baseline.spread_bps * 2.0),
            latency_ms=max(10.0, baseline.latency_ms * 1.5),
            latency_slippage_bps_per_second=max(
                baseline.latency_slippage_bps_per_second,
                baseline.latency_slippage_bps_per_second * 1.5,
            ),
            liquidity_score=min(1.0, max(0.05, baseline.liquidity_score * 0.45)),
            market_depth_notional_usd=max(100.0, baseline.market_depth_notional_usd * 0.35),
            notional_impact_coeff=max(0.10, baseline.notional_impact_coeff * 1.80),
        ),
        ExecutionScenario(
            name="combined_stress",
            spread_bps=max(2.0, baseline.spread_bps * 5.0),
            latency_ms=max(100.0, baseline.latency_ms * 6.0),
            latency_slippage_bps_per_second=max(
                baseline.latency_slippage_bps_per_second,
                baseline.latency_slippage_bps_per_second * 3.0,
            ),
            liquidity_score=min(1.0, max(0.05, baseline.liquidity_score * 0.30)),
            market_depth_notional_usd=max(75.0, baseline.market_depth_notional_usd * 0.20),
            notional_impact_coeff=max(0.25, baseline.notional_impact_coeff * 2.50),
        ),
    ]
    if scenario_set == "extended":
        scenarios.append(
            ExecutionScenario(
                name="extreme_gap_stress",
                spread_bps=max(4.0, baseline.spread_bps * 8.0),
                latency_ms=max(250.0, baseline.latency_ms * 10.0),
                latency_slippage_bps_per_second=max(
                    baseline.latency_slippage_bps_per_second,
                    baseline.latency_slippage_bps_per_second * 4.0,
                ),
                liquidity_score=min(1.0, max(0.02, baseline.liquidity_score * 0.15)),
                market_depth_notional_usd=max(50.0, baseline.market_depth_notional_usd * 0.10),
                notional_impact_coeff=max(0.50, baseline.notional_impact_coeff * 3.50),
            )
        )
    return scenarios


def _empty_state(*, starting_cash_usd: float, fee_bps: float) -> dict[str, Any]:
    return {
        "contract": "paper_portfolio_state.v1",
        "updated_at_utc": _utc_now_iso(),
        "starting_cash_usd": max(0.0, float(starting_cash_usd)),
        "cash_usd": max(0.0, float(starting_cash_usd)),
        "fee_bps": max(0.0, float(fee_bps)),
        "peak_equity_usd": max(0.0, float(starting_cash_usd)),
        "max_drawdown_ratio": 0.0,
        "positions": {},
    }


def _simulate_roundtrip(
    *,
    scenario: ExecutionScenario,
    symbol: str,
    mark_price: float,
    requested_notional_usd: float,
    starting_cash_usd: float,
    fee_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    state = _empty_state(starting_cash_usd=max(starting_cash_usd, requested_notional_usd * 3.0), fee_bps=fee_bps)
    buy = simulate_paper_trade_execution_step(
        state=state,
        symbol=symbol,
        intent_status="emitted",
        intent_action="buy",
        requested_notional_usd=max(0.0, float(requested_notional_usd)),
        mark_price=float(mark_price),
        fee_bps=max(0.0, float(fee_bps)),
        slippage_bps=max(0.0, float(slippage_bps)),
        spread_bps=scenario.spread_bps,
        latency_ms=scenario.latency_ms,
        latency_slippage_bps_per_second=scenario.latency_slippage_bps_per_second,
        liquidity_score=scenario.liquidity_score,
        market_depth_notional_usd=scenario.market_depth_notional_usd,
        notional_impact_coeff=scenario.notional_impact_coeff,
    )
    position_payload = dict(state.get("positions", {})).get(symbol, {})
    position_qty_after_buy = max(0.0, _safe_float(dict(position_payload).get("quantity"), 0.0))
    sell_requested_notional = (
        max(0.0, float(position_qty_after_buy) * float(mark_price))
        if mark_price > 0.0
        else max(0.0, _safe_float(buy.get("executed_notional_usd"), 0.0))
    )
    sell = simulate_paper_trade_execution_step(
        state=state,
        symbol=symbol,
        intent_status="emitted",
        intent_action="sell",
        requested_notional_usd=max(0.0, float(sell_requested_notional)),
        mark_price=float(mark_price),
        fee_bps=max(0.0, float(fee_bps)),
        slippage_bps=max(0.0, float(slippage_bps)),
        spread_bps=scenario.spread_bps,
        latency_ms=scenario.latency_ms,
        latency_slippage_bps_per_second=scenario.latency_slippage_bps_per_second,
        liquidity_score=scenario.liquidity_score,
        market_depth_notional_usd=scenario.market_depth_notional_usd,
        notional_impact_coeff=scenario.notional_impact_coeff,
    )

    buy_executed_notional = max(0.0, _safe_float(buy.get("executed_notional_usd"), 0.0))
    sell_executed_notional = max(0.0, _safe_float(sell.get("executed_notional_usd"), 0.0))
    buy_effective_slippage_bps = max(0.0, _safe_float(buy.get("effective_slippage_bps"), 0.0))
    sell_effective_slippage_bps = max(0.0, _safe_float(sell.get("effective_slippage_bps"), 0.0))
    buy_fee = max(0.0, _safe_float(buy.get("fee_usd"), 0.0))
    sell_fee = max(0.0, _safe_float(sell.get("fee_usd"), 0.0))

    notional_turnover = buy_executed_notional + sell_executed_notional
    modeled_drag_usd = (
        (buy_executed_notional * (buy_effective_slippage_bps / 10_000.0))
        + (sell_executed_notional * (sell_effective_slippage_bps / 10_000.0))
        + buy_fee
        + sell_fee
    )
    total_cost_drag_bps = (
        (modeled_drag_usd / max(1e-9, notional_turnover)) * 10_000.0
        if notional_turnover > 0.0
        else 0.0
    )
    buy_fill_ratio = max(0.0, min(1.0, _safe_float(buy.get("fill_ratio"), 0.0)))
    sell_fill_ratio = max(0.0, min(1.0, _safe_float(sell.get("fill_ratio"), 0.0)))
    average_fill_ratio = (buy_fill_ratio + sell_fill_ratio) / 2.0
    average_effective_slippage_bps = (buy_effective_slippage_bps + sell_effective_slippage_bps) / 2.0

    final_position_payload = dict(state.get("positions", {})).get(symbol, {})
    final_position_qty = max(0.0, _safe_float(dict(final_position_payload).get("quantity"), 0.0))
    drawdown_ratio = max(
        _safe_float(buy.get("max_drawdown_ratio"), 0.0),
        _safe_float(sell.get("max_drawdown_ratio"), 0.0),
        _safe_float(state.get("max_drawdown_ratio"), 0.0),
    )
    roundtrip_executed = (
        str(buy.get("execution_status")) == "executed"
        and str(sell.get("execution_status")) == "executed"
    )
    closed_position = final_position_qty <= 1e-8

    return {
        "scenario": scenario.name,
        "parameters": {
            "spread_bps": scenario.spread_bps,
            "latency_ms": scenario.latency_ms,
            "latency_slippage_bps_per_second": scenario.latency_slippage_bps_per_second,
            "liquidity_score": scenario.liquidity_score,
            "market_depth_notional_usd": scenario.market_depth_notional_usd,
            "notional_impact_coeff": scenario.notional_impact_coeff,
        },
        "buy": {
            "execution_status": str(buy.get("execution_status")),
            "reason": str(buy.get("reason")),
            "executed_notional_usd": buy_executed_notional,
            "fill_ratio": buy_fill_ratio,
            "effective_slippage_bps": buy_effective_slippage_bps,
            "fee_usd": buy_fee,
        },
        "sell": {
            "execution_status": str(sell.get("execution_status")),
            "reason": str(sell.get("reason")),
            "executed_notional_usd": sell_executed_notional,
            "fill_ratio": sell_fill_ratio,
            "effective_slippage_bps": sell_effective_slippage_bps,
            "fee_usd": sell_fee,
        },
        "notional_turnover_usd": notional_turnover,
        "modeled_drag_usd": modeled_drag_usd,
        "total_cost_drag_bps": total_cost_drag_bps,
        "average_fill_ratio": average_fill_ratio,
        "average_effective_slippage_bps": average_effective_slippage_bps,
        "realized_pnl_delta_usd": _safe_float(sell.get("realized_pnl_delta_usd"), 0.0),
        "cash_after_usd": _safe_float(state.get("cash_usd"), 0.0),
        "drawdown_ratio": max(0.0, min(1.0, drawdown_ratio)),
        "position_qty_after": final_position_qty,
        "roundtrip_executed": bool(roundtrip_executed),
        "roundtrip_closed_position": bool(closed_position),
    }


def _plot_sensitivity(*, rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["scenario"]) for row in rows]
    drag_bps = [float(row.get("total_cost_drag_bps", 0.0)) for row in rows]
    fill_ratio = [float(row.get("average_fill_ratio", 0.0)) for row in rows]

    x_positions = list(range(len(labels)))
    fig, axis_left = plt.subplots(figsize=(12, 6.8))
    axis_left.bar(x_positions, drag_bps, color="#2b6cb0", alpha=0.85)
    axis_left.set_ylabel("Total modeled cost drag (bps)")
    axis_left.set_xlabel("Execution realism scenario")
    axis_left.set_xticks(x_positions)
    axis_left.set_xticklabels(labels, rotation=20, ha="right")
    axis_left.grid(axis="y", linestyle="--", alpha=0.25)

    axis_right = axis_left.twinx()
    axis_right.plot(x_positions, fill_ratio, color="#d97706", marker="o", linewidth=2.0)
    axis_right.set_ylabel("Average fill ratio")
    axis_right.set_ylim(0.0, 1.05)

    plt.title("Execution realism sensitivity: cost drag vs fill quality")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=170)
    plt.close()


def _build_markdown_report(report: dict[str, Any]) -> str:
    status = "PASS" if bool(report.get("passed", False)) else "FAIL"
    thresholds = dict(report.get("thresholds", {}))
    failures = [f"- {entry}" for entry in list(report.get("failures", []))]
    warnings = [f"- {entry}" for entry in list(report.get("warnings", []))]
    if not failures:
        failures = ["- None"]
    if not warnings:
        warnings = ["- None"]

    rows = list(report.get("scenario_results", []))
    lines: list[str] = [
        "# Execution realism stress suite",
        f"- Status: **{status}**",
        f"- Created at: `{report.get('created_at_utc')}`",
        f"- Exchange: `{report.get('exchange')}`",
        f"- Symbol: `{report.get('symbol')}`",
        f"- Timeframe: `{report.get('timeframe')}`",
        f"- Source data path: `{report.get('source_data_path')}`",
        f"- Mark price: `{float(report.get('mark_price', 0.0)):.8f}`",
        "## Gate thresholds",
        f"- max_cost_drag_bps: `{float(thresholds.get('max_cost_drag_bps', 0.0)):.6f}`",
        f"- min_average_fill_ratio: `{float(thresholds.get('min_average_fill_ratio', 0.0)):.6f}`",
        "## Scenario summaries",
    ]
    for row in rows:
        lines.extend(
            [
                f"- `{row.get('scenario')}`",
                f"  - total_cost_drag_bps: `{float(row.get('total_cost_drag_bps', 0.0)):.6f}`",
                f"  - average_fill_ratio: `{float(row.get('average_fill_ratio', 0.0)):.6f}`",
                f"  - average_effective_slippage_bps: `{float(row.get('average_effective_slippage_bps', 0.0)):.6f}`",
                f"  - roundtrip_executed: `{bool(row.get('roundtrip_executed', False))}`",
                f"  - roundtrip_closed_position: `{bool(row.get('roundtrip_closed_position', False))}`",
                f"  - delta_vs_baseline_cost_drag_bps: `{float(dict(row.get('delta_vs_baseline', {})).get('total_cost_drag_bps', 0.0)):.6f}`",
            ]
        )

    lines.extend(
        [
            "## Failures",
            *failures,
            "## Warnings",
            *warnings,
            "## Artifacts",
            f"- Report JSON: `{dict(report.get('artifacts', {})).get('report_json_path')}`",
            f"- Report Markdown: `{dict(report.get('artifacts', {})).get('report_markdown_path')}`",
            f"- Sensitivity plot: `{dict(report.get('artifacts', {})).get('sensitivity_plot_path')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic execution realism stress scenarios and generate cost-drag sensitivity artifacts."
        )
    )
    parser.add_argument("--exchange", default=settings.default_exchange)
    parser.add_argument("--symbol", default=settings.default_symbol)
    parser.add_argument("--timeframe", default=settings.default_timeframe)
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument(
        "--scenario-set",
        choices=("default", "extended"),
        default="default",
        help="default=baseline+4 stress scenarios, extended=adds extreme gap stress.",
    )
    parser.add_argument(
        "--requested-notional-usd",
        type=float,
        default=float(settings.paper_trade_notional_usd),
    )
    parser.add_argument(
        "--starting-cash-usd",
        type=float,
        default=float(settings.paper_trade_starting_cash_usd),
    )
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=float(settings.paper_trade_fee_bps),
    )
    parser.add_argument(
        "--base-slippage-bps",
        type=float,
        default=float(settings.paper_trade_slippage_bps),
    )
    parser.add_argument(
        "--base-spread-bps",
        type=float,
        default=float(settings.execution_realism_spread_bps),
    )
    parser.add_argument(
        "--base-latency-ms",
        type=float,
        default=float(settings.execution_realism_latency_ms),
    )
    parser.add_argument(
        "--base-latency-slippage-bps-per-second",
        type=float,
        default=float(settings.execution_realism_latency_slippage_bps_per_second),
    )
    parser.add_argument(
        "--base-liquidity-score",
        type=float,
        default=float(settings.execution_realism_liquidity_score),
    )
    parser.add_argument(
        "--base-market-depth-notional-usd",
        type=float,
        default=float(settings.execution_realism_market_depth_notional_usd),
    )
    parser.add_argument(
        "--base-notional-impact-coeff",
        type=float,
        default=float(settings.execution_realism_notional_impact_coeff),
    )
    parser.add_argument(
        "--max-cost-drag-bps",
        type=float,
        default=500.0,
        help="Scenario gate threshold for modeled cost drag in basis points.",
    )
    parser.add_argument(
        "--min-average-fill-ratio",
        type=float,
        default=0.15,
        help="Scenario gate threshold for average fill ratio (0..1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory for suite artifacts.",
    )
    parser.add_argument(
        "--enforce-gate",
        action="store_true",
        help="Exit non-zero when any scenario violates the configured thresholds.",
    )
    args = parser.parse_args()

    ensure_data_root_ready(
        settings.quant_data_root,
        allow_unmounted=settings.allow_unmounted_data_root,
    )

    source_data_path = _resolve_source_data_path(
        quant_data_root=settings.quant_data_root,
        exchange=str(args.exchange),
        symbol=str(args.symbol),
        timeframe=str(args.timeframe),
        explicit_path=args.input_file,
    )
    mark_price, mark_price_timestamp = _load_mark_price(source_data_path)

    scenarios = _build_scenarios(
        scenario_set=str(args.scenario_set),
        spread_bps=max(0.0, float(args.base_spread_bps)),
        latency_ms=max(0.0, float(args.base_latency_ms)),
        latency_slippage_bps_per_second=max(0.0, float(args.base_latency_slippage_bps_per_second)),
        liquidity_score=max(0.0, min(1.0, float(args.base_liquidity_score))),
        market_depth_notional_usd=max(1.0, float(args.base_market_depth_notional_usd)),
        notional_impact_coeff=max(0.0, float(args.base_notional_impact_coeff)),
    )

    results = [
        _simulate_roundtrip(
            scenario=scenario,
            symbol=str(args.symbol),
            mark_price=float(mark_price),
            requested_notional_usd=max(0.0, float(args.requested_notional_usd)),
            starting_cash_usd=max(0.0, float(args.starting_cash_usd)),
            fee_bps=max(0.0, float(args.fee_bps)),
            slippage_bps=max(0.0, float(args.base_slippage_bps)),
        )
        for scenario in scenarios
    ]
    baseline = results[0]
    baseline_drag = float(baseline.get("total_cost_drag_bps", 0.0))
    baseline_fill = float(baseline.get("average_fill_ratio", 0.0))
    baseline_slippage = float(baseline.get("average_effective_slippage_bps", 0.0))
    for row in results:
        row["delta_vs_baseline"] = {
            "total_cost_drag_bps": float(row.get("total_cost_drag_bps", 0.0)) - baseline_drag,
            "average_fill_ratio": float(row.get("average_fill_ratio", 0.0)) - baseline_fill,
            "average_effective_slippage_bps": float(row.get("average_effective_slippage_bps", 0.0))
            - baseline_slippage,
        }

    failures: list[str] = []
    warnings: list[str] = []
    max_cost_drag_bps = max(0.0, float(args.max_cost_drag_bps))
    min_average_fill_ratio = max(0.0, min(1.0, float(args.min_average_fill_ratio)))
    if not bool(baseline.get("roundtrip_executed", False)):
        failures.append("baseline_roundtrip_not_executed")
    if not bool(baseline.get("roundtrip_closed_position", False)):
        warnings.append("baseline_roundtrip_left_open_position")
    for row in results:
        scenario_name = str(row.get("scenario"))
        if not bool(row.get("roundtrip_executed", False)):
            failures.append(f"{scenario_name}:roundtrip_not_executed")
        if not bool(row.get("roundtrip_closed_position", False)):
            warnings.append(f"{scenario_name}:roundtrip_left_open_position")
        if float(row.get("total_cost_drag_bps", 0.0)) > max_cost_drag_bps:
            failures.append(f"{scenario_name}:cost_drag_bps_exceeds_threshold")
        if float(row.get("average_fill_ratio", 0.0)) < min_average_fill_ratio:
            failures.append(f"{scenario_name}:average_fill_ratio_below_threshold")

    passed = len(failures) == 0
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            settings.quant_data_root
            / "curated"
            / "evaluations"
            / "execution_realism_stress_suite"
            / f"exchange={args.exchange}"
            / f"symbol={symbol_slug(str(args.symbol))}"
            / f"interval={args.timeframe}"
            / f"run_id={_run_id()}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    report_json_path = output_dir / "execution_realism_stress_suite.json"
    report_md_path = output_dir / "execution_realism_stress_suite.md"
    sensitivity_plot_path = output_dir / "execution_realism_cost_drag_sensitivity.png"

    _plot_sensitivity(rows=results, output_path=sensitivity_plot_path)

    report_payload: dict[str, Any] = {
        "contract": "execution_realism_stress_suite.v1",
        "created_at_utc": _utc_now_iso(),
        "passed": bool(passed),
        "exchange": str(args.exchange),
        "symbol": str(args.symbol),
        "timeframe": str(args.timeframe),
        "source_data_path": str(source_data_path),
        "mark_price": float(mark_price),
        "mark_price_timestamp_utc": mark_price_timestamp,
        "scenario_set": str(args.scenario_set),
        "thresholds": {
            "max_cost_drag_bps": max_cost_drag_bps,
            "min_average_fill_ratio": min_average_fill_ratio,
        },
        "base_execution_assumptions": {
            "fee_bps": max(0.0, float(args.fee_bps)),
            "base_slippage_bps": max(0.0, float(args.base_slippage_bps)),
            "requested_notional_usd": max(0.0, float(args.requested_notional_usd)),
            "starting_cash_usd": max(0.0, float(args.starting_cash_usd)),
            "spread_bps": max(0.0, float(args.base_spread_bps)),
            "latency_ms": max(0.0, float(args.base_latency_ms)),
            "latency_slippage_bps_per_second": max(
                0.0, float(args.base_latency_slippage_bps_per_second)
            ),
            "liquidity_score": max(0.0, min(1.0, float(args.base_liquidity_score))),
            "market_depth_notional_usd": max(1.0, float(args.base_market_depth_notional_usd)),
            "notional_impact_coeff": max(0.0, float(args.base_notional_impact_coeff)),
        },
        "scenario_results": results,
        "failures": failures,
        "warnings": warnings,
        "artifacts": {
            "output_dir": str(output_dir),
            "report_json_path": str(report_json_path),
            "report_markdown_path": str(report_md_path),
            "sensitivity_plot_path": str(sensitivity_plot_path),
        },
    }
    report_json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    report_md_path.write_text(_build_markdown_report(report_payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "PASS" if passed else "FAIL",
                "report_json_path": str(report_json_path),
                "report_markdown_path": str(report_md_path),
                "sensitivity_plot_path": str(sensitivity_plot_path),
                "scenario_count": len(results),
                "failure_count": len(failures),
                "warning_count": len(warnings),
            },
            indent=2,
        )
    )
    if bool(args.enforce_gate) and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
