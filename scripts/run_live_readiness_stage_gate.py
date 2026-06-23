#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_agents.config import ensure_data_root_ready, load_settings
from quant_agents.storage import symbol_slug


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_datetime_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return payload


def _try_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return _read_json(path, label="json payload")
    except Exception:
        return None


def _resolve_benchmark_summary_path(
    *,
    quant_data_root: Path,
    explicit_path: Path | None,
) -> Path | None:
    if explicit_path is not None:
        candidate = explicit_path.expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Benchmark summary path not found: {candidate}")
        return candidate
    benchmark_root = quant_data_root / "logs" / "analysis" / "regime-benchmark-gate"
    if not benchmark_root.exists():
        return None
    candidates = sorted(
        benchmark_root.glob("*/latest_summary.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        return None
    return candidates[-1].resolve()


def _extract_drawdown_ratio(outcome: dict[str, Any]) -> float | None:
    execution_diag = outcome.get("paper_execution_diagnostics")
    if isinstance(execution_diag, dict):
        drawdown_snapshot = execution_diag.get("drawdown_snapshot")
        if isinstance(drawdown_snapshot, dict):
            max_drawdown = _safe_float(drawdown_snapshot.get("max_drawdown_ratio"))
            if max_drawdown is not None:
                return max_drawdown
            drawdown = _safe_float(drawdown_snapshot.get("drawdown_ratio"))
            if drawdown is not None:
                return drawdown
    intent_diag = outcome.get("paper_intent_sizing_diagnostics")
    if isinstance(intent_diag, dict):
        drawdown = _safe_float(intent_diag.get("drawdown_ratio"))
        if drawdown is not None:
            return drawdown
    return None


def _collect_run_manifests(
    *,
    quant_data_root: Path,
    exchange: str,
    symbol: str,
    timeframe: str,
    lookback_days: int,
    max_manifests: int,
) -> list[dict[str, Any]]:
    orchestrator_root = quant_data_root / "logs" / "agents" / "openclaw-orchestrator"
    if not orchestrator_root.exists():
        return []
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=max(1, lookback_days))
        if lookback_days > 0
        else None
    )
    manifests: list[dict[str, Any]] = []
    for path in sorted(orchestrator_root.glob("*/*/run_manifest.json")):
        payload = _try_read_json(path)
        if not isinstance(payload, dict):
            continue
        scope = payload.get("scope")
        if not isinstance(scope, dict):
            continue
        if str(scope.get("exchange")) != exchange:
            continue
        if str(scope.get("symbol")) != symbol:
            continue
        if str(scope.get("timeframe")) != timeframe:
            continue
        created_at = _parse_datetime_utc(payload.get("created_at_utc"))
        if cutoff is not None and (created_at is None or created_at < cutoff):
            continue
        manifests.append(
            {
                "path": str(path.resolve()),
                "created_at_utc": created_at.isoformat() if created_at else None,
                "payload": payload,
            }
        )
    manifests.sort(
        key=lambda row: _parse_datetime_utc(row.get("created_at_utc")) or datetime.min.replace(
            tzinfo=timezone.utc
        )
    )
    if max_manifests > 0 and len(manifests) > max_manifests:
        manifests = manifests[-max_manifests:]
    return manifests


def _build_stage_status(
    *,
    checks: dict[str, bool],
) -> dict[str, bool]:
    shadow_ready = bool(checks.get("benchmark_gate_pass")) and bool(checks.get("incident_rate_within_limit"))
    paper_forward_ready = (
        shadow_ready
        and bool(checks.get("minimum_paper_runs"))
        and bool(checks.get("execution_success_rate"))
        and bool(checks.get("average_fill_ratio"))
    )
    limited_live_ready = (
        paper_forward_ready
        and bool(checks.get("max_drawdown_within_limit"))
        and bool(checks.get("realized_pnl_minimum"))
        and bool(checks.get("benchmark_cost_drag_within_limit"))
        and bool(checks.get("benchmark_priority_windows_covered"))
    )
    return {
        "shadow": shadow_ready,
        "paper_forward": paper_forward_ready,
        "limited_live": limited_live_ready,
    }


def _recommended_stage(stage_status: dict[str, bool]) -> str:
    if bool(stage_status.get("limited_live")):
        return "limited_live"
    if bool(stage_status.get("paper_forward")):
        return "paper_forward"
    if bool(stage_status.get("shadow")):
        return "shadow"
    return "blocked"


def _build_markdown_report(report: dict[str, Any]) -> str:
    status = "PASS" if bool(report.get("passed", False)) else "FAIL"
    stage_status = dict(report.get("stage_status", {}))
    checks = list(report.get("check_results", []))
    metrics = dict(report.get("metrics", {}))
    blockers = [f"- {item}" for item in list(report.get("blockers", []))]
    warnings = [f"- {item}" for item in list(report.get("warnings", []))]
    if not blockers:
        blockers = ["- None"]
    if not warnings:
        warnings = ["- None"]

    lines = [
        "# Live readiness stage-gate report",
        f"- Status: **{status}**",
        f"- Created at: `{report.get('created_at_utc')}`",
        f"- Exchange: `{report.get('exchange')}`",
        f"- Symbol: `{report.get('symbol')}`",
        f"- Timeframe: `{report.get('timeframe')}`",
        f"- Required stage: `{report.get('required_stage')}`",
        f"- Recommended stage: `{report.get('recommended_stage')}`",
        "## Stage status",
        f"- shadow: `{bool(stage_status.get('shadow', False))}`",
        f"- paper_forward: `{bool(stage_status.get('paper_forward', False))}`",
        f"- limited_live: `{bool(stage_status.get('limited_live', False))}`",
        "## Key metrics",
        f"- benchmark_gate_status: `{metrics.get('benchmark_gate_status')}`",
        f"- benchmark_mean_total_cost_return_drag: `{metrics.get('benchmark_mean_total_cost_return_drag')}`",
        f"- benchmark_skipped_priority_window_count: `{metrics.get('benchmark_skipped_priority_window_count')}`",
        f"- run_count: `{metrics.get('run_count')}`",
        f"- risk_approved_count: `{metrics.get('risk_approved_count')}`",
        f"- intent_emitted_count: `{metrics.get('intent_emitted_count')}`",
        f"- execution_count: `{metrics.get('execution_count')}`",
        f"- execution_success_rate: `{metrics.get('execution_success_rate')}`",
        f"- avg_fill_ratio: `{metrics.get('avg_fill_ratio')}`",
        f"- max_drawdown_ratio: `{metrics.get('max_drawdown_ratio')}`",
        f"- incident_rate: `{metrics.get('incident_rate')}`",
        f"- realized_pnl_delta_usd: `{metrics.get('realized_pnl_delta_usd')}`",
        "## Check results",
    ]
    for check in checks:
        name = str(check.get("name"))
        passed = bool(check.get("pass", False))
        observed = check.get("observed")
        expected = check.get("expected")
        lines.append(f"- [{'PASS' if passed else 'FAIL'}] `{name}` (observed={observed}, expected={expected})")

    lines.extend(
        [
            "## Blockers",
            *blockers,
            "## Warnings",
            *warnings,
            "## Artifacts",
            f"- Report JSON: `{dict(report.get('artifacts', {})).get('report_json_path')}`",
            f"- Report Markdown: `{dict(report.get('artifacts', {})).get('report_markdown_path')}`",
            f"- Benchmark summary: `{metrics.get('benchmark_summary_path')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate staged paper-to-live readiness gates using benchmark artifacts and recent agent-plane manifests."
        )
    )
    parser.add_argument("--exchange", default=settings.default_exchange)
    parser.add_argument("--symbol", default=settings.default_symbol)
    parser.add_argument("--timeframe", default=settings.default_timeframe)
    parser.add_argument("--benchmark-summary-path", type=Path, default=None)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument(
        "--max-manifests",
        type=int,
        default=500,
        help="Maximum number of recent manifests to include in readiness metrics.",
    )
    parser.add_argument(
        "--required-stage",
        choices=("shadow", "paper_forward", "limited_live"),
        default="paper_forward",
        help="Readiness stage required for PASS status.",
    )
    parser.add_argument(
        "--require-benchmark",
        dest="require_benchmark",
        action="store_true",
        help="Require benchmark summary availability for readiness PASS.",
    )
    parser.add_argument(
        "--no-require-benchmark",
        dest="require_benchmark",
        action="store_false",
        help="Allow readiness evaluation to proceed without benchmark summary.",
    )
    parser.set_defaults(require_benchmark=True)
    parser.add_argument("--min-paper-runs", type=int, default=5)
    parser.add_argument("--min-execution-success-rate", type=float, default=0.70)
    parser.add_argument("--min-average-fill-ratio", type=float, default=0.40)
    parser.add_argument("--max-incident-rate", type=float, default=0.20)
    parser.add_argument("--max-drawdown-ratio", type=float, default=0.20)
    parser.add_argument("--min-realized-pnl-usd", type=float, default=0.0)
    parser.add_argument("--max-benchmark-cost-drag", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--fail-on-unmet-stage",
        action="store_true",
        help="Exit non-zero when required stage gate is not met.",
    )
    args = parser.parse_args()

    ensure_data_root_ready(
        settings.quant_data_root,
        allow_unmounted=settings.allow_unmounted_data_root,
    )

    benchmark_summary_path = _resolve_benchmark_summary_path(
        quant_data_root=settings.quant_data_root,
        explicit_path=args.benchmark_summary_path,
    )
    benchmark_summary = (
        _read_json(benchmark_summary_path, label="benchmark summary")
        if benchmark_summary_path is not None
        else None
    )
    manifests = _collect_run_manifests(
        quant_data_root=settings.quant_data_root,
        exchange=str(args.exchange),
        symbol=str(args.symbol),
        timeframe=str(args.timeframe),
        lookback_days=max(0, int(args.lookback_days)),
        max_manifests=max(0, int(args.max_manifests)),
    )

    run_count = len(manifests)
    risk_approved_count = 0
    intent_emitted_count = 0
    execution_count = 0
    incident_count = 0
    fill_ratios: list[float] = []
    drawdowns: list[float] = []
    realized_pnl_delta_usd = 0.0

    for row in manifests:
        payload = dict(row.get("payload", {}))
        outcome = payload.get("outcome")
        if not isinstance(outcome, dict):
            outcome = {}
        steps = payload.get("steps")
        if not isinstance(steps, list):
            steps = []
        step_has_incident = False
        for step in steps:
            if not isinstance(step, dict):
                continue
            if str(step.get("status")) != "success":
                step_has_incident = True
                break
            errors = step.get("errors")
            if isinstance(errors, list) and len(errors) > 0:
                step_has_incident = True
                break
        if step_has_incident:
            incident_count += 1

        if outcome.get("risk_approved") is True:
            risk_approved_count += 1
        if str(outcome.get("intent_status")) == "emitted":
            intent_emitted_count += 1
        if str(outcome.get("paper_trade_execution_status")) == "executed":
            execution_count += 1

        fill_ratio = _safe_float(outcome.get("paper_execution_fill_ratio"))
        if fill_ratio is not None:
            fill_ratios.append(max(0.0, min(1.0, fill_ratio)))
        drawdown_ratio = _extract_drawdown_ratio(outcome)
        if drawdown_ratio is not None:
            drawdowns.append(max(0.0, min(1.0, drawdown_ratio)))

        artifacts = payload.get("artifacts")
        if isinstance(artifacts, dict):
            execution_path_value = artifacts.get("paper_trade_execution")
            if isinstance(execution_path_value, str) and execution_path_value.strip():
                execution_payload = _try_read_json(Path(execution_path_value).expanduser())
                if isinstance(execution_payload, dict):
                    realized_pnl_delta_usd += _safe_float(
                        execution_payload.get("realized_pnl_delta_usd"),
                        0.0,
                    ) or 0.0
                    if fill_ratio is None:
                        fallback_fill = _safe_float(execution_payload.get("fill_ratio"))
                        if fallback_fill is not None:
                            fill_ratios.append(max(0.0, min(1.0, fallback_fill)))

    execution_success_rate = (
        (execution_count / intent_emitted_count)
        if intent_emitted_count > 0
        else 0.0
    )
    avg_fill_ratio = (sum(fill_ratios) / len(fill_ratios)) if fill_ratios else None
    max_drawdown_ratio = max(drawdowns) if drawdowns else None
    incident_rate = (incident_count / run_count) if run_count > 0 else 1.0

    benchmark_gate_status = None
    benchmark_gate_pass = False
    benchmark_mean_total_cost_return_drag = None
    benchmark_skipped_priority_window_count = None
    if isinstance(benchmark_summary, dict):
        gate_outcome = benchmark_summary.get("gate_outcome")
        if isinstance(gate_outcome, dict):
            benchmark_gate_status = str(gate_outcome.get("status", "")).lower()
            benchmark_gate_pass = benchmark_gate_status == "pass"
        profile_summary = benchmark_summary.get("profile_summary")
        if isinstance(profile_summary, dict):
            regime_enabled = profile_summary.get("regime_enabled")
            if isinstance(regime_enabled, dict):
                benchmark_mean_total_cost_return_drag = _safe_float(
                    regime_enabled.get("mean_total_cost_return_drag")
                )
        benchmark_skipped_priority_window_count = (
            int(benchmark_summary.get("skipped_priority_window_count"))
            if benchmark_summary.get("skipped_priority_window_count") is not None
            else None
        )

    check_results: list[dict[str, Any]] = []
    blockers: list[str] = []
    warnings: list[str] = []

    def _append_check(name: str, passed: bool, observed: Any, expected: Any) -> None:
        check_results.append(
            {
                "name": name,
                "pass": bool(passed),
                "observed": observed,
                "expected": expected,
            }
        )
        if not passed:
            blockers.append(name)

    benchmark_present = benchmark_summary is not None
    if bool(args.require_benchmark):
        _append_check(
            "benchmark_available",
            benchmark_present,
            "present" if benchmark_present else "missing",
            "present",
        )
    else:
        if not benchmark_present:
            warnings.append("benchmark_summary_missing")

    _append_check(
        "benchmark_gate_pass",
        benchmark_gate_pass if benchmark_present else not bool(args.require_benchmark),
        benchmark_gate_status if benchmark_present else "missing",
        "pass",
    )
    _append_check(
        "benchmark_priority_windows_covered",
        (
            benchmark_skipped_priority_window_count is not None
            and benchmark_skipped_priority_window_count == 0
        )
        if benchmark_present
        else not bool(args.require_benchmark),
        benchmark_skipped_priority_window_count if benchmark_present else "missing",
        "0",
    )
    _append_check(
        "benchmark_cost_drag_within_limit",
        (
            benchmark_mean_total_cost_return_drag is not None
            and benchmark_mean_total_cost_return_drag <= float(args.max_benchmark_cost_drag)
        )
        if benchmark_present
        else not bool(args.require_benchmark),
        benchmark_mean_total_cost_return_drag if benchmark_present else "missing",
        f"<= {float(args.max_benchmark_cost_drag):.6f}",
    )

    _append_check(
        "minimum_paper_runs",
        run_count >= max(1, int(args.min_paper_runs)),
        run_count,
        f">= {max(1, int(args.min_paper_runs))}",
    )
    _append_check(
        "execution_success_rate",
        execution_success_rate >= max(0.0, min(1.0, float(args.min_execution_success_rate))),
        execution_success_rate,
        f">= {max(0.0, min(1.0, float(args.min_execution_success_rate))):.6f}",
    )
    _append_check(
        "average_fill_ratio",
        (avg_fill_ratio is not None)
        and (avg_fill_ratio >= max(0.0, min(1.0, float(args.min_average_fill_ratio)))),
        avg_fill_ratio,
        f">= {max(0.0, min(1.0, float(args.min_average_fill_ratio))):.6f}",
    )
    _append_check(
        "incident_rate_within_limit",
        incident_rate <= max(0.0, min(1.0, float(args.max_incident_rate))),
        incident_rate,
        f"<= {max(0.0, min(1.0, float(args.max_incident_rate))):.6f}",
    )
    _append_check(
        "max_drawdown_within_limit",
        (max_drawdown_ratio is not None)
        and (max_drawdown_ratio <= max(0.0, min(1.0, float(args.max_drawdown_ratio)))),
        max_drawdown_ratio,
        f"<= {max(0.0, min(1.0, float(args.max_drawdown_ratio))):.6f}",
    )
    _append_check(
        "realized_pnl_minimum",
        realized_pnl_delta_usd >= float(args.min_realized_pnl_usd),
        realized_pnl_delta_usd,
        f">= {float(args.min_realized_pnl_usd):.6f}",
    )

    check_map = {str(item["name"]): bool(item["pass"]) for item in check_results}
    stage_status = _build_stage_status(checks=check_map)
    recommended_stage = _recommended_stage(stage_status)
    required_stage = str(args.required_stage)
    passed = bool(stage_status.get(required_stage, False))

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            settings.quant_data_root
            / "curated"
            / "evaluations"
            / "live_readiness_stage_gate"
            / f"exchange={args.exchange}"
            / f"symbol={symbol_slug(str(args.symbol))}"
            / f"interval={args.timeframe}"
            / f"run_id={_run_id()}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    report_json_path = output_dir / "live_readiness_stage_gate_report.json"
    report_md_path = output_dir / "live_readiness_stage_gate_report.md"

    report_payload: dict[str, Any] = {
        "contract": "live_readiness_stage_gate_report.v1",
        "created_at_utc": _utc_now_iso(),
        "passed": bool(passed),
        "required_stage": required_stage,
        "recommended_stage": recommended_stage,
        "stage_status": stage_status,
        "exchange": str(args.exchange),
        "symbol": str(args.symbol),
        "timeframe": str(args.timeframe),
        "metrics": {
            "benchmark_summary_path": str(benchmark_summary_path) if benchmark_summary_path else None,
            "benchmark_gate_status": benchmark_gate_status,
            "benchmark_mean_total_cost_return_drag": benchmark_mean_total_cost_return_drag,
            "benchmark_skipped_priority_window_count": benchmark_skipped_priority_window_count,
            "run_count": run_count,
            "risk_approved_count": risk_approved_count,
            "intent_emitted_count": intent_emitted_count,
            "execution_count": execution_count,
            "execution_success_rate": execution_success_rate,
            "avg_fill_ratio": avg_fill_ratio,
            "max_drawdown_ratio": max_drawdown_ratio,
            "incident_count": incident_count,
            "incident_rate": incident_rate,
            "realized_pnl_delta_usd": realized_pnl_delta_usd,
        },
        "thresholds": {
            "min_paper_runs": max(1, int(args.min_paper_runs)),
            "min_execution_success_rate": max(0.0, min(1.0, float(args.min_execution_success_rate))),
            "min_average_fill_ratio": max(0.0, min(1.0, float(args.min_average_fill_ratio))),
            "max_incident_rate": max(0.0, min(1.0, float(args.max_incident_rate))),
            "max_drawdown_ratio": max(0.0, min(1.0, float(args.max_drawdown_ratio))),
            "min_realized_pnl_usd": float(args.min_realized_pnl_usd),
            "max_benchmark_cost_drag": float(args.max_benchmark_cost_drag),
        },
        "check_results": check_results,
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "manifest_sample_paths": [row["path"] for row in manifests[-20:]],
        "artifacts": {
            "output_dir": str(output_dir),
            "report_json_path": str(report_json_path),
            "report_markdown_path": str(report_md_path),
        },
    }

    report_json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    report_md_path.write_text(_build_markdown_report(report_payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "PASS" if passed else "FAIL",
                "required_stage": required_stage,
                "recommended_stage": recommended_stage,
                "report_json_path": str(report_json_path),
                "report_markdown_path": str(report_md_path),
                "run_count": run_count,
                "blocker_count": len(set(blockers)),
                "warning_count": len(set(warnings)),
            },
            indent=2,
        )
    )
    if bool(args.fail_on_unmet_stage) and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
