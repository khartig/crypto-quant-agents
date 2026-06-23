#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_agents.alternative_data_features import (
    ALTERNATIVE_DATA_FEATURE_COLUMNS,
    ALTERNATIVE_DATA_FEATURE_SCHEMA_VERSION,
)
from quant_agents.config import ensure_data_root_ready, load_settings
from quant_agents.priority2_features import PRIORITY2_FEATURE_COLUMNS
from quant_agents.storage import symbol_slug


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_contract_path(
    *,
    quant_data_root: Path,
    exchange: str,
    symbol: str,
    timeframe: str,
    explicit_contract_path: Path | None,
) -> Path:
    if explicit_contract_path is not None:
        candidate = explicit_contract_path.expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Priority2 contract path does not exist: {candidate}")
        return candidate

    base_dir = (
        quant_data_root
        / "curated"
        / "features"
        / "external"
        / f"exchange={exchange}"
        / f"symbol={symbol_slug(symbol)}"
        / f"interval={timeframe}"
    )
    pointer = base_dir / "latest_external_feature_contract_path.txt"
    if pointer.exists():
        pointer_value = pointer.read_text(encoding="utf-8").strip()
        if pointer_value:
            candidate = Path(pointer_value).expanduser().resolve()
            if candidate.exists():
                return candidate

    candidates = sorted(base_dir.glob("run_id=*/priority2_external_feature_contract.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No Priority2 external feature contracts found under: {base_dir}"
        )
    return candidates[-1].resolve()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected dict JSON payload in: {path}")
    return payload


def _format_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _build_markdown_report(report: dict[str, Any]) -> str:
    status = "PASS" if bool(report.get("passed", False)) else "FAIL"
    observed = dict(report.get("observed_metrics", {}))
    thresholds = dict(report.get("thresholds", {}))
    failure_lines = [f"- {item}" for item in report.get("failures", [])]
    warning_lines = [f"- {item}" for item in report.get("warnings", [])]
    if not failure_lines:
        failure_lines = ["- None"]
    if not warning_lines:
        warning_lines = ["- None"]

    coverage_ratio = float(observed.get("coverage_ratio", 0.0))
    alt_coverage = dict(observed.get("alternative_feature_coverage", {}))
    alt_schema = observed.get("alternative_data_feature_schema_version")
    latency = dict(observed.get("alignment_latency_seconds", {}))
    endpoint_errors = int(observed.get("endpoint_error_count", 0))

    lines = [
        "# Priority2 Alternative Data Quality + Latency Report",
        f"- Status: **{status}**",
        f"- Created at: `{report.get('created_at_utc')}`",
        f"- Source contract: `{report.get('source_contract_path')}`",
        "## Thresholds",
        f"- Min overall coverage ratio: `{thresholds.get('min_coverage_ratio')}`",
        f"- Min alternative-feature coverage: `{thresholds.get('min_alternative_feature_coverage')}`",
        f"- Max latency p95 seconds: `{thresholds.get('max_latency_p95_seconds')}`",
        f"- Require alternative schema version: `{thresholds.get('require_alternative_schema_version')}`",
        "## Observed metrics",
        f"- Overall coverage ratio: `{coverage_ratio:.6f}` ({_format_pct(coverage_ratio)})",
        f"- Alternative schema version: `{alt_schema}`",
        f"- Alignment latency p95: `{latency.get('p95')}`",
        f"- Endpoint error count: `{endpoint_errors}`",
        "### Alternative-feature coverage",
    ]
    for column in ALTERNATIVE_DATA_FEATURE_COLUMNS:
        value = float(alt_coverage.get(column, 0.0))
        lines.append(f"- `{column}`: `{value:.6f}` ({_format_pct(value)})")

    lines.extend(
        [
            "## Failures",
            *failure_lines,
            "## Warnings",
            *warning_lines,
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description="Validate Priority2 alternative-data feature quality and latency."
    )
    parser.add_argument("--exchange", default=settings.default_exchange)
    parser.add_argument("--symbol", default=settings.default_symbol)
    parser.add_argument("--timeframe", default=settings.default_timeframe)
    parser.add_argument("--contract-path", type=Path, default=None)
    parser.add_argument(
        "--min-coverage-ratio",
        type=float,
        default=float(settings.priority2_quality_min_external_raw_coverage),
    )
    parser.add_argument(
        "--min-alternative-feature-coverage",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--max-latency-p95-seconds",
        type=float,
        default=float(settings.priority2_quality_max_staleness_seconds),
    )
    parser.add_argument(
        "--allow-fallback-mode",
        action="store_true",
        help="Allow fallback_mode != none without failing the report.",
    )
    args = parser.parse_args()

    ensure_data_root_ready(
        settings.quant_data_root,
        allow_unmounted=settings.allow_unmounted_data_root,
    )

    contract_path = _resolve_contract_path(
        quant_data_root=settings.quant_data_root,
        exchange=str(args.exchange),
        symbol=str(args.symbol),
        timeframe=str(args.timeframe),
        explicit_contract_path=args.contract_path,
    )
    payload = _read_json(contract_path)

    coverage_ratio = float(payload.get("coverage_ratio", 0.0))
    column_coverage = dict(payload.get("column_coverage", {}))
    latency_stats = dict(payload.get("alignment_latency_seconds", {}))
    latency_p95 = latency_stats.get("p95")
    fallback_mode = str(payload.get("fallback_mode", "none") or "none")
    alternative_schema = payload.get("alternative_data_feature_schema_version")
    feature_columns = set(payload.get("feature_columns", []))
    endpoint_diagnostics = dict(payload.get("endpoint_diagnostics", {}))
    endpoint_error_count = sum(
        1
        for value in endpoint_diagnostics.values()
        if isinstance(value, dict) and str(value.get("status")) == "error"
    )

    failures: list[str] = []
    warnings: list[str] = []

    if str(payload.get("contract")) != "priority2_external_feature_retrieval.v1":
        failures.append(
            f"unexpected contract value: {payload.get('contract')!r}"
        )
    if not set(PRIORITY2_FEATURE_COLUMNS).issubset(feature_columns):
        missing = sorted(set(PRIORITY2_FEATURE_COLUMNS).difference(feature_columns))
        failures.append(f"missing required Priority2 feature columns: {missing}")

    if coverage_ratio < float(args.min_coverage_ratio):
        failures.append(
            f"coverage_ratio {coverage_ratio:.6f} below minimum {float(args.min_coverage_ratio):.6f}"
        )

    alternative_feature_coverage = {
        column: float(column_coverage.get(column, 0.0))
        for column in ALTERNATIVE_DATA_FEATURE_COLUMNS
    }
    for column, coverage in alternative_feature_coverage.items():
        if coverage < float(args.min_alternative_feature_coverage):
            failures.append(
                f"{column} coverage {coverage:.6f} below minimum {float(args.min_alternative_feature_coverage):.6f}"
            )

    if alternative_schema != ALTERNATIVE_DATA_FEATURE_SCHEMA_VERSION:
        failures.append(
            "alternative_data_feature_schema_version mismatch: "
            f"observed={alternative_schema!r}, expected={ALTERNATIVE_DATA_FEATURE_SCHEMA_VERSION!r}"
        )

    if latency_p95 is None:
        warnings.append("alignment latency p95 is missing")
    elif float(latency_p95) > float(args.max_latency_p95_seconds):
        failures.append(
            f"alignment latency p95 {float(latency_p95):.3f}s exceeds max {float(args.max_latency_p95_seconds):.3f}s"
        )

    if fallback_mode != "none":
        message = f"fallback_mode is {fallback_mode!r}"
        if args.allow_fallback_mode:
            warnings.append(message)
        else:
            failures.append(message)

    if endpoint_error_count > 0:
        warnings.append(f"{endpoint_error_count} endpoint(s) reported retrieval errors")

    report = {
        "contract": "priority2_alternative_data_quality_latency_report.v1",
        "created_at_utc": _utc_now_iso(),
        "passed": len(failures) == 0,
        "source_contract_path": str(contract_path),
        "exchange": payload.get("exchange", args.exchange),
        "symbol": payload.get("symbol", args.symbol),
        "timeframe": payload.get("timeframe", args.timeframe),
        "thresholds": {
            "min_coverage_ratio": float(args.min_coverage_ratio),
            "min_alternative_feature_coverage": float(args.min_alternative_feature_coverage),
            "max_latency_p95_seconds": float(args.max_latency_p95_seconds),
            "require_alternative_schema_version": ALTERNATIVE_DATA_FEATURE_SCHEMA_VERSION,
        },
        "observed_metrics": {
            "coverage_ratio": coverage_ratio,
            "alternative_feature_coverage": alternative_feature_coverage,
            "alternative_data_feature_schema_version": alternative_schema,
            "alignment_latency_seconds": latency_stats,
            "fallback_mode": fallback_mode,
            "endpoint_error_count": int(endpoint_error_count),
        },
        "failures": failures,
        "warnings": warnings,
        "artifacts": {
            "source_contract_path": str(contract_path),
        },
    }

    report_json_path = contract_path.parent / "priority2_alternative_data_quality_latency_report.json"
    report_md_path = contract_path.parent / "priority2_alternative_data_quality_latency_report.md"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md_path.write_text(_build_markdown_report(report), encoding="utf-8")
    report["artifacts"]["report_json_path"] = str(report_json_path)
    report["artifacts"]["report_markdown_path"] = str(report_md_path)
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "PASS" if report["passed"] else "FAIL",
                "report_json_path": str(report_json_path),
                "report_markdown_path": str(report_md_path),
                "coverage_ratio": coverage_ratio,
                "latency_p95_seconds": latency_p95,
                "failure_count": len(failures),
                "warning_count": len(warnings),
            },
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
