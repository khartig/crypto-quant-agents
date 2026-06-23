#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_PROFILES: tuple[str, ...] = ("regime_enabled", "regime_ablated")
REQUIRED_PROFILE_METRICS: tuple[str, ...] = (
    "approval_rate",
    "contradiction_rate",
    "mean_net_total_return",
    "mean_sharpe",
    "mean_max_drawdown",
    "mean_total_cost_return_drag",
)
REQUIRED_BENCHMARK_ARTIFACT_KEYS: tuple[str, ...] = (
    "history_summary_json",
    "history_summary_markdown",
    "history_metric_snapshot_json",
    "history_dataset_manifest_json",
    "history_delta_vs_baseline_json",
    "latest_summary_json",
    "latest_summary_markdown",
    "latest_metric_snapshot_json",
    "latest_dataset_manifest_json",
    "latest_delta_vs_baseline_json",
)
REQUIRED_RESULT_ARTIFACT_KEYS: tuple[str, ...] = (
    "backtest_evaluation",
    "confidence_calibration",
    "risk_decision",
    "run_manifest",
)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _validate_window_config(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"missing window config: {path}"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return [f"window config must be a JSON object: {path}"]
    if str(payload.get("contract", "")).strip() != "regime_window_config.v1":
        errors.append("window config contract must be regime_window_config.v1")
    windows = payload.get("windows")
    if not isinstance(windows, list) or not windows:
        errors.append("window config must include non-empty windows[]")
        return errors
    names: set[str] = set()
    priority_count = 0
    for row in windows:
        if not isinstance(row, dict):
            errors.append("every window entry must be an object")
            continue
        name = str(row.get("name", "")).strip()
        if not name:
            errors.append("window entry missing name")
            continue
        if name in names:
            errors.append(f"duplicate window name: {name}")
        names.add(name)
        if not str(row.get("start_utc", "")).strip():
            errors.append(f"window `{name}` missing start_utc")
        if not str(row.get("end_utc", "")).strip():
            errors.append(f"window `{name}` missing end_utc")
        if bool(row.get("priority", False)):
            priority_count += 1
    if priority_count <= 0:
        errors.append("window config must include at least one priority window")
    minimum_bars = payload.get("minimum_bars")
    if _safe_float(minimum_bars) is None or int(float(minimum_bars)) < 10:
        errors.append("window config minimum_bars must be numeric and >= 10")
    return errors


def _normalize_profile_metrics(raw_profiles: dict[str, Any]) -> tuple[dict[str, dict[str, Any]] | None, list[str]]:
    errors: list[str] = []
    output: dict[str, dict[str, Any]] = {}
    for profile in REQUIRED_PROFILES:
        section = raw_profiles.get(profile)
        if not isinstance(section, dict):
            errors.append(f"missing profile section `{profile}`")
            continue
        approval_rate = _safe_float(section.get("approval_rate"))
        contradiction_rate = _safe_float(section.get("contradiction_rate"))
        mean_net_total_return = _safe_float(
            section.get("mean_net_total_return", section.get("net_total_return"))
        )
        mean_sharpe = _safe_float(section.get("mean_sharpe", section.get("sharpe")))
        mean_max_drawdown = _safe_float(
            section.get("mean_max_drawdown", section.get("max_drawdown"))
        )
        mean_total_cost_return_drag = _safe_float(
            section.get("mean_total_cost_return_drag", section.get("total_cost_return_drag"))
        )
        values = {
            "approval_rate": approval_rate,
            "contradiction_rate": contradiction_rate,
            "mean_net_total_return": mean_net_total_return,
            "mean_sharpe": mean_sharpe,
            "mean_max_drawdown": mean_max_drawdown,
            "mean_total_cost_return_drag": mean_total_cost_return_drag,
        }
        missing = [key for key, value in values.items() if value is None]
        if missing:
            errors.append(
                f"profile `{profile}` missing required metric(s): {', '.join(missing)}"
            )
            continue
        reason_distribution = section.get("reason_code_distribution", {})
        if not isinstance(reason_distribution, dict):
            errors.append(f"profile `{profile}` reason_code_distribution must be an object")
            continue
        output[profile] = {
            **{key: float(value) for key, value in values.items() if value is not None},
            "reason_code_distribution": dict(reason_distribution),
        }
    if errors:
        return None, errors
    return output, []


def _validate_baseline(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"missing baseline file: {path}"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return [f"baseline must be a JSON object: {path}"]
    if str(payload.get("contract", "")).strip() != "regime_benchmark_baseline.v1":
        errors.append("baseline contract must be regime_benchmark_baseline.v1")
    source_summary_path = str(payload.get("source_summary_path", "")).strip()
    if not source_summary_path:
        errors.append("baseline missing source_summary_path")
    raw_profiles = payload.get("profile_metrics")
    if not isinstance(raw_profiles, dict):
        errors.append("baseline missing profile_metrics object")
        return errors
    _, profile_errors = _normalize_profile_metrics(raw_profiles)
    errors.extend(profile_errors)
    return errors


def _validate_pass_fail_doc(path: Path) -> list[str]:
    if not path.exists():
        return [f"missing pass/fail criteria doc: {path}"]
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return [f"pass/fail criteria doc is empty: {path}"]
    required_phrases = (
        "Mandatory artifacts",
        "Required core metrics",
        "Baseline delta checks",
    )
    errors: list[str] = []
    for phrase in required_phrases:
        if phrase not in content:
            errors.append(f"pass/fail criteria doc missing section phrase: {phrase}")
    return errors

def _validate_ablation_matrix(matrix: Any) -> list[str]:
    errors: list[str] = []
    if matrix is None:
        return errors
    if not isinstance(matrix, dict):
        return ["ablation_matrix must be an object when provided"]
    rows = matrix.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("ablation_matrix.rows must be a non-empty array")
        return errors
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"ablation_matrix.rows[{index}] must be an object")
            continue
        if not str(row.get("profile", "")).strip():
            errors.append(f"ablation_matrix.rows[{index}] missing profile")
        touchpoints = row.get("touchpoints")
        if not isinstance(touchpoints, dict):
            errors.append(f"ablation_matrix.rows[{index}] touchpoints must be an object")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            errors.append(f"ablation_matrix.rows[{index}] metrics must be an object")
    return errors


def _validate_cost_decomposition(payload: Any, *, label: str) -> list[str]:
    errors: list[str] = []
    if payload is None:
        return errors
    if not isinstance(payload, dict):
        return [f"{label} must be an object when provided"]
    by_profile_regime = payload.get("by_profile_regime_bucket")
    by_profile_regime_arm = payload.get("by_profile_regime_bucket_arm")
    if not isinstance(by_profile_regime, list):
        errors.append(f"{label}.by_profile_regime_bucket must be an array")
    if not isinstance(by_profile_regime_arm, list):
        errors.append(f"{label}.by_profile_regime_bucket_arm must be an array")
    return errors


def _validate_cost_stress(payload: Any) -> list[str]:
    errors: list[str] = []
    if payload is None:
        return errors
    if not isinstance(payload, dict):
        return ["cost_stress must be an object when provided"]
    scenarios = payload.get("scenarios")
    results = payload.get("results")
    summary = payload.get("summary")
    if not isinstance(scenarios, list) or not scenarios:
        errors.append("cost_stress.scenarios must be a non-empty array")
    if not isinstance(results, list) or not results:
        errors.append("cost_stress.results must be a non-empty array")
    if not isinstance(summary, dict):
        errors.append("cost_stress.summary must be an object")
    else:
        rows = summary.get("rows")
        if not isinstance(rows, list) or not rows:
            errors.append("cost_stress.summary.rows must be a non-empty array")
    errors.extend(
        _validate_cost_decomposition(
            payload.get("cost_decomposition"),
            label="cost_stress.cost_decomposition",
        )
    )
    return errors


def _validate_summary(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"summary file does not exist: {path}"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return [f"summary must be a JSON object: {path}"]
    profile_summary = payload.get("profile_summary")
    if not isinstance(profile_summary, dict):
        return ["summary missing profile_summary object"]
    _, profile_errors = _normalize_profile_metrics(profile_summary)
    errors.extend(profile_errors)
    for profile in REQUIRED_PROFILES:
        section = profile_summary.get(profile)
        if not isinstance(section, dict):
            continue
        directional_rate = _safe_float(section.get("directional_contradiction_rate"))
        quality_rate = _safe_float(section.get("quality_contradiction_rate"))
        if directional_rate is None:
            errors.append(f"profile `{profile}` missing directional_contradiction_rate")
        if quality_rate is None:
            errors.append(f"profile `{profile}` missing quality_contradiction_rate")

    window_comparison = payload.get("window_comparison")
    if not isinstance(window_comparison, list) or not window_comparison:
        errors.append("summary must include non-empty window_comparison[]")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        errors.append("summary must include non-empty results[]")
    else:
        for index, row in enumerate(results):
            if not isinstance(row, dict):
                errors.append(f"results[{index}] must be an object")
                continue
            if _safe_float(row.get("directional_contradiction_rate")) is None:
                errors.append(
                    f"results[{index}] missing directional_contradiction_rate"
                )
            if _safe_float(row.get("quality_contradiction_rate")) is None:
                errors.append(
                    f"results[{index}] missing quality_contradiction_rate"
                )
            if _safe_float(row.get("cost_pressure_score")) is None:
                errors.append(f"results[{index}] missing cost_pressure_score")
            regime_bucket = str(row.get("regime_bucket", "")).strip()
            if not regime_bucket:
                errors.append(f"results[{index}] missing regime_bucket")
            arm_cost = row.get("arm_cost_return_drag")
            if not isinstance(arm_cost, dict):
                errors.append(
                    f"results[{index}] arm_cost_return_drag must be an object"
                )
            artifacts = row.get("artifacts")
            if not isinstance(artifacts, dict):
                errors.append(f"results[{index}] artifacts must be an object")
            else:
                for key in REQUIRED_RESULT_ARTIFACT_KEYS:
                    value = artifacts.get(key)
                    if not isinstance(value, str) or not str(value).strip():
                        errors.append(
                            f"results[{index}] artifacts missing required key `{key}`"
                        )

    benchmark_artifacts = payload.get("benchmark_artifacts")
    if benchmark_artifacts is not None:
        if not isinstance(benchmark_artifacts, dict):
            errors.append("summary benchmark_artifacts must be an object when provided")
        else:
            for key in REQUIRED_BENCHMARK_ARTIFACT_KEYS:
                value = benchmark_artifacts.get(key)
                if not isinstance(value, str) or not str(value).strip():
                    errors.append(
                        f"summary benchmark_artifacts missing required key `{key}`"
                    )
    gate_outcome = payload.get("gate_outcome")
    if gate_outcome is not None:
        if not isinstance(gate_outcome, dict):
            errors.append("summary gate_outcome must be an object when provided")
        else:
            status = str(gate_outcome.get("status", "")).strip().lower()
            if status not in {"pass", "fail"}:
                errors.append("summary gate_outcome.status must be `pass` or `fail`")
            checks = gate_outcome.get("checks")
            if not isinstance(checks, list) or not checks:
                errors.append("summary gate_outcome.checks must be a non-empty array")
    errors.extend(_validate_ablation_matrix(payload.get("ablation_matrix")))
    errors.extend(
        _validate_cost_decomposition(
            payload.get("cost_decomposition"),
            label="cost_decomposition",
        )
    )
    errors.extend(_validate_cost_stress(payload.get("cost_stress")))
    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Presubmit validator for Priority 0 regime benchmark gate artifacts "
            "(window config, baseline snapshot, pass/fail criteria, optional summary)."
        )
    )
    parser.add_argument(
        "--window-config",
        default="scripts/regime_window_slices.json",
        help="Canonical window config path.",
    )
    parser.add_argument(
        "--baseline",
        default="doc/REGIME_BENCHMARK_BASELINE.json",
        help="Accepted baseline snapshot path.",
    )
    parser.add_argument(
        "--pass-fail-doc",
        default="doc/REGIME_BENCHMARK_PASS_FAIL_CRITERIA.md",
        help="Pass/fail criteria documentation path.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional benchmark summary JSON path to validate.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    window_config_path = Path(args.window_config).expanduser().resolve()
    baseline_path = Path(args.baseline).expanduser().resolve()
    pass_fail_doc_path = Path(args.pass_fail_doc).expanduser().resolve()
    summary_path = Path(args.summary_json).expanduser().resolve() if args.summary_json else None

    failures: list[str] = []
    failures.extend(_validate_window_config(window_config_path))
    failures.extend(_validate_baseline(baseline_path))
    failures.extend(_validate_pass_fail_doc(pass_fail_doc_path))
    if summary_path is not None:
        failures.extend(_validate_summary(summary_path))

    if failures:
        print("REGIME_BENCHMARK_GATE_VALIDATION_STATUS=FAIL")
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    print("REGIME_BENCHMARK_GATE_VALIDATION_STATUS=PASS")
    print(f"WINDOW_CONFIG={window_config_path}")
    print(f"BASELINE={baseline_path}")
    print(f"PASS_FAIL_DOC={pass_fail_doc_path}")
    if summary_path is not None:
        print(f"SUMMARY={summary_path}")


if __name__ == "__main__":
    main()
