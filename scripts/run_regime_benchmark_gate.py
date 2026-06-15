#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_agents.config import load_settings
from quant_agents.storage import symbol_slug

REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")
REQUIRED_PROFILES: tuple[str, ...] = ("regime_enabled", "regime_ablated")
REQUIRED_WINDOW_METRICS: tuple[str, ...] = (
    "approval_rate",
    "contradiction_rate",
    "net_total_return",
    "sharpe",
    "max_drawdown",
    "total_cost_return_drag",
)
REQUIRED_PROFILE_METRICS: tuple[str, ...] = (
    "approval_rate",
    "contradiction_rate",
    "mean_net_total_return",
    "mean_sharpe",
    "mean_max_drawdown",
    "mean_total_cost_return_drag",
)
REQUIRED_RESULT_ARTIFACT_KEYS: tuple[str, ...] = (
    "backtest_evaluation",
    "confidence_calibration",
    "risk_decision",
    "run_manifest",
)


@dataclass(frozen=True)
class WindowSpec:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    priority: bool


def _parse_timestamp(raw: str, *, now_utc: pd.Timestamp) -> pd.Timestamp:
    value = str(raw).strip()
    if value.lower() == "now":
        return now_utc
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_json(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _collect_scope_parquet_files(root: Path, exchange: str, symbol: str, timeframe: str) -> list[Path]:
    base = (
        root
        / "raw"
        / f"exchange={exchange}"
        / f"symbol={symbol_slug(symbol)}"
        / f"interval={timeframe}"
    )
    if not base.exists():
        raise FileNotFoundError(f"No raw dataset scope found: {base}")
    files = sorted(base.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {base}")
    return files


def _load_market_frame(source_files: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in source_files:
        frame = pd.read_parquet(path, columns=list(REQUIRED_COLUMNS))
        frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"])
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No market rows were loaded from source parquet file(s).")
    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return merged.reset_index(drop=True)


def _slice_window(frame: pd.DataFrame, window: WindowSpec) -> pd.DataFrame:
    scoped = frame.loc[(frame["timestamp"] >= window.start) & (frame["timestamp"] < window.end)].copy()
    scoped = scoped.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return scoped.reset_index(drop=True)


def _load_window_config(path: Path, *, now_utc: pd.Timestamp) -> tuple[list[WindowSpec], int | None, str]:
    if not path.exists():
        raise FileNotFoundError(f"Window config file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Window config payload must be a JSON object.")
    if str(payload.get("contract", "")).strip() != "regime_window_config.v1":
        raise ValueError("Window config contract must be regime_window_config.v1.")

    raw_windows = payload.get("windows")
    if not isinstance(raw_windows, list) or not raw_windows:
        raise ValueError("Window config must define a non-empty `windows` array.")

    windows: list[WindowSpec] = []
    names: set[str] = set()
    for item in raw_windows:
        if not isinstance(item, dict):
            raise ValueError("Each window definition must be a JSON object.")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("Window definition is missing `name`.")
        if name in names:
            raise ValueError(f"Duplicate window name detected: {name}")
        names.add(name)
        start_raw = item.get("start_utc")
        end_raw = item.get("end_utc")
        if start_raw is None or end_raw is None:
            raise ValueError(f"Window `{name}` must include start_utc and end_utc.")
        start = _parse_timestamp(str(start_raw), now_utc=now_utc)
        end = _parse_timestamp(str(end_raw), now_utc=now_utc)
        if not start < end:
            raise ValueError(f"Window `{name}` has invalid range: start must be < end.")
        windows.append(
            WindowSpec(
                name=name,
                start=start,
                end=end,
                priority=bool(item.get("priority", True)),
            )
        )
    windows = sorted(windows, key=lambda item: item.start)

    minimum_bars_raw = payload.get("minimum_bars")
    minimum_bars = None
    if minimum_bars_raw is not None:
        minimum_bars = max(10, int(minimum_bars_raw))

    return windows, minimum_bars, _sha256_file(path)


def _build_dataset_manifest(
    *,
    source_files: list[Path],
    market_frame: pd.DataFrame,
    windows: list[WindowSpec],
    minimum_bars: int,
    exchange: str,
    symbol: str,
    timeframe: str,
    source_mode: str,
) -> dict[str, Any]:
    source_entries: list[dict[str, Any]] = []
    for path in sorted(source_files):
        stat = path.stat()
        source_entries.append(
            {
                "path": str(path),
                "size_bytes": int(stat.st_size),
                "sha256": _sha256_file(path),
            }
        )
    source_entries = sorted(source_entries, key=lambda item: item["path"])
    data_start = pd.Timestamp(market_frame["timestamp"].iloc[0]).isoformat()
    data_end = pd.Timestamp(market_frame["timestamp"].iloc[-1]).isoformat()
    coverage: list[dict[str, Any]] = []
    for window in windows:
        scoped = _slice_window(market_frame, window)
        bar_count = int(len(scoped))
        coverage.append(
            {
                "name": window.name,
                "priority": bool(window.priority),
                "start_utc": window.start.isoformat(),
                "end_utc": window.end.isoformat(),
                "available_bars": bar_count,
                "required_minimum_bars": minimum_bars,
                "meets_minimum_bars": bool(bar_count >= minimum_bars),
            }
        )

    dataset_fingerprint_payload = {
        "scope": {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "source_mode": source_mode,
        },
        "source_entries": source_entries,
        "data_start_utc": data_start,
        "data_end_utc": data_end,
        "row_count": int(len(market_frame)),
    }
    dataset_sha256 = _hash_json(dataset_fingerprint_payload)
    return {
        "contract": "regime_canonical_dataset_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "source_mode": source_mode,
        },
        "row_count": int(len(market_frame)),
        "data_start_utc": data_start,
        "data_end_utc": data_end,
        "source_file_count": len(source_entries),
        "source_files": source_entries,
        "dataset_sha256": dataset_sha256,
        "window_coverage": coverage,
    }


def _compute_benchmark_id(
    *,
    dataset_sha256: str,
    window_config_sha256: str,
    exchange: str,
    symbol: str,
    timeframe: str,
    minimum_bars: int,
    strategy_model: str,
    ops_model: str,
    regime_min_confidence: float,
) -> str:
    payload = {
        "dataset_sha256": dataset_sha256,
        "window_config_sha256": window_config_sha256,
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "minimum_bars": minimum_bars,
        "strategy_model": strategy_model,
        "ops_model": ops_model,
        "regime_min_confidence": regime_min_confidence,
    }
    digest = _hash_json(payload)[:16]
    return f"regime-benchmark-{digest}"


def _new_history_dir(benchmark_root: Path) -> tuple[str, Path]:
    history_root = benchmark_root / "history"
    history_root.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    base_id = f"{now:%Y%m%dT%H%M%SZ}"
    for suffix in range(100):
        run_id = base_id if suffix == 0 else f"{base_id}_{suffix:02d}"
        candidate = history_root / run_id
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return run_id, candidate
    raise RuntimeError(f"Unable to allocate history directory under {history_root}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _validate_summary_contract(summary: dict[str, Any]) -> None:
    profile_summary = summary.get("profile_summary")
    if not isinstance(profile_summary, dict):
        raise RuntimeError("Summary payload missing profile_summary object.")
    for profile in REQUIRED_PROFILES:
        section = profile_summary.get(profile)
        if not isinstance(section, dict):
            raise RuntimeError(f"profile_summary is missing section for profile `{profile}`.")
        for metric in REQUIRED_PROFILE_METRICS:
            value = _safe_float(section.get(metric))
            if value is None:
                raise RuntimeError(
                    f"profile_summary[{profile}] is missing required metric `{metric}`."
                )
        reason_distribution = section.get("reason_code_distribution")
        if reason_distribution is not None and not isinstance(reason_distribution, dict):
            raise RuntimeError(
                f"profile_summary[{profile}].reason_code_distribution must be a JSON object."
            )
        directional_contradiction_rate = _safe_float(
            section.get("directional_contradiction_rate")
        )
        quality_contradiction_rate = _safe_float(section.get("quality_contradiction_rate"))
        mean_cost_pressure_score = _safe_float(section.get("mean_cost_pressure_score"))
        if directional_contradiction_rate is None:
            raise RuntimeError(
                f"profile_summary[{profile}] missing directional_contradiction_rate."
            )
        if quality_contradiction_rate is None:
            raise RuntimeError(
                f"profile_summary[{profile}] missing quality_contradiction_rate."
            )
        if mean_cost_pressure_score is None:
            raise RuntimeError(
                f"profile_summary[{profile}] missing mean_cost_pressure_score."
            )

    results = summary.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError("Summary payload must include non-empty results.")
    for row in results:
        if not isinstance(row, dict):
            raise RuntimeError("Every result row must be a JSON object.")
        profile = str(row.get("profile", ""))
        if profile not in REQUIRED_PROFILES:
            raise RuntimeError(f"Unexpected profile in results row: {profile}")
        for metric in REQUIRED_WINDOW_METRICS:
            value = _safe_float(row.get(metric))
            if value is None:
                raise RuntimeError(
                    f"Result row for profile `{profile}` is missing required metric `{metric}`."
                )
        for metric in (
            "directional_contradiction_rate",
            "quality_contradiction_rate",
            "cost_pressure_score",
        ):
            value = _safe_float(row.get(metric))
            if value is None:
                raise RuntimeError(
                    f"Result row for profile `{profile}` is missing required metric `{metric}`."
                )
        regime_bucket = str(row.get("regime_bucket", "")).strip()
        if not regime_bucket:
            raise RuntimeError(f"Result row for profile `{profile}` missing regime_bucket.")
        arm_cost = row.get("arm_cost_return_drag")
        if not isinstance(arm_cost, dict):
            raise RuntimeError(
                f"Result row for profile `{profile}` missing arm_cost_return_drag object."
            )
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, dict):
            raise RuntimeError("Result row is missing artifacts object.")
        for key in REQUIRED_RESULT_ARTIFACT_KEYS:
            if not isinstance(artifacts.get(key), str) or not str(artifacts.get(key)).strip():
                raise RuntimeError(f"Result row missing required artifact pointer `{key}`.")

    window_comparison = summary.get("window_comparison")
    if not isinstance(window_comparison, list) or not window_comparison:
        raise RuntimeError("Summary payload must include non-empty window_comparison.")
    for row in window_comparison:
        if not isinstance(row, dict):
            raise RuntimeError("Every window_comparison row must be a JSON object.")
        for profile in REQUIRED_PROFILES:
            section = row.get(profile)
            if not isinstance(section, dict):
                raise RuntimeError(f"window_comparison missing `{profile}` section.")
            for metric in REQUIRED_WINDOW_METRICS:
                value = _safe_float(section.get(metric))
                if value is None:
                    raise RuntimeError(
                        f"window_comparison `{profile}` missing required metric `{metric}`."
                    )

    ablation_matrix = summary.get("ablation_matrix")
    if ablation_matrix is None or not isinstance(ablation_matrix, dict):
        raise RuntimeError("Summary payload must include ablation_matrix object.")
    matrix_rows = ablation_matrix.get("rows")
    if not isinstance(matrix_rows, list) or not matrix_rows:
        raise RuntimeError("ablation_matrix.rows must be a non-empty array.")

    cost_decomposition = summary.get("cost_decomposition")
    if cost_decomposition is None or not isinstance(cost_decomposition, dict):
        raise RuntimeError("Summary payload must include cost_decomposition object.")
    if not isinstance(cost_decomposition.get("by_profile_regime_bucket"), list):
        raise RuntimeError(
            "cost_decomposition.by_profile_regime_bucket must be an array."
        )
    if not isinstance(cost_decomposition.get("by_profile_regime_bucket_arm"), list):
        raise RuntimeError(
            "cost_decomposition.by_profile_regime_bucket_arm must be an array."
        )


def _normalize_profile_metrics(raw_profiles: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for profile in REQUIRED_PROFILES:
        section = raw_profiles.get(profile)
        if not isinstance(section, dict):
            raise RuntimeError(f"Baseline missing profile section `{profile}`.")
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
            raise RuntimeError(
                f"Baseline profile `{profile}` is missing required metrics: {', '.join(missing)}"
            )
        reason_distribution_raw = section.get("reason_code_distribution", {})
        reason_distribution = (
            dict(reason_distribution_raw) if isinstance(reason_distribution_raw, dict) else {}
        )
        normalized[profile] = {
            **{key: float(value) for key, value in values.items() if value is not None},
            "reason_code_distribution": reason_distribution,
        }
    return normalized


def _extract_current_profile_metrics(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profile_summary = dict(summary.get("profile_summary", {}))
    return _normalize_profile_metrics(profile_summary)


def _load_baseline(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Baseline file is not a JSON object: {path}")

    if str(payload.get("contract", "")).strip() == "regime_benchmark_baseline.v1":
        profiles_raw = payload.get("profile_metrics")
        if not isinstance(profiles_raw, dict):
            raise RuntimeError("Baseline contract regime_benchmark_baseline.v1 requires profile_metrics.")
    else:
        profiles_raw = payload.get("profile_summary")
        if not isinstance(profiles_raw, dict):
            raise RuntimeError(
                "Baseline file must either be regime_benchmark_baseline.v1 or include profile_summary."
            )

    return {
        "path": str(path),
        "payload": payload,
        "profile_metrics": _normalize_profile_metrics(profiles_raw),
    }


def _reason_distribution_to_share_map(distribution: dict[str, Any]) -> dict[str, float]:
    normalized_counts: dict[str, float] = {}
    for code, raw in distribution.items():
        value = _safe_float(raw)
        if value is None:
            continue
        normalized_counts[str(code)] = max(0.0, float(value))
    total = sum(normalized_counts.values())
    if total <= 0.0:
        return {}
    return {code: (count / total) for code, count in normalized_counts.items()}


def _reason_code_drift(
    current_distribution: dict[str, Any],
    baseline_distribution: dict[str, Any],
    *,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    current_share = _reason_distribution_to_share_map(current_distribution)
    baseline_share = _reason_distribution_to_share_map(baseline_distribution)
    codes = sorted(set(current_share) | set(baseline_share))
    rows: list[dict[str, Any]] = []
    for code in codes:
        current_value = float(current_share.get(code, 0.0))
        baseline_value = float(baseline_share.get(code, 0.0))
        rows.append(
            {
                "reason_code": code,
                "current_share": current_value,
                "baseline_share": baseline_value,
                "delta_share": current_value - baseline_value,
            }
        )
    rows.sort(key=lambda item: (-abs(float(item["delta_share"])), item["reason_code"]))
    return rows[:top_n]


def _regime_minus_ablated(profile_metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    enabled = profile_metrics["regime_enabled"]
    ablated = profile_metrics["regime_ablated"]
    return {
        "approval_rate": float(enabled["approval_rate"]) - float(ablated["approval_rate"]),
        "contradiction_rate": float(enabled["contradiction_rate"]) - float(ablated["contradiction_rate"]),
        "mean_net_total_return": float(enabled["mean_net_total_return"]) - float(ablated["mean_net_total_return"]),
        "mean_sharpe": float(enabled["mean_sharpe"]) - float(ablated["mean_sharpe"]),
        "mean_max_drawdown": float(enabled["mean_max_drawdown"]) - float(ablated["mean_max_drawdown"]),
        "mean_total_cost_return_drag": float(enabled["mean_total_cost_return_drag"])
        - float(ablated["mean_total_cost_return_drag"]),
    }


def _compute_baseline_delta(
    *,
    current_metrics: dict[str, dict[str, Any]],
    baseline_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    profile_deltas: dict[str, dict[str, float]] = {}
    reason_code_drift: dict[str, list[dict[str, Any]]] = {}
    for profile in REQUIRED_PROFILES:
        current = current_metrics[profile]
        baseline = baseline_metrics[profile]
        profile_deltas[profile] = {
            "approval_rate": float(current["approval_rate"]) - float(baseline["approval_rate"]),
            "contradiction_rate": float(current["contradiction_rate"])
            - float(baseline["contradiction_rate"]),
            "mean_net_total_return": float(current["mean_net_total_return"])
            - float(baseline["mean_net_total_return"]),
            "mean_sharpe": float(current["mean_sharpe"]) - float(baseline["mean_sharpe"]),
            "mean_max_drawdown": float(current["mean_max_drawdown"])
            - float(baseline["mean_max_drawdown"]),
            "mean_total_cost_return_drag": float(current["mean_total_cost_return_drag"])
            - float(baseline["mean_total_cost_return_drag"]),
        }
        reason_code_drift[profile] = _reason_code_drift(
            dict(current.get("reason_code_distribution", {})),
            dict(baseline.get("reason_code_distribution", {})),
            top_n=10,
        )

    current_regime_minus_ablated = _regime_minus_ablated(current_metrics)
    baseline_regime_minus_ablated = _regime_minus_ablated(baseline_metrics)
    comparative_delta = {
        key: float(current_regime_minus_ablated[key]) - float(baseline_regime_minus_ablated[key])
        for key in current_regime_minus_ablated
    }
    return {
        "profile_deltas": profile_deltas,
        "reason_code_drift_top_changes": reason_code_drift,
        "regime_minus_ablated_current": current_regime_minus_ablated,
        "regime_minus_ablated_baseline": baseline_regime_minus_ablated,
        "regime_minus_ablated_delta": comparative_delta,
    }


def _build_gate_outcome(
    *,
    baseline_delta: dict[str, Any] | None,
    skipped_priority_window_count: int,
    require_baseline: bool,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = [
        {
            "name": "no_skipped_priority_windows",
            "pass": skipped_priority_window_count == 0,
            "observed": skipped_priority_window_count,
            "expected": "0",
        }
    ]
    if baseline_delta is None:
        checks.append(
            {
                "name": "baseline_available",
                "pass": not require_baseline,
                "observed": "missing",
                "expected": "present",
            }
        )
    else:
        checks.append(
            {
                "name": "baseline_available",
                "pass": True,
                "observed": "present",
                "expected": "present",
            }
        )
        enabled = dict(baseline_delta["profile_deltas"]["regime_enabled"])
        checks.extend(
            [
                {
                    "name": "net_total_return_delta_non_negative",
                    "pass": float(enabled["mean_net_total_return"]) >= 0.0,
                    "observed": float(enabled["mean_net_total_return"]),
                    "expected": ">= 0.0",
                },
                {
                    "name": "max_drawdown_delta_non_negative",
                    "pass": float(enabled["mean_max_drawdown"]) >= 0.0,
                    "observed": float(enabled["mean_max_drawdown"]),
                    "expected": ">= 0.0",
                },
                {
                    "name": "cost_drag_delta_non_positive",
                    "pass": float(enabled["mean_total_cost_return_drag"]) <= 0.0,
                    "observed": float(enabled["mean_total_cost_return_drag"]),
                    "expected": "<= 0.0",
                },
                {
                    "name": "contradiction_rate_delta_within_limit",
                    "pass": float(enabled["contradiction_rate"]) <= 0.02,
                    "observed": float(enabled["contradiction_rate"]),
                    "expected": "<= 0.02",
                },
                {
                    "name": "approval_rate_delta_within_limit",
                    "pass": float(enabled["approval_rate"]) >= -0.05,
                    "observed": float(enabled["approval_rate"]),
                    "expected": ">= -0.05",
                },
            ]
        )
    return {
        "status": "pass" if all(bool(check["pass"]) for check in checks) else "fail",
        "checks": checks,
    }


def _build_markdown_summary(
    *,
    benchmark_id: str,
    summary: dict[str, Any],
    dataset_manifest: dict[str, Any],
    window_config_path: Path,
    baseline_path: Path | None,
    baseline_delta: dict[str, Any] | None,
    gate_outcome: dict[str, Any],
    artifact_paths: dict[str, str],
) -> str:
    lines: list[str] = [
        "# Regime Benchmark Gate Summary",
        f"- benchmark_id: `{benchmark_id}`",
        f"- created_at_utc: `{datetime.now(timezone.utc).isoformat()}`",
        f"- gate_status: `{str(gate_outcome.get('status', 'fail')).upper()}`",
        f"- canonical_window_config: `{window_config_path}`",
    ]
    if baseline_path is not None:
        lines.append(f"- baseline_reference: `{baseline_path}`")

    lines.extend(
        [
            "",
            "## Dataset manifest",
            f"- dataset_sha256: `{dataset_manifest['dataset_sha256']}`",
            f"- source_file_count: `{dataset_manifest['source_file_count']}`",
            f"- row_count: `{dataset_manifest['row_count']}`",
            f"- data_start_utc: `{dataset_manifest['data_start_utc']}`",
            f"- data_end_utc: `{dataset_manifest['data_end_utc']}`",
            "",
            "## Window coverage",
        ]
    )
    for row in dataset_manifest.get("window_coverage", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            "- "
            + f"`{row.get('name')}` "
            + f"(priority={bool(row.get('priority'))}) "
            + f"bars={row.get('available_bars')}/{row.get('required_minimum_bars')} "
            + f"covered={bool(row.get('meets_minimum_bars'))}"
        )

    profile_summary = dict(summary.get("profile_summary", {}))
    lines.append("")
    lines.append("## Aggregate profile metrics")
    for profile in REQUIRED_PROFILES:
        section = dict(profile_summary.get(profile, {}))
        lines.extend(
            [
                f"### {profile}",
                f"- approval_rate: `{section.get('approval_rate')}`",
                f"- contradiction_rate: `{section.get('contradiction_rate')}`",
                f"- mean_net_total_return: `{section.get('mean_net_total_return')}`",
                f"- mean_sharpe: `{section.get('mean_sharpe')}`",
                f"- mean_max_drawdown: `{section.get('mean_max_drawdown')}`",
                f"- mean_total_cost_return_drag: `{section.get('mean_total_cost_return_drag')}`",
            ]
        )

    lines.append("")
    lines.append("## Baseline deltas")
    if baseline_delta is None:
        lines.append("- baseline delta unavailable for this run.")
    else:
        for profile in REQUIRED_PROFILES:
            delta_row = dict(baseline_delta["profile_deltas"][profile])
            lines.extend(
                [
                    f"### {profile} (current - baseline)",
                    f"- approval_rate: `{delta_row['approval_rate']}`",
                    f"- contradiction_rate: `{delta_row['contradiction_rate']}`",
                    f"- mean_net_total_return: `{delta_row['mean_net_total_return']}`",
                    f"- mean_sharpe: `{delta_row['mean_sharpe']}`",
                    f"- mean_max_drawdown: `{delta_row['mean_max_drawdown']}`",
                    f"- mean_total_cost_return_drag: `{delta_row['mean_total_cost_return_drag']}`",
                ]
            )
            drift = baseline_delta["reason_code_drift_top_changes"].get(profile, [])
            if drift:
                lines.append("- top_reason_code_drift:")
                for item in drift:
                    lines.append(
                        "  - "
                        + f"`{item['reason_code']}` "
                        + f"delta_share={item['delta_share']:.6f} "
                        + f"(current={item['current_share']:.6f}, baseline={item['baseline_share']:.6f})"
                    )

    lines.append("")
    lines.append("## Gate checks")
    for check in gate_outcome.get("checks", []):
        if not isinstance(check, dict):
            continue
        status = "PASS" if bool(check.get("pass")) else "FAIL"
        lines.append(
            "- "
            + f"[{status}] `{check.get('name')}` "
            + f"(observed={check.get('observed')}, expected={check.get('expected')})"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            f"- history_summary_json: `{artifact_paths['history_summary_json']}`",
            f"- history_summary_markdown: `{artifact_paths['history_summary_markdown']}`",
            f"- history_metric_snapshot_json: `{artifact_paths['history_metric_snapshot_json']}`",
            f"- history_dataset_manifest_json: `{artifact_paths['history_dataset_manifest_json']}`",
            f"- latest_summary_json: `{artifact_paths['latest_summary_json']}`",
            f"- latest_summary_markdown: `{artifact_paths['latest_summary_markdown']}`",
            f"- latest_metric_snapshot_json: `{artifact_paths['latest_metric_snapshot_json']}`",
            f"- latest_dataset_manifest_json: `{artifact_paths['latest_dataset_manifest_json']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _run_segmented_evaluation(
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    windows: list[WindowSpec],
    minimum_bars: int,
    output_json: Path,
    input_file: Path | None,
    step_retries: int,
    strategy_model: str | None,
    ops_model: str | None,
    regime_min_confidence: float | None,
) -> None:
    evaluate_script = Path(__file__).resolve().with_name("evaluate_agent_regime_windows.py")
    if not evaluate_script.exists():
        raise FileNotFoundError(f"Unable to locate evaluation script: {evaluate_script}")
    command: list[str] = [
        sys.executable,
        str(evaluate_script),
        "--exchange",
        exchange,
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--minimum-bars",
        str(max(10, int(minimum_bars))),
        "--step-retries",
        str(max(0, int(step_retries))),
        "--output-json",
        str(output_json),
        "--fail-on-insufficient-window",
    ]
    if input_file is not None:
        command.extend(["--input-file", str(input_file)])
    if strategy_model is not None:
        command.extend(["--strategy-model", strategy_model])
    if ops_model is not None:
        command.extend(["--ops-model", ops_model])
    if regime_min_confidence is not None:
        command.extend(["--regime-min-confidence", str(float(regime_min_confidence))])
    for window in windows:
        command.extend(
            [
                "--window",
                f"{window.name},{window.start.isoformat()},{window.end.isoformat()}",
            ]
        )
    subprocess.run(command, check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Priority 0 single-command benchmark gate: validates canonical data coverage, "
            "runs regime_enabled vs regime_ablated segmented evaluation, and writes deterministic "
            "JSON/markdown artifacts plus baseline deltas."
        )
    )
    parser.add_argument("--exchange", default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument(
        "--window-config",
        default=str(Path(__file__).resolve().with_name("regime_window_slices.json")),
        help="Canonical window config JSON (regime_window_config.v1).",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Optional explicit parquet input file. If omitted, merged scope raw files are used.",
    )
    parser.add_argument(
        "--minimum-bars",
        type=int,
        default=None,
        help="Override minimum bars per window. Defaults to window config minimum_bars or settings.",
    )
    parser.add_argument(
        "--step-retries",
        type=int,
        default=0,
        help="Retries per agent-plane step during segmented evaluation (default: 0).",
    )
    parser.add_argument("--strategy-model", default=None)
    parser.add_argument("--ops-model", default=None)
    parser.add_argument("--regime-min-confidence", type=float, default=None)
    parser.add_argument(
        "--baseline",
        default="doc/REGIME_BENCHMARK_BASELINE.json",
        help="Baseline JSON (baseline.v1 or summary with profile_summary).",
    )
    parser.add_argument(
        "--require-baseline",
        dest="require_baseline",
        action="store_true",
        help="Fail if baseline file is missing.",
    )
    parser.add_argument(
        "--no-require-baseline",
        dest="require_baseline",
        action="store_false",
        help="Allow runs without baseline deltas.",
    )
    parser.set_defaults(require_baseline=True)
    parser.add_argument(
        "--accept-as-baseline",
        action="store_true",
        help="Overwrite the baseline file with the current run metrics after successful execution.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root directory for benchmark artifacts. Defaults under QUANT_DATA_ROOT/logs/analysis/regime-benchmark-gate.",
    )
    parser.add_argument(
        "--benchmark-id",
        default=None,
        help="Optional benchmark id override. If omitted, derived deterministically from dataset+config.",
    )
    parser.add_argument(
        "--fail-on-gate-fail",
        dest="fail_on_gate_fail",
        action="store_true",
        help="Exit non-zero when gate checks fail.",
    )
    parser.add_argument(
        "--no-fail-on-gate-fail",
        dest="fail_on_gate_fail",
        action="store_false",
        help="Always exit zero after artifact generation, even on gate check failures.",
    )
    parser.set_defaults(fail_on_gate_fail=True)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    settings = load_settings()
    exchange = args.exchange or settings.default_exchange
    symbol = args.symbol or settings.default_symbol
    timeframe = args.timeframe or settings.default_timeframe
    now_utc = pd.Timestamp.now(tz="UTC")

    window_config_path = Path(args.window_config).expanduser().resolve()
    windows, config_minimum_bars, window_config_sha256 = _load_window_config(
        window_config_path,
        now_utc=now_utc,
    )

    minimum_bars = max(
        10,
        int(
            args.minimum_bars
            if args.minimum_bars is not None
            else (config_minimum_bars if config_minimum_bars is not None else settings.agent_minimum_bars)
        ),
    )
    input_file = Path(args.input_file).expanduser().resolve() if args.input_file else None
    source_files = (
        [input_file]
        if input_file is not None
        else _collect_scope_parquet_files(
            root=settings.quant_data_root,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
        )
    )
    source_mode = "single_input_file" if input_file is not None else "merged_scope_raw_files"
    market_frame = _load_market_frame(source_files)

    dataset_manifest = _build_dataset_manifest(
        source_files=source_files,
        market_frame=market_frame,
        windows=windows,
        minimum_bars=minimum_bars,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        source_mode=source_mode,
    )

    missing_priority_windows = [
        row
        for row in dataset_manifest["window_coverage"]
        if bool(row["priority"]) and not bool(row["meets_minimum_bars"])
    ]
    if missing_priority_windows:
        details = ", ".join(
            f"{row['name']}({row['available_bars']}/{row['required_minimum_bars']})"
            for row in missing_priority_windows
        )
        raise RuntimeError(
            "Priority window coverage check failed before segmented evaluation: "
            + details
        )

    strategy_model = args.strategy_model or settings.ollama_strategy_model
    ops_model = args.ops_model or settings.ollama_ops_model
    regime_min_confidence = (
        float(args.regime_min_confidence)
        if args.regime_min_confidence is not None
        else float(settings.risk_min_regime_confidence)
    )
    benchmark_id = args.benchmark_id or _compute_benchmark_id(
        dataset_sha256=dataset_manifest["dataset_sha256"],
        window_config_sha256=window_config_sha256,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        minimum_bars=minimum_bars,
        strategy_model=strategy_model,
        ops_model=ops_model,
        regime_min_confidence=regime_min_confidence,
    )
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else settings.quant_data_root / "logs" / "analysis" / "regime-benchmark-gate"
    )
    benchmark_root = output_root / benchmark_id
    benchmark_root.mkdir(parents=True, exist_ok=True)
    run_id, history_dir = _new_history_dir(benchmark_root)

    history_manifest_path = history_dir / "dataset_manifest.json"
    latest_manifest_path = benchmark_root / "latest_dataset_manifest.json"
    _write_json(history_manifest_path, dataset_manifest)
    _write_json(latest_manifest_path, dataset_manifest)

    history_summary_json = history_dir / "summary.json"
    _run_segmented_evaluation(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        windows=windows,
        minimum_bars=minimum_bars,
        output_json=history_summary_json,
        input_file=input_file,
        step_retries=max(0, int(args.step_retries)),
        strategy_model=args.strategy_model,
        ops_model=args.ops_model,
        regime_min_confidence=args.regime_min_confidence,
    )

    summary = json.loads(history_summary_json.read_text(encoding="utf-8"))
    _validate_summary_contract(summary)
    current_metrics = _extract_current_profile_metrics(summary)

    baseline_path = Path(args.baseline).expanduser().resolve() if args.baseline else None
    baseline = _load_baseline(baseline_path)
    if bool(args.require_baseline) and baseline is None:
        raise RuntimeError(f"Baseline is required but missing or invalid: {baseline_path}")

    baseline_delta = (
        _compute_baseline_delta(
            current_metrics=current_metrics,
            baseline_metrics=baseline["profile_metrics"],
        )
        if baseline is not None
        else None
    )

    priority_names = {window.name for window in windows if window.priority}
    skipped_windows = list(summary.get("skipped_windows", []))
    skipped_priority_windows = [
        row
        for row in skipped_windows
        if isinstance(row, dict) and str(row.get("name", "")) in priority_names
    ]
    gate_outcome = _build_gate_outcome(
        baseline_delta=baseline_delta,
        skipped_priority_window_count=len(skipped_priority_windows),
        require_baseline=bool(args.require_baseline),
    )

    history_summary_markdown = history_dir / "summary.md"
    history_metric_snapshot = history_dir / "metric_snapshot.json"
    history_delta_path = history_dir / "delta_vs_baseline.json"
    latest_summary_json = benchmark_root / "latest_summary.json"
    latest_summary_markdown = benchmark_root / "latest_summary.md"
    latest_metric_snapshot = benchmark_root / "latest_metric_snapshot.json"
    latest_delta_path = benchmark_root / "latest_delta_vs_baseline.json"

    artifact_paths = {
        "history_run_id": run_id,
        "history_dir": str(history_dir),
        "history_summary_json": str(history_summary_json),
        "history_summary_markdown": str(history_summary_markdown),
        "history_metric_snapshot_json": str(history_metric_snapshot),
        "history_dataset_manifest_json": str(history_manifest_path),
        "history_delta_vs_baseline_json": str(history_delta_path),
        "latest_summary_json": str(latest_summary_json),
        "latest_summary_markdown": str(latest_summary_markdown),
        "latest_metric_snapshot_json": str(latest_metric_snapshot),
        "latest_dataset_manifest_json": str(latest_manifest_path),
        "latest_delta_vs_baseline_json": str(latest_delta_path),
    }

    summary["contract"] = "segmented_regime_window_evaluation.v2"
    summary["benchmark_id"] = benchmark_id
    summary["benchmark_artifacts"] = artifact_paths
    summary["window_config"] = {
        "path": str(window_config_path),
        "sha256": window_config_sha256,
    }
    summary["dataset_manifest"] = {
        "path": str(history_manifest_path),
        "dataset_sha256": dataset_manifest["dataset_sha256"],
    }
    summary["baseline_reference"] = str(baseline_path) if baseline is not None else None
    summary["baseline_delta"] = baseline_delta
    summary["gate_outcome"] = gate_outcome
    summary["skipped_priority_windows"] = skipped_priority_windows
    summary["skipped_priority_window_count"] = int(len(skipped_priority_windows))

    _write_json(history_summary_json, summary)
    _write_json(latest_summary_json, summary)
    _write_json(history_delta_path, {"baseline_delta": baseline_delta})
    _write_json(latest_delta_path, {"baseline_delta": baseline_delta})

    metric_snapshot = {
        "contract": "regime_benchmark_metric_snapshot.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": benchmark_id,
        "history_run_id": run_id,
        "summary_json_path": str(history_summary_json),
        "profile_metrics": current_metrics,
        "regime_minus_ablated": _regime_minus_ablated(current_metrics),
        "baseline_delta": baseline_delta,
        "gate_outcome": gate_outcome,
    }
    _write_json(history_metric_snapshot, metric_snapshot)
    _write_json(latest_metric_snapshot, metric_snapshot)

    markdown = _build_markdown_summary(
        benchmark_id=benchmark_id,
        summary=summary,
        dataset_manifest=dataset_manifest,
        window_config_path=window_config_path,
        baseline_path=baseline_path if baseline is not None else None,
        baseline_delta=baseline_delta,
        gate_outcome=gate_outcome,
        artifact_paths=artifact_paths,
    )
    history_summary_markdown.write_text(markdown, encoding="utf-8")
    latest_summary_markdown.write_text(markdown, encoding="utf-8")

    if args.accept_as_baseline:
        if baseline_path is None:
            raise RuntimeError("Cannot accept baseline because --baseline path is not set.")
        baseline_payload = {
            "contract": "regime_benchmark_baseline.v1",
            "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
            "benchmark_id": benchmark_id,
            "source_summary_path": str(latest_summary_json),
            "profile_metrics": current_metrics,
        }
        _write_json(baseline_path, baseline_payload)

    gate_status = str(gate_outcome.get("status", "fail")).upper()
    print(f"REGIME_BENCHMARK_GATE_STATUS={gate_status}")
    print(f"REGIME_BENCHMARK_ID={benchmark_id}")
    print(f"REGIME_BENCHMARK_HISTORY_RUN_ID={run_id}")
    print(f"REGIME_BENCHMARK_HISTORY_DIR={history_dir}")
    print(f"REGIME_BENCHMARK_SUMMARY_JSON={history_summary_json}")
    print(f"REGIME_BENCHMARK_SUMMARY_MARKDOWN={history_summary_markdown}")
    print(f"REGIME_BENCHMARK_DATASET_MANIFEST={history_manifest_path}")
    print(f"REGIME_BENCHMARK_METRIC_SNAPSHOT={history_metric_snapshot}")
    if baseline_path is not None:
        print(f"REGIME_BENCHMARK_BASELINE_REFERENCE={baseline_path}")

    if gate_status != "PASS" and bool(args.fail_on_gate_fail):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
