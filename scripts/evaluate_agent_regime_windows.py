#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_agents.agent_plane import AgentPlaneConfig, RiskThresholds, run_agent_plane
from quant_agents.config import load_settings
from quant_agents.storage import symbol_slug

DEFAULT_WINDOW_SPECS: tuple[tuple[str, str, str], ...] = (
    ("uptrend_2025q2", "2025-04-01T00:00:00Z", "2025-08-01T00:00:00Z"),
    ("flat_2025nov_to_2026jan", "2025-11-01T00:00:00Z", "2026-02-01T00:00:00Z"),
    ("drawdown_2026latejan_to_mar", "2026-01-25T00:00:00Z", "2026-04-01T00:00:00Z"),
    ("rebound_2026apr", "2026-04-01T00:00:00Z", "2026-05-01T00:00:00Z"),
    ("decline_2026may_to_now", "2026-05-01T00:00:00Z", "now"),
)

REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class WindowSpec:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class EvaluationProfile:
    name: str
    regime_ablation_mode: bool
    regime_policy_mode: str
    regime_touchpoint_prompting_enabled: bool
    regime_touchpoint_calibration_enabled: bool
    regime_touchpoint_self_critique_enabled: bool
    regime_touchpoint_risk_gate_enabled: bool
    min_regime_confidence: float
    calibration_quality_penalty_strength: float
    calibration_directional_contradiction_penalty: float
    calibration_cost_pressure_penalty_strength: float


@dataclass(frozen=True)
class CostStressScenario:
    name: str
    fee_bps: float
    slippage_bps: float


def _parse_timestamp(raw: str, *, now_utc: pd.Timestamp) -> pd.Timestamp:
    value = raw.strip()
    if value.lower() == "now":
        return now_utc
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _parse_window(raw: str, *, now_utc: pd.Timestamp) -> WindowSpec:
    parts = [part.strip() for part in raw.split(",", 2)]
    if len(parts) != 3 or not parts[0] or not parts[1] or not parts[2]:
        raise ValueError(
            "Invalid --window value. Expected format: label,start,end "
            "(example: uptrend,2025-04-01,2025-08-01)."
        )
    start = _parse_timestamp(parts[1], now_utc=now_utc)
    end = _parse_timestamp(parts[2], now_utc=now_utc)
    if not start < end:
        raise ValueError(
            f"Invalid window range for `{parts[0]}`: start must be < end (start={start}, end={end})."
        )
    return WindowSpec(name=parts[0], start=start, end=end)


def _default_windows(now_utc: pd.Timestamp) -> list[WindowSpec]:
    windows: list[WindowSpec] = []
    for name, start_raw, end_raw in DEFAULT_WINDOW_SPECS:
        windows.append(
            WindowSpec(
                name=name,
                start=_parse_timestamp(start_raw, now_utc=now_utc),
                end=_parse_timestamp(end_raw, now_utc=now_utc),
            )
        )
    return windows


def _new_eval_root(root: Path) -> Path:
    now = datetime.now(timezone.utc)
    base = root / "logs" / "analysis" / "regime-window-evals" / f"{now:%Y-%m-%d}"
    base.mkdir(parents=True, exist_ok=True)
    base_id = f"{now:%Y%m%dT%H%M%SZ}"
    for suffix in range(100):
        run_id = base_id if suffix == 0 else f"{base_id}_{suffix:02d}"
        candidate = base / run_id
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise RuntimeError(f"Unable to allocate evaluation output directory under {base}")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_rate(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _delta(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return float(lhs) - float(rhs)


def _mean_optional(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def _classify_regime_bucket(regime_value: Any) -> str:
    regime = str(regime_value or "").strip().lower()
    if not regime:
        return "unknown"
    if "bull" in regime or "uptrend" in regime:
        return "bullish"
    if "bear" in regime or "downtrend" in regime or "decline" in regime:
        return "bearish"
    if "range" in regime or "sideways" in regime or "flat" in regime:
        return "range"
    if "volatile" in regime:
        return "volatile"
    return "unknown"


def _profile_to_payload(profile: EvaluationProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "regime_ablation_mode": bool(profile.regime_ablation_mode),
        "regime_policy_mode": profile.regime_policy_mode,
        "touchpoints": {
            "prompting": bool(profile.regime_touchpoint_prompting_enabled),
            "calibration": bool(profile.regime_touchpoint_calibration_enabled),
            "self_critique": bool(profile.regime_touchpoint_self_critique_enabled),
            "risk_gate": bool(profile.regime_touchpoint_risk_gate_enabled),
        },
        "min_regime_confidence": float(profile.min_regime_confidence),
        "calibration_quality_penalty_strength": float(
            profile.calibration_quality_penalty_strength
        ),
        "calibration_directional_contradiction_penalty": float(
            profile.calibration_directional_contradiction_penalty
        ),
        "calibration_cost_pressure_penalty_strength": float(
            profile.calibration_cost_pressure_penalty_strength
        ),
    }


def _build_profiles(
    *,
    settings,
    profile_set: str,
    regime_enabled_min_conf: float,
) -> list[EvaluationProfile]:
    base_quality_penalty_strength = float(settings.calibration_quality_penalty_strength)
    base_directional_penalty = float(settings.calibration_directional_contradiction_penalty)
    base_cost_penalty = float(settings.calibration_cost_pressure_penalty_strength)
    if profile_set == "priority1":
        return [
            EvaluationProfile(
                name="regime_v2_full",
                regime_ablation_mode=False,
                regime_policy_mode="conditional_v2",
                regime_touchpoint_prompting_enabled=True,
                regime_touchpoint_calibration_enabled=True,
                regime_touchpoint_self_critique_enabled=True,
                regime_touchpoint_risk_gate_enabled=True,
                min_regime_confidence=regime_enabled_min_conf,
                calibration_quality_penalty_strength=base_quality_penalty_strength,
                calibration_directional_contradiction_penalty=base_directional_penalty,
                calibration_cost_pressure_penalty_strength=base_cost_penalty,
            ),
            EvaluationProfile(
                name="regime_v2_no_prompting",
                regime_ablation_mode=False,
                regime_policy_mode="conditional_v2",
                regime_touchpoint_prompting_enabled=False,
                regime_touchpoint_calibration_enabled=True,
                regime_touchpoint_self_critique_enabled=True,
                regime_touchpoint_risk_gate_enabled=True,
                min_regime_confidence=regime_enabled_min_conf,
                calibration_quality_penalty_strength=base_quality_penalty_strength,
                calibration_directional_contradiction_penalty=base_directional_penalty,
                calibration_cost_pressure_penalty_strength=base_cost_penalty,
            ),
            EvaluationProfile(
                name="regime_v2_no_calibration",
                regime_ablation_mode=False,
                regime_policy_mode="conditional_v2",
                regime_touchpoint_prompting_enabled=True,
                regime_touchpoint_calibration_enabled=False,
                regime_touchpoint_self_critique_enabled=True,
                regime_touchpoint_risk_gate_enabled=True,
                min_regime_confidence=regime_enabled_min_conf,
                calibration_quality_penalty_strength=base_quality_penalty_strength,
                calibration_directional_contradiction_penalty=base_directional_penalty,
                calibration_cost_pressure_penalty_strength=base_cost_penalty,
            ),
            EvaluationProfile(
                name="regime_v2_no_self_critique",
                regime_ablation_mode=False,
                regime_policy_mode="conditional_v2",
                regime_touchpoint_prompting_enabled=True,
                regime_touchpoint_calibration_enabled=True,
                regime_touchpoint_self_critique_enabled=False,
                regime_touchpoint_risk_gate_enabled=True,
                min_regime_confidence=regime_enabled_min_conf,
                calibration_quality_penalty_strength=base_quality_penalty_strength,
                calibration_directional_contradiction_penalty=base_directional_penalty,
                calibration_cost_pressure_penalty_strength=base_cost_penalty,
            ),
            EvaluationProfile(
                name="regime_v2_no_risk_gate",
                regime_ablation_mode=False,
                regime_policy_mode="conditional_v2",
                regime_touchpoint_prompting_enabled=True,
                regime_touchpoint_calibration_enabled=True,
                regime_touchpoint_self_critique_enabled=True,
                regime_touchpoint_risk_gate_enabled=False,
                min_regime_confidence=regime_enabled_min_conf,
                calibration_quality_penalty_strength=base_quality_penalty_strength,
                calibration_directional_contradiction_penalty=base_directional_penalty,
                calibration_cost_pressure_penalty_strength=base_cost_penalty,
            ),
            EvaluationProfile(
                name="regime_legacy_full",
                regime_ablation_mode=False,
                regime_policy_mode="legacy",
                regime_touchpoint_prompting_enabled=True,
                regime_touchpoint_calibration_enabled=True,
                regime_touchpoint_self_critique_enabled=True,
                regime_touchpoint_risk_gate_enabled=True,
                min_regime_confidence=regime_enabled_min_conf,
                calibration_quality_penalty_strength=base_quality_penalty_strength,
                calibration_directional_contradiction_penalty=base_directional_penalty,
                calibration_cost_pressure_penalty_strength=base_cost_penalty,
            ),
            EvaluationProfile(
                name="regime_ablated",
                regime_ablation_mode=True,
                regime_policy_mode="legacy",
                regime_touchpoint_prompting_enabled=False,
                regime_touchpoint_calibration_enabled=False,
                regime_touchpoint_self_critique_enabled=False,
                regime_touchpoint_risk_gate_enabled=False,
                min_regime_confidence=regime_enabled_min_conf,
                calibration_quality_penalty_strength=base_quality_penalty_strength,
                calibration_directional_contradiction_penalty=base_directional_penalty,
                calibration_cost_pressure_penalty_strength=base_cost_penalty,
            ),
        ]

    return [
        EvaluationProfile(
            name="regime_enabled",
            regime_ablation_mode=False,
            regime_policy_mode=str(settings.regime_policy_mode),
            regime_touchpoint_prompting_enabled=bool(settings.regime_touchpoint_prompting_enabled),
            regime_touchpoint_calibration_enabled=bool(
                settings.regime_touchpoint_calibration_enabled
            ),
            regime_touchpoint_self_critique_enabled=bool(
                settings.regime_touchpoint_self_critique_enabled
            ),
            regime_touchpoint_risk_gate_enabled=bool(settings.regime_touchpoint_risk_gate_enabled),
            min_regime_confidence=regime_enabled_min_conf,
            calibration_quality_penalty_strength=base_quality_penalty_strength,
            calibration_directional_contradiction_penalty=base_directional_penalty,
            calibration_cost_pressure_penalty_strength=base_cost_penalty,
        ),
        EvaluationProfile(
            name="regime_ablated",
            regime_ablation_mode=True,
            regime_policy_mode=str(settings.regime_policy_mode),
            regime_touchpoint_prompting_enabled=False,
            regime_touchpoint_calibration_enabled=False,
            regime_touchpoint_self_critique_enabled=False,
            regime_touchpoint_risk_gate_enabled=False,
            min_regime_confidence=regime_enabled_min_conf,
            calibration_quality_penalty_strength=base_quality_penalty_strength,
            calibration_directional_contradiction_penalty=base_directional_penalty,
            calibration_cost_pressure_penalty_strength=base_cost_penalty,
        ),
    ]


def _build_cost_stress_scenarios(
    *,
    base_fee_bps: float,
    base_slippage_bps: float,
    multipliers: list[float],
) -> list[CostStressScenario]:
    scenarios: list[CostStressScenario] = [
        CostStressScenario(
            name="base",
            fee_bps=max(0.0, float(base_fee_bps)),
            slippage_bps=max(0.0, float(base_slippage_bps)),
        )
    ]
    seen: set[tuple[float, float]] = {
        (float(scenarios[0].fee_bps), float(scenarios[0].slippage_bps))
    }
    for raw_multiplier in multipliers:
        multiplier = max(0.1, float(raw_multiplier))
        fee = max(0.0, float(base_fee_bps) * multiplier)
        slippage = max(0.0, float(base_slippage_bps) * multiplier)
        key = (round(fee, 8), round(slippage, 8))
        if key in seen:
            continue
        seen.add(key)
        scenarios.append(
            CostStressScenario(
                name=f"stress_x{multiplier:.2f}",
                fee_bps=fee,
                slippage_bps=slippage,
            )
        )
    return scenarios


def _extract_arm_cost_map(backtest_payload: dict[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    arm_metrics = backtest_payload.get("arm_metrics")
    if not isinstance(arm_metrics, dict):
        return output
    for arm, metrics in arm_metrics.items():
        if not isinstance(metrics, dict):
            continue
        value = _safe_float(metrics.get("total_cost_return_drag"))
        if value is None:
            continue
        output[str(arm)] = float(value)
    return output


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


def _load_market_frame(*, input_file: Path | None, data_root: Path, exchange: str, symbol: str, timeframe: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if input_file is not None:
        candidates = [input_file]
    else:
        candidates = _collect_scope_parquet_files(
            root=data_root,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
        )

    for path in candidates:
        frame = pd.read_parquet(path, columns=list(REQUIRED_COLUMNS))
        frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"])
        frames.append(frame)

    if not frames:
        raise RuntimeError("No market rows were loaded from input parquet source(s).")

    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    merged = merged.reset_index(drop=True)
    return merged


def _slice_window(frame: pd.DataFrame, window: WindowSpec) -> pd.DataFrame:
    scoped = frame.loc[(frame["timestamp"] >= window.start) & (frame["timestamp"] < window.end)].copy()
    scoped = scoped.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    scoped = scoped.reset_index(drop=True)
    return scoped


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run segmented agent-plane evaluation across date-range windows and compare "
            "regime-enabled vs regime-ablated profiles."
        )
    )
    parser.add_argument("--exchange", default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument(
        "--input-file",
        default=None,
        help="Optional explicit parquet file. If omitted, all scope raw parquet files are merged.",
    )
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        help=(
            "Repeatable window in format label,start,end where start/end are ISO timestamps or `now`. "
            "Example: --window uptrend,2025-04-01,2025-08-01"
        ),
    )
    parser.add_argument(
        "--minimum-bars",
        type=int,
        default=None,
        help="Minimum bars required per window; defaults to AGENT_MINIMUM_BARS.",
    )
    parser.add_argument(
        "--step-retries",
        type=int,
        default=0,
        help="Retries per agent-plane step (default: 0 for deterministic segmented evaluation speed).",
    )
    parser.add_argument(
        "--strategy-model",
        default=None,
        help="Strategy model override; defaults to OLLAMA_STRATEGY_MODEL.",
    )
    parser.add_argument(
        "--ops-model",
        default=None,
        help="Ops model override; defaults to OLLAMA_OPS_MODEL.",
    )
    parser.add_argument(
        "--regime-min-confidence",
        type=float,
        default=None,
        help="Regime-enabled profile minimum regime confidence; defaults to RISK_MIN_REGIME_CONFIDENCE.",
    )
    parser.add_argument(
        "--profile-set",
        choices=["benchmark", "priority1"],
        default="benchmark",
        help=(
            "Profile bundle to evaluate: `benchmark` keeps the canonical two-profile gate "
            "(`regime_enabled`, `regime_ablated`), while `priority1` adds the component-level "
            "touchpoint matrix for Priority 1 redesign analysis."
        ),
    )
    parser.add_argument(
        "--enable-cost-stress",
        action="store_true",
        help=(
            "Run additional fee/slippage stress scenarios and include sensitivity summary "
            "plus cost-drag decomposition outputs."
        ),
    )
    parser.add_argument(
        "--cost-stress-multiplier",
        action="append",
        type=float,
        default=[],
        help=(
            "Repeatable multiplier for fee/slippage stress scenarios (applied to both fee and "
            "slippage bps). Example: --cost-stress-multiplier 1.5 --cost-stress-multiplier 2.0"
        ),
    )
    parser.add_argument(
        "--cost-stress-profile",
        action="append",
        default=[],
        help=(
            "Repeatable profile name to include in stress sweeps. "
            "Defaults to a compact canonical subset."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output summary JSON path; defaults under QUANT_DATA_ROOT/logs/analysis/regime-window-evals.",
    )
    parser.add_argument(
        "--fail-on-insufficient-window",
        action="store_true",
        help="Fail immediately when a window has fewer than minimum bars (default behavior is to skip and continue).",
    )
    return parser


def _evaluate_windows_for_profiles(
    *,
    settings,
    eval_settings,
    exchange: str,
    symbol: str,
    timeframe: str,
    strategy_model: str,
    ops_model: str,
    runnable_windows: list[WindowSpec],
    window_inputs: dict[str, Path],
    window_stats: dict[str, dict[str, Any]],
    profiles: list[EvaluationProfile],
    scenario: CostStressScenario,
    minimum_bars: int,
    step_retries: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for window in runnable_windows:
        source_path = window_inputs[window.name]
        for profile in profiles:
            thresholds = RiskThresholds(
                min_total_return=settings.risk_min_total_return,
                min_sharpe=settings.risk_min_sharpe,
                max_drawdown=settings.risk_max_drawdown,
                max_cost_return_drag=settings.risk_max_cost_return_drag,
                min_signal_confidence=settings.risk_min_signal_confidence,
                max_cost_pressure_score=settings.risk_max_cost_pressure_score,
                min_walkforward_quality_score=settings.risk_min_walkforward_quality_score,
                min_regime_confidence=float(profile.min_regime_confidence),
            )
            config = AgentPlaneConfig(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                strategy_model=strategy_model,
                ops_model=ops_model,
                step_retries=step_retries,
                thresholds=thresholds,
                backtest_fee_bps=float(scenario.fee_bps),
                backtest_slippage_bps=float(scenario.slippage_bps),
                walk_forward_fee_bps=float(scenario.fee_bps),
                walk_forward_slippage_bps=float(scenario.slippage_bps),
                paper_notional_usd=settings.paper_trade_notional_usd,
                paper_starting_cash_usd=settings.paper_trade_starting_cash_usd,
                paper_fee_bps=settings.paper_trade_fee_bps,
                paper_slippage_bps=settings.paper_trade_slippage_bps,
                minimum_bars=minimum_bars,
                regime_detector_mode=settings.regime_detector_mode,
                regime_policy_mode=str(profile.regime_policy_mode),
                regime_policy_min_actionable_confidence=settings.regime_policy_min_actionable_confidence,
                regime_policy_transition_confidence=settings.regime_policy_transition_confidence,
                regime_touchpoint_prompting_enabled=bool(
                    profile.regime_touchpoint_prompting_enabled
                ),
                regime_touchpoint_calibration_enabled=bool(
                    profile.regime_touchpoint_calibration_enabled
                ),
                regime_touchpoint_self_critique_enabled=bool(
                    profile.regime_touchpoint_self_critique_enabled
                ),
                regime_touchpoint_risk_gate_enabled=bool(
                    profile.regime_touchpoint_risk_gate_enabled
                ),
                regime_volatility_threshold=settings.regime_volatility_threshold,
                regime_trend_spread_threshold=settings.regime_trend_spread_threshold,
                regime_persistence_bars=settings.regime_persistence_bars,
                regime_ablation_mode=bool(profile.regime_ablation_mode),
                walk_forward_train_bars=settings.walk_forward_train_bars,
                walk_forward_validate_bars=settings.walk_forward_validate_bars,
                walk_forward_step_bars=settings.walk_forward_step_bars,
                walk_forward_min_windows=settings.walk_forward_min_windows,
                calibration_min_walkforward_sharpe=settings.calibration_min_walkforward_sharpe,
                calibration_confidence_floor=settings.calibration_confidence_floor,
                calibration_confidence_ceiling=settings.calibration_confidence_ceiling,
                calibration_max_contradictions=settings.calibration_max_contradictions,
                calibration_directional_edge_threshold=settings.calibration_directional_edge_threshold,
                calibration_quality_penalty_strength=float(
                    profile.calibration_quality_penalty_strength
                ),
                calibration_directional_contradiction_penalty=float(
                    profile.calibration_directional_contradiction_penalty
                ),
                calibration_cost_pressure_penalty_strength=float(
                    profile.calibration_cost_pressure_penalty_strength
                ),
                self_critique_min_score=settings.self_critique_min_score,
                self_critique_max_findings=settings.self_critique_max_findings,
                ops_report_verbosity=settings.ops_report_verbosity,
                ensemble_mode=settings.ensemble_mode,
                ensemble_enabled_arms=settings.ensemble_enabled_arms,
                ensemble_decay_horizon=settings.ensemble_decay_horizon,
                ensemble_exploration_weight=settings.ensemble_exploration_weight,
                ensemble_turnover_penalty_bps=settings.ensemble_turnover_penalty_bps,
                source_data_path=source_path,
            )

            run_result = run_agent_plane(eval_settings, config)
            risk = json.loads(run_result.risk_decision_path.read_text(encoding="utf-8"))
            calibration = json.loads(
                run_result.confidence_calibration_path.read_text(encoding="utf-8")
            )
            backtest = json.loads(run_result.backtest_evaluation_path.read_text(encoding="utf-8"))
            strategy_payload = json.loads(
                run_result.strategy_signal_path.read_text(encoding="utf-8")
            )
            manifest = json.loads(run_result.run_manifest_path.read_text(encoding="utf-8"))
            manifest_outcome = manifest.get("outcome", {}) if isinstance(manifest, dict) else {}
            phase1_regime = (
                manifest_outcome.get("phase1_regime")
                if isinstance(manifest_outcome, dict)
                else strategy_payload.get("regime")
            )
            if phase1_regime in {None, ""}:
                phase1_regime = strategy_payload.get("regime")
            regime_bucket = _classify_regime_bucket(phase1_regime)
            directional_contradiction_detected = bool(
                calibration.get(
                    "directional_contradiction_detected",
                    calibration.get("contradiction_detected", False),
                )
            )
            quality_contradiction_detected = bool(
                calibration.get("quality_contradiction_detected", False)
            )
            arm_cost_return_drag = _extract_arm_cost_map(backtest)

            row = {
                "scenario": scenario.name,
                "backtest_fee_bps": float(scenario.fee_bps),
                "backtest_slippage_bps": float(scenario.slippage_bps),
                "profile": profile.name,
                "window": window.name,
                "window_start_utc": window.start.isoformat(),
                "window_end_utc": window.end.isoformat(),
                "bar_count": window_stats[window.name]["bar_count"],
                "regime_ablation_mode": bool(profile.regime_ablation_mode),
                "regime_policy_mode": str(profile.regime_policy_mode),
                "regime_touchpoints": {
                    "prompting": bool(profile.regime_touchpoint_prompting_enabled),
                    "calibration": bool(profile.regime_touchpoint_calibration_enabled),
                    "self_critique": bool(profile.regime_touchpoint_self_critique_enabled),
                    "risk_gate": bool(profile.regime_touchpoint_risk_gate_enabled),
                },
                "data_start_utc": window_stats[window.name]["data_start_utc"],
                "data_end_utc": window_stats[window.name]["data_end_utc"],
                "phase1_regime": str(phase1_regime or "unknown"),
                "regime_bucket": regime_bucket,
                "risk_approved": bool(risk.get("approved", False)),
                "approval_rate": 1.0 if bool(risk.get("approved", False)) else 0.0,
                "contradiction_detected": bool(calibration.get("contradiction_detected", False)),
                "contradiction_rate": (
                    1.0 if bool(calibration.get("contradiction_detected", False)) else 0.0
                ),
                "directional_contradiction_detected": directional_contradiction_detected,
                "directional_contradiction_rate": (
                    1.0 if directional_contradiction_detected else 0.0
                ),
                "quality_contradiction_detected": quality_contradiction_detected,
                "quality_contradiction_rate": 1.0 if quality_contradiction_detected else 0.0,
                "cost_pressure_score": _safe_float(calibration.get("cost_pressure_score")),
                "reason_codes": [str(code) for code in risk.get("reason_codes", [])],
                "calibration_reason_codes": [
                    str(code) for code in calibration.get("reason_codes", [])
                ],
                "net_total_return": _safe_float(backtest.get("total_return")),
                "sharpe": _safe_float(backtest.get("sharpe")),
                "max_drawdown": _safe_float(backtest.get("max_drawdown")),
                "total_cost_return_drag": _safe_float(backtest.get("total_cost_return_drag")),
                "arm_cost_return_drag": arm_cost_return_drag,
                "agent_run_id": run_result.run_id,
                "agent_run_dir": str(run_result.run_dir),
                "artifacts": {
                    "strategy_proposal_signal": str(run_result.strategy_signal_path),
                    "backtest_evaluation": str(run_result.backtest_evaluation_path),
                    "confidence_calibration": str(run_result.confidence_calibration_path),
                    "risk_decision": str(run_result.risk_decision_path),
                    "run_manifest": str(run_result.run_manifest_path),
                },
            }
            results.append(row)

            print(
                "WINDOW_PROFILE_RESULT "
                + json.dumps(
                    {
                        "scenario": row["scenario"],
                        "window": row["window"],
                        "profile": row["profile"],
                        "approval_rate": row["approval_rate"],
                        "contradiction_rate": row["contradiction_rate"],
                        "directional_contradiction_rate": row[
                            "directional_contradiction_rate"
                        ],
                        "quality_contradiction_rate": row["quality_contradiction_rate"],
                        "cost_pressure_score": row["cost_pressure_score"],
                        "net_total_return": row["net_total_return"],
                        "sharpe": row["sharpe"],
                        "max_drawdown": row["max_drawdown"],
                        "total_cost_return_drag": row["total_cost_return_drag"],
                    },
                    sort_keys=True,
                )
            )
    return results


def _summarize_profile_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_profile[str(row["profile"])].append(row)

    profile_summary: dict[str, dict[str, Any]] = {}
    for profile_name, profile_rows in by_profile.items():
        reason_counts: Counter[str] = Counter()
        calibration_reason_counts: Counter[str] = Counter()
        for row in profile_rows:
            reason_counts.update(list(row.get("reason_codes", [])))
            calibration_reason_counts.update(list(row.get("calibration_reason_codes", [])))
        profile_summary[profile_name] = {
            "windows_evaluated": len(profile_rows),
            "approval_rate": _safe_rate(
                sum(1.0 for row in profile_rows if bool(row.get("risk_approved"))),
                len(profile_rows),
            ),
            "contradiction_rate": _safe_rate(
                sum(1.0 for row in profile_rows if bool(row.get("contradiction_detected"))),
                len(profile_rows),
            ),
            "directional_contradiction_rate": _safe_rate(
                sum(
                    1.0
                    for row in profile_rows
                    if bool(row.get("directional_contradiction_detected"))
                ),
                len(profile_rows),
            ),
            "quality_contradiction_rate": _safe_rate(
                sum(
                    1.0 for row in profile_rows if bool(row.get("quality_contradiction_detected"))
                ),
                len(profile_rows),
            ),
            "mean_cost_pressure_score": _mean_optional(
                [_safe_float(row.get("cost_pressure_score")) for row in profile_rows]
            ),
            "mean_net_total_return": _mean_optional(
                [_safe_float(row.get("net_total_return")) for row in profile_rows]
            ),
            "mean_sharpe": _mean_optional(
                [_safe_float(row.get("sharpe")) for row in profile_rows]
            ),
            "mean_max_drawdown": _mean_optional(
                [_safe_float(row.get("max_drawdown")) for row in profile_rows]
            ),
            "mean_total_cost_return_drag": _mean_optional(
                [_safe_float(row.get("total_cost_return_drag")) for row in profile_rows]
            ),
            "reason_code_distribution": dict(
                sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
            "calibration_reason_code_distribution": dict(
                sorted(
                    calibration_reason_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
        }
    return profile_summary


def _build_window_comparison(
    *,
    runnable_windows: list[WindowSpec],
    window_stats: dict[str, dict[str, Any]],
    by_window: dict[str, dict[str, dict[str, Any]]],
    left_profile: str,
    right_profile: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for window in runnable_windows:
        left_row = by_window.get(window.name, {}).get(left_profile)
        right_row = by_window.get(window.name, {}).get(right_profile)
        if left_row is None or right_row is None:
            continue
        left_payload = {
            "approval_rate": left_row.get("approval_rate"),
            "contradiction_rate": left_row.get("contradiction_rate"),
            "directional_contradiction_rate": left_row.get("directional_contradiction_rate"),
            "quality_contradiction_rate": left_row.get("quality_contradiction_rate"),
            "cost_pressure_score": left_row.get("cost_pressure_score"),
            "reason_codes": left_row.get("reason_codes"),
            "net_total_return": left_row.get("net_total_return"),
            "sharpe": left_row.get("sharpe"),
            "max_drawdown": left_row.get("max_drawdown"),
            "total_cost_return_drag": left_row.get("total_cost_return_drag"),
        }
        right_payload = {
            "approval_rate": right_row.get("approval_rate"),
            "contradiction_rate": right_row.get("contradiction_rate"),
            "directional_contradiction_rate": right_row.get("directional_contradiction_rate"),
            "quality_contradiction_rate": right_row.get("quality_contradiction_rate"),
            "cost_pressure_score": right_row.get("cost_pressure_score"),
            "reason_codes": right_row.get("reason_codes"),
            "net_total_return": right_row.get("net_total_return"),
            "sharpe": right_row.get("sharpe"),
            "max_drawdown": right_row.get("max_drawdown"),
            "total_cost_return_drag": right_row.get("total_cost_return_drag"),
        }
        delta_payload = {
            "approval_rate": _delta(
                _safe_float(left_row.get("approval_rate")),
                _safe_float(right_row.get("approval_rate")),
            ),
            "contradiction_rate": _delta(
                _safe_float(left_row.get("contradiction_rate")),
                _safe_float(right_row.get("contradiction_rate")),
            ),
            "directional_contradiction_rate": _delta(
                _safe_float(left_row.get("directional_contradiction_rate")),
                _safe_float(right_row.get("directional_contradiction_rate")),
            ),
            "quality_contradiction_rate": _delta(
                _safe_float(left_row.get("quality_contradiction_rate")),
                _safe_float(right_row.get("quality_contradiction_rate")),
            ),
            "cost_pressure_score": _delta(
                _safe_float(left_row.get("cost_pressure_score")),
                _safe_float(right_row.get("cost_pressure_score")),
            ),
            "net_total_return": _delta(
                _safe_float(left_row.get("net_total_return")),
                _safe_float(right_row.get("net_total_return")),
            ),
            "sharpe": _delta(
                _safe_float(left_row.get("sharpe")),
                _safe_float(right_row.get("sharpe")),
            ),
            "max_drawdown": _delta(
                _safe_float(left_row.get("max_drawdown")),
                _safe_float(right_row.get("max_drawdown")),
            ),
            "total_cost_return_drag": _delta(
                _safe_float(left_row.get("total_cost_return_drag")),
                _safe_float(right_row.get("total_cost_return_drag")),
            ),
        }
        row: dict[str, Any] = {
            "window": window.name,
            "window_start_utc": window.start.isoformat(),
            "window_end_utc": window.end.isoformat(),
            "bar_count": window_stats[window.name]["bar_count"],
            "left_profile": left_profile,
            "right_profile": right_profile,
            left_profile: left_payload,
            right_profile: right_payload,
            "delta_left_minus_right": delta_payload,
        }
        if left_profile == "regime_enabled" and right_profile == "regime_ablated":
            row["regime_enabled"] = left_payload
            row["regime_ablated"] = right_payload
            row["delta_regime_minus_ablated"] = delta_payload
        output.append(row)
    return output


def _build_ablation_matrix(
    *,
    profiles: list[EvaluationProfile],
    profile_summary: dict[str, dict[str, Any]],
    reference_profile: str,
) -> dict[str, Any] | None:
    reference_metrics = profile_summary.get(reference_profile)
    if reference_metrics is None:
        return None
    metric_keys = (
        "approval_rate",
        "contradiction_rate",
        "directional_contradiction_rate",
        "quality_contradiction_rate",
        "mean_cost_pressure_score",
        "mean_net_total_return",
        "mean_sharpe",
        "mean_max_drawdown",
        "mean_total_cost_return_drag",
    )
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        summary = profile_summary.get(profile.name)
        if summary is None:
            continue
        delta_vs_reference = {
            metric: _delta(
                _safe_float(summary.get(metric)),
                _safe_float(reference_metrics.get(metric)),
            )
            for metric in metric_keys
        }
        rows.append(
            {
                "profile": profile.name,
                "regime_policy_mode": profile.regime_policy_mode,
                "regime_ablation_mode": bool(profile.regime_ablation_mode),
                "touchpoints": {
                    "prompting": bool(profile.regime_touchpoint_prompting_enabled),
                    "calibration": bool(profile.regime_touchpoint_calibration_enabled),
                    "self_critique": bool(profile.regime_touchpoint_self_critique_enabled),
                    "risk_gate": bool(profile.regime_touchpoint_risk_gate_enabled),
                },
                "metrics": {metric: summary.get(metric) for metric in metric_keys},
                "delta_vs_reference": delta_vs_reference,
            }
        )
    return {
        "contract": "priority1_ablation_matrix.v1",
        "reference_profile": reference_profile,
        "rows": rows,
    }


def _build_cost_decomposition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    profile_regime_acc: dict[tuple[str, str], dict[str, float | int]] = defaultdict(
        lambda: {
            "run_count": 0,
            "sum_total_cost_return_drag": 0.0,
            "cost_return_drag_samples": 0,
            "sum_cost_pressure_score": 0.0,
            "cost_pressure_samples": 0,
        }
    )
    arm_regime_acc: dict[tuple[str, str, str], dict[str, float | int]] = defaultdict(
        lambda: {"run_count": 0, "sum_arm_cost_return_drag": 0.0, "arm_cost_samples": 0}
    )
    for row in rows:
        profile = str(row.get("profile"))
        regime_bucket = str(row.get("regime_bucket", "unknown"))
        profile_key = (profile, regime_bucket)
        profile_stats = profile_regime_acc[profile_key]
        profile_stats["run_count"] = int(profile_stats["run_count"]) + 1

        total_cost_drag = _safe_float(row.get("total_cost_return_drag"))
        if total_cost_drag is not None:
            profile_stats["sum_total_cost_return_drag"] = float(
                profile_stats["sum_total_cost_return_drag"]
            ) + float(total_cost_drag)
            profile_stats["cost_return_drag_samples"] = (
                int(profile_stats["cost_return_drag_samples"]) + 1
            )
        cost_pressure_score = _safe_float(row.get("cost_pressure_score"))
        if cost_pressure_score is not None:
            profile_stats["sum_cost_pressure_score"] = float(
                profile_stats["sum_cost_pressure_score"]
            ) + float(cost_pressure_score)
            profile_stats["cost_pressure_samples"] = (
                int(profile_stats["cost_pressure_samples"]) + 1
            )

        arm_cost_return_drag = row.get("arm_cost_return_drag", {})
        if isinstance(arm_cost_return_drag, dict):
            for arm, value in arm_cost_return_drag.items():
                arm_value = _safe_float(value)
                arm_key = (profile, regime_bucket, str(arm))
                arm_stats = arm_regime_acc[arm_key]
                arm_stats["run_count"] = int(arm_stats["run_count"]) + 1
                if arm_value is None:
                    continue
                arm_stats["sum_arm_cost_return_drag"] = float(
                    arm_stats["sum_arm_cost_return_drag"]
                ) + float(arm_value)
                arm_stats["arm_cost_samples"] = int(arm_stats["arm_cost_samples"]) + 1

    by_profile_regime: list[dict[str, Any]] = []
    for (profile, regime_bucket), stats in sorted(profile_regime_acc.items()):
        drag_samples = int(stats["cost_return_drag_samples"])
        pressure_samples = int(stats["cost_pressure_samples"])
        by_profile_regime.append(
            {
                "profile": profile,
                "regime_bucket": regime_bucket,
                "run_count": int(stats["run_count"]),
                "mean_total_cost_return_drag": (
                    float(stats["sum_total_cost_return_drag"]) / drag_samples
                    if drag_samples > 0
                    else None
                ),
                "mean_cost_pressure_score": (
                    float(stats["sum_cost_pressure_score"]) / pressure_samples
                    if pressure_samples > 0
                    else None
                ),
            }
        )

    by_arm_regime: list[dict[str, Any]] = []
    for (profile, regime_bucket, arm), stats in sorted(arm_regime_acc.items()):
        samples = int(stats["arm_cost_samples"])
        by_arm_regime.append(
            {
                "profile": profile,
                "regime_bucket": regime_bucket,
                "arm": arm,
                "run_count": int(stats["run_count"]),
                "mean_arm_cost_return_drag": (
                    float(stats["sum_arm_cost_return_drag"]) / samples if samples > 0 else None
                ),
            }
        )
    return {
        "contract": "cost_drag_decomposition.v1",
        "by_profile_regime_bucket": by_profile_regime,
        "by_profile_regime_bucket_arm": by_arm_regime,
    }


def _build_cost_stress_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_profile_scenario: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_profile_scenario[(str(row.get("profile")), str(row.get("scenario")))].append(row)

    scenario_rows: list[dict[str, Any]] = []
    for (profile, scenario), group in sorted(by_profile_scenario.items()):
        row = {
            "profile": profile,
            "scenario": scenario,
            "sample_count": len(group),
            "fee_bps": _mean_optional([_safe_float(item.get("backtest_fee_bps")) for item in group]),
            "slippage_bps": _mean_optional(
                [_safe_float(item.get("backtest_slippage_bps")) for item in group]
            ),
            "approval_rate": _safe_rate(
                sum(1.0 for item in group if bool(item.get("risk_approved"))),
                len(group),
            ),
            "contradiction_rate": _safe_rate(
                sum(1.0 for item in group if bool(item.get("contradiction_detected"))),
                len(group),
            ),
            "mean_cost_pressure_score": _mean_optional(
                [_safe_float(item.get("cost_pressure_score")) for item in group]
            ),
            "mean_net_total_return": _mean_optional(
                [_safe_float(item.get("net_total_return")) for item in group]
            ),
            "mean_sharpe": _mean_optional([_safe_float(item.get("sharpe")) for item in group]),
            "mean_max_drawdown": _mean_optional(
                [_safe_float(item.get("max_drawdown")) for item in group]
            ),
            "mean_total_cost_return_drag": _mean_optional(
                [_safe_float(item.get("total_cost_return_drag")) for item in group]
            ),
        }
        scenario_rows.append(row)

    base_by_profile: dict[str, dict[str, Any]] = {
        str(row["profile"]): row for row in scenario_rows if str(row.get("scenario")) == "base"
    }
    for row in scenario_rows:
        base_row = base_by_profile.get(str(row.get("profile")))
        if base_row is None:
            row["delta_vs_base"] = None
            continue
        row["delta_vs_base"] = {
            "approval_rate": _delta(
                _safe_float(row.get("approval_rate")),
                _safe_float(base_row.get("approval_rate")),
            ),
            "contradiction_rate": _delta(
                _safe_float(row.get("contradiction_rate")),
                _safe_float(base_row.get("contradiction_rate")),
            ),
            "mean_cost_pressure_score": _delta(
                _safe_float(row.get("mean_cost_pressure_score")),
                _safe_float(base_row.get("mean_cost_pressure_score")),
            ),
            "mean_net_total_return": _delta(
                _safe_float(row.get("mean_net_total_return")),
                _safe_float(base_row.get("mean_net_total_return")),
            ),
            "mean_sharpe": _delta(
                _safe_float(row.get("mean_sharpe")),
                _safe_float(base_row.get("mean_sharpe")),
            ),
            "mean_max_drawdown": _delta(
                _safe_float(row.get("mean_max_drawdown")),
                _safe_float(base_row.get("mean_max_drawdown")),
            ),
            "mean_total_cost_return_drag": _delta(
                _safe_float(row.get("mean_total_cost_return_drag")),
                _safe_float(base_row.get("mean_total_cost_return_drag")),
            ),
        }

    return {
        "contract": "cost_stress_summary.v1",
        "rows": scenario_rows,
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    settings = load_settings()
    exchange = args.exchange or settings.default_exchange
    symbol = args.symbol or settings.default_symbol
    timeframe = args.timeframe or settings.default_timeframe
    now_utc = pd.Timestamp.now(tz="UTC")

    windows = (
        [_parse_window(raw, now_utc=now_utc) for raw in args.window]
        if args.window
        else _default_windows(now_utc=now_utc)
    )
    windows = sorted(windows, key=lambda item: item.start)

    source_file = Path(args.input_file).expanduser().resolve() if args.input_file else None
    market_frame = _load_market_frame(
        input_file=source_file,
        data_root=settings.quant_data_root,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
    )

    evaluation_root = _new_eval_root(settings.quant_data_root)
    input_dir = evaluation_root / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    sandbox_root = evaluation_root / "sandbox"
    sandbox_root.mkdir(parents=True, exist_ok=True)
    eval_settings = replace(
        settings,
        quant_data_root=sandbox_root,
        allow_unmounted_data_root=True,
    )

    minimum_bars = max(
        10,
        int(args.minimum_bars if args.minimum_bars is not None else settings.agent_minimum_bars),
    )
    step_retries = max(0, int(args.step_retries))
    regime_enabled_min_conf = float(
        max(
            0.0,
            min(
                1.0,
                args.regime_min_confidence
                if args.regime_min_confidence is not None
                else settings.risk_min_regime_confidence,
            ),
        )
    )
    profiles = _build_profiles(
        settings=settings,
        profile_set=str(args.profile_set),
        regime_enabled_min_conf=regime_enabled_min_conf,
    )

    window_inputs: dict[str, Path] = {}
    window_stats: dict[str, dict[str, Any]] = {}
    skipped_windows: list[dict[str, Any]] = []
    runnable_windows: list[WindowSpec] = []
    for window in windows:
        scoped = _slice_window(market_frame, window)
        if len(scoped) < minimum_bars:
            message = (
                f"Window `{window.name}` has insufficient bars ({len(scoped)} < {minimum_bars}). "
                "Provide broader ranges or a larger source dataset."
            )
            if args.fail_on_insufficient_window:
                raise RuntimeError(message)
            skipped_windows.append(
                {
                    "name": window.name,
                    "start_utc": window.start.isoformat(),
                    "end_utc": window.end.isoformat(),
                    "available_bars": int(len(scoped)),
                    "required_minimum_bars": minimum_bars,
                    "reason": "insufficient_bars",
                }
            )
            print(
                "WINDOW_SKIPPED "
                + json.dumps(
                    {
                        "window": window.name,
                        "available_bars": int(len(scoped)),
                        "required_minimum_bars": minimum_bars,
                    },
                    sort_keys=True,
                )
            )
            continue
        target = input_dir / f"{window.name}.parquet"
        scoped.to_parquet(target, index=False)
        window_inputs[window.name] = target
        window_stats[window.name] = {
            "bar_count": int(len(scoped)),
            "data_start_utc": pd.Timestamp(scoped["timestamp"].iloc[0]).isoformat(),
            "data_end_utc": pd.Timestamp(scoped["timestamp"].iloc[-1]).isoformat(),
        }
        runnable_windows.append(window)

    if not runnable_windows:
        raise RuntimeError(
            "No runnable windows remain after applying minimum bar requirements. "
            "Provide broader ranges, reduce --minimum-bars, or use a dataset with longer history."
        )

    strategy_model = args.strategy_model or settings.ollama_strategy_model
    ops_model = args.ops_model or settings.ollama_ops_model
    base_scenario = CostStressScenario(
        name="base",
        fee_bps=float(settings.backtest_fee_bps),
        slippage_bps=float(settings.backtest_slippage_bps),
    )
    results = _evaluate_windows_for_profiles(
        settings=settings,
        eval_settings=eval_settings,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        strategy_model=strategy_model,
        ops_model=ops_model,
        runnable_windows=runnable_windows,
        window_inputs=window_inputs,
        window_stats=window_stats,
        profiles=profiles,
        scenario=base_scenario,
        minimum_bars=minimum_bars,
        step_retries=step_retries,
    )

    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_window: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in results:
        by_profile[str(row["profile"])].append(row)
        by_window[str(row["window"])][str(row["profile"])] = row

    comparison_pair: tuple[str, str] | None = None
    if "regime_enabled" in by_profile and "regime_ablated" in by_profile:
        comparison_pair = ("regime_enabled", "regime_ablated")
    elif "regime_v2_full" in by_profile and "regime_ablated" in by_profile:
        comparison_pair = ("regime_v2_full", "regime_ablated")

    profile_summary = _summarize_profile_rows(results)
    window_comparison = (
        _build_window_comparison(
            runnable_windows=runnable_windows,
            window_stats=window_stats,
            by_window=by_window,
            left_profile=comparison_pair[0],
            right_profile=comparison_pair[1],
        )
        if comparison_pair is not None
        else []
    )
    ablation_reference = "regime_v2_full" if args.profile_set == "priority1" else "regime_enabled"
    ablation_matrix = _build_ablation_matrix(
        profiles=profiles,
        profile_summary=profile_summary,
        reference_profile=ablation_reference,
    )
    cost_decomposition = _build_cost_decomposition(results)

    cost_stress_payload: dict[str, Any] | None = None
    if args.enable_cost_stress:
        requested_profiles = [
            str(name).strip()
            for name in list(args.cost_stress_profile or [])
            if str(name).strip()
        ]
        if not requested_profiles:
            if "regime_v2_full" in by_profile and "regime_ablated" in by_profile:
                requested_profiles = ["regime_v2_full", "regime_ablated"]
            elif "regime_enabled" in by_profile and "regime_ablated" in by_profile:
                requested_profiles = ["regime_enabled", "regime_ablated"]
            else:
                requested_profiles = [profile.name for profile in profiles[:2]]
        requested_profile_set = set(requested_profiles)
        stress_profiles = [profile for profile in profiles if profile.name in requested_profile_set]
        multipliers = (
            [float(value) for value in list(args.cost_stress_multiplier or [])]
            if args.cost_stress_multiplier
            else [1.5, 2.0, 3.0]
        )
        stress_scenarios = _build_cost_stress_scenarios(
            base_fee_bps=float(settings.backtest_fee_bps),
            base_slippage_bps=float(settings.backtest_slippage_bps),
            multipliers=multipliers,
        )
        stress_rows: list[dict[str, Any]] = [
            row for row in results if str(row.get("profile")) in requested_profile_set
        ]
        for scenario in stress_scenarios:
            if scenario.name == "base":
                continue
            stress_rows.extend(
                _evaluate_windows_for_profiles(
                    settings=settings,
                    eval_settings=eval_settings,
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy_model=strategy_model,
                    ops_model=ops_model,
                    runnable_windows=runnable_windows,
                    window_inputs=window_inputs,
                    window_stats=window_stats,
                    profiles=stress_profiles,
                    scenario=scenario,
                    minimum_bars=minimum_bars,
                    step_retries=step_retries,
                )
            )
        cost_stress_payload = {
            "contract": "cost_stress_report.v1",
            "profiles": requested_profiles,
            "scenarios": [
                {
                    "name": scenario.name,
                    "fee_bps": scenario.fee_bps,
                    "slippage_bps": scenario.slippage_bps,
                }
                for scenario in stress_scenarios
            ],
            "results": stress_rows,
            "summary": _build_cost_stress_summary(stress_rows),
            "cost_decomposition": _build_cost_decomposition(stress_rows),
        }

    output_contract = (
        "segmented_regime_window_evaluation.v2"
        if args.profile_set == "priority1" or args.enable_cost_stress
        else "segmented_regime_window_evaluation.v1"
    )
    output: dict[str, Any] = {
        "contract": output_contract,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "input_file": str(source_file) if source_file else None,
            "source_mode": "single_input_file" if source_file else "merged_scope_raw_files",
            "profile_set": str(args.profile_set),
        },
        "evaluation_root": str(evaluation_root),
        "sandbox_root": str(sandbox_root),
        "profiles": {profile.name: _profile_to_payload(profile) for profile in profiles},
        "windows": [
            {
                "name": window.name,
                "start_utc": window.start.isoformat(),
                "end_utc": window.end.isoformat(),
                **window_stats[window.name],
            }
            for window in runnable_windows
        ],
        "skipped_windows": skipped_windows,
        "results": results,
        "window_comparison": window_comparison,
        "profile_summary": profile_summary,
        "ablation_matrix": ablation_matrix,
        "cost_decomposition": cost_decomposition,
    }
    if cost_stress_payload is not None:
        output["cost_stress"] = cost_stress_payload

    output_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else (evaluation_root / "summary.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("SEGMENTED_REGIME_WINDOW_EVALUATION_STATUS=PASS")
    print(f"SEGMENTED_REGIME_WINDOW_EVALUATION_OUTPUT={output_path}")
    print("PROFILE_SUMMARY")
    print(json.dumps(profile_summary, sort_keys=True))


if __name__ == "__main__":
    main()
