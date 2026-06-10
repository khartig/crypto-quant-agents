from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal, TypeVar
import numpy as np

import pandas as pd

from quant_agents.agent_contracts import (
    BacktestEvaluation,
    DataQualitySignal,
    OpsReportContract,
    PaperTradeExecution,
    PaperTradeIntent,
    Recommendation,
    RiskDecision,
    StrategyProposalSignal,
    write_contract,
)
from quant_agents.backtest import (
    ENSEMBLE_STRATEGY_NAME,
    STRATEGY_NAME,
    SUPPORTED_STRATEGY_ARMS,
    run_ensemble_backtest,
)
from quant_agents.config import Settings
from quant_agents.ollama_client import OllamaClient
from quant_agents.paper_trading import execute_paper_trade_intent
from quant_agents.storage import latest_raw_dataset

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


@dataclass(frozen=True)
class RiskThresholds:
    min_total_return: float
    min_sharpe: float
    max_drawdown: float
    min_signal_confidence: float


@dataclass(frozen=True)
class AgentPlaneConfig:
    exchange: str
    symbol: str
    timeframe: str
    strategy_model: str
    ops_model: str
    step_retries: int
    thresholds: RiskThresholds
    paper_notional_usd: float
    paper_starting_cash_usd: float
    paper_fee_bps: float
    minimum_bars: int
    walk_forward_train_bars: int = 240
    walk_forward_validate_bars: int = 72
    walk_forward_step_bars: int = 72
    walk_forward_min_windows: int = 3
    calibration_min_walkforward_sharpe: float = 0.10
    calibration_confidence_floor: float = 0.05
    calibration_confidence_ceiling: float = 0.95
    calibration_max_contradictions: int = 0
    self_critique_min_score: float = 0.55
    self_critique_max_findings: int = 6
    ops_report_verbosity: str = "standard"
    ensemble_mode: Literal["single", "adaptive"] = "adaptive"
    ensemble_enabled_arms: tuple[str, ...] = (
        "sma_baseline",
        "technical_composite",
        "llm_context",
    )
    ensemble_decay_horizon: int = 96
    ensemble_exploration_weight: float = 0.08
    source_data_path: Path | None = None


@dataclass(frozen=True)
class StepExecutionRecord:
    step: str
    status: Literal["success", "fallback"]
    attempts: int
    started_at_utc: str
    finished_at_utc: str
    duration_ms: float
    errors: tuple[str, ...]
    step_dir: Path


@dataclass(frozen=True)
class AgentPlaneRunResult:
    run_id: str
    run_dir: Path
    source_data_path: Path
    data_quality_path: Path
    strategy_signal_path: Path
    backtest_evaluation_path: Path
    backtest_arm_attribution_path: Path | None
    phase1_feature_context_path: Path
    walkforward_evaluation_path: Path
    confidence_calibration_path: Path
    self_critique_signal_path: Path
    risk_decision_path: Path
    paper_trade_intent_path: Path
    paper_trade_execution_path: Path
    ops_report_markdown_path: Path
    ops_report_contract_path: Path
    run_manifest_path: Path
    ensemble_performance_update_path: Path | None
    risk_approved: bool
    intent_status: str
    paper_trade_execution_status: str
    intent_destination_path: Path | None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_agent_plane_run_dir(root: Path) -> tuple[str, Path]:
    now = datetime.now(timezone.utc)
    day_dir = root / "logs" / "agents" / "openclaw-orchestrator" / f"{now:%Y-%m-%d}"
    day_dir.mkdir(parents=True, exist_ok=True)

    base_run_id = f"{now:%Y%m%dT%H%M%SZ}"
    for suffix in range(100):
        run_id = base_run_id if suffix == 0 else f"{base_run_id}_{suffix:02d}"
        run_dir = day_dir / run_id
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_id, run_dir
    raise RuntimeError(f"Unable to allocate unique run id under {day_dir}")


def _timeframe_delta(timeframe: str) -> timedelta:
    mapping = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
    }
    return mapping.get(timeframe, timedelta(hours=1))


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _coerce_recommendation(value: Any) -> Recommendation:
    if not isinstance(value, str):
        raise ValueError("recommendation must be a string")
    normalized = value.strip().lower()
    if normalized == "buy":
        return "buy"
    if normalized == "sell":
        return "sell"
    if normalized == "hold":
        return "hold"
    raise ValueError(f"Unsupported recommendation: {value}")


def _sanitize_windows(fast_window: Any, slow_window: Any) -> tuple[int, int, list[str]]:
    warnings: list[str] = []
    try:
        fast = int(fast_window)
    except (TypeError, ValueError):
        fast = 20
        warnings.append("fast_window invalid; defaulted to 20")
    try:
        slow = int(slow_window)
    except (TypeError, ValueError):
        slow = 50
        warnings.append("slow_window invalid; defaulted to 50")

    if fast <= 0:
        fast = 20
        warnings.append("fast_window <= 0; defaulted to 20")
    if slow <= 0:
        slow = 50
        warnings.append("slow_window <= 0; defaulted to 50")
    if fast >= slow:
        fast, slow = 20, 50
        warnings.append("fast_window >= slow_window; reset to 20/50")
    return fast, slow, warnings


def _normalize_ensemble_mode(value: Any) -> Literal["single", "adaptive"]:
    if not isinstance(value, str):
        return "adaptive"
    normalized = value.strip().lower()
    if normalized == "single":
        return "single"
    if normalized == "adaptive":
        return "adaptive"
    return "adaptive"


def _normalize_enabled_arms(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = [part.strip().lower() for part in value.split(",") if part.strip()]
    elif isinstance(value, (tuple, list)):
        raw = [str(part).strip().lower() for part in value if str(part).strip()]
    else:
        raw = []
    deduped: list[str] = []
    for arm in raw:
        if arm in SUPPORTED_STRATEGY_ARMS and arm not in deduped:
            deduped.append(arm)
    if not deduped:
        return ("sma_baseline", "technical_composite", "llm_context")
    return tuple(deduped)


def _load_ensemble_weight_state(
    *,
    path: Path,
    enabled_arms: tuple[str, ...],
    decay_horizon: int,
    exploration_weight: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                payload = parsed
        except (OSError, json.JSONDecodeError):
            payload = {}
    raw_stats = payload.get("arm_stats")
    arm_stats_payload = raw_stats if isinstance(raw_stats, dict) else {}
    arm_stats: dict[str, dict[str, Any]] = {}
    equal_weight = 1.0 / max(1, len(enabled_arms))
    for arm in enabled_arms:
        raw = arm_stats_payload.get(arm)
        entry = raw if isinstance(raw, dict) else {}
        arm_stats[arm] = {
            "ewma_pnl": float(entry.get("ewma_pnl", 0.0)),
            "ewma_drawdown": float(entry.get("ewma_drawdown", 0.0)),
            "ewma_stability": float(entry.get("ewma_stability", 0.5)),
            "observations": max(0, int(entry.get("observations", 0))),
            "last_weight": float(entry.get("last_weight", equal_weight)),
        }
    history = payload.get("history")
    history_rows = list(history) if isinstance(history, list) else []
    return {
        "contract": "ensemble_weight_state.v1",
        "updated_at_utc": str(payload.get("updated_at_utc") or _utc_now_iso()),
        "decay_horizon": max(4, int(payload.get("decay_horizon", decay_horizon))),
        "exploration_weight": max(
            0.0,
            float(payload.get("exploration_weight", exploration_weight)),
        ),
        "arm_stats": arm_stats,
        "history": history_rows[-200:],
    }


def _build_sma_arm_vote(
    market_frame: pd.DataFrame,
    *,
    fast_window: int,
    slow_window: int,
) -> dict[str, Any]:
    close = pd.to_numeric(market_frame.get("close"), errors="coerce").ffill().bfill()
    if close.empty:
        return {
            "source": "deterministic",
            "recommendation": "hold",
            "confidence": 0.34,
            "action_scores": {"buy": 0.20, "sell": 0.20, "hold": 0.60},
            "reason_codes": ["sma_close_unavailable"],
            "metadata": {"fast_window": fast_window, "slow_window": slow_window},
        }
    ma_fast = close.rolling(window=max(2, fast_window), min_periods=max(2, fast_window // 2)).mean()
    ma_slow = close.rolling(window=max(3, slow_window), min_periods=max(3, slow_window // 2)).mean()
    fast_value = float(ma_fast.iloc[-1]) if pd.notna(ma_fast.iloc[-1]) else float(close.iloc[-1])
    slow_value = float(ma_slow.iloc[-1]) if pd.notna(ma_slow.iloc[-1]) else float(close.iloc[-1])
    spread = (fast_value / max(slow_value, 1e-9)) - 1.0
    strength = float(np.clip(abs(spread) * 28.0, 0.0, 0.45))
    if spread >= 0.001:
        recommendation: Recommendation = "buy"
        action_scores = _normalize_probability_votes(
            {"buy": 0.50 + strength, "sell": 0.10, "hold": 0.40 - strength}
        )
        reason_codes = ["sma_fast_above_slow"]
    elif spread <= -0.001:
        recommendation = "sell"
        action_scores = _normalize_probability_votes(
            {"buy": 0.10, "sell": 0.50 + strength, "hold": 0.40 - strength}
        )
        reason_codes = ["sma_fast_below_slow"]
    else:
        recommendation = "hold"
        action_scores = _normalize_probability_votes({"buy": 0.20, "sell": 0.20, "hold": 0.60})
        reason_codes = ["sma_spread_neutral"]
    confidence = float(action_scores.get(recommendation, 0.34))
    return {
        "source": "deterministic",
        "recommendation": recommendation,
        "confidence": confidence,
        "action_scores": action_scores,
        "reason_codes": reason_codes,
        "metadata": {
            "fast_window": int(fast_window),
            "slow_window": int(slow_window),
            "spread": spread,
        },
    }


def _build_technical_arm_vote(phase1_context: dict[str, Any]) -> dict[str, Any]:
    indicator_votes = _normalize_probability_votes(
        dict(phase1_context.get("indicator_votes", {}))
    )
    recommendation = max(indicator_votes, key=indicator_votes.get)
    confidence = float(indicator_votes.get(recommendation, 0.0))
    reason_codes = [str(code) for code in list(phase1_context.get("reason_codes", []))]
    if not reason_codes:
        reason_codes = ["technical_vote_fallback"]
    return {
        "source": "deterministic",
        "recommendation": recommendation,
        "confidence": confidence,
        "action_scores": indicator_votes,
        "reason_codes": reason_codes,
        "metadata": {
            "regime": str(phase1_context.get("regime", "unknown")),
        },
    }


def _build_llm_context_arm_vote(strategy_signal: StrategyProposalSignal) -> dict[str, Any]:
    recommendation = strategy_signal.recommendation
    confidence = float(np.clip(strategy_signal.confidence, 0.0, 1.0))
    action_scores = _normalize_probability_votes(
        dict(strategy_signal.indicator_votes)
        if strategy_signal.indicator_votes
        else {
            recommendation: confidence,
            "hold": max(0.0, 1.0 - confidence),
        }
    )
    return {
        "source": strategy_signal.source,
        "model": strategy_signal.model,
        "recommendation": recommendation,
        "confidence": confidence,
        "action_scores": action_scores,
        "reason_codes": list(strategy_signal.reason_codes),
        "metadata": {
            "regime": strategy_signal.regime,
            "warnings": list(strategy_signal.warnings),
        },
    }


def _build_strategy_arm_votes(
    *,
    market_frame: pd.DataFrame,
    phase1_context: dict[str, Any],
    llm_strategy_signal: StrategyProposalSignal,
    fast_window: int,
    slow_window: int,
    enabled_arms: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    votes: dict[str, dict[str, Any]] = {}
    for arm in enabled_arms:
        if arm == "sma_baseline":
            votes[arm] = _build_sma_arm_vote(
                market_frame,
                fast_window=fast_window,
                slow_window=slow_window,
            )
            continue
        if arm == "technical_composite":
            votes[arm] = _build_technical_arm_vote(phase1_context)
            continue
        if arm == "llm_context":
            votes[arm] = _build_llm_context_arm_vote(llm_strategy_signal)
            continue
    return votes


def _normalize_arm_weight_map(
    enabled_arms: tuple[str, ...],
    arm_weights: dict[str, float],
) -> dict[str, float]:
    if not enabled_arms:
        return {}
    sanitized = {arm: max(0.0, float(arm_weights.get(arm, 0.0))) for arm in enabled_arms}
    total = sum(sanitized.values())
    if total <= 0:
        equal = 1.0 / len(enabled_arms)
        return {arm: equal for arm in enabled_arms}
    return {arm: float(value / total) for arm, value in sanitized.items()}


def _bounded_weight_projection(
    weights: dict[str, float],
    *,
    floor: float,
    cap: float,
) -> dict[str, float]:
    if not weights:
        return {}
    clipped = {
        arm: min(max(floor, float(value)), cap)
        for arm, value in weights.items()
    }
    total = sum(clipped.values())
    if total <= 0:
        equal = 1.0 / len(clipped)
        return {arm: equal for arm in clipped}
    return {arm: float(value / total) for arm, value in clipped.items()}


def _compute_arm_weights(
    *,
    ensemble_mode: Literal["single", "adaptive"],
    enabled_arms: tuple[str, ...],
    arm_votes: dict[str, dict[str, Any]],
    ensemble_state: dict[str, Any],
    exploration_weight: float,
) -> tuple[dict[str, float], list[str]]:
    if not enabled_arms:
        return {}, ["ensemble_no_enabled_arms"]
    if ensemble_mode == "single":
        winner = max(
            enabled_arms,
            key=lambda arm: float(dict(arm_votes.get(arm, {})).get("confidence", 0.0)),
        )
        return (
            {arm: (1.0 if arm == winner else 0.0) for arm in enabled_arms},
            [f"ensemble_single_mode_winner={winner}"],
        )

    arm_stats = dict(ensemble_state.get("arm_stats", {}))
    raw_scores: dict[str, float] = {}
    for arm in enabled_arms:
        vote = dict(arm_votes.get(arm, {}))
        vote_confidence = float(np.clip(vote.get("confidence", 0.0), 0.0, 1.0))
        stats = dict(arm_stats.get(arm, {}))
        ewma_pnl = float(stats.get("ewma_pnl", 0.0))
        ewma_drawdown = float(stats.get("ewma_drawdown", 0.0))
        ewma_stability = float(np.clip(stats.get("ewma_stability", 0.5), 0.0, 1.0))
        performance_score = 0.5 + ewma_pnl - ewma_drawdown + ((ewma_stability - 0.5) * 0.2)
        raw_scores[arm] = max(
            0.0,
            (0.60 * vote_confidence) + (0.40 * performance_score),
        ) + max(0.0, exploration_weight)
    normalized = _normalize_arm_weight_map(enabled_arms, raw_scores)
    bounded = _bounded_weight_projection(normalized, floor=0.05, cap=0.80)
    return bounded, ["ensemble_adaptive_weights_applied"]


def _combine_arm_votes(
    *,
    arm_votes: dict[str, dict[str, Any]],
    arm_weights: dict[str, float],
) -> tuple[Recommendation, float, dict[str, float], list[str], list[str]]:
    if not arm_votes:
        return "hold", 0.0, {"buy": 0.0, "sell": 0.0, "hold": 1.0}, [], ["ensemble_no_arm_votes"]
    normalized_weights = _normalize_arm_weight_map(tuple(arm_votes.keys()), arm_weights)
    aggregate_scores = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
    for arm, vote in arm_votes.items():
        vote_payload = dict(vote)
        recommendation = str(vote_payload.get("recommendation", "hold")).lower()
        confidence = float(np.clip(vote_payload.get("confidence", 0.0), 0.0, 1.0))
        raw_scores = vote_payload.get("action_scores")
        if isinstance(raw_scores, dict):
            action_scores = _normalize_probability_votes(raw_scores)
        else:
            action_scores = _normalize_probability_votes(
                {recommendation: confidence, "hold": max(0.0, 1.0 - confidence)}
            )
        weight = float(normalized_weights.get(arm, 0.0))
        for action in ("buy", "sell", "hold"):
            aggregate_scores[action] += weight * float(action_scores.get(action, 0.0))
    aggregate_scores = _normalize_probability_votes(aggregate_scores)
    recommendation = max(aggregate_scores, key=aggregate_scores.get)
    confidence = float(aggregate_scores.get(recommendation, 0.0))
    selected_arms = sorted(
        [arm for arm in normalized_weights if normalized_weights[arm] > 0.0],
        key=lambda arm: normalized_weights[arm],
        reverse=True,
    )
    reason_codes = [f"ensemble_selected_{recommendation}"]
    for arm in selected_arms[:3]:
        reason_codes.append(f"arm_weight_{arm}_{normalized_weights[arm]:.3f}")
    return recommendation, confidence, aggregate_scores, selected_arms, reason_codes


def _update_ensemble_weight_state(
    *,
    state_path: Path,
    ensemble_state: dict[str, Any],
    selected_arms: list[str],
    arm_weights: dict[str, float],
    paper_trade_execution: PaperTradeExecution,
    decay_horizon: int,
    exploration_weight: float,
    run_id: str,
) -> dict[str, Any]:
    alpha = 2.0 / (max(4, decay_horizon) + 1.0)
    executed_notional = max(0.0, float(paper_trade_execution.executed_notional_usd))
    realized_pnl = float(paper_trade_execution.realized_pnl_delta_usd)
    pnl_ratio = realized_pnl / max(1.0, executed_notional) if executed_notional > 0 else 0.0
    drawdown = max(0.0, -pnl_ratio)
    executed = paper_trade_execution.execution_status == "executed"

    before_state = _json_safe(dict(ensemble_state))
    arm_stats = dict(ensemble_state.get("arm_stats", {}))
    normalized_weights = _normalize_arm_weight_map(tuple(arm_stats.keys()), arm_weights)
    for arm, raw_stats in arm_stats.items():
        stats = dict(raw_stats)
        participated = arm in selected_arms
        attributed_weight = float(normalized_weights.get(arm, 0.0)) if participated else 0.0
        pnl_signal = pnl_ratio * attributed_weight if participated and executed else 0.0
        drawdown_signal = drawdown * attributed_weight if participated and executed else 0.0
        stability_signal = 1.0 if participated and executed else 0.0
        stats["ewma_pnl"] = float(((1.0 - alpha) * float(stats.get("ewma_pnl", 0.0))) + (alpha * pnl_signal))
        stats["ewma_drawdown"] = float(
            ((1.0 - alpha) * float(stats.get("ewma_drawdown", 0.0))) + (alpha * drawdown_signal)
        )
        stats["ewma_stability"] = float(
            np.clip(
                ((1.0 - alpha) * float(stats.get("ewma_stability", 0.5))) + (alpha * stability_signal),
                0.0,
                1.0,
            )
        )
        stats["observations"] = int(max(0, int(stats.get("observations", 0)) + (1 if participated else 0)))
        stats["last_weight"] = float(normalized_weights.get(arm, 0.0))
        arm_stats[arm] = stats

    history = list(ensemble_state.get("history", []))
    history.append(
        {
            "run_id": run_id,
            "created_at_utc": _utc_now_iso(),
            "selected_arms": list(selected_arms),
            "execution_status": paper_trade_execution.execution_status,
            "executed_action": paper_trade_execution.executed_action,
            "pnl_ratio": pnl_ratio,
            "drawdown": drawdown,
        }
    )
    ensemble_state["contract"] = "ensemble_weight_state.v1"
    ensemble_state["updated_at_utc"] = _utc_now_iso()
    ensemble_state["decay_horizon"] = max(4, int(decay_horizon))
    ensemble_state["exploration_weight"] = max(0.0, float(exploration_weight))
    ensemble_state["arm_stats"] = arm_stats
    ensemble_state["history"] = history[-200:]
    _write_json(state_path, ensemble_state)
    return {
        "contract": "ensemble_weight_update.v1",
        "run_id": run_id,
        "created_at_utc": _utc_now_iso(),
        "state_path": str(state_path),
        "alpha": alpha,
        "selected_arms": list(selected_arms),
        "execution_status": paper_trade_execution.execution_status,
        "executed_action": paper_trade_execution.executed_action,
        "executed_notional_usd": executed_notional,
        "realized_pnl_delta_usd": realized_pnl,
        "pnl_ratio": pnl_ratio,
        "drawdown": drawdown,
        "before": before_state,
        "after": _json_safe(dict(ensemble_state)),
    }


def _run_step_with_retries(
    *,
    run_dir: Path,
    step_name: str,
    max_retries: int,
    runner: Callable[[], _T],
    fallback: Callable[[list[str]], _T] | None = None,
) -> tuple[_T, StepExecutionRecord]:
    step_dir = run_dir / "steps" / step_name
    step_dir.mkdir(parents=True, exist_ok=True)
    retries = max(0, max_retries)
    errors: list[str] = []
    first_started = datetime.now(timezone.utc)

    for attempt in range(1, retries + 2):
        started_at = datetime.now(timezone.utc)
        started_perf = perf_counter()
        try:
            result = runner()
            finished_at = datetime.now(timezone.utc)
            duration_ms = round((perf_counter() - started_perf) * 1000, 3)
            _write_json(
                step_dir / f"attempt_{attempt:02d}.json",
                {
                    "step": step_name,
                    "attempt": attempt,
                    "status": "success",
                    "started_at_utc": started_at.isoformat(),
                    "finished_at_utc": finished_at.isoformat(),
                    "duration_ms": duration_ms,
                },
            )
            record = StepExecutionRecord(
                step=step_name,
                status="success",
                attempts=attempt,
                started_at_utc=first_started.isoformat(),
                finished_at_utc=finished_at.isoformat(),
                duration_ms=duration_ms,
                errors=tuple(errors),
                step_dir=step_dir,
            )
            _write_json(step_dir / "step_result.json", asdict(record))
            return result, record
        except Exception as exc:
            finished_at = datetime.now(timezone.utc)
            duration_ms = round((perf_counter() - started_perf) * 1000, 3)
            error_message = f"{type(exc).__name__}: {exc}"
            errors.append(error_message)
            _write_json(
                step_dir / f"attempt_{attempt:02d}.json",
                {
                    "step": step_name,
                    "attempt": attempt,
                    "status": "failure",
                    "started_at_utc": started_at.isoformat(),
                    "finished_at_utc": finished_at.isoformat(),
                    "duration_ms": duration_ms,
                    "error": error_message,
                },
            )
            if attempt >= retries + 1:
                if fallback is None:
                    raise
                fallback_started = datetime.now(timezone.utc)
                fallback_started_perf = perf_counter()
                result = fallback(errors)
                fallback_finished = datetime.now(timezone.utc)
                fallback_duration_ms = round((perf_counter() - fallback_started_perf) * 1000, 3)
                _write_json(
                    step_dir / "fallback.json",
                    {
                        "step": step_name,
                        "status": "fallback",
                        "started_at_utc": fallback_started.isoformat(),
                        "finished_at_utc": fallback_finished.isoformat(),
                        "duration_ms": fallback_duration_ms,
                        "errors": errors,
                    },
                )
                record = StepExecutionRecord(
                    step=step_name,
                    status="fallback",
                    attempts=attempt,
                    started_at_utc=first_started.isoformat(),
                    finished_at_utc=fallback_finished.isoformat(),
                    duration_ms=fallback_duration_ms,
                    errors=tuple(errors),
                    step_dir=step_dir,
                )
                _write_json(step_dir / "step_result.json", asdict(record))
                return result, record


def _load_market_frame(source_data_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(source_data_path).sort_values("timestamp").reset_index(drop=True)
    if "timestamp" not in frame.columns:
        raise RuntimeError("Input parquet is missing required `timestamp` column.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if frame["timestamp"].isna().all():
        raise RuntimeError("Input parquet timestamp column could not be parsed.")
    return frame


def _market_snapshot(frame: pd.DataFrame) -> dict[str, float | int | str]:
    closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
    if closes.empty:
        raise RuntimeError("Input parquet contains no usable `close` values.")

    last_close = float(closes.iloc[-1])
    lookback_index = max(0, len(closes) - 25)
    close_24h_ago = float(closes.iloc[lookback_index])
    momentum_24h = float(last_close / close_24h_ago - 1.0) if close_24h_ago else 0.0
    recent_returns = closes.pct_change().dropna().tail(48)
    volatility = float(recent_returns.std(ddof=0)) if not recent_returns.empty else 0.0
    return {
        "bars": int(len(frame)),
        "last_close": last_close,
        "momentum_24h": momentum_24h,
        "volatility_48_bars": volatility,
        "start": str(frame["timestamp"].min()),
        "end": str(frame["timestamp"].max()),
    }


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _normalize_probability_votes(votes: dict[str, float]) -> dict[str, float]:
    clean = {
        "buy": max(0.0, float(votes.get("buy", 0.0))),
        "sell": max(0.0, float(votes.get("sell", 0.0))),
        "hold": max(0.0, float(votes.get("hold", 0.0))),
    }
    total = clean["buy"] + clean["sell"] + clean["hold"]
    if total <= 0:
        return {"buy": 0.0, "sell": 0.0, "hold": 1.0}
    return {key: float(value / total) for key, value in clean.items()}


def _compute_phase1_feature_context(frame: pd.DataFrame) -> dict[str, Any]:
    close = pd.to_numeric(frame.get("close"), errors="coerce").ffill().bfill()
    high = pd.to_numeric(frame.get("high"), errors="coerce").ffill().bfill()
    low = pd.to_numeric(frame.get("low"), errors="coerce").ffill().bfill()
    volume = pd.to_numeric(frame.get("volume"), errors="coerce").ffill().bfill()
    if close.empty or close.isna().all():
        raise RuntimeError("Cannot compute phase-1 feature context without usable close prices.")

    returns = close.pct_change()
    sma_fast = close.rolling(window=20, min_periods=5).mean()
    sma_slow = close.rolling(window=50, min_periods=10).mean()
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    volatility_48 = returns.rolling(window=48, min_periods=10).std(ddof=0)
    rsi_14 = _compute_rsi(close, period=14)
    bb_mid = close.rolling(window=20, min_periods=5).mean()
    bb_std = close.rolling(window=20, min_periods=5).std(ddof=0).fillna(0.0)
    bb_upper = bb_mid + (2.0 * bb_std)
    bb_lower = bb_mid - (2.0 * bb_std)
    hl_range_14 = ((high - low) / close.replace(0.0, np.nan)).rolling(window=14, min_periods=5).mean()
    volume_mean = volume.rolling(window=24, min_periods=5).mean().replace(0.0, np.nan)
    volume_zscore_24 = ((volume - volume_mean) / volume_mean).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    latest_close = float(close.iloc[-1])
    latest_sma_fast = float(sma_fast.iloc[-1]) if pd.notna(sma_fast.iloc[-1]) else latest_close
    latest_sma_slow = float(sma_slow.iloc[-1]) if pd.notna(sma_slow.iloc[-1]) else latest_close
    latest_macd = float(macd.iloc[-1]) if pd.notna(macd.iloc[-1]) else 0.0
    latest_macd_signal = float(macd_signal.iloc[-1]) if pd.notna(macd_signal.iloc[-1]) else 0.0
    latest_macd_hist = float(macd_hist.iloc[-1]) if pd.notna(macd_hist.iloc[-1]) else 0.0
    latest_rsi = float(rsi_14.iloc[-1]) if pd.notna(rsi_14.iloc[-1]) else 50.0
    latest_volatility = float(volatility_48.iloc[-1]) if pd.notna(volatility_48.iloc[-1]) else 0.0
    latest_bb_upper = float(bb_upper.iloc[-1]) if pd.notna(bb_upper.iloc[-1]) else latest_close
    latest_bb_lower = float(bb_lower.iloc[-1]) if pd.notna(bb_lower.iloc[-1]) else latest_close
    latest_hl_range = float(hl_range_14.iloc[-1]) if pd.notna(hl_range_14.iloc[-1]) else 0.0
    latest_volume_zscore = (
        float(volume_zscore_24.iloc[-1]) if pd.notna(volume_zscore_24.iloc[-1]) else 0.0
    )

    trend_spread = abs((latest_sma_fast / max(latest_sma_slow, 1e-9)) - 1.0)
    if latest_volatility >= 0.03:
        regime = "volatile"
    elif trend_spread >= 0.01 and latest_macd >= 0:
        regime = "bull_trend"
    elif trend_spread >= 0.01 and latest_macd < 0:
        regime = "bear_trend"
    else:
        regime = "range_bound"

    vote_counts = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
    reason_codes: list[str] = []
    if latest_rsi <= 35:
        vote_counts["buy"] += 1.0
        reason_codes.append("rsi_supports_buy")
    elif latest_rsi >= 65:
        vote_counts["sell"] += 1.0
        reason_codes.append("rsi_supports_sell")
    else:
        vote_counts["hold"] += 0.5
        reason_codes.append("rsi_neutral")

    if latest_macd_hist > 0:
        vote_counts["buy"] += 1.0
        reason_codes.append("macd_hist_positive")
    elif latest_macd_hist < 0:
        vote_counts["sell"] += 1.0
        reason_codes.append("macd_hist_negative")
    else:
        vote_counts["hold"] += 0.5
        reason_codes.append("macd_hist_flat")

    if latest_sma_fast > latest_sma_slow:
        vote_counts["buy"] += 1.0
        reason_codes.append("sma_fast_above_slow")
    elif latest_sma_fast < latest_sma_slow:
        vote_counts["sell"] += 1.0
        reason_codes.append("sma_fast_below_slow")
    else:
        vote_counts["hold"] += 0.5
        reason_codes.append("sma_spread_flat")

    if latest_close >= latest_bb_upper:
        vote_counts["sell"] += 0.75
        reason_codes.append("price_above_upper_band")
    elif latest_close <= latest_bb_lower:
        vote_counts["buy"] += 0.75
        reason_codes.append("price_below_lower_band")
    else:
        vote_counts["hold"] += 0.5
        reason_codes.append("price_inside_bands")

    if regime == "volatile":
        vote_counts["hold"] += 0.75
        reason_codes.append("regime_volatile")
    else:
        reason_codes.append(f"regime_{regime}")

    indicator_votes = _normalize_probability_votes(vote_counts)
    feature_snapshot: dict[str, float | int | str | bool] = {
        "close": latest_close,
        "sma_fast": latest_sma_fast,
        "sma_slow": latest_sma_slow,
        "sma_fast_spread": (latest_close / max(latest_sma_fast, 1e-9)) - 1.0,
        "sma_slow_spread": (latest_close / max(latest_sma_slow, 1e-9)) - 1.0,
        "macd": latest_macd,
        "macd_signal": latest_macd_signal,
        "macd_hist": latest_macd_hist,
        "rsi_14": latest_rsi,
        "volatility_48": latest_volatility,
        "bb_upper": latest_bb_upper,
        "bb_lower": latest_bb_lower,
        "hl_range_14": latest_hl_range,
        "volume_zscore_24": latest_volume_zscore,
    }
    return {
        "regime": regime,
        "indicator_votes": indicator_votes,
        "feature_snapshot": feature_snapshot,
        "reason_codes": sorted(set(reason_codes)),
    }


def _normalize_indicator_votes(value: Any, fallback: dict[str, float]) -> dict[str, float]:
    if not isinstance(value, dict):
        return fallback
    parsed: dict[str, float] = {}
    for key in ("buy", "sell", "hold"):
        raw = value.get(key)
        if raw is None:
            continue
        try:
            parsed[key] = float(raw)
        except (TypeError, ValueError):
            continue
    if not parsed:
        return fallback
    return _normalize_probability_votes(parsed)


def _normalize_regime(value: Any, fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    normalized = value.strip().lower().replace(" ", "_")
    if not normalized:
        return fallback
    return normalized


def _normalize_feature_snapshot(
    value: Any,
    fallback: dict[str, float | int | str | bool],
) -> dict[str, float | int | str | bool]:
    if not isinstance(value, dict):
        return fallback
    normalized: dict[str, float | int | str | bool] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            continue
        if isinstance(raw, bool):
            normalized[key] = raw
            continue
        if isinstance(raw, (int, float)):
            normalized[key] = float(raw)
            continue
        if isinstance(raw, str):
            trimmed = raw.strip()
            if not trimmed:
                continue
            try:
                normalized[key] = float(trimmed)
            except ValueError:
                normalized[key] = trimmed
    return normalized or fallback


def _normalize_reason_codes(value: Any, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return sorted(set(fallback))
    codes = [str(item).strip().lower().replace(" ", "_") for item in value if str(item).strip()]
    merged = sorted(set([*fallback, *codes]))
    return merged


def _periods_per_year(timeframe: str) -> int:
    mapping = {
        "1m": 525600,
        "5m": 105120,
        "15m": 35040,
        "1h": 8760,
        "4h": 2190,
        "1d": 365,
    }
    return mapping.get(timeframe, 365)


def _walkforward_quality_score(
    *,
    total_return: float,
    sharpe: float,
    max_drawdown: float,
    hit_rate: float,
) -> float:
    sharpe_score = (math.tanh(sharpe / 2.0) + 1.0) / 2.0
    return_score = float(np.clip((total_return + 0.05) / 0.20, 0.0, 1.0))
    drawdown_score = float(np.clip(1.0 - abs(min(0.0, max_drawdown)) / 0.40, 0.0, 1.0))
    hit_rate_score = float(np.clip(hit_rate, 0.0, 1.0))
    quality = (
        (0.35 * sharpe_score)
        + (0.30 * return_score)
        + (0.20 * drawdown_score)
        + (0.15 * hit_rate_score)
    )
    return float(np.clip(quality, 0.0, 1.0))


def _build_walkforward_diagnostics(window_rows: list[dict[str, Any]]) -> dict[str, Any]:
    confidence_rows = [
        (
            float(np.clip(row.get("quality_score", 0.0), 0.0, 1.0)),
            float(row.get("total_return", 0.0)),
        )
        for row in window_rows
    ]
    reliability_bins: list[dict[str, Any]] = []
    for index in range(5):
        lower = index / 5.0
        upper = (index + 1) / 5.0
        if index == 4:
            selected = [pair for pair in confidence_rows if lower <= pair[0] <= upper]
        else:
            selected = [pair for pair in confidence_rows if lower <= pair[0] < upper]
        if selected:
            avg_conf = float(np.mean([pair[0] for pair in selected]))
            avg_return = float(np.mean([pair[1] for pair in selected]))
            positive_rate = float(np.mean([1.0 if pair[1] > 0 else 0.0 for pair in selected]))
        else:
            avg_conf = 0.0
            avg_return = 0.0
            positive_rate = 0.0
        reliability_bins.append(
            {
                "bin": f"{lower:.1f}-{upper:.1f}",
                "count": len(selected),
                "avg_confidence": avg_conf,
                "avg_return": avg_return,
                "positive_rate": positive_rate,
            }
        )

    sorted_rows = sorted(confidence_rows, key=lambda pair: pair[0])
    deciles: list[dict[str, Any]] = []
    if sorted_rows:
        step = max(1, math.ceil(len(sorted_rows) / 10))
        for decile in range(10):
            start = decile * step
            end = min(len(sorted_rows), (decile + 1) * step)
            chunk = sorted_rows[start:end]
            if not chunk:
                deciles.append(
                    {
                        "decile": decile + 1,
                        "count": 0,
                        "avg_confidence": 0.0,
                        "avg_return": 0.0,
                    }
                )
                continue
            deciles.append(
                {
                    "decile": decile + 1,
                    "count": len(chunk),
                    "avg_confidence": float(np.mean([item[0] for item in chunk])),
                    "avg_return": float(np.mean([item[1] for item in chunk])),
                }
            )
    return {
        "reliability_bins": reliability_bins,
        "confidence_deciles": deciles,
    }


def _run_walkforward_evaluation(
    *,
    frame: pd.DataFrame,
    exchange: str,
    symbol: str,
    timeframe: str,
    source_data_path: Path,
    source_data_sha256: str,
    fast_window: int,
    slow_window: int,
    train_bars: int,
    validate_bars: int,
    step_bars: int,
    min_windows: int,
) -> dict[str, Any]:
    close_frame = frame[["timestamp", "close"]].copy()
    close_frame["close"] = pd.to_numeric(close_frame["close"], errors="coerce")
    close_frame = close_frame.dropna(subset=["timestamp", "close"]).reset_index(drop=True)
    required_bars = train_bars + validate_bars + slow_window
    if len(close_frame) < required_bars:
        raise RuntimeError(
            f"Insufficient bars for walk-forward evaluation: {len(close_frame)} < {required_bars}"
        )

    periods_per_year = _periods_per_year(timeframe)
    rows: list[dict[str, Any]] = []
    total_bars = len(close_frame)
    window_span = train_bars + validate_bars
    next_start = 0
    window_index = 0
    while next_start + window_span <= total_bars:
        window_index += 1
        segment = close_frame.iloc[next_start : next_start + window_span].copy().reset_index(drop=True)
        segment["returns"] = segment["close"].pct_change().fillna(0.0)
        segment["ma_fast"] = segment["close"].rolling(window=fast_window).mean()
        segment["ma_slow"] = segment["close"].rolling(window=slow_window).mean()
        segment["signal"] = (segment["ma_fast"] > segment["ma_slow"]).astype(float)
        segment["position"] = segment["signal"].shift(1).fillna(0.0)

        validation = segment.iloc[train_bars:].copy().reset_index(drop=True)
        validation["strategy_returns"] = validation["position"] * validation["returns"]
        validation["equity_curve"] = (1.0 + validation["strategy_returns"]).cumprod()
        if validation.empty:
            next_start += step_bars
            continue

        total_return = float(validation["equity_curve"].iloc[-1] - 1.0)
        mean_ret = float(validation["strategy_returns"].mean())
        std_ret = float(validation["strategy_returns"].std(ddof=0))
        sharpe = float(np.sqrt(periods_per_year) * mean_ret / std_ret) if std_ret > 0 else 0.0
        rolling_peak = validation["equity_curve"].cummax()
        drawdown = (validation["equity_curve"] / rolling_peak) - 1.0
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
        hit_rate = float((validation["strategy_returns"] > 0).mean()) if len(validation) else 0.0
        signal_flips = int(validation["signal"].diff().abs().fillna(0.0).sum())
        quality_score = _walkforward_quality_score(
            total_return=total_return,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            hit_rate=hit_rate,
        )

        rows.append(
            {
                "window_index": window_index,
                "train_start_utc": str(segment["timestamp"].iloc[0]),
                "train_end_utc": str(segment["timestamp"].iloc[train_bars - 1]),
                "validate_start_utc": str(validation["timestamp"].iloc[0]),
                "validate_end_utc": str(validation["timestamp"].iloc[-1]),
                "bars": int(len(validation)),
                "total_return": total_return,
                "annualized_return": float((1.0 + total_return) ** (periods_per_year / len(validation)) - 1.0),
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "hit_rate": hit_rate,
                "signal_flips": signal_flips,
                "quality_score": quality_score,
            }
        )
        next_start += step_bars

    if len(rows) < min_windows:
        raise RuntimeError(
            f"Walk-forward produced {len(rows)} windows; need at least {min_windows}"
        )

    total_returns = np.asarray([row["total_return"] for row in rows], dtype=float)
    sharpes = np.asarray([row["sharpe"] for row in rows], dtype=float)
    drawdowns = np.asarray([row["max_drawdown"] for row in rows], dtype=float)
    hit_rates = np.asarray([row["hit_rate"] for row in rows], dtype=float)
    quality_scores = np.asarray([row["quality_score"] for row in rows], dtype=float)

    avg_return = float(np.mean(total_returns))
    avg_sharpe = float(np.mean(sharpes))
    worst_drawdown = float(np.min(drawdowns))
    avg_hit_rate = float(np.mean(hit_rates))
    return_variance = float(np.std(total_returns))
    stability_score = float(np.clip(1.0 - min(1.0, return_variance / 0.08), 0.0, 1.0))
    quality_score = float(np.clip(float(np.mean(quality_scores)) * (0.7 + 0.3 * stability_score), 0.0, 1.0))
    diagnostics = _build_walkforward_diagnostics(rows)

    return {
        "contract": "walkforward_evaluation.v1",
        "created_at_utc": _utc_now_iso(),
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "source_data_path": str(source_data_path),
        "source_data_sha256": source_data_sha256,
        "strategy": STRATEGY_NAME,
        "fast_window": fast_window,
        "slow_window": slow_window,
        "train_bars": train_bars,
        "validate_bars": validate_bars,
        "step_bars": step_bars,
        "window_count": len(rows),
        "aggregate_total_return": avg_return,
        "aggregate_sharpe": avg_sharpe,
        "aggregate_max_drawdown": worst_drawdown,
        "aggregate_hit_rate": avg_hit_rate,
        "stability_score": stability_score,
        "quality_score": quality_score,
        "windows": rows,
        "diagnostics": diagnostics,
    }
def _run_self_critique(
    *,
    run_id: str,
    strategy_signal: StrategyProposalSignal,
    data_quality_signal: DataQualitySignal,
    market_frame: pd.DataFrame,
    confidence_calibration: dict[str, Any],
    min_score: float,
    max_findings: int,
    timeframe: str,
) -> dict[str, Any]:
    severity_rank = {"fail": 0, "block": 1}
    findings: list[dict[str, Any]] = []

    def add_finding(
        *,
        code: str,
        severity: Literal["block", "fail"],
        category: str,
        message: str,
        observed: Any,
        expected: Any,
    ) -> None:
        findings.append(
            {
                "code": code,
                "severity": severity,
                "category": category,
                "message": message,
                "observed": observed,
                "expected": expected,
            }
        )

    recommendation = strategy_signal.recommendation
    rationale = strategy_signal.rationale.lower()
    bullish_keywords = ("buy", "bull", "uptrend", "upside", "long", "breakout")
    bearish_keywords = ("sell", "bear", "downtrend", "downside", "short", "breakdown")
    bullish_score = sum(1 for token in bullish_keywords if token in rationale)
    bearish_score = sum(1 for token in bearish_keywords if token in rationale)

    if recommendation == "buy" and bearish_score > bullish_score:
        add_finding(
            code="contradiction_rationale_direction",
            severity="fail",
            category="rationale",
            message="Rationale language trends bearish while recommendation is buy.",
            observed={"bullish_score": bullish_score, "bearish_score": bearish_score},
            expected="bullish_score >= bearish_score",
        )
    if recommendation == "sell" and bullish_score > bearish_score:
        add_finding(
            code="contradiction_rationale_direction",
            severity="fail",
            category="rationale",
            message="Rationale language trends bullish while recommendation is sell.",
            observed={"bullish_score": bullish_score, "bearish_score": bearish_score},
            expected="bearish_score >= bullish_score",
        )

    regime = strategy_signal.regime.strip().lower()
    if recommendation == "buy" and ("bear" in regime or "downtrend" in regime):
        add_finding(
            code="contradiction_regime_recommendation",
            severity="block",
            category="regime",
            message="Recommendation buy is inconsistent with detected bearish regime.",
            observed=regime,
            expected="bull/uptrend or neutral regime",
        )
    if recommendation == "sell" and ("bull" in regime or "uptrend" in regime):
        add_finding(
            code="contradiction_regime_recommendation",
            severity="block",
            category="regime",
            message="Recommendation sell is inconsistent with detected bullish regime.",
            observed=regime,
            expected="bear/downtrend or neutral regime",
        )

    votes = _normalize_probability_votes(strategy_signal.indicator_votes)
    alignment = float(votes.get(recommendation, 0.0))
    if recommendation in {"buy", "sell"} and alignment < 0.34:
        add_finding(
            code="downgrade_indicator_alignment_low",
            severity="block",
            category="indicator_alignment",
            message="Indicator vote alignment is weak for the recommended direction.",
            observed=alignment,
            expected=">= 0.34",
        )

    if bool(confidence_calibration.get("contradiction_detected", False)):
        contradiction_severity = str(confidence_calibration.get("contradiction_severity", "block")).lower()
        severity: Literal["block", "fail"] = "fail" if contradiction_severity == "fail" else "block"
        add_finding(
            code="contradiction_confidence_walkforward",
            severity=severity,
            category="confidence_calibration",
            message="Confidence calibration detected a walk-forward contradiction.",
            observed={
                "quality_band": confidence_calibration.get("walkforward_quality_band"),
                "walkforward_sharpe": confidence_calibration.get("walkforward_sharpe"),
                "calibrated_confidence": confidence_calibration.get("calibrated_confidence"),
            },
            expected="no contradiction between actionable direction and walk-forward evidence",
        )

    frame_timestamps = pd.to_datetime(market_frame["timestamp"], utc=True, errors="coerce").dropna()
    if not frame_timestamps.empty:
        latest_bar = frame_timestamps.max().to_pydatetime()
        interval_seconds = _timeframe_delta(timeframe).total_seconds()
        stale_after_seconds = max(interval_seconds * 2.5, 3600)
        historical_replay_cutoff_seconds = 30 * 24 * 3600
        age_seconds = max(0.0, (datetime.now(timezone.utc) - latest_bar).total_seconds())
        if age_seconds <= historical_replay_cutoff_seconds and age_seconds > stale_after_seconds:
            add_finding(
                code="data_freshness_stale",
                severity="block",
                category="data_freshness",
                message="Latest market bar is older than the freshness threshold.",
                observed={"age_seconds": age_seconds, "latest_bar_utc": latest_bar.isoformat()},
                expected={"max_age_seconds": stale_after_seconds},
            )

    if not data_quality_signal.is_valid:
        add_finding(
            code="data_quality_invalid",
            severity="fail",
            category="data_quality",
            message="Data quality stage reported invalid input.",
            observed=data_quality_signal.anomalies,
            expected="no data quality anomalies",
        )

    penalty = 0.0
    for finding in findings:
        penalty += 0.35 if finding["severity"] == "fail" else 0.20
    score = float(np.clip(1.0 - penalty, 0.0, 1.0))

    if score < min_score:
        add_finding(
            code="self_critique_score_below_threshold",
            severity="block",
            category="self_critique",
            message="Aggregated self-critique score fell below configured threshold.",
            observed=score,
            expected=f">= {min_score}",
        )

    findings = sorted(findings, key=lambda item: (severity_rank.get(item["severity"], 99), item["code"]))
    if len(findings) > max_findings:
        findings = findings[:max_findings]

    reason_codes = [str(item["code"]) for item in findings]
    reason_code_details = {str(item["code"]): str(item["message"]) for item in findings}
    highest_severity: Literal["none", "block", "fail"]
    if any(item["severity"] == "fail" for item in findings):
        highest_severity = "fail"
    elif findings:
        highest_severity = "block"
    else:
        highest_severity = "none"

    passed = bool(score >= min_score and highest_severity != "fail")
    summary = (
        "No contradictions detected."
        if passed and not findings
        else f"{len(findings)} findings, highest_severity={highest_severity}, score={score:.3f}"
    )
    return {
        "contract": "self_critique_signal.v1",
        "run_id": run_id,
        "created_at_utc": _utc_now_iso(),
        "recommendation": recommendation,
        "strictness": {
            "min_score": min_score,
            "max_findings": max_findings,
        },
        "score": score,
        "pass": passed,
        "highest_severity": highest_severity,
        "reason_codes": reason_codes,
        "reason_code_details": reason_code_details,
        "findings": findings,
        "summary": summary,
    }


def _indicator_alignment_score(indicator_votes: dict[str, float], recommendation: Recommendation) -> float:
    if not indicator_votes:
        return 0.5
    votes = _normalize_probability_votes(indicator_votes)
    return float(np.clip(votes.get(recommendation, 0.0), 0.0, 1.0))


def _regime_alignment_adjustment(regime: str, recommendation: Recommendation) -> float:
    normalized = regime.strip().lower()
    if recommendation == "hold":
        return 0.05 if "range" in normalized or "sideways" in normalized else 0.0
    if "volatile" in normalized:
        return -0.05
    if recommendation == "buy":
        if "bull" in normalized or "uptrend" in normalized:
            return 0.07
        if "bear" in normalized or "downtrend" in normalized:
            return -0.10
    if recommendation == "sell":
        if "bear" in normalized or "downtrend" in normalized:
            return 0.07
        if "bull" in normalized or "uptrend" in normalized:
            return -0.10
    return 0.0

def _walkforward_quality_band(quality_score: float) -> Literal["high", "medium", "low", "very_low"]:
    score = float(np.clip(quality_score, 0.0, 1.0))
    if score >= 0.70:
        return "high"
    if score >= 0.55:
        return "medium"
    if score >= 0.40:
        return "low"
    return "very_low"


def _quality_band_sharpe_buffer(quality_band: Literal["high", "medium", "low", "very_low"]) -> float:
    if quality_band == "high":
        return 0.00
    if quality_band == "medium":
        return 0.05
    if quality_band == "low":
        return 0.10
    return 0.20


def _quality_band_contradiction_penalty(
    quality_band: Literal["high", "medium", "low", "very_low"],
) -> float:
    if quality_band == "high":
        return 0.85
    if quality_band == "medium":
        return 0.75
    if quality_band == "low":
        return 0.60
    return 0.45


def _quality_band_contradiction_severity(
    quality_band: Literal["high", "medium", "low", "very_low"],
) -> Literal["none", "block", "fail"]:
    if quality_band == "very_low":
        return "fail"
    return "block"


def _calibrate_confidence(
    *,
    run_id: str,
    strategy_signal: StrategyProposalSignal,
    walkforward_evaluation: dict[str, Any],
    min_walkforward_sharpe: float,
    confidence_floor: float,
    confidence_ceiling: float,
) -> dict[str, Any]:
    lower = min(confidence_floor, confidence_ceiling)
    upper = max(confidence_floor, confidence_ceiling)
    raw_confidence = float(np.clip(strategy_signal.confidence, 0.0, 1.0))
    walkforward_quality = float(np.clip(walkforward_evaluation.get("quality_score", 0.0), 0.0, 1.0))
    walkforward_quality_band = _walkforward_quality_band(walkforward_quality)
    walkforward_sharpe = float(walkforward_evaluation.get("aggregate_sharpe", 0.0))
    sharpe_buffer = _quality_band_sharpe_buffer(walkforward_quality_band)
    required_sharpe = min_walkforward_sharpe + sharpe_buffer
    contradiction_penalty = _quality_band_contradiction_penalty(walkforward_quality_band)
    indicator_alignment = _indicator_alignment_score(
        strategy_signal.indicator_votes,
        strategy_signal.recommendation,
    )
    regime_adjustment = _regime_alignment_adjustment(
        strategy_signal.regime,
        strategy_signal.recommendation,
    )

    reason_codes: list[str] = []
    calibrated_confidence = raw_confidence
    calibrated_confidence += (indicator_alignment - 0.5) * 0.20
    calibrated_confidence += (walkforward_quality - 0.5) * 0.35
    calibrated_confidence += regime_adjustment

    contradiction_detected = False
    contradiction_count = 0
    contradiction_severity: Literal["none", "block", "fail"] = "none"
    recommendation = strategy_signal.recommendation
    if recommendation in {"buy", "sell"}:
        if walkforward_quality_band == "very_low":
            contradiction_detected = True
            contradiction_count += 1
            contradiction_severity = "fail"
            reason_codes.append(f"{recommendation}_walkforward_quality_very_low")
        if walkforward_sharpe < required_sharpe:
            contradiction_detected = True
            contradiction_count += 1
            contradiction_severity = _quality_band_contradiction_severity(walkforward_quality_band)
            reason_codes.append("walkforward_sharpe_below_threshold")
            reason_codes.append("walkforward_sharpe_below_quality_band_threshold")
            reason_codes.append(
                f"{recommendation}_walkforward_{walkforward_quality_band}_sharpe_contradiction"
            )
        if contradiction_detected:
            calibrated_confidence *= contradiction_penalty
            reason_codes.append(f"walkforward_contradiction_{contradiction_severity}")

    if not strategy_signal.feature_snapshot:
        calibrated_confidence *= 0.95
        reason_codes.append("phase1_feature_snapshot_missing")
    if not strategy_signal.indicator_votes:
        calibrated_confidence *= 0.95
        reason_codes.append("phase1_indicator_votes_missing")
    if strategy_signal.regime in {"", "unknown"}:
        calibrated_confidence *= 0.95
        reason_codes.append("phase1_regime_missing")

    calibrated_confidence = float(np.clip(calibrated_confidence, lower, upper))
    diagnostics = {
        "reliability_bins": walkforward_evaluation.get("diagnostics", {}).get("reliability_bins", []),
        "confidence_deciles": walkforward_evaluation.get("diagnostics", {}).get("confidence_deciles", []),
        "contradiction_counts": {
            "current_run": contradiction_count,
            "severity_block": 1 if contradiction_detected and contradiction_severity == "block" else 0,
            "severity_fail": 1 if contradiction_detected and contradiction_severity == "fail" else 0,
        },
    }
    return {
        "contract": "confidence_calibration.v1",
        "run_id": run_id,
        "created_at_utc": _utc_now_iso(),
        "recommendation": strategy_signal.recommendation,
        "raw_confidence": raw_confidence,
        "calibrated_confidence": calibrated_confidence,
        "walkforward_quality_score": walkforward_quality,
        "walkforward_quality_band": walkforward_quality_band,
        "walkforward_sharpe": walkforward_sharpe,
        "contradiction_detected": contradiction_detected,
        "contradiction_severity": contradiction_severity,
        "contradiction_policy": {
            "required_walkforward_sharpe": required_sharpe,
            "quality_band_sharpe_buffer": sharpe_buffer,
            "quality_band_penalty_multiplier": contradiction_penalty,
        },
        "phase1_inputs": {
            "regime": strategy_signal.regime,
            "indicator_votes": strategy_signal.indicator_votes,
            "feature_snapshot": strategy_signal.feature_snapshot,
            "reason_codes": strategy_signal.reason_codes,
        },
        "components": {
            "indicator_alignment": indicator_alignment,
            "regime_adjustment": regime_adjustment,
            "walkforward_quality": walkforward_quality,
        },
        "reason_codes": sorted(set(reason_codes)),
        "diagnostics": diagnostics,
    }


def _strategy_prompt(
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    snapshot: dict[str, float | int | str],
    phase1_context: dict[str, Any],
    source_data_path: Path,
    source_data_sha256: str,
) -> str:
    return "\n".join(
        [
            "You are strategy-agent for a deterministic crypto quant pipeline.",
            "Return STRICT JSON only with keys:",
            '{"recommendation":"buy|sell|hold","confidence":0.0,"fast_window":20,"slow_window":50,"rationale":"...","indicator_votes":{"buy":0.0,"sell":0.0,"hold":1.0},"regime":"...","feature_snapshot":{"key":"value"},"reason_codes":["..."]}',
            "Rules:",
            "- confidence must be between 0 and 1.",
            "- fast_window and slow_window must be positive integers with slow_window > fast_window.",
            "- Keep rationale concise and risk-aware.",
            f"Exchange: {exchange}",
            f"Symbol: {symbol}",
            f"Timeframe: {timeframe}",
            f"Source data path: {source_data_path}",
            f"Source data sha256: {source_data_sha256}",
            f"Market snapshot: {json.dumps(snapshot, sort_keys=True)}",
            f"Phase-1 feature context (must be consumed): {json.dumps(phase1_context, sort_keys=True)}",
        ]
    )


def _normalize_report_verbosity(value: Any) -> Literal["compact", "standard", "verbose"]:
    if not isinstance(value, str):
        return "standard"
    lowered = value.strip().lower()
    if lowered == "compact":
        return "compact"
    if lowered == "verbose":
        return "verbose"
    if lowered == "standard":
        return "standard"
    return "standard"


def _ops_prompt(context: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are ops-report-agent for a deterministic crypto quant pipeline.",
            "Write concise markdown with sections:",
            "## Run summary",
            "## Self-critique",
            "## Deterministic gate outcome",
            "## Decision trace",
            "## Ensemble evidence",
            "## Paper intent",
            "## Paper execution",
            "## Follow-ups",
            "Use only facts from the context JSON and do not invent metrics.",
            json.dumps(context, indent=2, sort_keys=True),
        ]
    )


def _deterministic_ops_markdown(context: dict[str, Any]) -> str:
    verbosity = _normalize_report_verbosity(context.get("report_verbosity"))
    risk_approved = bool(context["risk"]["approved"])
    intent_status = str(context["intent"]["status"])
    execution_status = str(context["execution"]["status"])
    self_critique = context.get("self_critique", {})
    decision_trace = list(context.get("risk", {}).get("decision_trace", []))
    gate_transitions = list(context.get("risk", {}).get("gate_transition_sequence", []))
    reason_code_details = dict(context.get("risk", {}).get("reason_code_details", {}))

    lines = [
        f"# Agent Plane Ops Report ({context['run_id']})",
        "",
        "## Run summary",
        f"- Exchange: `{context['scope']['exchange']}`",
        f"- Symbol: `{context['scope']['symbol']}`",
        f"- Timeframe: `{context['scope']['timeframe']}`",
        f"- Strategy recommendation: `{context['proposal']['recommendation']}`",
        f"- Backtest status: `{context['backtest']['status']}`",
        f"- Ensemble mode: `{context.get('ensemble', {}).get('mode')}`",
        f"- Selected arms: `{context['proposal'].get('selected_arms', [])}`",
        "",
        "## Self-critique",
        f"- Pass: `{self_critique.get('pass')}`",
        f"- Score: `{self_critique.get('score')}`",
        f"- Highest severity: `{self_critique.get('highest_severity')}`",
        f"- Reason codes: `{self_critique.get('reason_codes', [])}`",
        "",
        "## Deterministic gate outcome",
        f"- Approved: `{risk_approved}`",
        f"- Deterministic gate: `{context['risk'].get('deterministic_gate')}`",
        f"- Reason codes: `{context['risk']['reason_codes']}`",
    ]
    if verbosity in {"standard", "verbose"} and reason_code_details:
        lines.append("- Reason-code details:")
        for code in sorted(reason_code_details):
            lines.append(f"  - `{code}`: {reason_code_details[code]}")

    lines.extend(
        [
            "",
            "## Decision trace",
            f"- Gate transition sequence: `{gate_transitions}`",
        ]
    )
    trace_rows = decision_trace if verbosity == "verbose" else decision_trace[:6]
    if trace_rows:
        for row in trace_rows:
            lines.append(
                "- "
                + f"`{row.get('gate')}` `{row.get('metric')}` observed=`{row.get('observed')}` "
                + f"threshold=`{row.get('threshold')}` pass=`{row.get('pass')}` "
                + f"reason=`{row.get('reason_code')}`"
            )
    if verbosity != "verbose" and len(decision_trace) > len(trace_rows):
        lines.append(f"- ... truncated `{len(decision_trace) - len(trace_rows)}` trace rows")

    lines.extend(
        [
            "",
            "## Ensemble evidence",
            f"- Arm weights: `{context['proposal'].get('arm_weights', {})}`",
            f"- Ensemble reason codes: `{context['proposal'].get('ensemble_reason_codes', [])}`",
            f"- Performance update path: `{context.get('ensemble', {}).get('performance_update_path')}`",
            f"- Weight state path: `{context.get('ensemble', {}).get('weight_state_path')}`",
            "",
            "## Paper intent",
            f"- Status: `{intent_status}`",
            f"- Action: `{context['intent']['action']}`",
            f"- Notional USD: `{context['intent']['notional_usd']}`",
            "",
            "## Paper execution",
            f"- Status: `{execution_status}`",
            f"- Executed action: `{context['execution']['executed_action']}`",
            f"- Executed notional USD: `{context['execution']['executed_notional_usd']}`",
            f"- Fee USD: `{context['execution']['fee_usd']}`",
            f"- Cash after USD: `{context['execution']['cash_after_usd']}`",
            f"- Position qty after: `{context['execution']['position_qty_after']}`",
            f"- Arm attribution: `{context['execution'].get('arm_attribution', {})}`",
            "",
            "## Follow-ups",
            "- Validate model availability in local Ollama if fallback mode was used.",
            "- Review deterministic risk thresholds for false positives/negatives.",
        ]
    )
    return "\n".join(lines)


def run_agent_plane(settings: Settings, config: AgentPlaneConfig) -> AgentPlaneRunResult:
    run_id, run_dir = _new_agent_plane_run_dir(settings.quant_data_root)
    logger.info("Agent-plane run initialized run_id=%s run_dir=%s", run_id, run_dir)

    source_data_path = (
        config.source_data_path
        if config.source_data_path is not None
        else latest_raw_dataset(settings.quant_data_root, config.exchange, config.symbol, config.timeframe)
    )
    source_data_path = source_data_path.expanduser().resolve()
    source_data_sha256 = _sha256_file(source_data_path)

    ollama = OllamaClient(
        settings.ollama_base_url,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    step_records: list[StepExecutionRecord] = []

    data_quality_path = run_dir / "data_quality_signal.json"
    phase1_feature_context_path = run_dir / "phase1_feature_context.json"
    strategy_signal_path = run_dir / "strategy_proposal_signal.json"
    backtest_evaluation_path = run_dir / "backtest_evaluation.json"
    walkforward_evaluation_path = run_dir / "walkforward_evaluation.json"
    confidence_calibration_path = run_dir / "confidence_calibration.json"
    self_critique_signal_path = run_dir / "self_critique_signal.json"
    risk_decision_path = run_dir / "risk_decision.json"
    paper_trade_intent_path = run_dir / "paper_trade_intent.json"
    paper_trade_execution_path = run_dir / "paper_trade_execution.json"
    ops_report_markdown_path = run_dir / "ops_report.md"
    ops_report_contract_path = run_dir / "ops_report_contract.json"
    run_manifest_path = run_dir / "run_manifest.json"
    ensemble_performance_update_path = run_dir / "ensemble_performance_update.json"
    ensemble_weight_state_path = (
        settings.quant_data_root / "paper-trading" / "state" / "ensemble_weight_state.json"
    )
    ensemble_state_path = (
        ensemble_weight_state_path
        if config.source_data_path is None
        else (run_dir / "ensemble_weight_state_replay.json")
    )

    def data_quality_runner() -> tuple[DataQualitySignal, pd.DataFrame]:
        frame = _load_market_frame(source_data_path)
        required_columns = ("open", "high", "low", "close", "volume")
        missing_columns = [col for col in required_columns if col not in frame.columns]
        null_value_count = (
            int(frame[list(required_columns)].isna().sum().sum()) if not missing_columns else len(frame)
        )
        duplicate_timestamp_count = int(frame["timestamp"].duplicated().sum())
        gap_count = int((frame["timestamp"].diff().dropna() > (_timeframe_delta(config.timeframe) * 1.5)).sum())

        anomalies: list[str] = []
        if missing_columns:
            anomalies.append(f"missing_columns={missing_columns}")
        if len(frame) < config.minimum_bars:
            anomalies.append(f"insufficient_bars={len(frame)}<{config.minimum_bars}")
        if null_value_count > 0:
            anomalies.append(f"null_value_count={null_value_count}")
        if duplicate_timestamp_count > 0:
            anomalies.append(f"duplicate_timestamp_count={duplicate_timestamp_count}")
        if gap_count > 0:
            anomalies.append(f"gap_count={gap_count}")

        signal = DataQualitySignal(
            contract="data_quality_signal.v1",
            run_id=run_id,
            created_at_utc=_utc_now_iso(),
            exchange=config.exchange,
            symbol=config.symbol,
            timeframe=config.timeframe,
            source_data_path=str(source_data_path),
            source_data_sha256=source_data_sha256,
            bar_count=int(len(frame)),
            gap_count=gap_count,
            null_value_count=null_value_count,
            duplicate_timestamp_count=duplicate_timestamp_count,
            is_valid=len(anomalies) == 0,
            anomalies=anomalies,
        )
        return signal, frame

    (data_quality_signal, market_frame), data_quality_step = _run_step_with_retries(
        run_dir=run_dir,
        step_name="data-quality-agent",
        max_retries=config.step_retries,
        runner=data_quality_runner,
    )
    write_contract(data_quality_path, data_quality_signal)
    step_records.append(data_quality_step)

    def phase1_feature_runner() -> dict[str, Any]:
        return {
            "contract": "phase1_feature_context.v1",
            "run_id": run_id,
            "created_at_utc": _utc_now_iso(),
            "exchange": config.exchange,
            "symbol": config.symbol,
            "timeframe": config.timeframe,
            "input_data_path": str(source_data_path),
            "input_data_sha256": source_data_sha256,
            **_compute_phase1_feature_context(market_frame),
        }

    phase1_feature_context, phase1_feature_step = _run_step_with_retries(
        run_dir=run_dir,
        step_name="phase1-feature-context",
        max_retries=config.step_retries,
        runner=phase1_feature_runner,
    )
    _write_json(phase1_feature_context_path, phase1_feature_context)
    step_records.append(phase1_feature_step)

    def strategy_runner() -> StrategyProposalSignal:
        available_models = ollama.list_models()
        if available_models and config.strategy_model not in available_models:
            raise RuntimeError(
                f"Configured strategy model `{config.strategy_model}` not found. Available models: {available_models}"
            )

        prompt = _strategy_prompt(
            exchange=config.exchange,
            symbol=config.symbol,
            timeframe=config.timeframe,
            snapshot=_market_snapshot(market_frame),
            phase1_context=phase1_feature_context,
            source_data_path=source_data_path,
            source_data_sha256=source_data_sha256,
        )
        raw_response = ollama.generate(
            model=config.strategy_model,
            prompt=prompt,
            system="Respond with strict JSON only.",
            temperature=0.1,
            format_json=True,
        )
        payload = json.loads(raw_response)
        if not isinstance(payload, dict):
            raise RuntimeError("Strategy model response was not a JSON object.")

        recommendation = _coerce_recommendation(payload.get("recommendation"))
        confidence = float(payload.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        fast_window, slow_window, warnings = _sanitize_windows(
            payload.get("fast_window"), payload.get("slow_window")
        )
        rationale = str(payload.get("rationale", "")).strip()
        if not rationale:
            raise RuntimeError("Strategy model returned empty rationale.")
        indicator_votes = _normalize_indicator_votes(
            payload.get("indicator_votes"),
            fallback=dict(phase1_feature_context.get("indicator_votes", {})),
        )
        regime = _normalize_regime(
            payload.get("regime"),
            fallback=str(phase1_feature_context.get("regime", "unknown")),
        )
        feature_snapshot = _normalize_feature_snapshot(
            payload.get("feature_snapshot"),
            fallback=dict(phase1_feature_context.get("feature_snapshot", {})),
        )
        reason_codes = _normalize_reason_codes(
            payload.get("reason_codes"),
            fallback=list(phase1_feature_context.get("reason_codes", [])),
        )

        return StrategyProposalSignal(
            contract="strategy_proposal_signal.v1",
            run_id=run_id,
            created_at_utc=_utc_now_iso(),
            source="ollama",
            model=config.strategy_model,
            exchange=config.exchange,
            symbol=config.symbol,
            timeframe=config.timeframe,
            input_data_path=str(source_data_path),
            input_data_sha256=source_data_sha256,
            recommendation=recommendation,
            confidence=confidence,
            fast_window=fast_window,
            slow_window=slow_window,
            rationale=rationale,
            raw_model_response=raw_response,
            warnings=warnings,
            indicator_votes=indicator_votes,
            regime=regime,
            feature_snapshot=feature_snapshot,
            reason_codes=reason_codes,
        )

    def strategy_fallback(errors: list[str]) -> StrategyProposalSignal:
        return StrategyProposalSignal(
            contract="strategy_proposal_signal.v1",
            run_id=run_id,
            created_at_utc=_utc_now_iso(),
            source="fallback",
            model=config.strategy_model,
            exchange=config.exchange,
            symbol=config.symbol,
            timeframe=config.timeframe,
            input_data_path=str(source_data_path),
            input_data_sha256=source_data_sha256,
            recommendation="hold",
            confidence=0.0,
            fast_window=20,
            slow_window=50,
            rationale="Fallback strategy due to model unavailability or invalid output.",
            raw_model_response=None,
            warnings=errors,
            indicator_votes=dict(phase1_feature_context.get("indicator_votes", {})),
            regime=str(phase1_feature_context.get("regime", "unknown")),
            feature_snapshot=dict(phase1_feature_context.get("feature_snapshot", {})),
            reason_codes=list(phase1_feature_context.get("reason_codes", [])),
        )

    strategy_signal, strategy_step = _run_step_with_retries(
        run_dir=run_dir,
        step_name="strategy-agent",
        max_retries=config.step_retries,
        runner=strategy_runner,
        fallback=strategy_fallback,
    )
    step_records.append(strategy_step)
    llm_strategy_signal = strategy_signal

    ensemble_mode = _normalize_ensemble_mode(config.ensemble_mode)
    enabled_arms = _normalize_enabled_arms(config.ensemble_enabled_arms)
    ensemble_state = _load_ensemble_weight_state(
        path=ensemble_state_path,
        enabled_arms=enabled_arms,
        decay_horizon=max(4, int(config.ensemble_decay_horizon)),
        exploration_weight=max(0.0, float(config.ensemble_exploration_weight)),
    )
    arm_votes = _build_strategy_arm_votes(
        market_frame=market_frame,
        phase1_context=phase1_feature_context,
        llm_strategy_signal=llm_strategy_signal,
        fast_window=llm_strategy_signal.fast_window,
        slow_window=llm_strategy_signal.slow_window,
        enabled_arms=enabled_arms,
    )
    arm_weights, weight_reason_codes = _compute_arm_weights(
        ensemble_mode=ensemble_mode,
        enabled_arms=enabled_arms,
        arm_votes=arm_votes,
        ensemble_state=ensemble_state,
        exploration_weight=max(0.0, float(config.ensemble_exploration_weight)),
    )
    (
        ensemble_recommendation,
        ensemble_confidence,
        ensemble_action_scores,
        selected_arms,
        combine_reason_codes,
    ) = _combine_arm_votes(
        arm_votes=arm_votes,
        arm_weights=arm_weights,
    )
    ensemble_reason_codes = sorted(
        set([*weight_reason_codes, *combine_reason_codes, f"ensemble_mode_{ensemble_mode}"])
    )
    strategy_signal = StrategyProposalSignal(
        contract="strategy_proposal_signal.v1",
        run_id=llm_strategy_signal.run_id,
        created_at_utc=_utc_now_iso(),
        source=llm_strategy_signal.source,
        model=llm_strategy_signal.model,
        exchange=llm_strategy_signal.exchange,
        symbol=llm_strategy_signal.symbol,
        timeframe=llm_strategy_signal.timeframe,
        input_data_path=llm_strategy_signal.input_data_path,
        input_data_sha256=llm_strategy_signal.input_data_sha256,
        recommendation=ensemble_recommendation,
        confidence=ensemble_confidence,
        fast_window=llm_strategy_signal.fast_window,
        slow_window=llm_strategy_signal.slow_window,
        rationale=(
            f"Ensemble decision ({ensemble_mode}) over arms {selected_arms}. "
            f"LLM context recommendation was {llm_strategy_signal.recommendation}. "
            f"{llm_strategy_signal.rationale}"
        ),
        raw_model_response=llm_strategy_signal.raw_model_response,
        warnings=list(llm_strategy_signal.warnings),
        indicator_votes=ensemble_action_scores,
        regime=llm_strategy_signal.regime,
        feature_snapshot=dict(llm_strategy_signal.feature_snapshot),
        reason_codes=sorted(
            set([*llm_strategy_signal.reason_codes, *ensemble_reason_codes])
        ),
        arm_votes=arm_votes,
        arm_weights=arm_weights,
        selected_arms=selected_arms,
        ensemble_reason_codes=ensemble_reason_codes,
    )
    write_contract(strategy_signal_path, strategy_signal)

    def backtest_runner() -> BacktestEvaluation:
        backtest_result = run_ensemble_backtest(
            settings=settings,
            exchange=config.exchange,
            symbol=config.symbol,
            timeframe=config.timeframe,
            fast_window=max(2, strategy_signal.fast_window),
            slow_window=max(3, strategy_signal.slow_window),
            enabled_arms=tuple(strategy_signal.selected_arms),
            arm_weights=dict(strategy_signal.arm_weights),
            llm_recommendation=llm_strategy_signal.recommendation,
            ensemble_mode=ensemble_mode,
            source_data_path=source_data_path,
            archive_run=False,
        )
        metrics_payload = backtest_result.ensemble_metrics or backtest_result.metrics
        return BacktestEvaluation(
            contract="backtest_evaluation.v1",
            run_id=run_id,
            created_at_utc=_utc_now_iso(),
            strategy=ENSEMBLE_STRATEGY_NAME,
            exchange=config.exchange,
            symbol=config.symbol,
            timeframe=config.timeframe,
            source_data_path=str(backtest_result.source_data_path),
            source_data_sha256=backtest_result.source_data_sha256,
            backtest_status="success",
            backtest_run_dir=str(backtest_result.run_dir),
            metrics_path=str(backtest_result.metrics_path),
            manifest_path=str(backtest_result.manifest_path),
            total_return=float(metrics_payload["total_return"]),
            annualized_return=float(metrics_payload["annualized_return"]),
            sharpe=float(metrics_payload["sharpe"]),
            max_drawdown=float(metrics_payload["max_drawdown"]),
            signal_flips=int(metrics_payload["signal_flips"]),
            bars=int(metrics_payload["bars"]),
            error_message=None,
            arm_metrics=dict(backtest_result.arm_metrics),
            ensemble_metrics=dict(backtest_result.ensemble_metrics),
            arm_attribution_path=(
                str(backtest_result.arm_attribution_path)
                if backtest_result.arm_attribution_path is not None
                else None
            ),
        )

    def backtest_fallback(errors: list[str]) -> BacktestEvaluation:
        return BacktestEvaluation(
            contract="backtest_evaluation.v1",
            run_id=run_id,
            created_at_utc=_utc_now_iso(),
            strategy=ENSEMBLE_STRATEGY_NAME,
            exchange=config.exchange,
            symbol=config.symbol,
            timeframe=config.timeframe,
            source_data_path=str(source_data_path),
            source_data_sha256=source_data_sha256,
            backtest_status="failed",
            backtest_run_dir=None,
            metrics_path=None,
            manifest_path=None,
            total_return=None,
            annualized_return=None,
            sharpe=None,
            max_drawdown=None,
            signal_flips=None,
            bars=None,
            error_message=errors[-1] if errors else "Unknown backtest failure",
            arm_metrics={},
            ensemble_metrics={},
            arm_attribution_path=None,
        )

    backtest_evaluation, backtest_step = _run_step_with_retries(
        run_dir=run_dir,
        step_name="backtest-agent",
        max_retries=config.step_retries,
        runner=backtest_runner,
        fallback=backtest_fallback,
    )
    write_contract(backtest_evaluation_path, backtest_evaluation)
    step_records.append(backtest_step)
    backtest_arm_attribution_path = (
        Path(backtest_evaluation.arm_attribution_path)
        if backtest_evaluation.arm_attribution_path
        else None
    )

    def walkforward_runner() -> dict[str, Any]:
        return _run_walkforward_evaluation(
            frame=market_frame,
            exchange=config.exchange,
            symbol=config.symbol,
            timeframe=config.timeframe,
            source_data_path=source_data_path,
            source_data_sha256=source_data_sha256,
            fast_window=strategy_signal.fast_window,
            slow_window=strategy_signal.slow_window,
            train_bars=max(20, config.walk_forward_train_bars),
            validate_bars=max(5, config.walk_forward_validate_bars),
            step_bars=max(5, config.walk_forward_step_bars),
            min_windows=max(1, config.walk_forward_min_windows),
        )

    def walkforward_fallback(errors: list[str]) -> dict[str, Any]:
        return {
            "contract": "walkforward_evaluation.v1",
            "created_at_utc": _utc_now_iso(),
            "exchange": config.exchange,
            "symbol": config.symbol,
            "timeframe": config.timeframe,
            "source_data_path": str(source_data_path),
            "source_data_sha256": source_data_sha256,
            "strategy": STRATEGY_NAME,
            "fast_window": strategy_signal.fast_window,
            "slow_window": strategy_signal.slow_window,
            "train_bars": max(20, config.walk_forward_train_bars),
            "validate_bars": max(5, config.walk_forward_validate_bars),
            "step_bars": max(5, config.walk_forward_step_bars),
            "window_count": 0,
            "aggregate_total_return": 0.0,
            "aggregate_sharpe": 0.0,
            "aggregate_max_drawdown": 0.0,
            "aggregate_hit_rate": 0.0,
            "stability_score": 0.0,
            "quality_score": 0.0,
            "windows": [],
            "diagnostics": {
                "reliability_bins": [],
                "confidence_deciles": [],
            },
            "warnings": errors,
        }

    walkforward_evaluation, walkforward_step = _run_step_with_retries(
        run_dir=run_dir,
        step_name="walkforward-agent",
        max_retries=config.step_retries,
        runner=walkforward_runner,
        fallback=walkforward_fallback,
    )
    _write_json(walkforward_evaluation_path, walkforward_evaluation)
    step_records.append(walkforward_step)

    def calibration_runner() -> dict[str, Any]:
        return _calibrate_confidence(
            run_id=run_id,
            strategy_signal=strategy_signal,
            walkforward_evaluation=walkforward_evaluation,
            min_walkforward_sharpe=config.calibration_min_walkforward_sharpe,
            confidence_floor=config.calibration_confidence_floor,
            confidence_ceiling=config.calibration_confidence_ceiling,
        )

    def calibration_fallback(errors: list[str]) -> dict[str, Any]:
        raw_confidence = float(np.clip(strategy_signal.confidence, 0.0, 1.0))
        lower = min(config.calibration_confidence_floor, config.calibration_confidence_ceiling)
        upper = max(config.calibration_confidence_floor, config.calibration_confidence_ceiling)
        calibrated_confidence = float(np.clip(raw_confidence, lower, upper))
        walkforward_quality = 0.0
        walkforward_quality_band = _walkforward_quality_band(walkforward_quality)
        return {
            "contract": "confidence_calibration.v1",
            "run_id": run_id,
            "created_at_utc": _utc_now_iso(),
            "recommendation": strategy_signal.recommendation,
            "raw_confidence": raw_confidence,
            "calibrated_confidence": calibrated_confidence,
            "walkforward_quality_score": walkforward_quality,
            "walkforward_quality_band": walkforward_quality_band,
            "walkforward_sharpe": 0.0,
            "contradiction_detected": False,
            "contradiction_severity": "none",
            "contradiction_policy": {
                "required_walkforward_sharpe": config.calibration_min_walkforward_sharpe,
                "quality_band_sharpe_buffer": _quality_band_sharpe_buffer(walkforward_quality_band),
                "quality_band_penalty_multiplier": _quality_band_contradiction_penalty(
                    walkforward_quality_band
                ),
            },
            "phase1_inputs": {
                "regime": strategy_signal.regime,
                "indicator_votes": strategy_signal.indicator_votes,
                "feature_snapshot": strategy_signal.feature_snapshot,
                "reason_codes": strategy_signal.reason_codes,
            },
            "components": {
                "indicator_alignment": 0.5,
                "regime_adjustment": 0.0,
                "walkforward_quality": 0.0,
            },
            "reason_codes": ["calibration_fallback", *errors],
            "diagnostics": {
                "reliability_bins": [],
                "confidence_deciles": [],
                "contradiction_counts": {
                    "current_run": 0,
                    "severity_block": 0,
                    "severity_fail": 0,
                },
            },
        }

    confidence_calibration, calibration_step = _run_step_with_retries(
        run_dir=run_dir,
        step_name="confidence-calibration-agent",
        max_retries=config.step_retries,
        runner=calibration_runner,
        fallback=calibration_fallback,
    )
    _write_json(confidence_calibration_path, confidence_calibration)
    step_records.append(calibration_step)

    def self_critique_runner() -> dict[str, Any]:
        return _run_self_critique(
            run_id=run_id,
            strategy_signal=strategy_signal,
            data_quality_signal=data_quality_signal,
            market_frame=market_frame,
            confidence_calibration=confidence_calibration,
            min_score=config.self_critique_min_score,
            max_findings=max(1, config.self_critique_max_findings),
            timeframe=config.timeframe,
        )

    def self_critique_fallback(errors: list[str]) -> dict[str, Any]:
        return {
            "contract": "self_critique_signal.v1",
            "run_id": run_id,
            "created_at_utc": _utc_now_iso(),
            "recommendation": strategy_signal.recommendation,
            "strictness": {
                "min_score": config.self_critique_min_score,
                "max_findings": max(1, config.self_critique_max_findings),
            },
            "score": 0.0,
            "pass": False,
            "highest_severity": "fail",
            "reason_codes": ["self_critique_unavailable", *errors],
            "reason_code_details": {
                "self_critique_unavailable": "Self-critique stage failed and fallback was used.",
            },
            "findings": [
                {
                    "code": "self_critique_unavailable",
                    "severity": "fail",
                    "category": "self_critique",
                    "message": "Self-critique stage failed and fallback was used.",
                    "observed": errors,
                    "expected": "self-critique stage execution success",
                }
            ],
            "summary": "Self-critique fallback triggered.",
        }

    self_critique_signal, self_critique_step = _run_step_with_retries(
        run_dir=run_dir,
        step_name="self-critique-agent",
        max_retries=config.step_retries,
        runner=self_critique_runner,
        fallback=self_critique_fallback,
    )
    _write_json(self_critique_signal_path, self_critique_signal)
    step_records.append(self_critique_step)

    def risk_runner() -> RiskDecision:
        fail_reasons: list[str] = []
        block_reasons: list[str] = []
        reason_code_details: dict[str, str] = {}
        decision_trace: list[dict[str, Any]] = []
        gate_transition_sequence: list[str] = ["start"]
        calibration_reason_codes = [str(code) for code in confidence_calibration.get("reason_codes", [])]
        walkforward_quality_score = float(confidence_calibration.get("walkforward_quality_score", 0.0))
        walkforward_quality_band = str(
            confidence_calibration.get("walkforward_quality_band", _walkforward_quality_band(walkforward_quality_score))
        )
        contradiction_detected = bool(confidence_calibration.get("contradiction_detected", False))
        contradiction_severity = str(confidence_calibration.get("contradiction_severity", "none")).lower()
        contradiction_count = int(
            confidence_calibration.get("diagnostics", {})
            .get("contradiction_counts", {})
            .get("current_run", 0)
        )
        self_critique_pass = bool(self_critique_signal.get("pass", False))
        self_critique_score = float(self_critique_signal.get("score", 0.0))
        self_critique_highest = str(self_critique_signal.get("highest_severity", "none")).lower()
        self_critique_reason_codes = [str(code) for code in self_critique_signal.get("reason_codes", [])]
        self_critique_reason_details = {
            str(code): str(detail)
            for code, detail in dict(self_critique_signal.get("reason_code_details", {})).items()
        }
        self_critique_severity_by_code = {
            str(item.get("code")): str(item.get("severity", "block")).lower()
            for item in list(self_critique_signal.get("findings", []))
            if item.get("code")
        }
        action = strategy_signal.recommendation
        arm_votes = dict(strategy_signal.arm_votes)
        arm_weights = _normalize_arm_weight_map(tuple(arm_votes.keys()), dict(strategy_signal.arm_weights))
        selected_arms = list(strategy_signal.selected_arms) or sorted(
            [arm for arm, weight in arm_weights.items() if weight > 0.0],
            key=lambda arm: arm_weights.get(arm, 0.0),
            reverse=True,
        )
        ensemble_reason_codes = [str(code) for code in list(strategy_signal.ensemble_reason_codes)]

        def register_reason(reason_code: str, severity: Literal["block", "fail"], detail: str) -> None:
            if severity == "fail":
                fail_reasons.append(reason_code)
            else:
                block_reasons.append(reason_code)
            reason_code_details.setdefault(reason_code, detail)

        def append_transition(marker: str) -> None:
            if not gate_transition_sequence or gate_transition_sequence[-1] != marker:
                gate_transition_sequence.append(marker)

        def trace_check(
            *,
            gate: str,
            metric: str,
            observed_value: Any,
            threshold: Any,
            passed: bool,
            failure_reason_code: str | None = None,
            failure_severity: Literal["block", "fail"] = "block",
            failure_detail: str = "",
        ) -> None:
            decision_trace.append(
                {
                    "gate": gate,
                    "metric": metric,
                    "observed": observed_value,
                    "threshold": threshold,
                    "pass": passed,
                    "severity": "none" if passed else failure_severity,
                    "reason_code": failure_reason_code if not passed else None,
                }
            )
            if passed:
                append_transition(f"{gate}:pass")
                return
            append_transition(f"{gate}:{failure_severity}")
            if failure_reason_code:
                register_reason(
                    reason_code=failure_reason_code,
                    severity=failure_severity,
                    detail=failure_detail or f"Gate `{gate}` failed metric `{metric}`.",
                )

        observed: dict[str, Any] = {
            "data_quality_valid": data_quality_signal.is_valid,
            "backtest_status": backtest_evaluation.backtest_status,
            "total_return": backtest_evaluation.total_return,
            "sharpe": backtest_evaluation.sharpe,
            "max_drawdown": backtest_evaluation.max_drawdown,
            "recommendation": action,
            "raw_recommendation_confidence": confidence_calibration.get("raw_confidence"),
            "calibrated_confidence": confidence_calibration.get("calibrated_confidence"),
            "walkforward_quality_score": walkforward_quality_score,
            "walkforward_quality_band": walkforward_quality_band,
            "walkforward_sharpe": confidence_calibration.get("walkforward_sharpe"),
            "walkforward_window_count": walkforward_evaluation.get("window_count", 0),
            "contradiction_detected": contradiction_detected,
            "contradiction_severity": contradiction_severity,
            "contradiction_count": contradiction_count,
            "phase1_regime": strategy_signal.regime,
            "phase1_indicator_votes": strategy_signal.indicator_votes,
            "phase1_feature_snapshot": strategy_signal.feature_snapshot,
            "self_critique_pass": self_critique_pass,
            "self_critique_score": self_critique_score,
            "self_critique_highest_severity": self_critique_highest,
            "self_critique_reason_codes": self_critique_reason_codes,
            "arm_votes": arm_votes,
            "arm_weights": arm_weights,
            "selected_arms": selected_arms,
            "ensemble_reason_codes": ensemble_reason_codes,
        }
        thresholds = {
            "min_total_return": config.thresholds.min_total_return,
            "min_sharpe": config.thresholds.min_sharpe,
            "max_drawdown": config.thresholds.max_drawdown,
            "min_signal_confidence": config.thresholds.min_signal_confidence,
            "min_walkforward_sharpe": config.calibration_min_walkforward_sharpe,
            "max_contradictions": float(config.calibration_max_contradictions),
            "walkforward_quality_low_cutoff": 0.40,
            "walkforward_quality_medium_cutoff": 0.55,
            "walkforward_quality_high_cutoff": 0.70,
            "self_critique_min_score": config.self_critique_min_score,
            "ensemble_min_selected_arms": 1.0,
        }

        trace_check(
            gate="data_quality",
            metric="is_valid",
            observed_value=data_quality_signal.is_valid,
            threshold=True,
            passed=data_quality_signal.is_valid,
            failure_reason_code="data_quality_invalid",
            failure_severity="fail",
            failure_detail="Data quality stage reported invalid source data.",
        )

        backtest_success = backtest_evaluation.backtest_status == "success"
        trace_check(
            gate="backtest",
            metric="status",
            observed_value=backtest_evaluation.backtest_status,
            threshold="success",
            passed=backtest_success,
            failure_reason_code="backtest_failed",
            failure_severity="fail",
            failure_detail="Backtest stage did not complete successfully.",
        )
        total_return = backtest_evaluation.total_return
        trace_check(
            gate="backtest",
            metric="total_return",
            observed_value=total_return,
            threshold=f">= {config.thresholds.min_total_return}",
            passed=backtest_success and total_return is not None and total_return >= config.thresholds.min_total_return,
            failure_reason_code="total_return_below_threshold",
            failure_severity="fail",
            failure_detail="Backtest total return fell below the configured threshold.",
        )
        sharpe = backtest_evaluation.sharpe
        trace_check(
            gate="backtest",
            metric="sharpe",
            observed_value=sharpe,
            threshold=f">= {config.thresholds.min_sharpe}",
            passed=backtest_success and sharpe is not None and sharpe >= config.thresholds.min_sharpe,
            failure_reason_code="sharpe_below_threshold",
            failure_severity="fail",
            failure_detail="Backtest Sharpe ratio fell below the configured threshold.",
        )
        max_drawdown = backtest_evaluation.max_drawdown
        trace_check(
            gate="backtest",
            metric="max_drawdown",
            observed_value=max_drawdown,
            threshold=f">= {config.thresholds.max_drawdown}",
            passed=backtest_success and max_drawdown is not None and max_drawdown >= config.thresholds.max_drawdown,
            failure_reason_code="max_drawdown_exceeded",
            failure_severity="fail",
            failure_detail="Backtest max drawdown exceeded allowable bounds.",
        )

        actionable = action in {"buy", "sell"}
        trace_check(
            gate="recommendation",
            metric="actionable_direction",
            observed_value=action,
            threshold="buy|sell",
            passed=actionable,
            failure_reason_code="non_actionable_recommendation",
            failure_severity="block",
            failure_detail="Recommendation was hold and cannot emit a trade intent.",
        )
        trace_check(
            gate="ensemble",
            metric="arm_votes_present",
            observed_value=len(arm_votes),
            threshold=">= 1",
            passed=len(arm_votes) >= 1,
            failure_reason_code="ensemble_arm_votes_missing",
            failure_severity="fail",
            failure_detail="No arm vote evidence was available for ensemble decisioning.",
        )
        trace_check(
            gate="ensemble",
            metric="selected_arms_present",
            observed_value=len(selected_arms),
            threshold=">= 1",
            passed=len(selected_arms) >= 1,
            failure_reason_code="ensemble_selected_arms_missing",
            failure_severity="fail",
            failure_detail="Ensemble selected-arm list is empty.",
        )
        trace_check(
            gate="ensemble",
            metric="arm_weights_sum",
            observed_value=sum(arm_weights.values()),
            threshold="~ 1.0",
            passed=abs(sum(arm_weights.values()) - 1.0) <= 0.05 if arm_weights else False,
            failure_reason_code="ensemble_arm_weights_invalid",
            failure_severity="fail",
            failure_detail="Ensemble arm weights were missing or not normalized.",
        )

        if actionable:
            quality_very_low = walkforward_quality_band == "very_low"
            trace_check(
                gate="calibration",
                metric="walkforward_quality_band",
                observed_value=walkforward_quality_band,
                threshold="not very_low",
                passed=not quality_very_low,
                failure_reason_code=f"risk_fail_{action}_walkforward_quality_very_low",
                failure_severity="fail",
                failure_detail="Walk-forward quality band is very_low for an actionable recommendation.",
            )

            quality_low = walkforward_quality_band == "low"
            trace_check(
                gate="calibration",
                metric="walkforward_quality_low_block",
                observed_value=walkforward_quality_band,
                threshold="not low",
                passed=not quality_low,
                failure_reason_code=f"risk_block_{action}_walkforward_quality_low",
                failure_severity="block",
                failure_detail="Walk-forward quality band is low and requires blocking execution.",
            )

            contradiction_failure_reason = (
                f"risk_fail_{action}_walkforward_contradiction_{walkforward_quality_band}"
                if contradiction_severity == "fail"
                else f"risk_block_{action}_walkforward_contradiction_{walkforward_quality_band}"
            )
            contradiction_failure_severity: Literal["block", "fail"] = (
                "fail" if contradiction_severity == "fail" else "block"
            )
            trace_check(
                gate="calibration",
                metric="walkforward_contradiction",
                observed_value={
                    "contradiction_detected": contradiction_detected,
                    "contradiction_severity": contradiction_severity,
                },
                threshold={"contradiction_detected": False},
                passed=not contradiction_detected,
                failure_reason_code=contradiction_failure_reason,
                failure_severity=contradiction_failure_severity,
                failure_detail=(
                    "Confidence calibration detected an actionable contradiction against walk-forward evidence."
                ),
            )

        calibrated_confidence = float(confidence_calibration.get("calibrated_confidence", 0.0))
        trace_check(
            gate="calibration",
            metric="calibrated_confidence",
            observed_value=calibrated_confidence,
            threshold=f">= {config.thresholds.min_signal_confidence}",
            passed=calibrated_confidence >= config.thresholds.min_signal_confidence,
            failure_reason_code="calibrated_confidence_below_threshold",
            failure_severity="block",
            failure_detail="Calibrated confidence is below deterministic minimum confidence.",
        )
        raw_confidence = float(confidence_calibration.get("raw_confidence", 0.0))
        trace_check(
            gate="calibration",
            metric="confidence_downgrade_guard",
            observed_value={"raw_confidence": raw_confidence, "calibrated_confidence": calibrated_confidence},
            threshold="raw >= min_signal_confidence implies calibrated >= min_signal_confidence",
            passed=not (
                raw_confidence >= config.thresholds.min_signal_confidence
                and calibrated_confidence < config.thresholds.min_signal_confidence
            ),
            failure_reason_code="confidence_downgraded_by_calibration",
            failure_severity="block",
            failure_detail="Calibration downgraded confidence below executable threshold.",
        )
        trace_check(
            gate="calibration",
            metric="contradiction_count",
            observed_value=contradiction_count,
            threshold=f"<= {config.calibration_max_contradictions}",
            passed=contradiction_count <= config.calibration_max_contradictions,
            failure_reason_code="calibration_contradiction_limit_exceeded",
            failure_severity="fail",
            failure_detail="Current run contradiction count exceeded configured limit.",
        )
        if contradiction_count > config.calibration_max_contradictions and actionable:
            register_reason(
                reason_code=f"risk_fail_{action}_calibration_contradiction_limit_exceeded",
                severity="fail",
                detail="Actionable recommendation exceeded calibration contradiction limit.",
            )

        trace_check(
            gate="self_critique",
            metric="pass",
            observed_value={"pass": self_critique_pass, "highest_severity": self_critique_highest},
            threshold={"pass": True},
            passed=self_critique_pass,
            failure_reason_code=(
                "self_critique_failed" if self_critique_highest == "fail" else "self_critique_blocked"
            ),
            failure_severity="fail" if self_critique_highest == "fail" else "block",
            failure_detail="Self-critique stage flagged contradictions requiring gate intervention.",
        )
        trace_check(
            gate="self_critique",
            metric="score",
            observed_value=self_critique_score,
            threshold=f">= {config.self_critique_min_score}",
            passed=self_critique_score >= config.self_critique_min_score,
            failure_reason_code="self_critique_score_below_threshold",
            failure_severity="block",
            failure_detail="Self-critique score is below configured minimum threshold.",
        )

        for reason_code in self_critique_reason_codes:
            severity = self_critique_severity_by_code.get(reason_code, "block")
            failure_severity: Literal["block", "fail"] = "fail" if severity == "fail" else "block"
            should_gate = failure_severity == "fail" or not self_critique_pass
            if should_gate:
                register_reason(
                    reason_code=reason_code,
                    severity=failure_severity,
                    detail=self_critique_reason_details.get(
                        reason_code,
                        "Self-critique stage generated this reason code.",
                    ),
                )
            decision_trace.append(
                {
                    "gate": "self_critique_reason",
                    "metric": reason_code,
                    "observed": severity,
                    "threshold": "none",
                    "pass": not should_gate,
                    "severity": failure_severity if should_gate else "advisory",
                    "reason_code": reason_code if should_gate else None,
                }
            )

        for reason_code in calibration_reason_codes:
            severity: Literal["block", "fail"] = (
                "fail" if "very_low" in reason_code or "contradiction_fail" in reason_code else "block"
            )
            register_reason(
                reason_code=reason_code,
                severity=severity,
                detail=f"Confidence calibration propagated reason code `{reason_code}`.",
            )
            decision_trace.append(
                {
                    "gate": "calibration_reason",
                    "metric": reason_code,
                    "observed": severity,
                    "threshold": "none",
                    "pass": False,
                    "severity": severity,
                    "reason_code": reason_code,
                }
            )
        for reason_code in ensemble_reason_codes:
            decision_trace.append(
                {
                    "gate": "ensemble_reason",
                    "metric": reason_code,
                    "observed": "advisory",
                    "threshold": "none",
                    "pass": True,
                    "severity": "advisory",
                    "reason_code": reason_code,
                }
            )

        reasons = sorted(set([*fail_reasons, *block_reasons]))
        approved = len(reasons) == 0
        append_transition("deterministic_gate:pass" if approved else "deterministic_gate:fail")
        return RiskDecision(
            contract="risk_decision.v1",
            run_id=run_id,
            created_at_utc=_utc_now_iso(),
            approved=approved,
            reason_codes=reasons,
            thresholds=thresholds,
            observed=observed,
            recommendation=strategy_signal.recommendation,
            recommendation_confidence=calibrated_confidence,
            deterministic_gate="pass" if approved else "fail",
            arm_votes=arm_votes,
            arm_weights=arm_weights,
            selected_arms=selected_arms,
            ensemble_reason_codes=ensemble_reason_codes,
            decision_trace=decision_trace,
            reason_code_details=reason_code_details,
            gate_transition_sequence=gate_transition_sequence,
        )

    risk_decision, risk_step = _run_step_with_retries(
        run_dir=run_dir,
        step_name="risk-review-agent",
        max_retries=config.step_retries,
        runner=risk_runner,
    )
    write_contract(risk_decision_path, risk_decision)
    step_records.append(risk_step)

    def action_runner() -> PaperTradeIntent:
        actionable = risk_decision.approved and strategy_signal.recommendation in {"buy", "sell"}
        now = datetime.now(timezone.utc)
        destination_path: str | None = None
        if actionable:
            destination = (
                settings.quant_data_root / "paper-trading" / f"{now:%Y-%m-%d}" / f"paper_trade_intent_{run_id}.json"
            )
            destination_path = str(destination)

        if actionable:
            reason = "deterministic_gate_passed"
            action: Recommendation = strategy_signal.recommendation
            status: Literal["emitted", "blocked"] = "emitted"
        else:
            reason = ",".join(risk_decision.reason_codes) if risk_decision.reason_codes else "blocked"
            action = "hold"
            status = "blocked"

        return PaperTradeIntent(
            contract="paper_trade_intent.v1",
            run_id=run_id,
            created_at_utc=_utc_now_iso(),
            mode="paper",
            status=status,
            exchange=config.exchange,
            symbol=config.symbol,
            timeframe=config.timeframe,
            action=action,
            notional_usd=float(config.paper_notional_usd),
            risk_approved=risk_decision.approved,
            reason=reason,
            destination_path=destination_path,
            arm_votes=dict(risk_decision.arm_votes),
            arm_weights=dict(risk_decision.arm_weights),
            selected_arms=list(risk_decision.selected_arms),
            ensemble_reason_codes=list(risk_decision.ensemble_reason_codes),
        )

    paper_trade_intent, action_step = _run_step_with_retries(
        run_dir=run_dir,
        step_name="execution-gateway",
        max_retries=config.step_retries,
        runner=action_runner,
    )
    write_contract(paper_trade_intent_path, paper_trade_intent)
    step_records.append(action_step)

    intent_destination_path: Path | None = None
    if paper_trade_intent.destination_path:
        intent_destination_path = Path(paper_trade_intent.destination_path)
        write_contract(intent_destination_path, paper_trade_intent)

    def paper_execution_runner() -> PaperTradeExecution:
        closes = pd.to_numeric(market_frame["close"], errors="coerce").dropna()
        mark_price = float(closes.iloc[-1]) if not closes.empty else None
        return execute_paper_trade_intent(
            quant_data_root=settings.quant_data_root,
            run_id=run_id,
            created_at_utc=_utc_now_iso(),
            exchange=config.exchange,
            symbol=config.symbol,
            timeframe=config.timeframe,
            intent=paper_trade_intent,
            mark_price=mark_price,
            starting_cash_usd=config.paper_starting_cash_usd,
            fee_bps=config.paper_fee_bps,
        )

    paper_trade_execution, paper_execution_step = _run_step_with_retries(
        run_dir=run_dir,
        step_name="paper-trading-executor",
        max_retries=config.step_retries,
        runner=paper_execution_runner,
    )
    write_contract(paper_trade_execution_path, paper_trade_execution)
    step_records.append(paper_execution_step)
    ensemble_performance_update = _update_ensemble_weight_state(
        state_path=ensemble_state_path,
        ensemble_state=ensemble_state,
        selected_arms=list(paper_trade_execution.selected_arms),
        arm_weights=dict(paper_trade_execution.arm_weights),
        paper_trade_execution=paper_trade_execution,
        decay_horizon=max(4, int(config.ensemble_decay_horizon)),
        exploration_weight=max(0.0, float(config.ensemble_exploration_weight)),
        run_id=run_id,
    )
    _write_json(ensemble_performance_update_path, ensemble_performance_update)

    ops_context = {
        "run_id": run_id,
        "report_verbosity": _normalize_report_verbosity(config.ops_report_verbosity),
        "scope": {
            "exchange": config.exchange,
            "symbol": config.symbol,
            "timeframe": config.timeframe,
            "source_data_path": str(source_data_path),
            "source_data_sha256": source_data_sha256,
        },
        "proposal": {
            "source": strategy_signal.source,
            "recommendation": strategy_signal.recommendation,
            "confidence": strategy_signal.confidence,
            "regime": strategy_signal.regime,
            "indicator_votes": strategy_signal.indicator_votes,
            "feature_snapshot": strategy_signal.feature_snapshot,
            "reason_codes": strategy_signal.reason_codes,
            "fast_window": strategy_signal.fast_window,
            "slow_window": strategy_signal.slow_window,
            "rationale": strategy_signal.rationale,
            "arm_votes": strategy_signal.arm_votes,
            "arm_weights": strategy_signal.arm_weights,
            "selected_arms": strategy_signal.selected_arms,
            "ensemble_reason_codes": strategy_signal.ensemble_reason_codes,
        },
        "phase1": phase1_feature_context,
        "backtest": {
            "status": backtest_evaluation.backtest_status,
            "total_return": backtest_evaluation.total_return,
            "annualized_return": backtest_evaluation.annualized_return,
            "sharpe": backtest_evaluation.sharpe,
            "max_drawdown": backtest_evaluation.max_drawdown,
            "arm_metrics": backtest_evaluation.arm_metrics,
            "ensemble_metrics": backtest_evaluation.ensemble_metrics,
            "arm_attribution_path": backtest_evaluation.arm_attribution_path,
        },
        "walkforward": walkforward_evaluation,
        "calibration": confidence_calibration,
        "self_critique": self_critique_signal,
        "risk": {
            "approved": risk_decision.approved,
            "reason_codes": risk_decision.reason_codes,
            "thresholds": risk_decision.thresholds,
            "deterministic_gate": risk_decision.deterministic_gate,
            "decision_trace": risk_decision.decision_trace,
            "reason_code_details": risk_decision.reason_code_details,
            "gate_transition_sequence": risk_decision.gate_transition_sequence,
            "arm_votes": risk_decision.arm_votes,
            "arm_weights": risk_decision.arm_weights,
            "selected_arms": risk_decision.selected_arms,
            "ensemble_reason_codes": risk_decision.ensemble_reason_codes,
        },
        "intent": {
            "status": paper_trade_intent.status,
            "action": paper_trade_intent.action,
            "notional_usd": paper_trade_intent.notional_usd,
            "destination_path": paper_trade_intent.destination_path,
        },
        "execution": {
            "status": paper_trade_execution.execution_status,
            "executed_action": paper_trade_execution.executed_action,
            "executed_notional_usd": paper_trade_execution.executed_notional_usd,
            "fee_usd": paper_trade_execution.fee_usd,
            "cash_after_usd": paper_trade_execution.cash_after_usd,
            "position_qty_after": paper_trade_execution.position_qty_after,
            "position_avg_entry_after": paper_trade_execution.position_avg_entry_after,
            "reason": paper_trade_execution.reason,
            "portfolio_state_path": paper_trade_execution.portfolio_state_path,
            "fills_log_path": paper_trade_execution.fills_log_path,
            "execution_record_path": paper_trade_execution.execution_record_path,
            "arm_attribution": paper_trade_execution.arm_attribution,
        },
        "ensemble": {
            "mode": ensemble_mode,
            "enabled_arms": list(enabled_arms),
            "weight_state_path": str(ensemble_state_path),
            "performance_update_path": str(ensemble_performance_update_path),
            "performance_update": ensemble_performance_update,
        },
    }

    def reporting_runner() -> dict[str, Any]:
        available_models = ollama.list_models()
        if available_models and config.ops_model not in available_models:
            raise RuntimeError(
                f"Configured ops model `{config.ops_model}` not found. Available models: {available_models}"
            )
        markdown = ollama.generate(
            model=config.ops_model,
            prompt=_ops_prompt(ops_context),
            system="Respond with markdown only.",
            temperature=0.1,
            format_json=False,
        )
        if not markdown.strip():
            raise RuntimeError("Ops report model returned empty markdown.")
        return {
            "source": "ollama",
            "model": config.ops_model,
            "markdown": markdown.strip(),
            "warnings": [],
        }

    def reporting_fallback(errors: list[str]) -> dict[str, Any]:
        return {
            "source": "fallback",
            "model": None,
            "markdown": _deterministic_ops_markdown(ops_context),
            "warnings": errors,
        }

    report_payload, reporting_step = _run_step_with_retries(
        run_dir=run_dir,
        step_name="ops-report-agent",
        max_retries=config.step_retries,
        runner=reporting_runner,
        fallback=reporting_fallback,
    )
    step_records.append(reporting_step)

    ops_report_markdown_path.write_text(str(report_payload["markdown"]) + "\n", encoding="utf-8")
    report_source = str(report_payload.get("source", "deterministic")).strip().lower()
    report_source_literal: Literal["ollama", "fallback", "deterministic"]
    if report_source == "ollama":
        report_source_literal = "ollama"
    elif report_source == "fallback":
        report_source_literal = "fallback"
    else:
        report_source_literal = "deterministic"
    ops_report_contract = OpsReportContract(
        contract="ops_report.v1",
        run_id=run_id,
        created_at_utc=_utc_now_iso(),
        source=report_source_literal,
        model=report_payload.get("model"),
        summary_markdown_path=str(ops_report_markdown_path),
        summary_markdown=str(report_payload["markdown"]),
        artifact_paths={
            "data_quality_signal": str(data_quality_path),
            "phase1_feature_context": str(phase1_feature_context_path),
            "strategy_proposal_signal": str(strategy_signal_path),
            "backtest_evaluation": str(backtest_evaluation_path),
            "walkforward_evaluation": str(walkforward_evaluation_path),
            "confidence_calibration": str(confidence_calibration_path),
            "self_critique_signal": str(self_critique_signal_path),
            "risk_decision": str(risk_decision_path),
            "paper_trade_intent": str(paper_trade_intent_path),
            "paper_trade_execution": str(paper_trade_execution_path),
            "paper_trade_destination": str(intent_destination_path) if intent_destination_path else "",
            "paper_portfolio_state": paper_trade_execution.portfolio_state_path or "",
            "paper_fills_log": paper_trade_execution.fills_log_path or "",
            "paper_execution_record": paper_trade_execution.execution_record_path or "",
            "backtest_arm_attribution": backtest_evaluation.arm_attribution_path or "",
            "ensemble_weight_state": str(ensemble_state_path),
            "ensemble_performance_update": str(ensemble_performance_update_path),
        },
        arm_votes=dict(risk_decision.arm_votes),
        arm_weights=dict(risk_decision.arm_weights),
        selected_arms=list(risk_decision.selected_arms),
        ensemble_reason_codes=list(risk_decision.ensemble_reason_codes),
        decision_trace=list(risk_decision.decision_trace),
        reason_code_details=dict(risk_decision.reason_code_details),
        gate_transition_sequence=list(risk_decision.gate_transition_sequence),
        report_verbosity=_normalize_report_verbosity(config.ops_report_verbosity),
        warnings=list(report_payload.get("warnings", [])),
    )
    write_contract(ops_report_contract_path, ops_report_contract)

    run_manifest = {
        "contract": "agent_plane_run_manifest.v1",
        "run_id": run_id,
        "created_at_utc": _utc_now_iso(),
        "scope": {
            "exchange": config.exchange,
            "symbol": config.symbol,
            "timeframe": config.timeframe,
            "source_data_path": str(source_data_path),
            "source_data_sha256": source_data_sha256,
        },
        "config": {
            "strategy_model": config.strategy_model,
            "ops_model": config.ops_model,
            "step_retries": config.step_retries,
            "minimum_bars": config.minimum_bars,
            "walk_forward_train_bars": config.walk_forward_train_bars,
            "walk_forward_validate_bars": config.walk_forward_validate_bars,
            "walk_forward_step_bars": config.walk_forward_step_bars,
            "walk_forward_min_windows": config.walk_forward_min_windows,
            "calibration_min_walkforward_sharpe": config.calibration_min_walkforward_sharpe,
            "calibration_confidence_floor": config.calibration_confidence_floor,
            "calibration_confidence_ceiling": config.calibration_confidence_ceiling,
            "calibration_max_contradictions": config.calibration_max_contradictions,
            "self_critique_min_score": config.self_critique_min_score,
            "self_critique_max_findings": config.self_critique_max_findings,
            "ops_report_verbosity": _normalize_report_verbosity(config.ops_report_verbosity),
            "ensemble_mode": ensemble_mode,
            "ensemble_enabled_arms": list(enabled_arms),
            "ensemble_decay_horizon": max(4, int(config.ensemble_decay_horizon)),
            "ensemble_exploration_weight": max(0.0, float(config.ensemble_exploration_weight)),
            "paper_notional_usd": config.paper_notional_usd,
            "paper_starting_cash_usd": config.paper_starting_cash_usd,
            "paper_fee_bps": config.paper_fee_bps,
            "thresholds": _json_safe(asdict(config.thresholds)),
        },
        "steps": [_json_safe(asdict(record)) for record in step_records],
        "artifacts": {
            "run_dir": str(run_dir),
            "data_quality_signal": str(data_quality_path),
            "phase1_feature_context": str(phase1_feature_context_path),
            "strategy_proposal_signal": str(strategy_signal_path),
            "backtest_evaluation": str(backtest_evaluation_path),
            "walkforward_evaluation": str(walkforward_evaluation_path),
            "confidence_calibration": str(confidence_calibration_path),
            "self_critique_signal": str(self_critique_signal_path),
            "risk_decision": str(risk_decision_path),
            "paper_trade_intent": str(paper_trade_intent_path),
            "paper_trade_execution": str(paper_trade_execution_path),
            "ops_report_markdown": str(ops_report_markdown_path),
            "ops_report_contract": str(ops_report_contract_path),
            "backtest_arm_attribution": str(backtest_arm_attribution_path)
            if backtest_arm_attribution_path
            else None,
            "ensemble_weight_state": str(ensemble_state_path),
            "ensemble_performance_update": str(ensemble_performance_update_path),
            "paper_trade_destination": str(intent_destination_path) if intent_destination_path else None,
            "paper_portfolio_state": paper_trade_execution.portfolio_state_path,
            "paper_fills_log": paper_trade_execution.fills_log_path,
            "paper_execution_record": paper_trade_execution.execution_record_path,
        },
        "outcome": {
            "risk_approved": risk_decision.approved,
            "intent_status": paper_trade_intent.status,
            "paper_trade_execution_status": paper_trade_execution.execution_status,
            "deterministic_gate": risk_decision.deterministic_gate,
            "calibrated_confidence": confidence_calibration.get("calibrated_confidence"),
            "walkforward_quality_score": confidence_calibration.get("walkforward_quality_score"),
            "self_critique_score": self_critique_signal.get("score"),
            "self_critique_pass": self_critique_signal.get("pass"),
            "decision_trace_entries": len(risk_decision.decision_trace),
            "selected_arms": list(risk_decision.selected_arms),
            "ensemble_reason_codes": list(risk_decision.ensemble_reason_codes),
        },
    }
    _write_json(run_manifest_path, run_manifest)
    logger.info(
        "Agent-plane run complete run_id=%s risk_approved=%s intent_status=%s execution_status=%s",
        run_id,
        risk_decision.approved,
        paper_trade_intent.status,
        paper_trade_execution.execution_status,
    )

    return AgentPlaneRunResult(
        run_id=run_id,
        run_dir=run_dir,
        source_data_path=source_data_path,
        data_quality_path=data_quality_path,
        strategy_signal_path=strategy_signal_path,
        backtest_evaluation_path=backtest_evaluation_path,
        backtest_arm_attribution_path=backtest_arm_attribution_path,
        phase1_feature_context_path=phase1_feature_context_path,
        walkforward_evaluation_path=walkforward_evaluation_path,
        confidence_calibration_path=confidence_calibration_path,
        self_critique_signal_path=self_critique_signal_path,
        risk_decision_path=risk_decision_path,
        paper_trade_intent_path=paper_trade_intent_path,
        paper_trade_execution_path=paper_trade_execution_path,
        ops_report_markdown_path=ops_report_markdown_path,
        ops_report_contract_path=ops_report_contract_path,
        run_manifest_path=run_manifest_path,
        ensemble_performance_update_path=ensemble_performance_update_path,
        risk_approved=risk_decision.approved,
        intent_status=paper_trade_intent.status,
        paper_trade_execution_status=paper_trade_execution.execution_status,
        intent_destination_path=intent_destination_path,
    )
