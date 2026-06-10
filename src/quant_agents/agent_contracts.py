from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Recommendation = Literal["buy", "sell", "hold"]
IntentStatus = Literal["emitted", "blocked"]
ExecutionStatus = Literal["executed", "skipped", "rejected"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_contract(path: Path, contract: Any) -> None:
    payload = _json_safe(asdict(contract))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class DataQualitySignal:
    contract: str
    run_id: str
    created_at_utc: str
    exchange: str
    symbol: str
    timeframe: str
    source_data_path: str
    source_data_sha256: str
    bar_count: int
    gap_count: int
    null_value_count: int
    duplicate_timestamp_count: int
    is_valid: bool
    anomalies: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StrategyProposalSignal:
    contract: str
    run_id: str
    created_at_utc: str
    source: Literal["ollama", "fallback"]
    model: str
    exchange: str
    symbol: str
    timeframe: str
    input_data_path: str
    input_data_sha256: str
    recommendation: Recommendation
    confidence: float
    fast_window: int
    slow_window: int
    rationale: str
    raw_model_response: str | None
    warnings: list[str] = field(default_factory=list)
    indicator_votes: dict[str, float] = field(default_factory=dict)
    regime: str = "unknown"
    feature_snapshot: dict[str, float | int | str | bool] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)
    arm_votes: dict[str, dict[str, Any]] = field(default_factory=dict)
    arm_weights: dict[str, float] = field(default_factory=dict)
    selected_arms: list[str] = field(default_factory=list)
    ensemble_reason_codes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BacktestEvaluation:
    contract: str
    run_id: str
    created_at_utc: str
    strategy: str
    exchange: str
    symbol: str
    timeframe: str
    source_data_path: str
    source_data_sha256: str
    backtest_status: Literal["success", "failed"]
    backtest_run_dir: str | None
    metrics_path: str | None
    manifest_path: str | None
    total_return: float | None
    annualized_return: float | None
    sharpe: float | None
    max_drawdown: float | None
    signal_flips: int | None
    bars: int | None
    error_message: str | None
    arm_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    ensemble_metrics: dict[str, Any] = field(default_factory=dict)
    arm_attribution_path: str | None = None


@dataclass(frozen=True)
class RiskDecision:
    contract: str
    run_id: str
    created_at_utc: str
    approved: bool
    reason_codes: list[str]
    thresholds: dict[str, float]
    observed: dict[str, Any]
    recommendation: Recommendation
    recommendation_confidence: float
    deterministic_gate: Literal["pass", "fail"]
    arm_votes: dict[str, dict[str, Any]] = field(default_factory=dict)
    arm_weights: dict[str, float] = field(default_factory=dict)
    selected_arms: list[str] = field(default_factory=list)
    ensemble_reason_codes: list[str] = field(default_factory=list)
    decision_trace: list[dict[str, Any]] = field(default_factory=list)
    reason_code_details: dict[str, str] = field(default_factory=dict)
    gate_transition_sequence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PaperTradeIntent:
    contract: str
    run_id: str
    created_at_utc: str
    mode: Literal["paper"]
    status: IntentStatus
    exchange: str
    symbol: str
    timeframe: str
    action: Recommendation
    notional_usd: float
    risk_approved: bool
    reason: str
    destination_path: str | None
    arm_votes: dict[str, dict[str, Any]] = field(default_factory=dict)
    arm_weights: dict[str, float] = field(default_factory=dict)
    selected_arms: list[str] = field(default_factory=list)
    ensemble_reason_codes: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class PaperTradeExecution:
    contract: str
    run_id: str
    created_at_utc: str
    mode: Literal["paper"]
    exchange: str
    symbol: str
    timeframe: str
    intent_status: IntentStatus
    intent_action: Recommendation
    execution_status: ExecutionStatus
    executed_action: Recommendation
    requested_notional_usd: float
    executed_notional_usd: float
    executed_quantity: float
    mark_price: float | None
    fee_usd: float
    cash_after_usd: float | None
    position_qty_after: float | None
    position_avg_entry_after: float | None
    realized_pnl_delta_usd: float
    reason: str
    portfolio_state_path: str | None
    fills_log_path: str | None
    execution_record_path: str | None
    arm_votes: dict[str, dict[str, Any]] = field(default_factory=dict)
    arm_weights: dict[str, float] = field(default_factory=dict)
    selected_arms: list[str] = field(default_factory=list)
    ensemble_reason_codes: list[str] = field(default_factory=list)
    arm_attribution: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class OpsReportContract:
    contract: str
    run_id: str
    created_at_utc: str
    source: Literal["ollama", "fallback", "deterministic"]
    model: str | None
    summary_markdown_path: str
    summary_markdown: str
    artifact_paths: dict[str, str]
    arm_votes: dict[str, dict[str, Any]] = field(default_factory=dict)
    arm_weights: dict[str, float] = field(default_factory=dict)
    selected_arms: list[str] = field(default_factory=list)
    ensemble_reason_codes: list[str] = field(default_factory=list)
    decision_trace: list[dict[str, Any]] = field(default_factory=list)
    reason_code_details: dict[str, str] = field(default_factory=dict)
    gate_transition_sequence: list[str] = field(default_factory=list)
    report_verbosity: str = "standard"
    warnings: list[str] = field(default_factory=list)
