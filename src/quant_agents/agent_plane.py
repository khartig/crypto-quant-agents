from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal, TypeVar

import pandas as pd

from quant_agents.agent_contracts import (
    BacktestEvaluation,
    DataQualitySignal,
    OpsReportContract,
    PaperTradeIntent,
    Recommendation,
    RiskDecision,
    StrategyProposalSignal,
    write_contract,
)
from quant_agents.backtest import STRATEGY_NAME, run_sma_backtest
from quant_agents.config import Settings
from quant_agents.ollama_client import OllamaClient
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
    minimum_bars: int
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
    risk_decision_path: Path
    paper_trade_intent_path: Path
    ops_report_markdown_path: Path
    ops_report_contract_path: Path
    run_manifest_path: Path
    risk_approved: bool
    intent_status: str
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


def _strategy_prompt(
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    snapshot: dict[str, float | int | str],
    source_data_path: Path,
    source_data_sha256: str,
) -> str:
    return "\n".join(
        [
            "You are strategy-agent for a deterministic crypto quant pipeline.",
            "Return STRICT JSON only with keys:",
            '{"recommendation":"buy|sell|hold","confidence":0.0,"fast_window":20,"slow_window":50,"rationale":"..."}',
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
        ]
    )


def _ops_prompt(context: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are ops-report-agent for a deterministic crypto quant pipeline.",
            "Write concise markdown with sections:",
            "## Run summary",
            "## Deterministic gate outcome",
            "## Paper intent",
            "## Follow-ups",
            "Use only facts from the context JSON and do not invent metrics.",
            json.dumps(context, indent=2, sort_keys=True),
        ]
    )


def _deterministic_ops_markdown(context: dict[str, Any]) -> str:
    risk_approved = bool(context["risk"]["approved"])
    intent_status = str(context["intent"]["status"])
    return "\n".join(
        [
            f"# Agent Plane Ops Report ({context['run_id']})",
            "",
            "## Run summary",
            f"- Exchange: `{context['scope']['exchange']}`",
            f"- Symbol: `{context['scope']['symbol']}`",
            f"- Timeframe: `{context['scope']['timeframe']}`",
            f"- Strategy recommendation: `{context['proposal']['recommendation']}`",
            f"- Backtest status: `{context['backtest']['status']}`",
            "",
            "## Deterministic gate outcome",
            f"- Approved: `{risk_approved}`",
            f"- Reason codes: `{context['risk']['reason_codes']}`",
            "",
            "## Paper intent",
            f"- Status: `{intent_status}`",
            f"- Action: `{context['intent']['action']}`",
            f"- Notional USD: `{context['intent']['notional_usd']}`",
            "",
            "## Follow-ups",
            "- Validate model availability in local Ollama if fallback mode was used.",
            "- Review deterministic risk thresholds for false positives/negatives.",
        ]
    )


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
    strategy_signal_path = run_dir / "strategy_proposal_signal.json"
    backtest_evaluation_path = run_dir / "backtest_evaluation.json"
    risk_decision_path = run_dir / "risk_decision.json"
    paper_trade_intent_path = run_dir / "paper_trade_intent.json"
    ops_report_markdown_path = run_dir / "ops_report.md"
    ops_report_contract_path = run_dir / "ops_report_contract.json"
    run_manifest_path = run_dir / "run_manifest.json"

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
        )

    strategy_signal, strategy_step = _run_step_with_retries(
        run_dir=run_dir,
        step_name="strategy-agent",
        max_retries=config.step_retries,
        runner=strategy_runner,
        fallback=strategy_fallback,
    )
    write_contract(strategy_signal_path, strategy_signal)
    step_records.append(strategy_step)

    def backtest_runner() -> BacktestEvaluation:
        backtest_result = run_sma_backtest(
            settings=settings,
            exchange=config.exchange,
            symbol=config.symbol,
            timeframe=config.timeframe,
            fast_window=strategy_signal.fast_window,
            slow_window=strategy_signal.slow_window,
            source_data_path=source_data_path,
            archive_run=False,
        )
        return BacktestEvaluation(
            contract="backtest_evaluation.v1",
            run_id=run_id,
            created_at_utc=_utc_now_iso(),
            strategy=STRATEGY_NAME,
            exchange=config.exchange,
            symbol=config.symbol,
            timeframe=config.timeframe,
            source_data_path=str(backtest_result.source_data_path),
            source_data_sha256=backtest_result.source_data_sha256,
            backtest_status="success",
            backtest_run_dir=str(backtest_result.run_dir),
            metrics_path=str(backtest_result.metrics_path),
            manifest_path=str(backtest_result.manifest_path),
            total_return=float(backtest_result.metrics["total_return"]),
            annualized_return=float(backtest_result.metrics["annualized_return"]),
            sharpe=float(backtest_result.metrics["sharpe"]),
            max_drawdown=float(backtest_result.metrics["max_drawdown"]),
            signal_flips=int(backtest_result.metrics["signal_flips"]),
            bars=int(backtest_result.metrics["bars"]),
            error_message=None,
        )

    def backtest_fallback(errors: list[str]) -> BacktestEvaluation:
        return BacktestEvaluation(
            contract="backtest_evaluation.v1",
            run_id=run_id,
            created_at_utc=_utc_now_iso(),
            strategy=STRATEGY_NAME,
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

    def risk_runner() -> RiskDecision:
        reasons: list[str] = []
        observed: dict[str, float | int | str | bool | None] = {
            "data_quality_valid": data_quality_signal.is_valid,
            "backtest_status": backtest_evaluation.backtest_status,
            "total_return": backtest_evaluation.total_return,
            "sharpe": backtest_evaluation.sharpe,
            "max_drawdown": backtest_evaluation.max_drawdown,
            "recommendation": strategy_signal.recommendation,
            "recommendation_confidence": strategy_signal.confidence,
        }
        thresholds = {
            "min_total_return": config.thresholds.min_total_return,
            "min_sharpe": config.thresholds.min_sharpe,
            "max_drawdown": config.thresholds.max_drawdown,
            "min_signal_confidence": config.thresholds.min_signal_confidence,
        }

        if not data_quality_signal.is_valid:
            reasons.append("data_quality_invalid")
        if backtest_evaluation.backtest_status != "success":
            reasons.append("backtest_failed")
        else:
            total_return = backtest_evaluation.total_return
            sharpe = backtest_evaluation.sharpe
            max_drawdown = backtest_evaluation.max_drawdown
            if total_return is None or total_return < config.thresholds.min_total_return:
                reasons.append("total_return_below_threshold")
            if sharpe is None or sharpe < config.thresholds.min_sharpe:
                reasons.append("sharpe_below_threshold")
            if max_drawdown is None or max_drawdown < config.thresholds.max_drawdown:
                reasons.append("max_drawdown_exceeded")

        if strategy_signal.recommendation not in {"buy", "sell"}:
            reasons.append("non_actionable_recommendation")
        if strategy_signal.confidence < config.thresholds.min_signal_confidence:
            reasons.append("signal_confidence_below_threshold")

        approved = len(reasons) == 0
        return RiskDecision(
            contract="risk_decision.v1",
            run_id=run_id,
            created_at_utc=_utc_now_iso(),
            approved=approved,
            reason_codes=sorted(set(reasons)),
            thresholds=thresholds,
            observed=observed,
            recommendation=strategy_signal.recommendation,
            recommendation_confidence=strategy_signal.confidence,
            deterministic_gate="pass" if approved else "fail",
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

    ops_context = {
        "run_id": run_id,
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
            "fast_window": strategy_signal.fast_window,
            "slow_window": strategy_signal.slow_window,
            "rationale": strategy_signal.rationale,
        },
        "backtest": {
            "status": backtest_evaluation.backtest_status,
            "total_return": backtest_evaluation.total_return,
            "annualized_return": backtest_evaluation.annualized_return,
            "sharpe": backtest_evaluation.sharpe,
            "max_drawdown": backtest_evaluation.max_drawdown,
        },
        "risk": {
            "approved": risk_decision.approved,
            "reason_codes": risk_decision.reason_codes,
            "thresholds": risk_decision.thresholds,
        },
        "intent": {
            "status": paper_trade_intent.status,
            "action": paper_trade_intent.action,
            "notional_usd": paper_trade_intent.notional_usd,
            "destination_path": paper_trade_intent.destination_path,
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
            "strategy_proposal_signal": str(strategy_signal_path),
            "backtest_evaluation": str(backtest_evaluation_path),
            "risk_decision": str(risk_decision_path),
            "paper_trade_intent": str(paper_trade_intent_path),
            "paper_trade_destination": str(intent_destination_path) if intent_destination_path else "",
        },
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
            "paper_notional_usd": config.paper_notional_usd,
            "thresholds": _json_safe(asdict(config.thresholds)),
        },
        "steps": [_json_safe(asdict(record)) for record in step_records],
        "artifacts": {
            "run_dir": str(run_dir),
            "data_quality_signal": str(data_quality_path),
            "strategy_proposal_signal": str(strategy_signal_path),
            "backtest_evaluation": str(backtest_evaluation_path),
            "risk_decision": str(risk_decision_path),
            "paper_trade_intent": str(paper_trade_intent_path),
            "ops_report_markdown": str(ops_report_markdown_path),
            "ops_report_contract": str(ops_report_contract_path),
            "paper_trade_destination": str(intent_destination_path) if intent_destination_path else None,
        },
        "outcome": {
            "risk_approved": risk_decision.approved,
            "intent_status": paper_trade_intent.status,
            "deterministic_gate": risk_decision.deterministic_gate,
        },
    }
    _write_json(run_manifest_path, run_manifest)
    logger.info(
        "Agent-plane run complete run_id=%s risk_approved=%s intent_status=%s",
        run_id,
        risk_decision.approved,
        paper_trade_intent.status,
    )

    return AgentPlaneRunResult(
        run_id=run_id,
        run_dir=run_dir,
        source_data_path=source_data_path,
        data_quality_path=data_quality_path,
        strategy_signal_path=strategy_signal_path,
        backtest_evaluation_path=backtest_evaluation_path,
        risk_decision_path=risk_decision_path,
        paper_trade_intent_path=paper_trade_intent_path,
        ops_report_markdown_path=ops_report_markdown_path,
        ops_report_contract_path=ops_report_contract_path,
        run_manifest_path=run_manifest_path,
        risk_approved=risk_decision.approved,
        intent_status=paper_trade_intent.status,
        intent_destination_path=intent_destination_path,
    )
