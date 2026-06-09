from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from quant_agents.agent_plane import AgentPlaneConfig, RiskThresholds, run_agent_plane
from quant_agents.config import ensure_data_root_ready, load_settings
from quant_agents.storage import ensure_phase1_tree
JobStatus = Literal["queued", "running", "succeeded", "blocked", "failed"]
TERMINAL_JOB_STATUSES: frozenset[str] = frozenset({"succeeded", "blocked", "failed"})
REQUIRED_ARTIFACT_KEYS: tuple[str, ...] = (
    "data_quality_signal",
    "walkforward_evaluation",
    "confidence_calibration",
    "self_critique_signal",
    "strategy_proposal_signal",
    "backtest_evaluation",
    "risk_decision",
    "paper_trade_intent",
    "paper_trade_execution",
    "ops_report_markdown",
    "ops_report_contract",
    "run_manifest",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def _try_read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return _read_json_object(path, label="json document")
    except Exception:
        return None


@dataclass(frozen=True)
class OpenClawOrchestrationRequest:
    exchange: str
    symbol: str
    timeframe: str
    strategy_model: str
    ops_model: str
    step_retries: int
    minimum_bars: int
    min_total_return: float
    min_sharpe: float
    max_drawdown: float
    min_signal_confidence: float
    walk_forward_train_bars: int
    walk_forward_validate_bars: int
    walk_forward_step_bars: int
    walk_forward_min_windows: int
    calibration_min_walkforward_sharpe: float
    calibration_confidence_floor: float
    calibration_confidence_ceiling: float
    calibration_max_contradictions: int
    self_critique_min_score: float
    self_critique_max_findings: int
    ops_report_verbosity: str
    paper_notional_usd: float
    paper_starting_cash_usd: float
    paper_fee_bps: float
    source_data_path: str | None = None

    @staticmethod
    def from_dict(payload: dict[str, Any], settings) -> OpenClawOrchestrationRequest:
        return OpenClawOrchestrationRequest(
            exchange=str(payload.get("exchange") or settings.default_exchange),
            symbol=str(payload.get("symbol") or settings.default_symbol),
            timeframe=str(payload.get("timeframe") or settings.default_timeframe),
            strategy_model=str(payload.get("strategy_model") or settings.ollama_strategy_model),
            ops_model=str(payload.get("ops_model") or settings.ollama_ops_model),
            step_retries=max(0, int(payload.get("step_retries", settings.agent_step_retries))),
            minimum_bars=max(10, int(payload.get("minimum_bars", settings.agent_minimum_bars))),
            min_total_return=float(payload.get("min_total_return", settings.risk_min_total_return)),
            min_sharpe=float(payload.get("min_sharpe", settings.risk_min_sharpe)),
            max_drawdown=float(payload.get("max_drawdown", settings.risk_max_drawdown)),
            min_signal_confidence=float(
                payload.get("min_signal_confidence", settings.risk_min_signal_confidence)
            ),
            walk_forward_train_bars=max(
                50,
                int(payload.get("walk_forward_train_bars", settings.walk_forward_train_bars)),
            ),
            walk_forward_validate_bars=max(
                10,
                int(payload.get("walk_forward_validate_bars", settings.walk_forward_validate_bars)),
            ),
            walk_forward_step_bars=max(
                10,
                int(payload.get("walk_forward_step_bars", settings.walk_forward_step_bars)),
            ),
            walk_forward_min_windows=max(
                1,
                int(payload.get("walk_forward_min_windows", settings.walk_forward_min_windows)),
            ),
            calibration_min_walkforward_sharpe=float(
                payload.get(
                    "calibration_min_walkforward_sharpe",
                    settings.calibration_min_walkforward_sharpe,
                )
            ),
            calibration_confidence_floor=float(
                payload.get("calibration_confidence_floor", settings.calibration_confidence_floor)
            ),
            calibration_confidence_ceiling=float(
                payload.get("calibration_confidence_ceiling", settings.calibration_confidence_ceiling)
            ),
            calibration_max_contradictions=max(
                0,
                int(
                    payload.get(
                        "calibration_max_contradictions",
                        settings.calibration_max_contradictions,
                    )
                ),
            ),
            self_critique_min_score=float(
                payload.get("self_critique_min_score", settings.self_critique_min_score)
            ),
            self_critique_max_findings=max(
                1,
                int(payload.get("self_critique_max_findings", settings.self_critique_max_findings)),
            ),
            ops_report_verbosity=str(
                payload.get("ops_report_verbosity", settings.ops_report_verbosity)
            ).strip().lower(),
            paper_notional_usd=float(payload.get("paper_notional_usd", settings.paper_trade_notional_usd)),
            paper_starting_cash_usd=float(
                payload.get("paper_starting_cash_usd", settings.paper_trade_starting_cash_usd)
            ),
            paper_fee_bps=float(payload.get("paper_fee_bps", settings.paper_trade_fee_bps)),
            source_data_path=(str(payload["source_data_path"]) if payload.get("source_data_path") else None),
        )

    def to_agent_plane_config(self) -> AgentPlaneConfig:
        thresholds = RiskThresholds(
            min_total_return=self.min_total_return,
            min_sharpe=self.min_sharpe,
            max_drawdown=self.max_drawdown,
            min_signal_confidence=self.min_signal_confidence,
        )
        return AgentPlaneConfig(
            exchange=self.exchange,
            symbol=self.symbol,
            timeframe=self.timeframe,
            strategy_model=self.strategy_model,
            ops_model=self.ops_model,
            step_retries=self.step_retries,
            thresholds=thresholds,
            paper_notional_usd=self.paper_notional_usd,
            paper_starting_cash_usd=self.paper_starting_cash_usd,
            paper_fee_bps=self.paper_fee_bps,
            minimum_bars=self.minimum_bars,
            walk_forward_train_bars=self.walk_forward_train_bars,
            walk_forward_validate_bars=self.walk_forward_validate_bars,
            walk_forward_step_bars=self.walk_forward_step_bars,
            walk_forward_min_windows=self.walk_forward_min_windows,
            calibration_min_walkforward_sharpe=self.calibration_min_walkforward_sharpe,
            calibration_confidence_floor=self.calibration_confidence_floor,
            calibration_confidence_ceiling=self.calibration_confidence_ceiling,
            calibration_max_contradictions=self.calibration_max_contradictions,
            self_critique_min_score=self.self_critique_min_score,
            self_critique_max_findings=self.self_critique_max_findings,
            ops_report_verbosity=self.ops_report_verbosity,
            source_data_path=Path(self.source_data_path).expanduser().resolve()
            if self.source_data_path
            else None,
        )

def run_openclaw_orchestration(
    request: OpenClawOrchestrationRequest, *, settings=None
) -> dict[str, Any]:
    resolved_settings = settings or load_settings()
    ensure_data_root_ready(
        resolved_settings.quant_data_root,
        allow_unmounted=resolved_settings.allow_unmounted_data_root,
    )
    ensure_phase1_tree(resolved_settings.quant_data_root)
    result = run_agent_plane(resolved_settings, request.to_agent_plane_config())
    payload = {
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "source_data_path": str(result.source_data_path),
        "risk_approved": result.risk_approved,
        "intent_status": result.intent_status,
        "paper_trade_execution_status": result.paper_trade_execution_status,
        "artifacts": {
            "data_quality_signal": str(result.data_quality_path),
            "phase1_feature_context": str(result.phase1_feature_context_path),
            "strategy_proposal_signal": str(result.strategy_signal_path),
            "backtest_evaluation": str(result.backtest_evaluation_path),
            "walkforward_evaluation": str(result.walkforward_evaluation_path),
            "confidence_calibration": str(result.confidence_calibration_path),
            "self_critique_signal": str(result.self_critique_signal_path),
            "risk_decision": str(result.risk_decision_path),
            "paper_trade_intent": str(result.paper_trade_intent_path),
            "paper_trade_execution": str(result.paper_trade_execution_path),
            "ops_report_markdown": str(result.ops_report_markdown_path),
            "ops_report_contract": str(result.ops_report_contract_path),
            "run_manifest": str(result.run_manifest_path),
            "paper_trade_destination": (
                str(result.intent_destination_path) if result.intent_destination_path else None
            ),
        },
    }
    return payload

def verify_orchestration_gate(response: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, bool] = {}

    run_id = str(response.get("run_id") or "")
    run_dir = str(response.get("run_dir") or "")
    if not run_id:
        errors.append("missing_run_id")
    if not run_dir:
        errors.append("missing_run_dir")

    artifacts = response.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        errors.append("missing_artifacts_map")

    artifact_paths: dict[str, str] = {}
    for key in REQUIRED_ARTIFACT_KEYS:
        raw_path = artifacts.get(key)
        if not isinstance(raw_path, str) or not raw_path.strip():
            checks[f"artifact:{key}:present"] = False
            errors.append(f"missing_artifact:{key}")
            continue
        artifact_path = Path(raw_path).expanduser()
        exists = artifact_path.exists()
        checks[f"artifact:{key}:present"] = exists
        if not exists:
            errors.append(f"artifact_not_found:{key}:{artifact_path}")
            continue
        artifact_paths[key] = str(artifact_path)

    def _load_contract(key: str) -> dict[str, Any] | None:
        path_str = artifact_paths.get(key)
        if not path_str:
            return None
        path = Path(path_str)
        try:
            contract = _read_json_object(path, label=f"{key} contract")
            checks[f"{key}:json_object"] = True
            return contract
        except Exception as exc:
            checks[f"{key}:json_object"] = False
            errors.append(f"invalid_contract:{key}:{type(exc).__name__}")
            return None

    data_quality = _load_contract("data_quality_signal")
    confidence_calibration = _load_contract("confidence_calibration")
    self_critique = _load_contract("self_critique_signal")
    backtest = _load_contract("backtest_evaluation")
    risk = _load_contract("risk_decision")
    intent = _load_contract("paper_trade_intent")
    execution = _load_contract("paper_trade_execution")
    ops_report_contract = _load_contract("ops_report_contract")
    run_manifest = _load_contract("run_manifest")

    if data_quality is not None:
        is_valid = data_quality.get("is_valid") is True
        checks["data_quality:is_valid"] = is_valid
        if not is_valid:
            errors.append("data_quality_not_valid")

    if backtest is not None:
        backtest_success = str(backtest.get("backtest_status")) == "success"
        checks["backtest:success"] = backtest_success
        if not backtest_success:
            errors.append("backtest_not_success")

    if confidence_calibration is not None:
        calibrated_confidence_present = isinstance(
            confidence_calibration.get("calibrated_confidence"),
            (int, float),
        )
        reason_codes_list = isinstance(confidence_calibration.get("reason_codes"), list)
        contradiction_field_present = isinstance(confidence_calibration.get("contradiction_detected"), bool)
        checks["confidence_calibration:calibrated_confidence_present"] = calibrated_confidence_present
        checks["confidence_calibration:reason_codes_list"] = reason_codes_list
        checks["confidence_calibration:contradiction_field_present"] = contradiction_field_present
        if not calibrated_confidence_present:
            errors.append("confidence_calibration_missing_calibrated_confidence")
        if not reason_codes_list:
            errors.append("confidence_calibration_reason_codes_invalid")
        if not contradiction_field_present:
            errors.append("confidence_calibration_contradiction_flag_invalid")

    if self_critique is not None:
        self_critique_pass = self_critique.get("pass") is True
        self_critique_score_present = isinstance(self_critique.get("score"), (int, float))
        self_critique_reasons_present = isinstance(self_critique.get("reason_codes"), list)
        self_critique_findings_present = isinstance(self_critique.get("findings"), list)
        checks["self_critique:pass"] = self_critique_pass
        checks["self_critique:score_present"] = self_critique_score_present
        checks["self_critique:reason_codes_present"] = self_critique_reasons_present
        checks["self_critique:findings_present"] = self_critique_findings_present
        if not self_critique_pass:
            errors.append("self_critique_not_pass")
        if not self_critique_score_present:
            errors.append("self_critique_score_missing")
        if not self_critique_reasons_present:
            errors.append("self_critique_reason_codes_invalid")
        if not self_critique_findings_present:
            errors.append("self_critique_findings_invalid")

    if risk is not None:
        risk_approved = risk.get("approved") is True
        gate_pass = str(risk.get("deterministic_gate")) == "pass"
        decision_trace_present = isinstance(risk.get("decision_trace"), list) and len(
            list(risk.get("decision_trace") or [])
        ) > 0
        reason_code_details_present = isinstance(risk.get("reason_code_details"), dict)
        gate_transition_present = isinstance(risk.get("gate_transition_sequence"), list) and len(
            list(risk.get("gate_transition_sequence") or [])
        ) > 0
        checks["risk:approved"] = risk_approved
        checks["risk:deterministic_gate_pass"] = gate_pass
        checks["risk:decision_trace_present"] = decision_trace_present
        checks["risk:reason_code_details_present"] = reason_code_details_present
        checks["risk:gate_transition_present"] = gate_transition_present
        if not risk_approved:
            errors.append("risk_not_approved")
        if not gate_pass:
            errors.append("risk_gate_not_pass")
        if not decision_trace_present:
            errors.append("risk_decision_trace_missing")
        if not reason_code_details_present:
            errors.append("risk_reason_code_details_missing")
        if not gate_transition_present:
            errors.append("risk_gate_transition_sequence_missing")

    if intent is not None:
        intent_emitted = str(intent.get("status")) == "emitted"
        intent_actionable = str(intent.get("action")) in {"buy", "sell"}
        intent_risk_approved = intent.get("risk_approved") is True
        checks["intent:emitted"] = intent_emitted
        checks["intent:actionable"] = intent_actionable
        checks["intent:risk_approved"] = intent_risk_approved
        if not intent_emitted:
            errors.append("intent_not_emitted")
        if not intent_actionable:
            errors.append("intent_not_actionable")
        if not intent_risk_approved:
            errors.append("intent_risk_not_approved")

    if execution is not None:
        execution_status_ok = str(execution.get("execution_status")) == "executed"
        execution_intent_status_ok = str(execution.get("intent_status")) == "emitted"
        try:
            executed_notional_usd = float(execution.get("executed_notional_usd", 0.0))
        except (TypeError, ValueError):
            executed_notional_usd = 0.0
        executed_notional_positive = executed_notional_usd > 0.0
        checks["execution:executed"] = execution_status_ok
        checks["execution:intent_status_emitted"] = execution_intent_status_ok
        checks["execution:executed_notional_positive"] = executed_notional_positive
        if not execution_status_ok:
            errors.append("execution_not_executed")
        if not execution_intent_status_ok:
            errors.append("execution_intent_status_not_emitted")
        if not executed_notional_positive:
            errors.append("execution_notional_not_positive")

    if ops_report_contract is not None:
        report_trace_present = isinstance(ops_report_contract.get("decision_trace"), list)
        report_reason_details_present = isinstance(ops_report_contract.get("reason_code_details"), dict)
        report_gate_sequence_present = isinstance(ops_report_contract.get("gate_transition_sequence"), list)
        report_verbosity = str(ops_report_contract.get("report_verbosity", "")).lower()
        report_verbosity_valid = report_verbosity in {"compact", "standard", "verbose"}
        checks["ops_report_contract:decision_trace_present"] = report_trace_present
        checks["ops_report_contract:reason_code_details_present"] = report_reason_details_present
        checks["ops_report_contract:gate_transition_present"] = report_gate_sequence_present
        checks["ops_report_contract:report_verbosity_valid"] = report_verbosity_valid
        if not report_trace_present:
            errors.append("ops_report_contract_decision_trace_missing")
        if not report_reason_details_present:
            errors.append("ops_report_contract_reason_code_details_missing")
        if not report_gate_sequence_present:
            errors.append("ops_report_contract_gate_transition_sequence_missing")
        if not report_verbosity_valid:
            errors.append("ops_report_contract_report_verbosity_invalid")

    if run_manifest is not None:
        artifacts_obj = run_manifest.get("artifacts")
        if isinstance(artifacts_obj, dict):
            manifest_self_critique_present = isinstance(artifacts_obj.get("self_critique_signal"), str) and bool(
                str(artifacts_obj.get("self_critique_signal"))
            )
            checks["manifest:self_critique_artifact_present"] = manifest_self_critique_present
            if not manifest_self_critique_present:
                errors.append("manifest_self_critique_artifact_missing")
        else:
            checks["manifest:artifacts_present"] = False
            errors.append("manifest_artifacts_missing")

        manifest_config = run_manifest.get("config")
        if isinstance(manifest_config, dict):
            config_has_min_score = "self_critique_min_score" in manifest_config
            config_has_max_findings = "self_critique_max_findings" in manifest_config
            config_has_report_verbosity = "ops_report_verbosity" in manifest_config
            checks["manifest:config_self_critique_min_score_present"] = config_has_min_score
            checks["manifest:config_self_critique_max_findings_present"] = config_has_max_findings
            checks["manifest:config_ops_report_verbosity_present"] = config_has_report_verbosity
            if not config_has_min_score:
                errors.append("manifest_config_self_critique_min_score_missing")
            if not config_has_max_findings:
                errors.append("manifest_config_self_critique_max_findings_missing")
            if not config_has_report_verbosity:
                errors.append("manifest_config_ops_report_verbosity_missing")
        else:
            checks["manifest:config_present"] = False
            errors.append("manifest_config_missing")

        outcome = run_manifest.get("outcome")
        if isinstance(outcome, dict):
            manifest_gate_pass = str(outcome.get("deterministic_gate")) == "pass"
            manifest_risk_approved = outcome.get("risk_approved") is True
            manifest_intent_emitted = str(outcome.get("intent_status")) == "emitted"
            manifest_executed = str(outcome.get("paper_trade_execution_status")) == "executed"
            manifest_self_critique_pass = outcome.get("self_critique_pass") is True
            try:
                manifest_decision_trace_entries = int(outcome.get("decision_trace_entries", 0))
            except (TypeError, ValueError):
                manifest_decision_trace_entries = 0
            checks["manifest:deterministic_gate_pass"] = manifest_gate_pass
            checks["manifest:risk_approved"] = manifest_risk_approved
            checks["manifest:intent_emitted"] = manifest_intent_emitted
            checks["manifest:execution_executed"] = manifest_executed
            checks["manifest:self_critique_pass"] = manifest_self_critique_pass
            checks["manifest:decision_trace_entries_positive"] = manifest_decision_trace_entries > 0
            if not manifest_gate_pass:
                errors.append("manifest_gate_not_pass")
            if not manifest_risk_approved:
                errors.append("manifest_risk_not_approved")
            if not manifest_intent_emitted:
                errors.append("manifest_intent_not_emitted")
            if not manifest_executed:
                errors.append("manifest_execution_not_executed")
            if not manifest_self_critique_pass:
                errors.append("manifest_self_critique_not_pass")
            if manifest_decision_trace_entries <= 0:
                errors.append("manifest_decision_trace_entries_missing")
        else:
            checks["manifest:outcome_present"] = False
            errors.append("manifest_outcome_missing")

    top_level_risk_approved = response.get("risk_approved") is True
    top_level_intent_emitted = str(response.get("intent_status")) == "emitted"
    top_level_execution_ok = str(response.get("paper_trade_execution_status")) == "executed"
    checks["response:risk_approved"] = top_level_risk_approved
    checks["response:intent_emitted"] = top_level_intent_emitted
    checks["response:execution_executed"] = top_level_execution_ok
    if not top_level_risk_approved:
        errors.append("response_risk_not_approved")
    if not top_level_intent_emitted:
        errors.append("response_intent_not_emitted")
    if not top_level_execution_ok:
        errors.append("response_execution_not_executed")

    deduped_errors = sorted(set(errors))
    return {
        "contract": "openclaw_execution_gate.v1",
        "checked_at_utc": _utc_now_iso(),
        "run_id": run_id,
        "run_dir": run_dir,
        "passed": len(deduped_errors) == 0,
        "checks": checks,
        "errors": deduped_errors,
        "artifact_paths": artifact_paths,
    }


def _job_root_dir(settings) -> Path:
    return settings.quant_data_root / "logs" / "agents" / "openclaw-supervisor"


def _job_dir(settings, job_id: str) -> Path:
    return _job_root_dir(settings) / job_id


def _job_status_path(settings, job_id: str) -> Path:
    return _job_dir(settings, job_id) / "status.json"


def _read_job_status(settings, job_id: str) -> dict[str, Any]:
    status_path = _job_status_path(settings, job_id)
    if not status_path.exists():
        raise FileNotFoundError(f"Unknown orchestration job id: {job_id}")
    return _read_json_object(status_path, label=f"job status ({job_id})")


def _write_job_status(settings, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    payload["updated_at_utc"] = _utc_now_iso()
    _atomic_write_json(_job_status_path(settings, job_id), payload)
    return payload


def _update_job_status(settings, job_id: str, **changes: Any) -> dict[str, Any]:
    payload = _read_job_status(settings, job_id)
    payload.update(changes)
    return _write_job_status(settings, job_id, payload)


def _new_job_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"


def submit_orchestration_job(request_payload: dict[str, Any], settings) -> dict[str, Any]:
    ensure_data_root_ready(
        settings.quant_data_root,
        allow_unmounted=settings.allow_unmounted_data_root,
    )
    ensure_phase1_tree(settings.quant_data_root)

    job_id = _new_job_id()
    job_dir = _job_dir(settings, job_id)
    job_dir.mkdir(parents=True, exist_ok=False)
    request_path = job_dir / "request.json"
    result_path = job_dir / "result.json"
    verification_path = job_dir / "verification_gate.json"
    worker_log_path = job_dir / "worker.log"

    _atomic_write_json(request_path, request_payload)
    status_payload = {
        "contract": "openclaw_orchestration_job.v1",
        "job_id": job_id,
        "status": "queued",
        "created_at_utc": _utc_now_iso(),
        "updated_at_utc": _utc_now_iso(),
        "request_path": str(request_path),
        "result_path": str(result_path),
        "verification_path": str(verification_path),
        "worker_log_path": str(worker_log_path),
        "worker_pid": None,
        "started_at_utc": None,
        "finished_at_utc": None,
        "run_id": None,
        "run_dir": None,
        "error": None,
        "traceback": None,
    }
    _atomic_write_json(_job_status_path(settings, job_id), status_payload)

    command = [
        sys.executable,
        "-m",
        "quant_agents.openclaw_native",
        "--job-mode",
        "run",
        "--job-id",
        job_id,
    ]
    try:
        with worker_log_path.open("a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except Exception as exc:
        return _update_job_status(
            settings,
            job_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at_utc=_utc_now_iso(),
        )

    return _update_job_status(
        settings,
        job_id,
        worker_pid=process.pid,
    )


def run_submitted_job(job_id: str) -> dict[str, Any]:
    settings = load_settings()
    status = _read_job_status(settings, job_id)

    request_path = Path(str(status["request_path"]))
    result_path = Path(str(status["result_path"]))
    verification_path = Path(str(status["verification_path"]))

    _update_job_status(
        settings,
        job_id,
        status="running",
        started_at_utc=_utc_now_iso(),
    )
    try:
        request_payload = _read_json_object(request_path, label=f"job request ({job_id})")
        request = OpenClawOrchestrationRequest.from_dict(request_payload, settings)
        result = run_openclaw_orchestration(request, settings=settings)
        _atomic_write_json(result_path, result)

        verification = verify_orchestration_gate(result)
        _atomic_write_json(verification_path, verification)
        if verification["passed"]:
            return _update_job_status(
                settings,
                job_id,
                status="succeeded",
                run_id=result.get("run_id"),
                run_dir=result.get("run_dir"),
                finished_at_utc=_utc_now_iso(),
                error=None,
                traceback=None,
            )

        return _update_job_status(
            settings,
            job_id,
            status="blocked",
            run_id=result.get("run_id"),
            run_dir=result.get("run_dir"),
            finished_at_utc=_utc_now_iso(),
            error=";".join(verification["errors"]),
            traceback=None,
        )
    except Exception as exc:
        return _update_job_status(
            settings,
            job_id,
            status="failed",
            finished_at_utc=_utc_now_iso(),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )


def get_job_snapshot(settings, job_id: str) -> dict[str, Any]:
    status = _read_job_status(settings, job_id)
    result = _try_read_json_object(Path(str(status["result_path"])))
    verification = _try_read_json_object(Path(str(status["verification_path"])))
    return {
        "job": status,
        "result": result,
        "verification_gate": verification,
    }


def wait_for_job_snapshot(
    settings,
    job_id: str,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    deadline = None if timeout_seconds <= 0 else (time.time() + timeout_seconds)
    interval = max(0.1, poll_interval_seconds)
    while True:
        snapshot = get_job_snapshot(settings, job_id)
        status = str(snapshot["job"].get("status") or "")
        if status in TERMINAL_JOB_STATUSES:
            return snapshot
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for job {job_id} after {timeout_seconds} seconds"
            )
        time.sleep(interval)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-openclaw-entrypoint",
        description=(
            "OpenClaw-native orchestration entrypoint with deterministic async supervision "
            "and strict artifact/risk verification gate."
        ),
    )
    parser.add_argument(
        "--job-mode",
        choices=["submit", "status", "wait", "run-sync", "run"],
        default="submit",
        help=(
            "submit: enqueue async job (default); status: read job snapshot; "
            "wait: block until terminal; run-sync: run inline with strict gate; "
            "run: internal worker mode."
        ),
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="Job id for --job-mode=status|wait|run.",
    )
    parser.add_argument("--request-json", default=None, help="JSON object string for orchestration request.")
    parser.add_argument("--request-file", default=None, help="Path to JSON request payload.")
    parser.add_argument(
        "--wait-for-completion",
        action="store_true",
        help="When --job-mode=submit, wait for terminal status after enqueue.",
    )
    parser.add_argument(
        "--wait-timeout-seconds",
        type=float,
        default=0.0,
        help="Timeout for --job-mode=wait or --wait-for-completion (<=0 means no timeout).",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=2.0,
        help="Poll interval used by wait modes.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write command output JSON.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print command output JSON to stdout.",
    )
    return parser


def _read_payload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.request_file:
        raw = Path(args.request_file).expanduser().read_text(encoding="utf-8")
        payload = json.loads(raw)
    elif args.request_json:
        payload = json.loads(args.request_json)
    else:
        payload = {}
    if not isinstance(payload, dict):
        raise RuntimeError("OpenClaw request payload must be a JSON object.")
    return payload

def _emit_output(payload: dict[str, Any], args: argparse.Namespace) -> None:
    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.print_json or not args.output_json:
        print(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)

    settings = load_settings()
    if args.job_mode == "run":
        if not args.job_id:
            parser.error("--job-id is required for --job-mode=run")
        run_submitted_job(args.job_id)
        return

    if args.job_mode == "submit":
        request_payload = _read_payload_from_args(args)
        job_status = submit_orchestration_job(request_payload, settings)
        snapshot = {
            "job": job_status,
            "result": None,
            "verification_gate": None,
        }
        if args.wait_for_completion:
            snapshot = wait_for_job_snapshot(
                settings,
                str(job_status["job_id"]),
                timeout_seconds=float(args.wait_timeout_seconds),
                poll_interval_seconds=float(args.poll_interval_seconds),
            )
        _emit_output(snapshot, args)
        if args.wait_for_completion and snapshot["job"].get("status") != "succeeded":
            raise SystemExit(2)
        return

    if args.job_mode in {"status", "wait"}:
        if not args.job_id:
            parser.error("--job-id is required for --job-mode=status|wait")
        if args.job_mode == "status":
            snapshot = get_job_snapshot(settings, args.job_id)
        else:
            snapshot = wait_for_job_snapshot(
                settings,
                args.job_id,
                timeout_seconds=float(args.wait_timeout_seconds),
                poll_interval_seconds=float(args.poll_interval_seconds),
            )
        _emit_output(snapshot, args)
        if args.job_mode == "wait" and snapshot["job"].get("status") != "succeeded":
            raise SystemExit(2)
        return

    if args.job_mode == "run-sync":
        request_payload = _read_payload_from_args(args)
        request = OpenClawOrchestrationRequest.from_dict(request_payload, settings)
        result = run_openclaw_orchestration(request, settings=settings)
        verification = verify_orchestration_gate(result)
        output = {
            "job": None,
            "result": result,
            "verification_gate": verification,
        }
        _emit_output(output, args)
        if not verification["passed"]:
            raise SystemExit(2)
        return

    raise RuntimeError(f"Unsupported job mode: {args.job_mode}")


if __name__ == "__main__":
    main()
