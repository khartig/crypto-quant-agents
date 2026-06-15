#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import quant_agents.agent_plane as agent_plane
from quant_agents.agent_plane import AgentPlaneConfig, RiskThresholds, run_agent_plane
from quant_agents.config import load_settings


class _FakeOllamaClient:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def list_models(self) -> list[str]:
        return ["fake-strategy", "fake-ops"]

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: str,
        temperature: float,
        format_json: bool,
    ) -> str:
        if format_json:
            return json.dumps(
                {
                    "recommendation": "buy",
                    "confidence": 0.81,
                    "fast_window": 20,
                    "slow_window": 50,
                    "rationale": "Bullish breakout with improving momentum and upside continuation.",
                    "indicator_votes": {
                        "buy": 0.75,
                        "sell": 0.15,
                        "hold": 0.10,
                    },
                    "regime": "bull_trend",
                    "feature_snapshot": {
                        "macd_hist": 0.008,
                        "rsi_14": 56.0,
                        "volatility_20": 0.02,
                    },
                    "reason_codes": ["synthetic_phase2_suite_signal"],
                }
            )
        return "## Synthetic ops report\n- deterministic suite fallback\n"


def _build_market_data(path: Path, bars: int = 520) -> None:
    index = np.arange(bars)
    close = 100.0 + (index * 0.07) + np.sin(index / 9.0) * 1.4
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                end=datetime.now(timezone.utc).replace(microsecond=0),
                periods=bars,
                freq="h",
                tz="UTC",
            ),
            "open": close - 0.15,
            "high": close + 0.35,
            "low": close - 0.45,
            "close": close,
            "volume": 1400.0 + (np.cos(index / 6.0) * 90.0),
        }
    )
    frame.to_parquet(path, index=False)


def _walkforward_payload(quality_score: float, aggregate_sharpe: float) -> dict[str, Any]:
    return {
        "contract": "walkforward_evaluation.v1",
        "created_at_utc": "2026-01-01T00:00:00Z",
        "exchange": "kraken",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "source_data_path": "/tmp/phase2_suite.parquet",
        "source_data_sha256": "synthetic",
        "strategy": "sma_crossover",
        "fast_window": 20,
        "slow_window": 50,
        "train_bars": 240,
        "validate_bars": 72,
        "step_bars": 72,
        "window_count": 3,
        "aggregate_total_return": 0.03,
        "aggregate_sharpe": float(aggregate_sharpe),
        "aggregate_max_drawdown": -0.08,
        "aggregate_hit_rate": 0.59,
        "stability_score": 0.76,
        "quality_score": float(quality_score),
        "windows": [],
        "diagnostics": {
            "reliability_bins": [],
            "confidence_deciles": [],
        },
    }


def _run_phase2_suite() -> tuple[list[dict[str, Any]], list[str]]:
    scenarios = [
        {
            "name": "phase2_high_quality_pass",
            "quality": 0.78,
            "sharpe": 0.18,
            "expect_approved": True,
            "expect_contradiction": False,
            "expect_directional_contradiction": False,
            "expect_quality_contradiction": False,
            "required_reason": None,
        },
        {
            "name": "phase2_high_quality_sharpe_contradiction",
            "quality": 0.74,
            "sharpe": 0.03,
            "expect_approved": False,
            "expect_contradiction": True,
            "expect_directional_contradiction": False,
            "expect_quality_contradiction": True,
            "required_reason": "risk_block_buy_walkforward_quality_contradiction_high",
        },
    ]

    original_ollama_client = agent_plane.OllamaClient
    original_walkforward_runner = agent_plane._run_walkforward_evaluation

    failures: list[str] = []
    results: list[dict[str, Any]] = []
    try:
        agent_plane.OllamaClient = _FakeOllamaClient
        for scenario in scenarios:
            with tempfile.TemporaryDirectory(prefix="phase2-verifier-") as temp_root:
                root = Path(temp_root)
                source_data_path = root / "phase2_suite_input.parquet"
                _build_market_data(source_data_path)

                settings = replace(
                    load_settings(),
                    quant_data_root=root,
                    allow_unmounted_data_root=True,
                    ollama_strategy_model="fake-strategy",
                    ollama_ops_model="fake-ops",
                )
                config = AgentPlaneConfig(
                    exchange="kraken",
                    symbol="BTC/USDT",
                    timeframe="1h",
                    strategy_model="fake-strategy",
                    ops_model="fake-ops",
                    step_retries=0,
                    thresholds=RiskThresholds(
                        min_total_return=-1.0,
                        min_sharpe=-1.0,
                        max_drawdown=-1.0,
                        max_cost_return_drag=1.0,
                        min_signal_confidence=0.10,
                    ),
                    backtest_fee_bps=5.0,
                    backtest_slippage_bps=2.5,
                    walk_forward_fee_bps=5.0,
                    walk_forward_slippage_bps=2.5,
                    paper_notional_usd=100.0,
                    paper_starting_cash_usd=10000.0,
                    paper_fee_bps=5.0,
                    paper_slippage_bps=1.0,
                    minimum_bars=120,
                    walk_forward_train_bars=240,
                    walk_forward_validate_bars=72,
                    walk_forward_step_bars=72,
                    walk_forward_min_windows=1,
                    calibration_min_walkforward_sharpe=0.10,
                    calibration_confidence_floor=0.05,
                    calibration_confidence_ceiling=0.95,
                    calibration_max_contradictions=5,
                    self_critique_min_score=0.55,
                    self_critique_max_findings=6,
                    ops_report_verbosity="standard",
                    source_data_path=source_data_path,
                )

                def _walkforward_override(**_kwargs: Any) -> dict[str, Any]:
                    return _walkforward_payload(
                        quality_score=float(scenario["quality"]),
                        aggregate_sharpe=float(scenario["sharpe"]),
                    )

                agent_plane._run_walkforward_evaluation = _walkforward_override

                run_result_a = run_agent_plane(settings, config)
                calibration_a = json.loads(run_result_a.confidence_calibration_path.read_text(encoding="utf-8"))
                risk_a = json.loads(run_result_a.risk_decision_path.read_text(encoding="utf-8"))

                run_result_b = run_agent_plane(settings, config)
                calibration_b = json.loads(run_result_b.confidence_calibration_path.read_text(encoding="utf-8"))
                risk_b = json.loads(run_result_b.risk_decision_path.read_text(encoding="utf-8"))

                observed = {
                    "scenario": str(scenario["name"]),
                    "approved": bool(risk_a.get("approved", False)),
                    "quality_band": str(calibration_a.get("walkforward_quality_band")),
                    "contradiction": bool(calibration_a.get("contradiction_detected", False)),
                    "directional_contradiction": bool(
                        calibration_a.get("directional_contradiction_detected", False)
                    ),
                    "quality_contradiction": bool(
                        calibration_a.get("quality_contradiction_detected", False)
                    ),
                    "contradiction_severity": str(calibration_a.get("contradiction_severity")),
                    "calibrated_confidence": calibration_a.get("calibrated_confidence"),
                    "cost_pressure_score": calibration_a.get("cost_pressure_score"),
                    "risk_reason_codes": list(risk_a.get("reason_codes", [])),
                }
                results.append(observed)

                if observed["approved"] != scenario["expect_approved"]:
                    failures.append(
                        f"{scenario['name']}: expected approved={scenario['expect_approved']} got {observed['approved']}"
                    )
                if observed["contradiction"] != scenario["expect_contradiction"]:
                    failures.append(
                        f"{scenario['name']}: expected contradiction={scenario['expect_contradiction']} got {observed['contradiction']}"
                    )
                if observed["directional_contradiction"] != scenario["expect_directional_contradiction"]:
                    failures.append(
                        f"{scenario['name']}: expected directional_contradiction={scenario['expect_directional_contradiction']} got {observed['directional_contradiction']}"
                    )
                if observed["quality_contradiction"] != scenario["expect_quality_contradiction"]:
                    failures.append(
                        f"{scenario['name']}: expected quality_contradiction={scenario['expect_quality_contradiction']} got {observed['quality_contradiction']}"
                    )
                required_reason = scenario["required_reason"]
                if required_reason and required_reason not in observed["risk_reason_codes"]:
                    failures.append(
                        f"{scenario['name']}: missing reason code {required_reason}; got {observed['risk_reason_codes']}"
                    )

                for required_calibration_key in (
                    "calibrated_confidence",
                    "walkforward_quality_score",
                    "walkforward_sharpe",
                    "directional_contradiction_detected",
                    "quality_contradiction_detected",
                    "cost_pressure_score",
                    "diagnostics",
                    "reason_codes",
                ):
                    if required_calibration_key not in calibration_a:
                        failures.append(
                            f"{scenario['name']}: calibration missing key {required_calibration_key}"
                        )

                risk_observed = risk_a.get("observed", {})
                if not isinstance(risk_observed, dict):
                    failures.append(f"{scenario['name']}: risk observed payload missing")
                else:
                    for required_observed_key in (
                        "calibrated_confidence",
                        "walkforward_quality_score",
                        "walkforward_sharpe",
                    ):
                        if required_observed_key not in risk_observed:
                            failures.append(
                                f"{scenario['name']}: risk observed missing key {required_observed_key}"
                            )

                calibration_projection_a = {
                    "calibrated_confidence": calibration_a.get("calibrated_confidence"),
                    "walkforward_quality_score": calibration_a.get("walkforward_quality_score"),
                    "walkforward_quality_band": calibration_a.get("walkforward_quality_band"),
                    "walkforward_sharpe": calibration_a.get("walkforward_sharpe"),
                    "contradiction_detected": calibration_a.get("contradiction_detected"),
                    "contradiction_severity": calibration_a.get("contradiction_severity"),
                    "reason_codes": calibration_a.get("reason_codes"),
                }
                calibration_projection_b = {
                    "calibrated_confidence": calibration_b.get("calibrated_confidence"),
                    "walkforward_quality_score": calibration_b.get("walkforward_quality_score"),
                    "walkforward_quality_band": calibration_b.get("walkforward_quality_band"),
                    "walkforward_sharpe": calibration_b.get("walkforward_sharpe"),
                    "contradiction_detected": calibration_b.get("contradiction_detected"),
                    "contradiction_severity": calibration_b.get("contradiction_severity"),
                    "reason_codes": calibration_b.get("reason_codes"),
                }
                risk_projection_a = {
                    "approved": risk_a.get("approved"),
                    "deterministic_gate": risk_a.get("deterministic_gate"),
                    "reason_codes": risk_a.get("reason_codes"),
                    "recommendation_confidence": risk_a.get("recommendation_confidence"),
                    "observed_calibrated_confidence": (
                        risk_a.get("observed", {}).get("calibrated_confidence")
                        if isinstance(risk_a.get("observed"), dict)
                        else None
                    ),
                    "observed_walkforward_quality_score": (
                        risk_a.get("observed", {}).get("walkforward_quality_score")
                        if isinstance(risk_a.get("observed"), dict)
                        else None
                    ),
                }
                risk_projection_b = {
                    "approved": risk_b.get("approved"),
                    "deterministic_gate": risk_b.get("deterministic_gate"),
                    "reason_codes": risk_b.get("reason_codes"),
                    "recommendation_confidence": risk_b.get("recommendation_confidence"),
                    "observed_calibrated_confidence": (
                        risk_b.get("observed", {}).get("calibrated_confidence")
                        if isinstance(risk_b.get("observed"), dict)
                        else None
                    ),
                    "observed_walkforward_quality_score": (
                        risk_b.get("observed", {}).get("walkforward_quality_score")
                        if isinstance(risk_b.get("observed"), dict)
                        else None
                    ),
                }
                if calibration_projection_a != calibration_projection_b:
                    failures.append(
                        f"{scenario['name']}: calibration output is not deterministic across identical replay inputs"
                    )
                if risk_projection_a != risk_projection_b:
                    failures.append(
                        f"{scenario['name']}: risk output is not deterministic across identical replay inputs"
                    )
    finally:
        agent_plane.OllamaClient = original_ollama_client
        agent_plane._run_walkforward_evaluation = original_walkforward_runner

    return results, failures


def main() -> None:
    results, failures = _run_phase2_suite()

    print("PHASE2_COMPLETION_RESULTS")
    for item in results:
        print(json.dumps(item, sort_keys=True))

    if failures:
        print("PHASE2_COMPLETION_STATUS=FAIL")
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    print("PHASE2_COMPLETION_STATUS=PASS")


if __name__ == "__main__":
    main()
