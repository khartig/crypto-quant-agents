#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from dataclasses import replace
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
                    "confidence": 0.82,
                    "fast_window": 20,
                    "slow_window": 50,
                    "rationale": "Synthetic suite signal",
                    "indicator_votes": {
                        "buy": 0.72,
                        "sell": 0.18,
                        "hold": 0.10,
                    },
                    "regime": "bull_trend",
                    "feature_snapshot": {"macd_hist": 0.01, "rsi_14": 48.0},
                    "reason_codes": ["synthetic_suite_signal"],
                }
            )
        return "## Synthetic ops report\n- ok\n"


def _build_market_data(path: Path, bars: int = 420) -> None:
    index = np.arange(bars)
    close = 100.0 + (index * 0.05) + np.sin(index / 7.0) * 1.8
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=bars, freq="h", tz="UTC"),
            "open": close - 0.2,
            "high": close + 0.4,
            "low": close - 0.5,
            "close": close,
            "volume": 1000.0 + (np.cos(index / 5.0) * 60.0),
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
        "source_data_path": "/tmp/suite.parquet",
        "source_data_sha256": "synthetic",
        "strategy": "sma_crossover",
        "fast_window": 20,
        "slow_window": 50,
        "train_bars": 240,
        "validate_bars": 72,
        "step_bars": 72,
        "window_count": 3,
        "aggregate_total_return": 0.02,
        "aggregate_sharpe": float(aggregate_sharpe),
        "aggregate_max_drawdown": -0.08,
        "aggregate_hit_rate": 0.56,
        "stability_score": 0.72,
        "quality_score": float(quality_score),
        "windows": [],
        "diagnostics": {
            "reliability_bins": [],
            "confidence_deciles": [],
        },
    }


def _run_suite() -> tuple[list[dict[str, Any]], list[str]]:
    scenarios = [
        {
            "name": "high_no_contradiction",
            "quality": 0.75,
            "sharpe": 0.12,
            "expect_band": "high",
            "expect_approved": True,
            "expect_contradiction": False,
            "expect_directional_contradiction": False,
            "expect_quality_contradiction": False,
            "required_reason": None,
        },
        {
            "name": "high_sharpe_contradiction_block",
            "quality": 0.75,
            "sharpe": 0.08,
            "expect_band": "high",
            "expect_approved": False,
            "expect_contradiction": True,
            "expect_directional_contradiction": False,
            "expect_quality_contradiction": True,
            "required_reason": "risk_block_buy_walkforward_quality_contradiction_high",
        },
        {
            "name": "medium_sharpe_contradiction_block",
            "quality": 0.60,
            "sharpe": 0.13,
            "expect_band": "medium",
            "expect_approved": False,
            "expect_contradiction": True,
            "expect_directional_contradiction": False,
            "expect_quality_contradiction": True,
            "required_reason": "risk_block_buy_walkforward_quality_contradiction_medium",
        },
        {
            "name": "low_quality_block",
            "quality": 0.42,
            "sharpe": 0.25,
            "expect_band": "low",
            "expect_approved": False,
            "expect_contradiction": False,
            "expect_directional_contradiction": False,
            "expect_quality_contradiction": False,
            "required_reason": "risk_block_buy_walkforward_quality_low",
        },
        {
            "name": "very_low_quality_fail",
            "quality": 0.20,
            "sharpe": 0.35,
            "expect_band": "very_low",
            "expect_approved": False,
            "expect_contradiction": True,
            "expect_directional_contradiction": False,
            "expect_quality_contradiction": True,
            "required_reason": "risk_fail_buy_walkforward_quality_very_low",
        },
    ]

    original_ollama_client = agent_plane.OllamaClient
    original_walkforward_runner = agent_plane._run_walkforward_evaluation

    failures: list[str] = []
    results: list[dict[str, Any]] = []
    try:
        agent_plane.OllamaClient = _FakeOllamaClient
        for scenario in scenarios:
            with tempfile.TemporaryDirectory(prefix="walkforward-policy-suite-") as temp_root:
                root = Path(temp_root)
                source_data_path = root / "suite_input.parquet"
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
                        min_sharpe=-1_000_000_000.0,
                        max_drawdown=-1.0,
                        max_cost_return_drag=1.0,
                        min_signal_confidence=0.55,
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
                    source_data_path=source_data_path,
                )

                def _walkforward_override(**_kwargs: Any) -> dict[str, Any]:
                    return _walkforward_payload(
                        quality_score=float(scenario["quality"]),
                        aggregate_sharpe=float(scenario["sharpe"]),
                    )

                agent_plane._run_walkforward_evaluation = _walkforward_override
                run_result = run_agent_plane(settings, config)

                calibration = json.loads(
                    run_result.confidence_calibration_path.read_text(encoding="utf-8")
                )
                risk = json.loads(run_result.risk_decision_path.read_text(encoding="utf-8"))

                observed = {
                    "scenario": str(scenario["name"]),
                    "approved": bool(risk.get("approved", False)),
                    "quality_band": str(calibration.get("walkforward_quality_band")),
                    "contradiction": bool(calibration.get("contradiction_detected", False)),
                    "directional_contradiction": bool(
                        calibration.get("directional_contradiction_detected", False)
                    ),
                    "quality_contradiction": bool(
                        calibration.get("quality_contradiction_detected", False)
                    ),
                    "contradiction_severity": str(calibration.get("contradiction_severity")),
                    "cost_pressure_score": calibration.get("cost_pressure_score"),
                    "reason_codes": list(risk.get("reason_codes", [])),
                }
                results.append(observed)

                if observed["quality_band"] != scenario["expect_band"]:
                    failures.append(
                        f"{scenario['name']}: expected band {scenario['expect_band']} got {observed['quality_band']}"
                    )
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
                if required_reason and required_reason not in observed["reason_codes"]:
                    failures.append(
                        f"{scenario['name']}: missing reason code {required_reason}; got {observed['reason_codes']}"
                    )
                if calibration.get("cost_pressure_score") is None:
                    failures.append(f"{scenario['name']}: missing calibration cost_pressure_score")
    finally:
        agent_plane.OllamaClient = original_ollama_client
        agent_plane._run_walkforward_evaluation = original_walkforward_runner

    return results, failures


def main() -> None:
    results, failures = _run_suite()

    print("WALKFORWARD_POLICY_SUITE_RESULTS")
    for item in results:
        print(json.dumps(item, sort_keys=True))

    if failures:
        print("WALKFORWARD_POLICY_SUITE_STATUS=FAIL")
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    print("WALKFORWARD_POLICY_SUITE_STATUS=PASS")


if __name__ == "__main__":
    main()
