#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from quant_agents.regime_detection import RegimeDetectionConfig, detect_regime_from_frame


def _build_frame(close_values: np.ndarray) -> pd.DataFrame:
    close = pd.Series(close_values.astype(float))
    high = close * 1.002
    low = close * 0.998
    open_price = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.full(len(close), 1000.0))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=len(close), freq="h", tz="UTC"),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _assert(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _run_suite() -> tuple[list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    results: list[dict[str, Any]] = []

    bullish_close = np.linspace(100.0, 140.0, 320)
    bearish_close = np.linspace(140.0, 100.0, 320)
    range_close = np.full(320, 100.0) + (np.sin(np.arange(320) / 8.0) * 0.2)
    volatile_close = np.empty(320, dtype=float)
    volatile_close[0] = 100.0
    for idx in range(1, len(volatile_close)):
        direction = 1.0 if idx % 2 == 0 else -1.0
        volatile_close[idx] = volatile_close[idx - 1] * (1.0 + (direction * 0.05))

    scenarios = [
        ("heuristic_bull", bullish_close, "bull_trend"),
        ("heuristic_bear", bearish_close, "bear_trend"),
        ("heuristic_range", range_close, "range_bound"),
        ("heuristic_volatile", volatile_close, "volatile"),
    ]

    for name, close_values, expected in scenarios:
        frame = _build_frame(close_values)
        result = detect_regime_from_frame(
            frame,
            config=RegimeDetectionConfig(
                mode="heuristic",
                volatility_threshold=0.03,
                trend_spread_threshold=0.01,
                persistence_bars=3,
            ),
        )
        results.append(
            {
                "scenario": name,
                "expected_regime": expected,
                "observed_regime": result.get("regime"),
                "regime_confidence": result.get("regime_confidence"),
                "regime_transition": result.get("regime_transition"),
            }
        )
        _assert(
            str(result.get("regime")) == expected,
            f"{name}: expected regime={expected}, got {result.get('regime')}",
            failures,
        )
        _assert(
            0.0 <= float(result.get("regime_confidence", -1.0)) <= 1.0,
            f"{name}: regime_confidence out of range [0,1]",
            failures,
        )
        _assert(
            isinstance(result.get("diagnostics"), dict),
            f"{name}: diagnostics payload missing",
            failures,
        )

    deterministic_frame = _build_frame(bullish_close)
    deterministic_config = RegimeDetectionConfig(
        mode="score",
        volatility_threshold=0.03,
        trend_spread_threshold=0.01,
        persistence_bars=4,
    )
    run_a = detect_regime_from_frame(deterministic_frame, config=deterministic_config)
    run_b = detect_regime_from_frame(deterministic_frame, config=deterministic_config)
    results.append(
        {
            "scenario": "score_mode_deterministic",
            "regime_a": run_a.get("regime"),
            "regime_b": run_b.get("regime"),
            "equal": run_a == run_b,
        }
    )
    _assert(
        run_a == run_b,
        "score_mode_deterministic: repeated runs on identical input are not deterministic",
        failures,
    )
    _assert(
        str(run_a.get("regime")) in {"bull_trend", "bear_trend", "range_bound", "volatile"},
        "score_mode_deterministic: regime output is outside expected label set",
        failures,
    )
    return results, failures


def main() -> None:
    results, failures = _run_suite()
    print("REGIME_DETECTION_SUITE_RESULTS")
    for row in results:
        print(json.dumps(row, sort_keys=True))

    if failures:
        print("REGIME_DETECTION_SUITE_STATUS=FAIL")
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    print("REGIME_DETECTION_SUITE_STATUS=PASS")


if __name__ == "__main__":
    main()
