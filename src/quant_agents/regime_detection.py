from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

RegimeLabel = Literal["bull_trend", "bear_trend", "range_bound", "volatile"]


@dataclass(frozen=True)
class RegimeDetectionConfig:
    mode: Literal["heuristic", "score"] = "score"
    volatility_threshold: float = 0.03
    trend_spread_threshold: float = 0.01
    persistence_bars: int = 3


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _normalize_mode(mode: str) -> Literal["heuristic", "score"]:
    normalized = str(mode).strip().lower()
    if normalized == "heuristic":
        return "heuristic"
    return "score"


def _classify_heuristic(
    *,
    trend_spread: float,
    macd: float,
    volatility: float,
    config: RegimeDetectionConfig,
) -> RegimeLabel:
    if volatility >= config.volatility_threshold:
        return "volatile"
    if trend_spread >= config.trend_spread_threshold and macd >= 0.0:
        return "bull_trend"
    if trend_spread >= config.trend_spread_threshold and macd < 0.0:
        return "bear_trend"
    return "range_bound"


def _classify_score(
    *,
    trend_spread: float,
    macd: float,
    volatility: float,
    rsi: float,
    config: RegimeDetectionConfig,
) -> RegimeLabel:
    trend_ratio = float(
        np.clip(trend_spread / max(config.trend_spread_threshold, 1e-6), 0.0, 3.0)
    )
    vol_ratio = float(
        np.clip(volatility / max(config.volatility_threshold, 1e-6), 0.0, 3.0)
    )
    momentum_sign = float(np.sign(macd))
    rsi_bias = float(np.clip((rsi - 50.0) / 50.0, -1.0, 1.0))

    volatile_score = vol_ratio + (0.15 * trend_ratio)
    bull_score = (1.20 * trend_ratio) + (0.30 * max(0.0, momentum_sign)) + (0.20 * max(0.0, rsi_bias))
    bear_score = (1.20 * trend_ratio) + (0.30 * max(0.0, -momentum_sign)) + (0.20 * max(0.0, -rsi_bias))
    range_score = (
        (1.10 * max(0.0, 1.0 - min(1.0, trend_ratio)))
        + (0.35 * max(0.0, 1.0 - min(1.0, vol_ratio)))
        + (0.15 * max(0.0, 1.0 - min(1.0, abs(momentum_sign))))
    )

    scores: dict[RegimeLabel, float] = {
        "bull_trend": bull_score,
        "bear_trend": bear_score,
        "range_bound": range_score,
        "volatile": volatile_score,
    }
    return max(scores, key=scores.get)


def _latest_stable_regime(
    raw_regimes: list[RegimeLabel],
    persistence_bars: int,
) -> tuple[RegimeLabel, int]:
    if not raw_regimes:
        return "range_bound", 0
    window_size = max(1, int(persistence_bars))
    window = raw_regimes[-window_size:]
    counts = Counter(window)
    latest = window[-1]
    stable = latest
    stable_count = int(counts.get(latest, 0))
    for regime, count in counts.items():
        if count > stable_count:
            stable = regime
            stable_count = int(count)
    return stable, stable_count


def detect_regime_from_frame(
    frame: pd.DataFrame,
    *,
    config: RegimeDetectionConfig,
) -> dict[str, Any]:
    close = pd.to_numeric(frame.get("close"), errors="coerce").ffill().bfill()
    if close.empty or close.isna().all():
        return {
            "regime": "unknown",
            "regime_confidence": 0.0,
            "regime_transition": "unknown",
            "reason_codes": ["regime_close_unavailable"],
            "diagnostics": {
                "mode": _normalize_mode(config.mode),
                "persistence_bars": max(1, int(config.persistence_bars)),
            },
        }

    sma_fast = close.rolling(window=20, min_periods=5).mean()
    sma_slow = close.rolling(window=50, min_periods=10).mean()
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    volatility_48 = close.pct_change().rolling(window=48, min_periods=10).std(ddof=0)
    rsi_14 = _compute_rsi(close, period=14)

    mode = _normalize_mode(config.mode)
    raw_regimes: list[RegimeLabel] = []
    trend_spread_series = (sma_fast / sma_slow.replace(0.0, np.nan) - 1.0).abs().fillna(0.0)
    macd_series = macd.fillna(0.0)
    volatility_series = volatility_48.fillna(0.0)
    rsi_series = rsi_14.fillna(50.0)
    for trend_spread, macd_value, volatility_value, rsi_value in zip(
        trend_spread_series.to_numpy(dtype=float),
        macd_series.to_numpy(dtype=float),
        volatility_series.to_numpy(dtype=float),
        rsi_series.to_numpy(dtype=float),
    ):
        if mode == "heuristic":
            raw = _classify_heuristic(
                trend_spread=float(trend_spread),
                macd=float(macd_value),
                volatility=float(volatility_value),
                config=config,
            )
        else:
            raw = _classify_score(
                trend_spread=float(trend_spread),
                macd=float(macd_value),
                volatility=float(volatility_value),
                rsi=float(rsi_value),
                config=config,
            )
        raw_regimes.append(raw)

    persistence_bars = max(1, int(config.persistence_bars))
    regime, support = _latest_stable_regime(raw_regimes, persistence_bars)
    if len(raw_regimes) > 1:
        previous_regime, _ = _latest_stable_regime(raw_regimes[:-1], persistence_bars)
    else:
        previous_regime = regime
    transition = "none" if regime == previous_regime else f"{previous_regime}->{regime}"

    latest_trend_spread = float(trend_spread_series.iloc[-1]) if not trend_spread_series.empty else 0.0
    latest_macd = float(macd_series.iloc[-1]) if not macd_series.empty else 0.0
    latest_volatility = float(volatility_series.iloc[-1]) if not volatility_series.empty else 0.0
    latest_rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
    trend_ratio = float(np.clip(latest_trend_spread / max(config.trend_spread_threshold, 1e-6), 0.0, 1.5))
    vol_ratio = float(np.clip(latest_volatility / max(config.volatility_threshold, 1e-6), 0.0, 1.5))
    persistence_ratio = float(np.clip(support / max(1, persistence_bars), 0.0, 1.0))

    if regime == "volatile":
        base_confidence = 0.46 + (0.42 * min(1.0, vol_ratio))
    elif regime == "bull_trend":
        base_confidence = 0.42 + (0.30 * min(1.0, trend_ratio)) + (0.20 if latest_macd >= 0.0 else 0.0)
    elif regime == "bear_trend":
        base_confidence = 0.42 + (0.30 * min(1.0, trend_ratio)) + (0.20 if latest_macd < 0.0 else 0.0)
    else:
        base_confidence = (
            0.45
            + (0.28 * max(0.0, 1.0 - min(1.0, trend_ratio)))
            + (0.17 * max(0.0, 1.0 - min(1.0, vol_ratio)))
        )
    regime_confidence = float(np.clip((0.80 * base_confidence) + (0.20 * persistence_ratio), 0.0, 1.0))

    reason_codes = [
        f"regime_mode_{mode}",
        f"regime_{regime}",
        f"regime_transition_{transition.replace('->', '_to_')}",
    ]
    if latest_volatility >= config.volatility_threshold:
        reason_codes.append("regime_volatility_above_threshold")
    else:
        reason_codes.append("regime_volatility_below_threshold")
    if latest_trend_spread >= config.trend_spread_threshold:
        reason_codes.append("regime_trend_spread_above_threshold")
    else:
        reason_codes.append("regime_trend_spread_below_threshold")
    if support < max(1, persistence_bars // 2):
        reason_codes.append("regime_low_persistence_support")

    diagnostics: dict[str, float | int | str | bool] = {
        "mode": mode,
        "latest_raw_regime": raw_regimes[-1] if raw_regimes else "unknown",
        "previous_regime": previous_regime,
        "trend_spread": latest_trend_spread,
        "trend_spread_threshold": float(max(1e-6, config.trend_spread_threshold)),
        "macd": latest_macd,
        "volatility_48": latest_volatility,
        "volatility_threshold": float(max(1e-6, config.volatility_threshold)),
        "rsi_14": latest_rsi,
        "persistence_bars": persistence_bars,
        "persistence_support": int(support),
        "persistence_ratio": persistence_ratio,
        "transition_detected": bool(transition != "none"),
    }
    return {
        "regime": regime,
        "regime_confidence": regime_confidence,
        "regime_transition": transition,
        "reason_codes": sorted(set(reason_codes)),
        "diagnostics": diagnostics,
    }
