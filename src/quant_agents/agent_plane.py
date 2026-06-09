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
from quant_agents.backtest import STRATEGY_NAME, run_sma_backtest
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
    phase1_feature_context_path: Path
    walkforward_evaluation_path: Path
    confidence_calibration_path: Path
    risk_decision_path: Path
    paper_trade_intent_path: Path
    paper_trade_execution_path: Path
    ops_report_markdown_path: Path
    ops_report_contract_path: Path
    run_manifest_path: Path
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


def _ops_prompt(context: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are ops-report-agent for a deterministic crypto quant pipeline.",
            "Write concise markdown with sections:",
            "## Run summary",
            "## Deterministic gate outcome",
            "## Paper intent",
            "## Paper execution",
            "## Follow-ups",
            "Use only facts from the context JSON and do not invent metrics.",
            json.dumps(context, indent=2, sort_keys=True),
        ]
    )


def _deterministic_ops_markdown(context: dict[str, Any]) -> str:
    risk_approved = bool(context["risk"]["approved"])
    intent_status = str(context["intent"]["status"])
    execution_status = str(context["execution"]["status"])
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
            "## Paper execution",
            f"- Status: `{execution_status}`",
            f"- Executed action: `{context['execution']['executed_action']}`",
            f"- Executed notional USD: `{context['execution']['executed_notional_usd']}`",
            f"- Fee USD: `{context['execution']['fee_usd']}`",
            f"- Cash after USD: `{context['execution']['cash_after_usd']}`",
            f"- Position qty after: `{context['execution']['position_qty_after']}`",
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
    phase1_feature_context_path = run_dir / "phase1_feature_context.json"
    strategy_signal_path = run_dir / "strategy_proposal_signal.json"
    backtest_evaluation_path = run_dir / "backtest_evaluation.json"
    walkforward_evaluation_path = run_dir / "walkforward_evaluation.json"
    confidence_calibration_path = run_dir / "confidence_calibration.json"
    risk_decision_path = run_dir / "risk_decision.json"
    paper_trade_intent_path = run_dir / "paper_trade_intent.json"
    paper_trade_execution_path = run_dir / "paper_trade_execution.json"
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

    def risk_runner() -> RiskDecision:
        fail_reasons: list[str] = []
        block_reasons: list[str] = []
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
        action = strategy_signal.recommendation
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
        }

        if not data_quality_signal.is_valid:
            fail_reasons.append("data_quality_invalid")
        if backtest_evaluation.backtest_status != "success":
            fail_reasons.append("backtest_failed")
        else:
            total_return = backtest_evaluation.total_return
            sharpe = backtest_evaluation.sharpe
            max_drawdown = backtest_evaluation.max_drawdown
            if total_return is None or total_return < config.thresholds.min_total_return:
                fail_reasons.append("total_return_below_threshold")
            if sharpe is None or sharpe < config.thresholds.min_sharpe:
                fail_reasons.append("sharpe_below_threshold")
            if max_drawdown is None or max_drawdown < config.thresholds.max_drawdown:
                fail_reasons.append("max_drawdown_exceeded")

        if action not in {"buy", "sell"}:
            block_reasons.append("non_actionable_recommendation")
        else:
            if walkforward_quality_band == "very_low":
                fail_reasons.append(f"risk_fail_{action}_walkforward_quality_very_low")
            elif walkforward_quality_band == "low":
                block_reasons.append(f"risk_block_{action}_walkforward_quality_low")

            if contradiction_detected:
                if contradiction_severity == "fail":
                    fail_reasons.append(
                        f"risk_fail_{action}_walkforward_contradiction_{walkforward_quality_band}"
                    )
                else:
                    block_reasons.append(
                        f"risk_block_{action}_walkforward_contradiction_{walkforward_quality_band}"
                    )
        calibrated_confidence = float(confidence_calibration.get("calibrated_confidence", 0.0))
        if calibrated_confidence < config.thresholds.min_signal_confidence:
            block_reasons.append("calibrated_confidence_below_threshold")
        if (
            float(confidence_calibration.get("raw_confidence", 0.0)) >= config.thresholds.min_signal_confidence
            and calibrated_confidence < config.thresholds.min_signal_confidence
        ):
            block_reasons.append("confidence_downgraded_by_calibration")
        if contradiction_count > config.calibration_max_contradictions:
            fail_reasons.append("calibration_contradiction_limit_exceeded")
            if action in {"buy", "sell"}:
                fail_reasons.append(f"risk_fail_{action}_calibration_contradiction_limit_exceeded")

        reasons = sorted(set([*fail_reasons, *block_reasons, *calibration_reason_codes]))
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
            recommendation_confidence=calibrated_confidence,
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
            "regime": strategy_signal.regime,
            "indicator_votes": strategy_signal.indicator_votes,
            "feature_snapshot": strategy_signal.feature_snapshot,
            "reason_codes": strategy_signal.reason_codes,
            "fast_window": strategy_signal.fast_window,
            "slow_window": strategy_signal.slow_window,
            "rationale": strategy_signal.rationale,
        },
        "phase1": phase1_feature_context,
        "backtest": {
            "status": backtest_evaluation.backtest_status,
            "total_return": backtest_evaluation.total_return,
            "annualized_return": backtest_evaluation.annualized_return,
            "sharpe": backtest_evaluation.sharpe,
            "max_drawdown": backtest_evaluation.max_drawdown,
        },
        "walkforward": walkforward_evaluation,
        "calibration": confidence_calibration,
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
            "risk_decision": str(risk_decision_path),
            "paper_trade_intent": str(paper_trade_intent_path),
            "paper_trade_execution": str(paper_trade_execution_path),
            "paper_trade_destination": str(intent_destination_path) if intent_destination_path else "",
            "paper_portfolio_state": paper_trade_execution.portfolio_state_path or "",
            "paper_fills_log": paper_trade_execution.fills_log_path or "",
            "paper_execution_record": paper_trade_execution.execution_record_path or "",
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
            "walk_forward_train_bars": config.walk_forward_train_bars,
            "walk_forward_validate_bars": config.walk_forward_validate_bars,
            "walk_forward_step_bars": config.walk_forward_step_bars,
            "walk_forward_min_windows": config.walk_forward_min_windows,
            "calibration_min_walkforward_sharpe": config.calibration_min_walkforward_sharpe,
            "calibration_confidence_floor": config.calibration_confidence_floor,
            "calibration_confidence_ceiling": config.calibration_confidence_ceiling,
            "calibration_max_contradictions": config.calibration_max_contradictions,
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
            "risk_decision": str(risk_decision_path),
            "paper_trade_intent": str(paper_trade_intent_path),
            "paper_trade_execution": str(paper_trade_execution_path),
            "ops_report_markdown": str(ops_report_markdown_path),
            "ops_report_contract": str(ops_report_contract_path),
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
        phase1_feature_context_path=phase1_feature_context_path,
        walkforward_evaluation_path=walkforward_evaluation_path,
        confidence_calibration_path=confidence_calibration_path,
        risk_decision_path=risk_decision_path,
        paper_trade_intent_path=paper_trade_intent_path,
        paper_trade_execution_path=paper_trade_execution_path,
        ops_report_markdown_path=ops_report_markdown_path,
        ops_report_contract_path=ops_report_contract_path,
        run_manifest_path=run_manifest_path,
        risk_approved=risk_decision.approved,
        intent_status=paper_trade_intent.status,
        paper_trade_execution_status=paper_trade_execution.execution_status,
        intent_destination_path=intent_destination_path,
    )
