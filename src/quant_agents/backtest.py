from __future__ import annotations
import hashlib

import json
import logging
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_agents.config import Settings
from quant_agents.storage import latest_raw_dataset, new_backtest_run_dir

logger = logging.getLogger(__name__)

STRATEGY_NAME = "sma_crossover"
ENSEMBLE_STRATEGY_NAME = "adaptive_ensemble"
SUPPORTED_STRATEGY_ARMS: tuple[str, ...] = ("sma_baseline", "technical_composite", "llm_context")


@dataclass(frozen=True)
class BacktestResult:
    strategy: str
    source_data_path: Path
    source_data_sha256: str
    run_dir: Path
    equity_path: Path
    metrics_path: Path
    manifest_path: Path
    archive_path: Path | None
    metrics: dict[str, Any]
    arm_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    ensemble_metrics: dict[str, Any] = field(default_factory=dict)
    arm_attribution_path: Path | None = None


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

def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def archive_backtest_run(root: Path, run_dir: Path, strategy_name: str = STRATEGY_NAME) -> Path:
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Backtest run directory not found: {run_dir}")

    run_id = run_dir.name
    if len(run_id) >= 6 and run_id[:6].isdigit():
        month_key = f"{run_id[:4]}-{run_id[4:6]}"
    else:
        now = datetime.now(timezone.utc)
        month_key = f"{now:%Y-%m}"

    archive_dir = root / "archive" / "monthly" / month_key / "backtests" / strategy_name
    archive_dir.mkdir(parents=True, exist_ok=True)

    archive_path = archive_dir / f"{run_id}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar_handle:
        tar_handle.add(run_dir, arcname=run_id)

    checksum = _sha256_file(archive_path)
    checksum_path = Path(str(archive_path) + ".sha256")
    checksum_path.write_text(f"{checksum}  {archive_path.name}\n", encoding="utf-8")
    return archive_path


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)

def _compute_turnover_units(position: pd.Series) -> pd.Series:
    return position.diff().abs().fillna(position.abs())


def _compute_cost_adjusted_returns(
    *,
    position: pd.Series,
    gross_returns: pd.Series,
    fee_bps: float,
    slippage_bps: float,
) -> tuple[pd.Series, pd.Series, pd.Series, float]:
    turnover_units = _compute_turnover_units(position.astype(float))
    one_way_cost_bps = max(0.0, float(fee_bps)) + max(0.0, float(slippage_bps))
    cost_rate = one_way_cost_bps / 10_000.0
    cost_returns = turnover_units * cost_rate
    net_returns = gross_returns.astype(float) - cost_returns
    return turnover_units, cost_returns, net_returns, one_way_cost_bps


def _performance_stats(
    frame: pd.DataFrame,
    *,
    returns_column: str,
    equity_column: str,
    timeframe: str,
) -> dict[str, float | int | str]:
    periods_per_year = _periods_per_year(timeframe)
    total_return = float(frame[equity_column].iloc[-1] - 1.0)
    bars = len(frame)
    annualized_return = float((1 + total_return) ** (periods_per_year / bars) - 1) if bars else 0.0
    ret_mean = float(frame[returns_column].mean())
    ret_std = float(frame[returns_column].std(ddof=0))
    sharpe = float(np.sqrt(periods_per_year) * ret_mean / ret_std) if ret_std > 0 else 0.0
    rolling_peak = frame[equity_column].cummax()
    drawdown = (frame[equity_column] / rolling_peak) - 1.0
    max_drawdown = float(drawdown.min())
    buy_hold_return = float(frame["close"].iloc[-1] / frame["close"].iloc[0] - 1.0)
    return {
        "bars": bars,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "buy_and_hold_return": buy_hold_return,
        "data_start": str(frame["timestamp"].min()),
        "data_end": str(frame["timestamp"].max()),
    }


def _metrics_from_position(
    frame: pd.DataFrame,
    *,
    position_column: str,
    gross_returns_column: str,
    gross_equity_column: str,
    net_returns_column: str | None,
    net_equity_column: str | None,
    turnover_column: str | None,
    cost_returns_column: str | None,
    one_way_cost_bps: float | None,
    timeframe: str,
) -> dict[str, float | int | str]:
    gross_stats = _performance_stats(
        frame,
        returns_column=gross_returns_column,
        equity_column=gross_equity_column,
        timeframe=timeframe,
    )
    resolved_net_returns = net_returns_column or gross_returns_column
    resolved_net_equity = net_equity_column or gross_equity_column
    net_stats = _performance_stats(
        frame,
        returns_column=resolved_net_returns,
        equity_column=resolved_net_equity,
        timeframe=timeframe,
    )
    signal_flips = int(frame[position_column].diff().abs().fillna(0).sum())
    turnover_units = (
        float(frame[turnover_column].sum())
        if turnover_column and turnover_column in frame.columns
        else float(signal_flips)
    )
    total_cost_return_drag = (
        float(frame[cost_returns_column].sum())
        if cost_returns_column and cost_returns_column in frame.columns
        else 0.0
    )
    effective_cost_per_turnover_bps = (
        float((total_cost_return_drag / turnover_units) * 10_000.0)
        if turnover_units > 0
        else 0.0
    )
    break_even_one_way_cost_bps = (
        float((max(0.0, float(gross_stats["total_return"])) / turnover_units) * 10_000.0)
        if turnover_units > 0
        else 0.0
    )
    return {
        "bars": int(net_stats["bars"]),
        "total_return": float(net_stats["total_return"]),
        "annualized_return": float(net_stats["annualized_return"]),
        "sharpe": float(net_stats["sharpe"]),
        "max_drawdown": float(net_stats["max_drawdown"]),
        "gross_total_return": float(gross_stats["total_return"]),
        "gross_annualized_return": float(gross_stats["annualized_return"]),
        "gross_sharpe": float(gross_stats["sharpe"]),
        "gross_max_drawdown": float(gross_stats["max_drawdown"]),
        "net_total_return": float(net_stats["total_return"]),
        "net_annualized_return": float(net_stats["annualized_return"]),
        "net_sharpe": float(net_stats["sharpe"]),
        "net_max_drawdown": float(net_stats["max_drawdown"]),
        "buy_and_hold_return": float(net_stats["buy_and_hold_return"]),
        "signal_flips": signal_flips,
        "turnover_units": turnover_units,
        "total_cost_return_drag": total_cost_return_drag,
        "effective_cost_per_turnover_bps": effective_cost_per_turnover_bps,
        "break_even_one_way_cost_bps": break_even_one_way_cost_bps,
        "one_way_trading_cost_bps": float(one_way_cost_bps or 0.0),
        "data_start": str(net_stats["data_start"]),
        "data_end": str(net_stats["data_end"]),
    }


def _normalize_weights(
    enabled_arms: tuple[str, ...],
    arm_weights: dict[str, float] | None,
) -> dict[str, float]:
    if not enabled_arms:
        return {}
    provided = arm_weights or {}
    sanitized = {arm: max(0.0, float(provided.get(arm, 0.0))) for arm in enabled_arms}
    total = sum(sanitized.values())
    if total <= 0:
        equal = 1.0 / len(enabled_arms)
        return {arm: equal for arm in enabled_arms}
    return {arm: value / total for arm, value in sanitized.items()}


def _technical_composite_position(close: pd.Series) -> pd.Series:
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    rsi_14 = _compute_rsi(close, period=14)
    ret_4 = close.pct_change(periods=4).fillna(0.0)

    bullish_score = (
        (macd_hist > 0).astype(float)
        + (rsi_14 < 52.0).astype(float)
        + (ret_4 > 0).astype(float)
    )
    bearish_score = (
        (macd_hist < 0).astype(float)
        + (rsi_14 > 48.0).astype(float)
        + (ret_4 < 0).astype(float)
    )
    return (bullish_score >= (bearish_score + 1.0)).astype(float)


def _build_arm_position_series(
    frame: pd.DataFrame,
    *,
    arm_name: str,
    fast_window: int,
    slow_window: int,
    llm_recommendation: str,
) -> pd.Series:
    close = frame["close"].astype(float)
    if arm_name == "sma_baseline":
        ma_fast = close.rolling(window=fast_window).mean()
        ma_slow = close.rolling(window=slow_window).mean()
        return (ma_fast > ma_slow).astype(float).fillna(0.0)
    if arm_name == "technical_composite":
        return _technical_composite_position(close).fillna(0.0)
    if arm_name == "llm_context":
        if llm_recommendation == "buy":
            trend_filter = close > close.rolling(window=max(8, fast_window // 2), min_periods=2).mean()
            return trend_filter.astype(float).fillna(0.0)
        return pd.Series(np.zeros(len(close), dtype=float), index=close.index)
    return pd.Series(np.zeros(len(close), dtype=float), index=close.index)


def run_ensemble_backtest(
    settings: Settings,
    exchange: str,
    symbol: str,
    timeframe: str,
    *,
    fast_window: int,
    slow_window: int,
    enabled_arms: tuple[str, ...],
    arm_weights: dict[str, float] | None,
    llm_recommendation: str,
    ensemble_mode: str,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    source_data_path: Path | None = None,
    archive_run: bool = False,
) -> BacktestResult:
    if fast_window <= 0 or slow_window <= 0:
        raise ValueError("Moving-average windows must be positive integers.")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window.")

    source_file = source_data_path or latest_raw_dataset(settings.quant_data_root, exchange, symbol, timeframe)
    if not source_file.exists():
        raise FileNotFoundError(f"Backtest input file does not exist: {source_file}")
    source_hash = _sha256_file(source_file)

    logger.info("Loading market data from %s", source_file)
    frame = pd.read_parquet(source_file).sort_values("timestamp").reset_index(drop=True)
    if len(frame) < (slow_window + 5):
        raise RuntimeError(
            f"Insufficient data for backtest: {len(frame)} rows, need at least {slow_window + 5}."
        )

    frame["returns"] = frame["close"].pct_change().fillna(0.0)
    resolved_arms = tuple(
        arm for arm in enabled_arms if arm in SUPPORTED_STRATEGY_ARMS
    ) or ("sma_baseline",)
    resolved_weights = _normalize_weights(resolved_arms, arm_weights)

    fee_bps = max(0.0, float(fee_bps))
    slippage_bps = max(0.0, float(slippage_bps))
    arm_attribution = pd.DataFrame({"timestamp": frame["timestamp"], "close": frame["close"], "returns": frame["returns"]})
    arm_metrics: dict[str, dict[str, Any]] = {}
    for arm in resolved_arms:
        position_col = f"{arm}_position"
        gross_returns_col = f"{arm}_strategy_returns_gross"
        cost_returns_col = f"{arm}_cost_returns"
        turnover_col = f"{arm}_turnover_units"
        net_returns_col = f"{arm}_strategy_returns_net"
        returns_col = f"{arm}_strategy_returns"
        gross_equity_col = f"{arm}_equity_curve_gross"
        net_equity_col = f"{arm}_equity_curve_net"
        equity_col = f"{arm}_equity_curve"
        position_series = _build_arm_position_series(
            frame,
            arm_name=arm,
            fast_window=fast_window,
            slow_window=slow_window,
            llm_recommendation=llm_recommendation,
        ).clip(lower=0.0, upper=1.0)
        arm_frame = pd.DataFrame(
            {
                "timestamp": frame["timestamp"],
                "close": frame["close"],
                "returns": frame["returns"],
                position_col: position_series.astype(float),
            }
        )
        arm_frame[gross_returns_col] = arm_frame[position_col].shift(1).fillna(0.0) * arm_frame["returns"]
        (
            arm_frame[turnover_col],
            arm_frame[cost_returns_col],
            arm_frame[net_returns_col],
            one_way_cost_bps,
        ) = _compute_cost_adjusted_returns(
            position=arm_frame[position_col],
            gross_returns=arm_frame[gross_returns_col],
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        arm_frame[returns_col] = arm_frame[net_returns_col]
        arm_frame[gross_equity_col] = (1.0 + arm_frame[gross_returns_col]).cumprod()
        arm_frame[net_equity_col] = (1.0 + arm_frame[net_returns_col]).cumprod()
        arm_frame[equity_col] = arm_frame[net_equity_col]
        frame[position_col] = arm_frame[position_col]
        arm_metrics[arm] = _metrics_from_position(
            arm_frame,
            position_column=position_col,
            gross_returns_column=gross_returns_col,
            gross_equity_column=gross_equity_col,
            net_returns_column=net_returns_col,
            net_equity_column=net_equity_col,
            turnover_column=turnover_col,
            cost_returns_column=cost_returns_col,
            one_way_cost_bps=one_way_cost_bps,
            timeframe=timeframe,
        )
        arm_metrics[arm]["weight"] = float(resolved_weights.get(arm, 0.0))
        arm_attribution[position_col] = arm_frame[position_col]
        arm_attribution[gross_returns_col] = arm_frame[gross_returns_col]
        arm_attribution[cost_returns_col] = arm_frame[cost_returns_col]
        arm_attribution[turnover_col] = arm_frame[turnover_col]
        arm_attribution[net_returns_col] = arm_frame[net_returns_col]
        arm_attribution[returns_col] = arm_frame[returns_col]

    frame["ensemble_position"] = 0.0
    for arm in resolved_arms:
        frame["ensemble_position"] += frame[f"{arm}_position"] * float(resolved_weights.get(arm, 0.0))
    frame["ensemble_position"] = frame["ensemble_position"].clip(lower=0.0, upper=1.0)
    frame["ensemble_strategy_returns_gross"] = frame["ensemble_position"].shift(1).fillna(0.0) * frame["returns"]
    (
        frame["ensemble_turnover_units"],
        frame["ensemble_cost_returns"],
        frame["ensemble_strategy_returns_net"],
        ensemble_one_way_cost_bps,
    ) = _compute_cost_adjusted_returns(
        position=frame["ensemble_position"],
        gross_returns=frame["ensemble_strategy_returns_gross"],
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    frame["ensemble_strategy_returns"] = frame["ensemble_strategy_returns_net"]
    frame["equity_curve_gross"] = (1.0 + frame["ensemble_strategy_returns_gross"]).cumprod()
    frame["equity_curve_net"] = (1.0 + frame["ensemble_strategy_returns_net"]).cumprod()
    frame["equity_curve"] = frame["equity_curve_net"]

    ensemble_metrics = _metrics_from_position(
        frame[
            [
                "timestamp",
                "close",
                "ensemble_position",
                "ensemble_strategy_returns_gross",
                "ensemble_strategy_returns_net",
                "ensemble_turnover_units",
                "ensemble_cost_returns",
                "equity_curve_gross",
                "equity_curve_net",
            ]
        ].copy(),
        position_column="ensemble_position",
        gross_returns_column="ensemble_strategy_returns_gross",
        gross_equity_column="equity_curve_gross",
        net_returns_column="ensemble_strategy_returns_net",
        net_equity_column="equity_curve_net",
        turnover_column="ensemble_turnover_units",
        cost_returns_column="ensemble_cost_returns",
        one_way_cost_bps=ensemble_one_way_cost_bps,
        timeframe=timeframe,
    )
    ensemble_metrics["strategy"] = ENSEMBLE_STRATEGY_NAME
    ensemble_metrics["ensemble_mode"] = str(ensemble_mode)
    ensemble_metrics["enabled_arms"] = list(resolved_arms)
    ensemble_metrics["llm_recommendation"] = llm_recommendation

    run_dir = new_backtest_run_dir(settings.quant_data_root, ENSEMBLE_STRATEGY_NAME)
    equity_path = run_dir / "equity_curve.parquet"
    metrics_path = run_dir / "metrics.json"
    manifest_path = run_dir / "run_manifest.json"
    arm_attribution_path = run_dir / "arm_attribution.parquet"

    frame[
        [
            "timestamp",
            "close",
            "ensemble_position",
            "ensemble_strategy_returns",
            "ensemble_strategy_returns_gross",
            "ensemble_cost_returns",
            "ensemble_turnover_units",
            "equity_curve",
            "equity_curve_gross",
        ]
    ].to_parquet(equity_path, index=False)
    arm_attribution["ensemble_position"] = frame["ensemble_position"]
    arm_attribution["ensemble_strategy_returns_gross"] = frame["ensemble_strategy_returns_gross"]
    arm_attribution["ensemble_cost_returns"] = frame["ensemble_cost_returns"]
    arm_attribution["ensemble_turnover_units"] = frame["ensemble_turnover_units"]
    arm_attribution["ensemble_strategy_returns_net"] = frame["ensemble_strategy_returns_net"]
    arm_attribution["ensemble_strategy_returns"] = frame["ensemble_strategy_returns"]
    arm_attribution.to_parquet(arm_attribution_path, index=False)

    metrics: dict[str, Any] = {
        "strategy": ENSEMBLE_STRATEGY_NAME,
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "source_data_path": str(source_file),
        "source_data_sha256": source_hash,
        **ensemble_metrics,
        "arm_metrics": arm_metrics,
        "arm_weights": {arm: float(resolved_weights.get(arm, 0.0)) for arm in resolved_arms},
        "cost_model": {
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "one_way_trading_cost_bps": fee_bps + slippage_bps,
        },
        "arm_attribution_path": str(arm_attribution_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": ENSEMBLE_STRATEGY_NAME,
        "parameters": {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "fast_window": fast_window,
            "slow_window": slow_window,
            "ensemble_mode": ensemble_mode,
            "enabled_arms": list(resolved_arms),
            "arm_weights": {arm: float(resolved_weights.get(arm, 0.0)) for arm in resolved_arms},
            "llm_recommendation": llm_recommendation,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
        },
        "source_data": {
            "path": str(source_file),
            "sha256": source_hash,
            "rows": int(len(frame)),
            "start": str(frame["timestamp"].min()),
            "end": str(frame["timestamp"].max()),
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "equity_curve_path": str(equity_path),
            "metrics_path": str(metrics_path),
            "arm_attribution_path": str(arm_attribution_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    archive_path: Path | None = None
    if archive_run:
        archive_path = archive_backtest_run(settings.quant_data_root, run_dir, ENSEMBLE_STRATEGY_NAME)
        logger.info("Archived ensemble backtest run to %s", archive_path)
    logger.info("Ensemble backtest artifacts written to %s", run_dir)

    return BacktestResult(
        strategy=ENSEMBLE_STRATEGY_NAME,
        source_data_path=source_file,
        source_data_sha256=source_hash,
        run_dir=run_dir,
        equity_path=equity_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        archive_path=archive_path,
        metrics=metrics,
        arm_metrics=arm_metrics,
        ensemble_metrics=ensemble_metrics,
        arm_attribution_path=arm_attribution_path,
    )


def run_sma_backtest(
    settings: Settings,
    exchange: str,
    symbol: str,
    timeframe: str,
    fast_window: int = 20,
    slow_window: int = 50,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    source_data_path: Path | None = None,
    archive_run: bool = False,
) -> BacktestResult:
    if fast_window <= 0 or slow_window <= 0:
        raise ValueError("Moving-average windows must be positive integers.")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window.")
    source_file = source_data_path or latest_raw_dataset(settings.quant_data_root, exchange, symbol, timeframe)
    if not source_file.exists():
        raise FileNotFoundError(f"Backtest input file does not exist: {source_file}")

    source_hash = _sha256_file(source_file)
    logger.info("Loading market data from %s", source_file)
    df = pd.read_parquet(source_file).sort_values("timestamp").reset_index(drop=True)

    if len(df) < (slow_window + 5):
        raise RuntimeError(
            f"Insufficient data for backtest: {len(df)} rows, need at least {slow_window + 5}."
        )

    df["returns"] = df["close"].pct_change().fillna(0.0)
    df["ma_fast"] = df["close"].rolling(window=fast_window).mean()
    df["ma_slow"] = df["close"].rolling(window=slow_window).mean()
    df["signal"] = (df["ma_fast"] > df["ma_slow"]).astype(float)
    df["position"] = df["signal"].shift(1).fillna(0.0)
    fee_bps = max(0.0, float(fee_bps))
    slippage_bps = max(0.0, float(slippage_bps))
    df["strategy_returns_gross"] = df["position"] * df["returns"]
    (
        df["turnover_units"],
        df["cost_returns"],
        df["strategy_returns_net"],
        one_way_cost_bps,
    ) = _compute_cost_adjusted_returns(
        position=df["position"],
        gross_returns=df["strategy_returns_gross"],
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    df["strategy_returns"] = df["strategy_returns_net"]
    df["equity_curve_gross"] = (1.0 + df["strategy_returns_gross"]).cumprod()
    df["equity_curve_net"] = (1.0 + df["strategy_returns_net"]).cumprod()
    df["equity_curve"] = df["equity_curve_net"]
    computed_metrics = _metrics_from_position(
        df[
            [
                "timestamp",
                "close",
                "position",
                "strategy_returns_gross",
                "strategy_returns_net",
                "turnover_units",
                "cost_returns",
                "equity_curve_gross",
                "equity_curve_net",
            ]
        ].copy(),
        position_column="position",
        gross_returns_column="strategy_returns_gross",
        gross_equity_column="equity_curve_gross",
        net_returns_column="strategy_returns_net",
        net_equity_column="equity_curve_net",
        turnover_column="turnover_units",
        cost_returns_column="cost_returns",
        one_way_cost_bps=one_way_cost_bps,
        timeframe=timeframe,
    )
    bars = int(computed_metrics["bars"])

    run_dir = new_backtest_run_dir(settings.quant_data_root, STRATEGY_NAME)
    equity_path = run_dir / "equity_curve.parquet"
    metrics_path = run_dir / "metrics.json"
    manifest_path = run_dir / "run_manifest.json"

    df[
        [
            "timestamp",
            "close",
            "ma_fast",
            "ma_slow",
            "position",
            "strategy_returns",
            "strategy_returns_gross",
            "cost_returns",
            "turnover_units",
            "equity_curve",
            "equity_curve_gross",
        ]
    ].to_parquet(equity_path, index=False)

    metrics: dict[str, float | int | str] = {
        "strategy": STRATEGY_NAME,
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "source_data_path": str(source_file),
        "source_data_sha256": source_hash,
        "bars": bars,
        "fast_window": fast_window,
        "slow_window": slow_window,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        **computed_metrics,
    }

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": STRATEGY_NAME,
        "parameters": {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "fast_window": fast_window,
            "slow_window": slow_window,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
        },
        "source_data": {
            "path": str(source_file),
            "sha256": source_hash,
            "rows": bars,
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max()),
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "equity_curve_path": str(equity_path),
            "metrics_path": str(metrics_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    archive_path: Path | None = None
    if archive_run:
        archive_path = archive_backtest_run(settings.quant_data_root, run_dir, STRATEGY_NAME)
        logger.info("Archived backtest run to %s", archive_path)
    logger.info("Backtest artifacts written to %s", run_dir)

    return BacktestResult(
        strategy=STRATEGY_NAME,
        source_data_path=source_file,
        source_data_sha256=source_hash,
        run_dir=run_dir,
        equity_path=equity_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        archive_path=archive_path,
        metrics=metrics,
    )

