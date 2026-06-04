from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant_agents.config import Settings
from quant_agents.storage import latest_raw_dataset, new_backtest_run_dir

logger = logging.getLogger(__name__)

STRATEGY_NAME = "sma_crossover"


@dataclass(frozen=True)
class BacktestResult:
    strategy: str
    source_data_path: Path
    run_dir: Path
    equity_path: Path
    metrics_path: Path
    metrics: dict[str, float | int | str]


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


def run_sma_backtest(
    settings: Settings,
    exchange: str,
    symbol: str,
    timeframe: str,
    fast_window: int = 20,
    slow_window: int = 50,
) -> BacktestResult:
    if fast_window <= 0 or slow_window <= 0:
        raise ValueError("Moving-average windows must be positive integers.")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window.")

    source_file = latest_raw_dataset(settings.quant_data_root, exchange, symbol, timeframe)
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
    df["strategy_returns"] = df["position"] * df["returns"]
    df["equity_curve"] = (1.0 + df["strategy_returns"]).cumprod()

    periods_per_year = _periods_per_year(timeframe)
    total_return = float(df["equity_curve"].iloc[-1] - 1.0)
    bars = len(df)
    annualized_return = float((1 + total_return) ** (periods_per_year / bars) - 1) if bars else 0.0
    ret_mean = float(df["strategy_returns"].mean())
    ret_std = float(df["strategy_returns"].std(ddof=0))
    sharpe = float(np.sqrt(periods_per_year) * ret_mean / ret_std) if ret_std > 0 else 0.0
    rolling_peak = df["equity_curve"].cummax()
    drawdown = (df["equity_curve"] / rolling_peak) - 1.0
    max_drawdown = float(drawdown.min())
    buy_hold_return = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1.0)
    trade_flips = int(df["signal"].diff().abs().fillna(0).sum())

    run_dir = new_backtest_run_dir(settings.quant_data_root, STRATEGY_NAME)
    equity_path = run_dir / "equity_curve.parquet"
    metrics_path = run_dir / "metrics.json"

    df[
        [
            "timestamp",
            "close",
            "ma_fast",
            "ma_slow",
            "position",
            "strategy_returns",
            "equity_curve",
        ]
    ].to_parquet(equity_path, index=False)

    metrics: dict[str, float | int | str] = {
        "strategy": STRATEGY_NAME,
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "source_data_path": str(source_file),
        "bars": bars,
        "fast_window": fast_window,
        "slow_window": slow_window,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "buy_and_hold_return": buy_hold_return,
        "signal_flips": trade_flips,
        "data_start": str(df["timestamp"].min()),
        "data_end": str(df["timestamp"].max()),
    }

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Backtest artifacts written to %s", run_dir)

    return BacktestResult(
        strategy=STRATEGY_NAME,
        source_data_path=source_file,
        run_dir=run_dir,
        equity_path=equity_path,
        metrics_path=metrics_path,
        metrics=metrics,
    )

