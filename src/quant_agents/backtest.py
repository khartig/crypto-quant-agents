from __future__ import annotations
import hashlib

import json
import logging
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
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
    source_data_sha256: str
    run_dir: Path
    equity_path: Path
    metrics_path: Path
    manifest_path: Path
    archive_path: Path | None
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


def run_sma_backtest(
    settings: Settings,
    exchange: str,
    symbol: str,
    timeframe: str,
    fast_window: int = 20,
    slow_window: int = 50,
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
    manifest_path = run_dir / "run_manifest.json"

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
        "source_data_sha256": source_hash,
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

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": STRATEGY_NAME,
        "parameters": {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "fast_window": fast_window,
            "slow_window": slow_window,
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

