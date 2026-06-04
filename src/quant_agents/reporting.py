from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from quant_agents.backtest import STRATEGY_NAME
from quant_agents.config import Settings
from quant_agents.storage import latest_backtest_run_dir, latest_raw_dataset


@dataclass(frozen=True)
class ReportResult:
    report_path: Path
    strategy_run_dir: Path


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


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def generate_daily_report(
    settings: Settings,
    exchange: str,
    symbol: str,
    timeframe: str,
    strategy_name: str = STRATEGY_NAME,
) -> ReportResult:
    raw_file = latest_raw_dataset(settings.quant_data_root, exchange, symbol, timeframe)
    backtest_dir = latest_backtest_run_dir(settings.quant_data_root, strategy_name)
    metrics_path = backtest_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    df = pd.read_parquet(raw_file).sort_values("timestamp").reset_index(drop=True)
    bar_delta = _timeframe_delta(timeframe)
    gaps = int((df["timestamp"].diff().dropna() > (bar_delta * 1.5)).sum())

    now = datetime.now(timezone.utc)
    report_dir = (
        settings.quant_data_root / "logs" / "agents" / "ops-report-agent" / f"{now:%Y-%m-%d}"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"daily_report_{now:%Y%m%dT%H%M%SZ}.md"

    content = "\n".join(
        [
            f"# Phase 1 Daily Quant Ops Report ({now:%Y-%m-%d %H:%M UTC})",
            "",
            "## Scope",
            f"- Exchange: `{exchange}`",
            f"- Symbol: `{symbol}`",
            f"- Timeframe: `{timeframe}`",
            f"- Strategy: `{strategy_name}`",
            "",
            "## Data ingestion status",
            f"- Source file: `{raw_file}`",
            f"- Bars: `{len(df)}`",
            f"- Range: `{df['timestamp'].min()}` → `{df['timestamp'].max()}`",
            f"- Gap count (>1.5x expected interval): `{gaps}`",
            "",
            "## Backtest snapshot",
            f"- Run directory: `{backtest_dir}`",
            f"- Total return: `{_format_pct(float(metrics['total_return']))}`",
            f"- Annualized return: `{_format_pct(float(metrics['annualized_return']))}`",
            f"- Sharpe (simple): `{float(metrics['sharpe']):.3f}`",
            f"- Max drawdown: `{_format_pct(float(metrics['max_drawdown']))}`",
            f"- Buy-and-hold return: `{_format_pct(float(metrics['buy_and_hold_return']))}`",
            f"- Signal flips: `{int(metrics['signal_flips'])}`",
            "",
            "## Artifact pointers",
            f"- Metrics JSON: `{metrics_path}`",
            f"- Equity curve parquet: `{backtest_dir / 'equity_curve.parquet'}`",
            "",
            "## Next actions",
            "- Review gap count and backtest behavior for anomalies.",
            "- If metrics are stable, proceed to paper-trading simulation loop.",
        ]
    )
    report_path.write_text(content, encoding="utf-8")
    return ReportResult(report_path=report_path, strategy_run_dir=backtest_dir)

