from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisualizationResult:
    run_dir: Path
    output_dir: Path
    price_signals_path: Path
    equity_drawdown_path: Path
    returns_diagnostics_path: Path
    buy_trigger_count: int
    sell_trigger_count: int


def _latest_agent_plane_run_dir(root: Path) -> Path:
    base = root / "logs" / "agents" / "openclaw-orchestrator"
    if not base.exists():
        raise FileNotFoundError(f"No agent-plane runs found: {base}")
    candidates = sorted([p for p in base.glob("*/*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No agent-plane run directories found under: {base}")
    return candidates[-1]


def _load_backtest_evaluation(run_dir: Path) -> dict[str, object]:
    path = run_dir / "backtest_evaluation.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing backtest evaluation artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected JSON payload in {path}")
    return payload


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Equity curve dataset is missing required columns: {missing}")


def generate_run_visuals(
    *,
    quant_data_root: Path,
    run_dir: Path | None = None,
    output_dir: Path | None = None,
) -> VisualizationResult:
    selected_run_dir = (
        _latest_agent_plane_run_dir(quant_data_root)
        if run_dir is None
        else run_dir.expanduser().resolve()
    )
    selected_run_dir = selected_run_dir.resolve()
    if not selected_run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {selected_run_dir}")

    backtest_eval = _load_backtest_evaluation(selected_run_dir)
    status = str(backtest_eval.get("backtest_status"))
    if status != "success":
        raise RuntimeError(
            f"Run {selected_run_dir.name} has no successful backtest (status={status})."
        )

    backtest_run_dir_raw = backtest_eval.get("backtest_run_dir")
    if not isinstance(backtest_run_dir_raw, str) or not backtest_run_dir_raw.strip():
        raise RuntimeError(f"Run {selected_run_dir.name} does not include backtest run directory.")
    backtest_run_dir = Path(backtest_run_dir_raw).expanduser().resolve()
    equity_curve_path = backtest_run_dir / "equity_curve.parquet"
    if not equity_curve_path.exists():
        raise FileNotFoundError(f"Backtest equity curve file missing: {equity_curve_path}")

    # Keep matplotlib import local so non-visual commands don't require it at import time.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.read_parquet(equity_curve_path).sort_values("timestamp").reset_index(drop=True)
    _require_columns(
        frame,
        ("timestamp", "close", "ma_fast", "ma_slow", "position", "strategy_returns", "equity_curve"),
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["position_change"] = frame["position"].diff().fillna(frame["position"])
    buy_points = frame[frame["position_change"] > 0]
    sell_points = frame[frame["position_change"] < 0]

    target_output_dir = (
        (selected_run_dir / "visuals")
        if output_dir is None
        else output_dir.expanduser().resolve()
    )
    target_output_dir.mkdir(parents=True, exist_ok=True)

    price_signals_path = target_output_dir / "price_sma_triggers.png"
    equity_drawdown_path = target_output_dir / "equity_vs_buyhold_drawdown.png"
    returns_diagnostics_path = target_output_dir / "returns_quality_diagnostics.png"

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(frame["timestamp"], frame["close"], label="Close", color="black", linewidth=1.4)
    ax.plot(frame["timestamp"], frame["ma_fast"], label="MA Fast", color="#1f77b4", linewidth=1.2)
    ax.plot(frame["timestamp"], frame["ma_slow"], label="MA Slow", color="#ff7f0e", linewidth=1.2)
    if not buy_points.empty:
        ax.scatter(
            buy_points["timestamp"],
            buy_points["close"],
            marker="^",
            s=90,
            color="green",
            label="Buy trigger",
            zorder=5,
        )
    if not sell_points.empty:
        ax.scatter(
            sell_points["timestamp"],
            sell_points["close"],
            marker="v",
            s=90,
            color="red",
            label="Sell trigger",
            zorder=5,
        )
    ax.set_title(f"Price, SMA Crossover Signals, and Triggers\nRun {selected_run_dir.name}")
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(price_signals_path, dpi=160)
    plt.close(fig)

    strategy_equity = frame["equity_curve"]
    buy_hold_equity = frame["close"] / frame["close"].iloc[0]
    strategy_drawdown = strategy_equity / strategy_equity.cummax() - 1.0
    buy_hold_drawdown = buy_hold_equity / buy_hold_equity.cummax() - 1.0

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True, height_ratios=[2.2, 1])
    axes[0].plot(
        frame["timestamp"],
        strategy_equity,
        label="Strategy equity",
        color="#2ca02c",
        linewidth=1.5,
    )
    axes[0].plot(
        frame["timestamp"],
        buy_hold_equity,
        label="Buy & hold equity",
        color="#9467bd",
        linewidth=1.3,
        alpha=0.9,
    )
    axes[0].set_ylabel("Normalized equity")
    axes[0].set_title("Strategy Equity vs Buy-and-Hold")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(
        frame["timestamp"],
        strategy_drawdown,
        label="Strategy drawdown",
        color="#d62728",
        linewidth=1.2,
    )
    axes[1].plot(
        frame["timestamp"],
        buy_hold_drawdown,
        label="Buy & hold drawdown",
        color="#8c564b",
        linewidth=1.1,
        alpha=0.9,
    )
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Timestamp (UTC)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    axes[1].axhline(0, color="black", linewidth=0.8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(equity_drawdown_path, dpi=160)
    plt.close(fig)

    returns = frame["strategy_returns"].fillna(0.0)
    rolling_window = min(48, max(12, len(returns) // 5))
    rolling_mean = returns.rolling(window=rolling_window).mean()
    rolling_std = returns.rolling(window=rolling_window).std(ddof=0)
    rolling_ratio = (rolling_mean / rolling_std).replace([float("inf"), float("-inf")], pd.NA)

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=False, height_ratios=[1.4, 1])
    axes[0].plot(
        frame["timestamp"],
        returns,
        color="#17becf",
        linewidth=1.0,
        label="Per-bar strategy return",
    )
    axes[0].plot(
        frame["timestamp"],
        rolling_mean,
        color="#1f77b4",
        linewidth=1.2,
        label=f"Rolling mean ({rolling_window})",
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Strategy Return Stream and Rolling Mean")
    axes[0].set_ylabel("Return")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(
        frame["timestamp"],
        rolling_ratio,
        color="#ff7f0e",
        linewidth=1.2,
        label=f"Rolling mean/std ({rolling_window})",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Rolling Return Quality Proxy (Mean / Std)")
    axes[1].set_xlabel("Timestamp (UTC)")
    axes[1].set_ylabel("Ratio")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(returns_diagnostics_path, dpi=160)
    plt.close(fig)

    logger.info(
        "Generated run visuals run_dir=%s output_dir=%s buy_triggers=%s sell_triggers=%s",
        selected_run_dir,
        target_output_dir,
        len(buy_points),
        len(sell_points),
    )

    return VisualizationResult(
        run_dir=selected_run_dir,
        output_dir=target_output_dir,
        price_signals_path=price_signals_path,
        equity_drawdown_path=equity_drawdown_path,
        returns_diagnostics_path=returns_diagnostics_path,
        buy_trigger_count=int(len(buy_points)),
        sell_trigger_count=int(len(sell_points)),
    )
