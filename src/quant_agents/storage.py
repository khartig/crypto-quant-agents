from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


PHASE1_TREE = (
    "raw",
    "curated/features",
    "curated/training",
    "backtests",
    "paper-trading",
    "paper-trading/state",
    "models/ollama-cache",
    "models/trigger-models",
    "logs/agents",
    "logs/agents/model-predictor",
    "logs/agents/trigger-monitor",
    "archive/monthly",
)


def ensure_phase1_tree(root: Path) -> None:
    for rel_path in PHASE1_TREE:
        (root / rel_path).mkdir(parents=True, exist_ok=True)


def symbol_slug(symbol: str) -> str:
    return symbol.replace("/", "-").replace(":", "-")


def raw_dataset_dir(root: Path, exchange: str, symbol: str, timeframe: str, ts: datetime) -> Path:
    return (
        root
        / "raw"
        / f"exchange={exchange}"
        / f"symbol={symbol_slug(symbol)}"
        / f"interval={timeframe}"
        / f"year={ts:%Y}"
        / f"month={ts:%m}"
    )


def latest_raw_dataset(root: Path, exchange: str, symbol: str, timeframe: str) -> Path:
    base = (
        root
        / "raw"
        / f"exchange={exchange}"
        / f"symbol={symbol_slug(symbol)}"
        / f"interval={timeframe}"
    )
    if not base.exists():
        raise FileNotFoundError(f"No raw dataset directory found: {base}")

    candidates = sorted(base.rglob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No raw parquet files found under: {base}")

    return candidates[-1]


def new_backtest_run_dir(root: Path, strategy_name: str) -> Path:
    base_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_dir = root / "backtests" / strategy_name
    base_dir.mkdir(parents=True, exist_ok=True)
    for suffix in range(100):
        run_id = base_run_id if suffix == 0 else f"{base_run_id}_{suffix:02d}"
        path = base_dir / run_id
        if path.exists():
            continue
        path.mkdir(parents=True, exist_ok=False)
        return path
    raise RuntimeError(f"Unable to allocate unique backtest run directory under {base_dir}")


def latest_backtest_run_dir(root: Path, strategy_name: str) -> Path:
    base = root / "backtests" / strategy_name
    if not base.exists():
        raise FileNotFoundError(f"No backtest directory found: {base}")

    candidates = sorted([p for p in base.iterdir() if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No backtest runs found under: {base}")

    return candidates[-1]

