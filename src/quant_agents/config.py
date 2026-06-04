from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    quant_data_root: Path
    default_exchange: str
    default_symbol: str
    default_timeframe: str


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        quant_data_root=Path(os.getenv("QUANT_DATA_ROOT", "/mnt/quant-data")).expanduser(),
        default_exchange=os.getenv("EXCHANGE_ID", "kraken"),
        default_symbol=os.getenv("SYMBOL", "BTC/USDT"),
        default_timeframe=os.getenv("TIMEFRAME", "1h"),
    )


def ensure_data_root_ready(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"QUANT_DATA_ROOT does not exist: {path}. "
            "Mount external storage and set QUANT_DATA_ROOT accordingly."
        )

    allow_unmounted = os.getenv("ALLOW_UNMOUNTED_DATA_ROOT", "0") == "1"
    if not allow_unmounted and not path.is_mount():
        raise RuntimeError(
            f"QUANT_DATA_ROOT is not a mountpoint: {path}. "
            "This guard prevents accidental writes to internal storage. "
            "Set ALLOW_UNMOUNTED_DATA_ROOT=1 to bypass intentionally."
        )

