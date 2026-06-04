from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

REQUIRED_EXCHANGE_SECRET_ENV_VARS: Final[tuple[str, ...]] = (
    "EXCHANGE_API_KEY",
    "EXCHANGE_API_SECRET",
)
OPTIONAL_EXCHANGE_SECRET_ENV_VARS: Final[tuple[str, ...]] = ("EXCHANGE_API_PASSPHRASE",)


@dataclass(frozen=True)
class Settings:
    quant_data_root: Path
    default_exchange: str
    default_symbol: str
    default_timeframe: str
    allow_unmounted_data_root: bool
    require_exchange_secrets: bool
    exchange_api_key: str | None
    exchange_api_secret: str | None
    exchange_api_passphrase: str | None


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        quant_data_root=Path(os.getenv("QUANT_DATA_ROOT", "/mnt/quant-data")).expanduser(),
        default_exchange=os.getenv("EXCHANGE_ID", "kraken"),
        default_symbol=os.getenv("SYMBOL", "BTC/USDT"),
        default_timeframe=os.getenv("TIMEFRAME", "1h"),
        allow_unmounted_data_root=_as_bool(os.getenv("ALLOW_UNMOUNTED_DATA_ROOT"), default=False),
        require_exchange_secrets=_as_bool(os.getenv("REQUIRE_EXCHANGE_SECRETS"), default=False),
        exchange_api_key=os.getenv("EXCHANGE_API_KEY") or None,
        exchange_api_secret=os.getenv("EXCHANGE_API_SECRET") or None,
        exchange_api_passphrase=os.getenv("EXCHANGE_API_PASSPHRASE") or None,
    )


def missing_exchange_secrets(settings: Settings) -> list[str]:
    missing: list[str] = []
    if not settings.exchange_api_key:
        missing.append("EXCHANGE_API_KEY")
    if not settings.exchange_api_secret:
        missing.append("EXCHANGE_API_SECRET")
    return missing


def ensure_exchange_secrets_ready(settings: Settings) -> None:
    missing = missing_exchange_secrets(settings)
    if missing:
        raise RuntimeError(
            "Missing required exchange secrets: "
            + ", ".join(missing)
            + ". Populate them in your shell environment or .env file."
        )


def ensure_data_root_ready(path: Path, allow_unmounted: bool = False) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"QUANT_DATA_ROOT does not exist: {path}. "
            "Mount external storage and set QUANT_DATA_ROOT accordingly."
        )

    if not allow_unmounted and not path.is_mount():
        raise RuntimeError(
            f"QUANT_DATA_ROOT is not a mountpoint: {path}. "
            "This guard prevents accidental writes to internal storage. "
            "Set ALLOW_UNMOUNTED_DATA_ROOT=1 to bypass intentionally."
        )

