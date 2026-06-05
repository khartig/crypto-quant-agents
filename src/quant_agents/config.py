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
    ollama_base_url: str
    ollama_timeout_seconds: float
    ollama_strategy_model: str
    ollama_ops_model: str
    agent_step_retries: int
    agent_minimum_bars: int
    risk_min_total_return: float
    risk_min_sharpe: float
    risk_max_drawdown: float
    risk_min_signal_confidence: float
    paper_trade_notional_usd: float
    paper_trade_starting_cash_usd: float
    paper_trade_fee_bps: float
    paper_account_provider: str
    paper_account_exchange: str
    paper_account_sandbox: bool
    paper_account_timeout_seconds: float
    paper_account_api_key: str | None
    paper_account_api_secret: str | None
    paper_account_api_passphrase: str | None
    tradingview_base_url: str


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _as_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


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
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        ollama_timeout_seconds=max(1.0, _as_float(os.getenv("OLLAMA_TIMEOUT_SECONDS"), default=180.0)),
        ollama_strategy_model=os.getenv("OLLAMA_STRATEGY_MODEL", "llama3.1:8b"),
        ollama_ops_model=os.getenv("OLLAMA_OPS_MODEL", "llama3.1:8b"),
        agent_step_retries=max(0, _as_int(os.getenv("AGENT_STEP_RETRIES"), default=2)),
        agent_minimum_bars=max(10, _as_int(os.getenv("AGENT_MINIMUM_BARS"), default=120)),
        risk_min_total_return=_as_float(os.getenv("RISK_MIN_TOTAL_RETURN"), default=0.0),
        risk_min_sharpe=_as_float(os.getenv("RISK_MIN_SHARPE"), default=0.0),
        risk_max_drawdown=_as_float(os.getenv("RISK_MAX_DRAWDOWN"), default=-0.20),
        risk_min_signal_confidence=_as_float(os.getenv("RISK_MIN_SIGNAL_CONFIDENCE"), default=0.55),
        paper_trade_notional_usd=_as_float(os.getenv("PAPER_TRADE_NOTIONAL_USD"), default=100.0),
        paper_trade_starting_cash_usd=max(
            0.0,
            _as_float(os.getenv("PAPER_TRADE_STARTING_CASH_USD"), default=10000.0),
        ),
        paper_trade_fee_bps=max(
            0.0,
            _as_float(os.getenv("PAPER_TRADE_FEE_BPS"), default=5.0),
        ),
        paper_account_provider=os.getenv("PAPER_ACCOUNT_PROVIDER", "tradingview"),
        paper_account_exchange=os.getenv("PAPER_ACCOUNT_EXCHANGE", "kraken"),
        paper_account_sandbox=_as_bool(os.getenv("PAPER_ACCOUNT_SANDBOX"), default=True),
        paper_account_timeout_seconds=max(
            1.0,
            _as_float(os.getenv("PAPER_ACCOUNT_TIMEOUT_SECONDS"), default=15.0),
        ),
        paper_account_api_key=os.getenv("PAPER_ACCOUNT_API_KEY") or None,
        paper_account_api_secret=os.getenv("PAPER_ACCOUNT_API_SECRET") or None,
        paper_account_api_passphrase=os.getenv("PAPER_ACCOUNT_API_PASSPHRASE") or None,
        tradingview_base_url=os.getenv("TRADINGVIEW_BASE_URL", "https://www.tradingview.com"),
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

