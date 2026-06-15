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
    regime_detector_mode: str
    regime_policy_mode: str
    regime_policy_min_actionable_confidence: float
    regime_policy_transition_confidence: float
    regime_touchpoint_prompting_enabled: bool
    regime_touchpoint_calibration_enabled: bool
    regime_touchpoint_self_critique_enabled: bool
    regime_touchpoint_risk_gate_enabled: bool
    regime_volatility_threshold: float
    regime_trend_spread_threshold: float
    regime_persistence_bars: int
    regime_ablation_mode: bool
    risk_min_total_return: float
    risk_min_sharpe: float
    risk_max_drawdown: float
    risk_max_cost_return_drag: float
    risk_max_cost_pressure_score: float
    risk_min_signal_confidence: float
    risk_min_walkforward_quality_score: float
    risk_min_regime_confidence: float
    backtest_fee_bps: float
    backtest_slippage_bps: float
    walk_forward_fee_bps: float
    walk_forward_slippage_bps: float
    walk_forward_train_bars: int
    walk_forward_validate_bars: int
    walk_forward_step_bars: int
    walk_forward_min_windows: int
    calibration_min_walkforward_sharpe: float
    calibration_confidence_floor: float
    calibration_confidence_ceiling: float
    calibration_max_contradictions: int
    calibration_directional_edge_threshold: float
    calibration_quality_penalty_strength: float
    calibration_directional_contradiction_penalty: float
    calibration_cost_pressure_penalty_strength: float
    self_critique_min_score: float
    self_critique_max_findings: int
    ops_report_verbosity: str
    ensemble_mode: str
    ensemble_enabled_arms: tuple[str, ...]
    ensemble_decay_horizon: int
    ensemble_exploration_weight: float
    ensemble_turnover_penalty_bps: float
    paper_trade_notional_usd: float
    paper_trade_starting_cash_usd: float
    paper_trade_fee_bps: float
    paper_trade_slippage_bps: float
    paper_account_provider: str
    paper_account_exchange: str
    paper_account_sandbox: bool
    paper_account_timeout_seconds: float
    paper_account_api_key: str | None
    paper_account_api_secret: str | None
    paper_account_api_passphrase: str | None
    tradingview_base_url: str
    trigger_model_horizon_bars: int
    trigger_model_buy_threshold: float
    trigger_model_sell_threshold: float
    trigger_model_min_train_samples: int
    trigger_model_cost_bps: float
    trigger_model_optimize_thresholds: bool
    trigger_monitor_poll_seconds: float
    trigger_monitor_signal_confidence: float
    trigger_monitor_webhook_url: str | None
    trigger_monitor_notify_on_hold: bool


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

def _as_report_verbosity(value: str | None, default: str = "standard") -> str:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"compact", "standard", "verbose"}:
        return normalized
    return default

def _as_ensemble_mode(value: str | None, default: str = "adaptive") -> str:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"single", "adaptive"}:
        return normalized
    return default


def _as_regime_mode(value: str | None, default: str = "score") -> str:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"heuristic", "score"}:
        return normalized
    return default

def _as_regime_policy_mode(value: str | None, default: str = "legacy") -> str:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"legacy", "conditional_v2"}:
        return normalized
    return default


def _as_csv_tuple(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    items = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    return items or default


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
        ollama_timeout_seconds=max(1.0, _as_float(os.getenv("OLLAMA_TIMEOUT_SECONDS"), default=600.0)),
        ollama_strategy_model=os.getenv("OLLAMA_STRATEGY_MODEL", "llama3.1:8b"),
        ollama_ops_model=os.getenv("OLLAMA_OPS_MODEL", "llama3.1:8b"),
        agent_step_retries=max(0, _as_int(os.getenv("AGENT_STEP_RETRIES"), default=2)),
        agent_minimum_bars=max(10, _as_int(os.getenv("AGENT_MINIMUM_BARS"), default=120)),
        regime_detector_mode=_as_regime_mode(
            os.getenv("REGIME_DETECTOR_MODE"),
            default="score",
        ),
        regime_policy_mode=_as_regime_policy_mode(
            os.getenv("REGIME_POLICY_MODE"),
            default="legacy",
        ),
        regime_policy_min_actionable_confidence=min(
            1.0,
            max(
                0.0,
                _as_float(
                    os.getenv("REGIME_POLICY_MIN_ACTIONABLE_CONFIDENCE"),
                    default=0.50,
                ),
            ),
        ),
        regime_policy_transition_confidence=min(
            1.0,
            max(
                0.0,
                _as_float(
                    os.getenv("REGIME_POLICY_TRANSITION_CONFIDENCE"),
                    default=0.65,
                ),
            ),
        ),
        regime_touchpoint_prompting_enabled=_as_bool(
            os.getenv("REGIME_TOUCHPOINT_PROMPTING_ENABLED"),
            default=True,
        ),
        regime_touchpoint_calibration_enabled=_as_bool(
            os.getenv("REGIME_TOUCHPOINT_CALIBRATION_ENABLED"),
            default=True,
        ),
        regime_touchpoint_self_critique_enabled=_as_bool(
            os.getenv("REGIME_TOUCHPOINT_SELF_CRITIQUE_ENABLED"),
            default=True,
        ),
        regime_touchpoint_risk_gate_enabled=_as_bool(
            os.getenv("REGIME_TOUCHPOINT_RISK_GATE_ENABLED"),
            default=True,
        ),
        regime_volatility_threshold=max(
            0.0001,
            _as_float(os.getenv("REGIME_VOLATILITY_THRESHOLD"), default=0.03),
        ),
        regime_trend_spread_threshold=max(
            0.0001,
            _as_float(os.getenv("REGIME_TREND_SPREAD_THRESHOLD"), default=0.01),
        ),
        regime_persistence_bars=max(
            1,
            _as_int(os.getenv("REGIME_PERSISTENCE_BARS"), default=3),
        ),
        regime_ablation_mode=_as_bool(
            os.getenv("REGIME_ABLATION_MODE"),
            default=False,
        ),
        risk_min_total_return=_as_float(os.getenv("RISK_MIN_TOTAL_RETURN"), default=0.0),
        risk_min_sharpe=_as_float(os.getenv("RISK_MIN_SHARPE"), default=0.0),
        risk_max_drawdown=_as_float(os.getenv("RISK_MAX_DRAWDOWN"), default=-0.20),
        risk_max_cost_return_drag=max(
            0.0,
            _as_float(os.getenv("RISK_MAX_COST_RETURN_DRAG"), default=0.05),
        ),
        risk_max_cost_pressure_score=max(
            0.0,
            _as_float(os.getenv("RISK_MAX_COST_PRESSURE_SCORE"), default=0.95),
        ),
        risk_min_signal_confidence=_as_float(os.getenv("RISK_MIN_SIGNAL_CONFIDENCE"), default=0.55),
        risk_min_walkforward_quality_score=min(
            1.0,
            max(
                0.0,
                _as_float(os.getenv("RISK_MIN_WALKFORWARD_QUALITY_SCORE"), default=0.43),
            ),
        ),
        risk_min_regime_confidence=min(
            1.0,
            max(
                0.0,
                _as_float(os.getenv("RISK_MIN_REGIME_CONFIDENCE"), default=0.45),
            ),
        ),
        backtest_fee_bps=max(
            0.0,
            _as_float(os.getenv("BACKTEST_FEE_BPS"), default=5.0),
        ),
        backtest_slippage_bps=max(
            0.0,
            _as_float(os.getenv("BACKTEST_SLIPPAGE_BPS"), default=2.5),
        ),
        walk_forward_fee_bps=max(
            0.0,
            _as_float(os.getenv("WALK_FORWARD_FEE_BPS"), default=5.0),
        ),
        walk_forward_slippage_bps=max(
            0.0,
            _as_float(os.getenv("WALK_FORWARD_SLIPPAGE_BPS"), default=2.5),
        ),
        walk_forward_train_bars=max(
            50,
            _as_int(os.getenv("WALK_FORWARD_TRAIN_BARS"), default=240),
        ),
        walk_forward_validate_bars=max(
            10,
            _as_int(os.getenv("WALK_FORWARD_VALIDATE_BARS"), default=72),
        ),
        walk_forward_step_bars=max(
            10,
            _as_int(os.getenv("WALK_FORWARD_STEP_BARS"), default=72),
        ),
        walk_forward_min_windows=max(
            1,
            _as_int(os.getenv("WALK_FORWARD_MIN_WINDOWS"), default=3),
        ),
        calibration_min_walkforward_sharpe=_as_float(
            os.getenv("CALIBRATION_MIN_WALKFORWARD_SHARPE"),
            default=0.10,
        ),
        calibration_confidence_floor=min(
            1.0,
            max(
                0.0,
                _as_float(os.getenv("CALIBRATION_CONFIDENCE_FLOOR"), default=0.05),
            ),
        ),
        calibration_confidence_ceiling=min(
            1.0,
            max(
                0.0,
                _as_float(os.getenv("CALIBRATION_CONFIDENCE_CEILING"), default=0.95),
            ),
        ),
        calibration_max_contradictions=max(
            0,
            _as_int(os.getenv("CALIBRATION_MAX_CONTRADICTIONS"), default=0),
        ),
        calibration_directional_edge_threshold=_as_float(
            os.getenv("CALIBRATION_DIRECTIONAL_EDGE_THRESHOLD"),
            default=0.0,
        ),
        calibration_quality_penalty_strength=min(
            1.0,
            max(
                0.0,
                _as_float(
                    os.getenv("CALIBRATION_QUALITY_PENALTY_STRENGTH"),
                    default=0.25,
                ),
            ),
        ),
        calibration_directional_contradiction_penalty=min(
            1.0,
            max(
                0.0,
                _as_float(
                    os.getenv("CALIBRATION_DIRECTIONAL_CONTRADICTION_PENALTY"),
                    default=0.35,
                ),
            ),
        ),
        calibration_cost_pressure_penalty_strength=min(
            1.0,
            max(
                0.0,
                _as_float(
                    os.getenv("CALIBRATION_COST_PRESSURE_PENALTY_STRENGTH"),
                    default=0.30,
                ),
            ),
        ),
        self_critique_min_score=min(
            1.0,
            max(
                0.0,
                _as_float(os.getenv("SELF_CRITIQUE_MIN_SCORE"), default=0.55),
            ),
        ),
        self_critique_max_findings=max(
            1,
            _as_int(os.getenv("SELF_CRITIQUE_MAX_FINDINGS"), default=6),
        ),
        ops_report_verbosity=_as_report_verbosity(
            os.getenv("OPS_REPORT_VERBOSITY"),
            default="standard",
        ),
        ensemble_mode=_as_ensemble_mode(
            os.getenv("AGENT_ENSEMBLE_MODE"),
            default="adaptive",
        ),
        ensemble_enabled_arms=_as_csv_tuple(
            os.getenv("AGENT_ENSEMBLE_ARMS"),
            default=("sma_baseline", "technical_composite", "llm_context"),
        ),
        ensemble_decay_horizon=max(
            4,
            _as_int(os.getenv("AGENT_ENSEMBLE_DECAY_HORIZON"), default=96),
        ),
        ensemble_exploration_weight=min(
            0.50,
            max(
                0.0,
                _as_float(os.getenv("AGENT_ENSEMBLE_EXPLORATION_WEIGHT"), default=0.08),
            ),
        ),
        ensemble_turnover_penalty_bps=max(
            0.0,
            _as_float(os.getenv("AGENT_ENSEMBLE_TURNOVER_PENALTY_BPS"), default=8.0),
        ),
        paper_trade_notional_usd=_as_float(os.getenv("PAPER_TRADE_NOTIONAL_USD"), default=100.0),
        paper_trade_starting_cash_usd=max(
            0.0,
            _as_float(os.getenv("PAPER_TRADE_STARTING_CASH_USD"), default=10000.0),
        ),
        paper_trade_fee_bps=max(
            0.0,
            _as_float(os.getenv("PAPER_TRADE_FEE_BPS"), default=5.0),
        ),
        paper_trade_slippage_bps=max(
            0.0,
            _as_float(os.getenv("PAPER_TRADE_SLIPPAGE_BPS"), default=1.0),
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
        trigger_model_horizon_bars=max(
            1,
            _as_int(os.getenv("TRIGGER_MODEL_HORIZON_BARS"), default=6),
        ),
        trigger_model_buy_threshold=_as_float(
            os.getenv("TRIGGER_MODEL_BUY_THRESHOLD"),
            default=0.012,
        ),
        trigger_model_sell_threshold=abs(
            _as_float(os.getenv("TRIGGER_MODEL_SELL_THRESHOLD"), default=0.003)
        ),
        trigger_model_min_train_samples=max(
            20,
            _as_int(os.getenv("TRIGGER_MODEL_MIN_TRAIN_SAMPLES"), default=120),
        ),
        trigger_model_cost_bps=max(
            0.0,
            _as_float(os.getenv("TRIGGER_MODEL_COST_BPS"), default=7.5),
        ),
        trigger_model_optimize_thresholds=_as_bool(
            os.getenv("TRIGGER_MODEL_OPTIMIZE_THRESHOLDS"),
            default=True,
        ),
        trigger_monitor_poll_seconds=max(
            5.0,
            _as_float(os.getenv("TRIGGER_MONITOR_POLL_SECONDS"), default=300.0),
        ),
        trigger_monitor_signal_confidence=min(
            1.0,
            max(
                0.0,
                _as_float(
                    os.getenv("TRIGGER_MONITOR_SIGNAL_CONFIDENCE"),
                    default=0.60,
                ),
            ),
        ),
        trigger_monitor_webhook_url=os.getenv("TRIGGER_MONITOR_WEBHOOK_URL") or None,
        trigger_monitor_notify_on_hold=_as_bool(
            os.getenv("TRIGGER_MONITOR_NOTIFY_ON_HOLD"),
            default=False,
        ),
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

