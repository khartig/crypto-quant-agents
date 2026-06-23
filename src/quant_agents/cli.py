from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant_agents.agent_plane import AgentPlaneConfig, RiskThresholds, run_agent_plane
from quant_agents.backtest import STRATEGY_NAME, archive_backtest_run, run_sma_backtest
from quant_agents.config import (
    ensure_data_root_ready,
    ensure_exchange_secrets_ready,
    load_settings,
)
from quant_agents.doctor import format_doctor_report, run_doctor
from quant_agents.ingestion import fetch_ohlcv_to_parquet
from quant_agents.logging_utils import configure_logging
from quant_agents.metrics import tracked_operation
from quant_agents.orderbook_features import (
    DEFAULT_STABLE_ORDERBOOK_FEATURE_COLUMNS,
    ORDERBOOK_FEATURE_COLUMNS,
    normalize_orderbook_feature_columns,
)
from quant_agents.orderbook_ingestion import capture_orderbook_snapshots_to_parquet
from quant_agents.orderbook_retrieval import retrieve_orderbook_features
from quant_agents.paper_account import run_paper_account_probe
from quant_agents.priority2_features import (
    DEFAULT_STABLE_PRIORITY2_FEATURE_COLUMNS,
    PRIORITY2_FEATURE_COLUMNS,
    normalize_priority2_feature_columns,
)
from quant_agents.ranked_features import (
    DEFAULT_STABLE_RANKED_FEATURE_COLUMNS,
    RANKED_FEATURE_COLUMNS,
    normalize_ranked_feature_columns,
)
from quant_agents.priority2_retrieval import retrieve_priority2_external_features
from quant_agents.reporting import generate_daily_report
from quant_agents.storage import ensure_phase1_tree, latest_backtest_run_dir
from quant_agents.trigger_model import (
    monitor_trigger_signals,
    predict_trigger_signal,
    train_trigger_model,
)
from quant_agents.visualization import generate_run_visuals

logger = logging.getLogger(__name__)


def _parse_ensemble_arms(value: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, tuple):
        items = value
    elif isinstance(value, list):
        items = tuple(str(item) for item in value)
    else:
        items = tuple(part.strip() for part in str(value).split(","))
    normalized = tuple(item.strip().lower() for item in items if item and item.strip())
    return normalized

def _parse_priority2_feature_columns(
    value: str | tuple[str, ...] | list[str] | None,
    *,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    fallback = normalize_priority2_feature_columns(default)
    if value is None:
        return fallback
    if isinstance(value, tuple):
        items = value
    elif isinstance(value, list):
        items = tuple(str(item).strip() for item in value)
    else:
        raw = str(value).strip()
        if not raw:
            return fallback
        lowered = raw.lower()
        if lowered in {"all", "*"}:
            return tuple(PRIORITY2_FEATURE_COLUMNS)
        if lowered in {"stable", "default"}:
            return tuple(DEFAULT_STABLE_PRIORITY2_FEATURE_COLUMNS)
        items = tuple(part.strip() for part in raw.split(","))
    normalized = normalize_priority2_feature_columns(items)
    return normalized or fallback


def _parse_ranked_feature_columns(
    value: str | tuple[str, ...] | list[str] | None,
    *,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    fallback = normalize_ranked_feature_columns(default)
    if value is None:
        return fallback
    if isinstance(value, tuple):
        items = value
    elif isinstance(value, list):
        items = tuple(str(item).strip() for item in value)
    else:
        raw = str(value).strip()
        if not raw:
            return fallback
        lowered = raw.lower()
        if lowered in {"all", "*"}:
            return tuple(RANKED_FEATURE_COLUMNS)
        if lowered in {"stable", "default"}:
            return tuple(DEFAULT_STABLE_RANKED_FEATURE_COLUMNS)
        items = tuple(part.strip() for part in raw.split(","))
    normalized = normalize_ranked_feature_columns(items)
    return normalized or fallback


def _parse_orderbook_feature_columns(
    value: str | tuple[str, ...] | list[str] | None,
    *,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    fallback = normalize_orderbook_feature_columns(default)
    if value is None:
        return fallback
    if isinstance(value, tuple):
        items = value
    elif isinstance(value, list):
        items = tuple(str(item).strip() for item in value)
    else:
        raw = str(value).strip()
        if not raw:
            return fallback
        lowered = raw.lower()
        if lowered in {"all", "*"}:
            return tuple(ORDERBOOK_FEATURE_COLUMNS)
        if lowered in {"stable", "default"}:
            return tuple(DEFAULT_STABLE_ORDERBOOK_FEATURE_COLUMNS)
        items = tuple(part.strip() for part in raw.split(","))
    normalized = normalize_orderbook_feature_columns(items)
    return normalized or fallback


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-agents",
        description="Deterministic crypto quant pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Fetch and store OHLCV data.")
    ingest.add_argument("--exchange", default=None)
    ingest.add_argument("--symbol", default=None)
    ingest.add_argument("--timeframe", default=None)
    ingest.add_argument("--limit", type=int, default=1000)
    orderbook_capture = subparsers.add_parser(
        "capture-orderbook",
        help="Capture live order book snapshots and persist them to raw storage.",
    )
    orderbook_capture.add_argument("--exchange", default=None)
    orderbook_capture.add_argument("--symbol", default=None)
    orderbook_capture.add_argument(
        "--sample-count",
        type=int,
        default=120,
        help="Number of order book snapshots to collect in this capture run.",
    )
    orderbook_capture.add_argument(
        "--sample-interval-seconds",
        type=float,
        default=None,
        help="Polling interval between snapshots in seconds (default from settings).",
    )
    orderbook_capture.add_argument(
        "--depth-limit",
        type=int,
        default=None,
        help="Order book depth levels requested per snapshot (default from settings).",
    )
    orderbook_retrieve = subparsers.add_parser(
        "retrieve-orderbook-features",
        help="Build aligned order book feature artifacts from captured snapshots.",
    )
    orderbook_retrieve.add_argument("--exchange", default=None)
    orderbook_retrieve.add_argument("--symbol", default=None)
    orderbook_retrieve.add_argument("--timeframe", default=None)
    orderbook_retrieve.add_argument(
        "--input-file",
        default=None,
        help="Optional explicit market parquet input file for retrieval window alignment.",
    )
    orderbook_retrieve.add_argument(
        "--snapshot-source-path",
        default=None,
        help="Optional explicit order book snapshot parquet path.",
    )
    orderbook_retrieve.add_argument(
        "--orderbook-feature-columns",
        default=None,
        help=(
            "Comma-separated order book feature columns to enable "
            "(for example orderbook_spread_feature,orderbook_depth_imbalance_feature). "
            "Use 'stable' for the default stable set or 'all' for full order book set."
        ),
    )
    priority2_retrieve = subparsers.add_parser(
        "retrieve-priority2-features",
        help="Build canonical external Priority 2 feature artifacts from derivatives/whale data sources.",
    )
    priority2_retrieve.add_argument("--exchange", default=None)
    priority2_retrieve.add_argument("--symbol", default=None)
    priority2_retrieve.add_argument("--timeframe", default=None)
    priority2_retrieve.add_argument(
        "--input-file",
        default=None,
        help="Optional explicit market parquet input file for retrieval window alignment.",
    )
    priority2_retrieve.add_argument(
        "--provider",
        choices=["binance_futures_public", "okx_public"],
        default=None,
        help="External retrieval provider (default from settings).",
    )
    priority2_retrieve.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="HTTP timeout for provider requests (default from settings).",
    )
    priority2_retrieve.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Per-request page size for provider endpoints (default from settings).",
    )
    priority2_retrieve.add_argument(
        "--base-url",
        default=None,
        help="Provider base URL override (default from settings).",
    )
    priority2_retrieve.add_argument(
        "--local-feature-overrides-path",
        default=None,
        help="Optional local CSV/JSON/Parquet feature file used to override/augment provider output.",
    )

    backtest = subparsers.add_parser("backtest", help="Run baseline SMA crossover backtest.")
    backtest.add_argument("--exchange", default=None)
    backtest.add_argument("--symbol", default=None)
    backtest.add_argument("--timeframe", default=None)
    backtest.add_argument("--fast-window", type=int, default=24)
    backtest.add_argument("--slow-window", type=int, default=60)
    backtest.add_argument(
        "--fee-bps",
        type=float,
        default=None,
        help="One-way fee in bps used for cost-adjusted backtest metrics (default from settings).",
    )
    backtest.add_argument(
        "--slippage-bps",
        type=float,
        default=None,
        help="One-way slippage in bps used for cost-adjusted backtest metrics (default from settings).",
    )
    backtest.add_argument(
        "--input-file",
        default=None,
        help="Optional explicit parquet input file for reproducible re-runs.",
    )
    backtest.add_argument(
        "--archive",
        action="store_true",
        help="Archive this run under archive/monthly after completion.",
    )

    report = subparsers.add_parser("report", help="Generate daily markdown operations report.")
    report.add_argument("--exchange", default=None)
    report.add_argument("--symbol", default=None)
    report.add_argument("--timeframe", default=None)
    archive = subparsers.add_parser(
        "archive-backtest",
        help="Archive a backtest run directory under archive/monthly.",
    )
    archive.add_argument(
        "--run-dir",
        default=None,
        help="Run directory to archive. Defaults to latest run for the given strategy.",
    )
    archive.add_argument(
        "--strategy",
        default=STRATEGY_NAME,
        help="Strategy name used to find latest run when --run-dir is omitted.",
    )

    daily = subparsers.add_parser(
        "run-daily",
        help="Run ingest + backtest + report in sequence.",
    )
    daily.add_argument("--exchange", default=None)
    daily.add_argument("--symbol", default=None)
    daily.add_argument("--timeframe", default=None)
    daily.add_argument("--limit", type=int, default=1000)
    daily.add_argument("--fast-window", type=int, default=24)
    daily.add_argument("--slow-window", type=int, default=60)
    daily.add_argument(
        "--fee-bps",
        type=float,
        default=None,
        help="One-way fee in bps used by daily backtest stage (default from settings).",
    )
    daily.add_argument(
        "--slippage-bps",
        type=float,
        default=None,
        help="One-way slippage in bps used by daily backtest stage (default from settings).",
    )
    daily.add_argument(
        "--archive-backtest",
        action="store_true",
        help="Archive generated backtest run after completion.",
    )
    daily.add_argument(
        "--require-secrets",
        action="store_true",
        help="Fail if exchange API secrets are missing.",
    )
    agent_plane = subparsers.add_parser(
        "agent-plane",
        help="Run OpenClaw-style agent-plane orchestration with deterministic risk gating.",
    )
    agent_plane.add_argument("--exchange", default=None)
    agent_plane.add_argument("--symbol", default=None)
    agent_plane.add_argument("--timeframe", default=None)
    agent_plane.add_argument(
        "--input-file",
        default=None,
        help="Optional explicit parquet input file used by data-quality, strategy, and backtest phases.",
    )
    agent_plane.add_argument(
        "--strategy-model",
        default=None,
        help="Ollama model for strategy proposal generation (default from settings).",
    )
    agent_plane.add_argument(
        "--ops-model",
        default=None,
        help="Ollama model for ops report generation (default from settings).",
    )
    agent_plane.add_argument(
        "--step-retries",
        type=int,
        default=None,
        help="Retries per orchestration step before fallback.",
    )
    agent_plane.add_argument(
        "--minimum-bars",
        type=int,
        default=None,
        help="Minimum bars required for data-quality pass.",
    )
    agent_plane.add_argument(
        "--regime-enabled",
        dest="regime_enabled",
        action="store_true",
        help="Enable regime logic and regime touchpoint contributions.",
    )
    agent_plane.add_argument(
        "--no-regime-enabled",
        dest="regime_enabled",
        action="store_false",
        help="Disable regime logic globally (forces regime-off behavior across touchpoints).",
    )
    agent_plane.add_argument(
        "--regime-detector-mode",
        choices=["heuristic", "score"],
        default=None,
        help="Regime detector mode used in phase-1 feature context.",
    )
    agent_plane.add_argument(
        "--regime-volatility-threshold",
        type=float,
        default=None,
        help="Regime detector volatility threshold (rolling std units).",
    )
    agent_plane.add_argument(
        "--regime-trend-spread-threshold",
        type=float,
        default=None,
        help="Regime detector absolute SMA trend-spread threshold.",
    )
    agent_plane.add_argument(
        "--regime-persistence-bars",
        type=int,
        default=None,
        help="Regime detector persistence window for transition smoothing.",
    )
    agent_plane.add_argument(
        "--regime-ablation-mode",
        dest="regime_ablation_mode",
        action="store_true",
        help="Disable all regime contributions (prompting, calibration terms, self-critique regime checks, and risk regime gate).",
    )
    agent_plane.add_argument(
        "--no-regime-ablation-mode",
        dest="regime_ablation_mode",
        action="store_false",
        help="Enable normal regime contributions (default behavior unless env enables ablation).",
    )
    agent_plane.set_defaults(regime_ablation_mode=None)
    agent_plane.add_argument(
        "--priority2-features-enabled",
        dest="priority2_features_enabled",
        action="store_true",
        help="Enable Priority 2 feature expansion in phase-1 context.",
    )
    agent_plane.add_argument(
        "--no-priority2-features-enabled",
        dest="priority2_features_enabled",
        action="store_false",
        help="Disable Priority 2 feature expansion in phase-1 context.",
    )
    agent_plane.add_argument(
        "--priority2-external-features-path",
        default=None,
        help="Optional CSV/JSON/Parquet path with externally computed Priority 2 feature columns.",
    )
    agent_plane.add_argument(
        "--priority2-feature-columns",
        default=None,
        help=(
            "Comma-separated Priority 2 feature columns to enable "
            "(for example open_interest_feature,participant_positioning_feature). "
            "Use 'stable' for the default stable pair or 'all' for full Priority 2 set."
        ),
    )
    agent_plane.add_argument(
        "--priority2-quality-gate-enabled",
        dest="priority2_quality_gate_enabled",
        action="store_true",
        help="Enable Priority 2 external-data quality gate in phase-1 context (default from settings).",
    )
    agent_plane.add_argument(
        "--no-priority2-quality-gate-enabled",
        dest="priority2_quality_gate_enabled",
        action="store_false",
        help="Disable Priority 2 external-data quality gate in phase-1 context.",
    )
    agent_plane.add_argument(
        "--priority2-quality-min-external-raw-coverage",
        type=float,
        default=None,
        help="Priority 2 quality gate: minimum required external raw coverage ratio.",
    )
    agent_plane.add_argument(
        "--priority2-quality-min-non-zero-coverage",
        type=float,
        default=None,
        help="Priority 2 quality gate: minimum required non-zero feature coverage ratio.",
    )
    agent_plane.add_argument(
        "--priority2-quality-max-fallback-rate",
        type=float,
        default=None,
        help="Priority 2 quality gate: maximum allowed fallback/imputation ratio.",
    )
    agent_plane.add_argument(
        "--priority2-quality-max-staleness-seconds",
        type=float,
        default=None,
        help="Priority 2 quality gate: maximum allowed external feature staleness in seconds.",
    )
    agent_plane.add_argument(
        "--regime-policy-mode",
        choices=["legacy", "conditional_v2"],
        default=None,
        help="Regime policy behavior at recommendation time (legacy additive behavior vs explicit conditional policy v2).",
    )
    agent_plane.add_argument(
        "--regime-policy-min-actionable-confidence",
        type=float,
        default=None,
        help="Conditional regime policy: minimum regime confidence before actionable recommendations remain buy/sell.",
    )
    agent_plane.add_argument(
        "--regime-policy-transition-confidence",
        type=float,
        default=None,
        help="Conditional regime policy: minimum regime confidence required during transitions before allowing actionable recommendations.",
    )
    agent_plane.add_argument(
        "--regime-touchpoint-prompting-enabled",
        dest="regime_touchpoint_prompting_enabled",
        action="store_true",
        help="Enable regime influence at prompting/recommendation touchpoint.",
    )
    agent_plane.add_argument(
        "--no-regime-touchpoint-prompting-enabled",
        dest="regime_touchpoint_prompting_enabled",
        action="store_false",
        help="Disable regime influence at prompting/recommendation touchpoint.",
    )
    agent_plane.add_argument(
        "--regime-touchpoint-calibration-enabled",
        dest="regime_touchpoint_calibration_enabled",
        action="store_true",
        help="Enable regime influence in confidence calibration.",
    )
    agent_plane.add_argument(
        "--no-regime-touchpoint-calibration-enabled",
        dest="regime_touchpoint_calibration_enabled",
        action="store_false",
        help="Disable regime influence in confidence calibration.",
    )
    agent_plane.add_argument(
        "--regime-touchpoint-self-critique-enabled",
        dest="regime_touchpoint_self_critique_enabled",
        action="store_true",
        help="Enable regime contradiction checks in self-critique.",
    )
    agent_plane.add_argument(
        "--no-regime-touchpoint-self-critique-enabled",
        dest="regime_touchpoint_self_critique_enabled",
        action="store_false",
        help="Disable regime contradiction checks in self-critique.",
    )
    agent_plane.add_argument(
        "--regime-touchpoint-risk-gate-enabled",
        dest="regime_touchpoint_risk_gate_enabled",
        action="store_true",
        help="Enable regime confidence gate checks in deterministic risk stage.",
    )
    agent_plane.add_argument(
        "--no-regime-touchpoint-risk-gate-enabled",
        dest="regime_touchpoint_risk_gate_enabled",
        action="store_false",
        help="Disable regime confidence gate checks in deterministic risk stage.",
    )
    agent_plane.set_defaults(
        regime_enabled=None,
        priority2_features_enabled=None,
        priority2_quality_gate_enabled=None,
        regime_touchpoint_prompting_enabled=None,
        regime_touchpoint_calibration_enabled=None,
        regime_touchpoint_self_critique_enabled=None,
        regime_touchpoint_risk_gate_enabled=None,
    )
    agent_plane.add_argument(
        "--fast-window",
        type=int,
        default=None,
        help="Strategy/backtest fast SMA window override (defaults to 24).",
    )
    agent_plane.add_argument(
        "--slow-window",
        type=int,
        default=None,
        help="Strategy/backtest slow SMA window override (defaults to 60).",
    )
    agent_plane.add_argument(
        "--min-total-return",
        type=float,
        default=None,
        help="Deterministic risk gate: minimum backtest total return.",
    )
    agent_plane.add_argument(
        "--min-sharpe",
        type=float,
        default=None,
        help="Deterministic risk gate: minimum backtest sharpe.",
    )
    agent_plane.add_argument(
        "--max-drawdown",
        type=float,
        default=None,
        help="Deterministic risk gate: minimum allowed max drawdown (negative value).",
    )
    agent_plane.add_argument(
        "--max-cost-return-drag",
        type=float,
        default=None,
        help="Deterministic risk gate: maximum allowed cost drag in return units.",
    )
    agent_plane.add_argument(
        "--max-cost-pressure-score",
        type=float,
        default=None,
        help="Deterministic risk gate: maximum allowed calibration cost-pressure score.",
    )
    agent_plane.add_argument(
        "--min-signal-confidence",
        type=float,
        default=None,
        help="Deterministic risk gate: minimum strategy confidence.",
    )
    agent_plane.add_argument(
        "--min-walkforward-quality-score",
        type=float,
        default=None,
        help="Deterministic risk gate: minimum walk-forward quality score for actionable recommendations.",
    )
    agent_plane.add_argument(
        "--min-regime-confidence",
        type=float,
        default=None,
        help="Deterministic risk gate: minimum regime confidence for actionable recommendations.",
    )
    agent_plane.add_argument(
        "--backtest-fee-bps",
        type=float,
        default=None,
        help="One-way fee bps applied in agent-plane backtest evaluation.",
    )
    agent_plane.add_argument(
        "--backtest-slippage-bps",
        type=float,
        default=None,
        help="One-way slippage bps applied in agent-plane backtest evaluation.",
    )
    agent_plane.add_argument(
        "--walkforward-fee-bps",
        type=float,
        default=None,
        help="One-way fee bps applied in walk-forward evaluation.",
    )
    agent_plane.add_argument(
        "--walkforward-slippage-bps",
        type=float,
        default=None,
        help="One-way slippage bps applied in walk-forward evaluation.",
    )
    agent_plane.add_argument(
        "--walkforward-train-bars",
        type=int,
        default=None,
        help="Phase 2: train bars per walk-forward window.",
    )
    agent_plane.add_argument(
        "--walkforward-validate-bars",
        type=int,
        default=None,
        help="Phase 2: validate bars per walk-forward window.",
    )
    agent_plane.add_argument(
        "--walkforward-step-bars",
        type=int,
        default=None,
        help="Phase 2: step size in bars between walk-forward windows.",
    )
    agent_plane.add_argument(
        "--walkforward-min-windows",
        type=int,
        default=None,
        help="Phase 2: minimum number of walk-forward windows required.",
    )
    agent_plane.add_argument(
        "--calibration-min-walkforward-sharpe",
        type=float,
        default=None,
        help="Phase 2: minimum walk-forward sharpe before contradiction penalty.",
    )
    agent_plane.add_argument(
        "--calibration-confidence-floor",
        type=float,
        default=None,
        help="Phase 2: lower clamp for calibrated confidence.",
    )
    agent_plane.add_argument(
        "--calibration-confidence-ceiling",
        type=float,
        default=None,
        help="Phase 2: upper clamp for calibrated confidence.",
    )
    agent_plane.add_argument(
        "--calibration-max-contradictions",
        type=int,
        default=None,
        help="Phase 2: maximum allowed contradiction events before risk block.",
    )
    agent_plane.add_argument(
        "--calibration-directional-edge-threshold",
        type=float,
        default=None,
        help="Minimum walk-forward return edge magnitude required to avoid directional contradiction penalties.",
    )
    agent_plane.add_argument(
        "--calibration-quality-penalty-strength",
        type=float,
        default=None,
        help="Strength (0..1) of quality-band confidence penalty in calibration.",
    )
    agent_plane.add_argument(
        "--calibration-directional-contradiction-penalty",
        type=float,
        default=None,
        help="Penalty multiplier strength (0..1) applied when directional contradiction is detected.",
    )
    agent_plane.add_argument(
        "--calibration-cost-pressure-penalty-strength",
        type=float,
        default=None,
        help="Penalty multiplier strength (0..1) applied as cost-pressure score increases.",
    )
    agent_plane.add_argument(
        "--self-critique-min-score",
        type=float,
        default=None,
        help="Phase 3: minimum self-critique score required before final risk approval.",
    )
    agent_plane.add_argument(
        "--self-critique-max-findings",
        type=int,
        default=None,
        help="Phase 3: maximum findings retained in the self-critique artifact.",
    )
    agent_plane.add_argument(
        "--ops-report-verbosity",
        choices=["compact", "standard", "verbose"],
        default=None,
        help="Phase 3: deterministic ops report detail level.",
    )
    agent_plane.add_argument(
        "--ensemble-mode",
        choices=["single", "adaptive"],
        default=None,
        help="Phase 4: ensemble combiner mode.",
    )
    agent_plane.add_argument(
        "--ensemble-arms",
        default=None,
        help="Phase 4: comma-separated enabled strategy arms (for example sma_baseline,technical_composite,llm_context).",
    )
    agent_plane.add_argument(
        "--ensemble-decay-horizon",
        type=int,
        default=None,
        help="Phase 4: decay horizon for adaptive arm-performance weighting.",
    )
    agent_plane.add_argument(
        "--ensemble-exploration-weight",
        type=float,
        default=None,
        help="Phase 4: minimum exploration mass injected before adaptive normalization.",
    )
    agent_plane.add_argument(
        "--ensemble-turnover-penalty-bps",
        type=float,
        default=None,
        help="Phase 4: turnover penalty in bps applied during adaptive arm weighting.",
    )
    agent_plane.add_argument(
        "--paper-notional-usd",
        type=float,
        default=None,
        help="Notional USD for emitted paper intents.",
    )
    agent_plane.add_argument(
        "--paper-starting-cash-usd",
        type=float,
        default=None,
        help="Starting cash used by deterministic paper execution ledger.",
    )
    agent_plane.add_argument(
        "--paper-fee-bps",
        type=float,
        default=None,
        help="Per-trade fee in basis points used by deterministic paper execution.",
    )
    agent_plane.add_argument(
        "--paper-slippage-bps",
        type=float,
        default=None,
        help="Per-trade slippage in basis points used by deterministic paper execution.",
    )
    agent_plane.add_argument(
        "--paper-sizing-enabled",
        dest="paper_sizing_enabled",
        action="store_true",
        help="Enable volatility/confidence-based risk-budget sizing for paper intents.",
    )
    agent_plane.add_argument(
        "--no-paper-sizing-enabled",
        dest="paper_sizing_enabled",
        action="store_false",
        help="Disable adaptive risk-budget sizing and use fallback sizing profile.",
    )
    agent_plane.add_argument(
        "--paper-sizing-target-annual-volatility",
        type=float,
        default=None,
        help="Target annualized volatility used by paper notional scaling.",
    )
    agent_plane.add_argument(
        "--paper-sizing-confidence-floor",
        type=float,
        default=None,
        help="Lower confidence bound used to normalize confidence-based sizing.",
    )
    agent_plane.add_argument(
        "--paper-sizing-confidence-ceiling",
        type=float,
        default=None,
        help="Upper confidence bound used to normalize confidence-based sizing.",
    )
    agent_plane.add_argument(
        "--paper-sizing-min-fraction",
        type=float,
        default=None,
        help="Minimum sizing fraction relative to base paper notional.",
    )
    agent_plane.add_argument(
        "--paper-sizing-max-fraction",
        type=float,
        default=None,
        help="Maximum sizing fraction relative to base paper notional.",
    )
    agent_plane.add_argument(
        "--paper-sizing-drawdown-throttle-start",
        type=float,
        default=None,
        help="Drawdown ratio where paper sizing throttle begins.",
    )
    agent_plane.add_argument(
        "--paper-sizing-drawdown-kill-switch",
        type=float,
        default=None,
        help="Drawdown ratio where actionable paper intents are blocked.",
    )
    agent_plane.add_argument(
        "--paper-sizing-fallback-notional-usd",
        type=float,
        default=None,
        help="Fallback paper notional used when adaptive sizing cannot resolve volatility.",
    )
    agent_plane.add_argument(
        "--execution-realism-spread-bps",
        type=float,
        default=None,
        help="Quoted spread in bps injected into deterministic paper execution.",
    )
    agent_plane.add_argument(
        "--execution-realism-latency-ms",
        type=float,
        default=None,
        help="Execution latency in milliseconds used by deterministic fill simulation.",
    )
    agent_plane.add_argument(
        "--execution-realism-latency-slippage-bps-per-second",
        type=float,
        default=None,
        help="Additional slippage bps applied per second of execution latency.",
    )
    agent_plane.add_argument(
        "--execution-realism-liquidity-score",
        type=float,
        default=None,
        help="Liquidity score (0..1) driving partial-fill realism in deterministic execution.",
    )
    agent_plane.add_argument(
        "--execution-realism-market-depth-notional-usd",
        type=float,
        default=None,
        help="Approximate top-of-book market depth in USD used for impact estimation.",
    )
    agent_plane.add_argument(
        "--execution-realism-notional-impact-coeff",
        type=float,
        default=None,
        help="Impact coefficient applied to notional-vs-depth slippage pressure.",
    )
    agent_plane.set_defaults(paper_sizing_enabled=None)
    paper_account = subparsers.add_parser(
        "paper-account-check",
        help="Validate connectivity to configured paper-account provider.",
    )
    paper_account.add_argument(
        "--provider",
        choices=["tradingview", "ccxt"],
        default=None,
        help="Paper account provider override (default from settings).",
    )
    paper_account.add_argument(
        "--exchange",
        default=None,
        help="CCXT exchange id when provider=ccxt (default from settings).",
    )
    paper_account.add_argument(
        "--sandbox",
        dest="paper_sandbox",
        action="store_true",
        help="Enable CCXT sandbox mode (provider=ccxt).",
    )
    paper_account.add_argument(
        "--no-sandbox",
        dest="paper_sandbox",
        action="store_false",
        help="Disable CCXT sandbox mode (provider=ccxt).",
    )
    paper_account.set_defaults(paper_sandbox=None)
    paper_account.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Provider connectivity timeout override.",
    )
    paper_account.add_argument(
        "--tradingview-base-url",
        default=None,
        help="TradingView base URL override for provider=tradingview.",
    )
    visualize = subparsers.add_parser(
        "visualize-run",
        help="Generate readable backtest/strategy evaluation charts for an agent-plane run.",
    )
    visualize.add_argument(
        "--run-dir",
        default=None,
        help="Agent-plane run directory path. Defaults to the latest openclaw-orchestrator run.",
    )
    visualize.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for generated charts. Defaults to <run-dir>/visuals.",
    )
    trigger_train = subparsers.add_parser(
        "train-trigger-model",
        help="Train deterministic buy/sell/hold trigger model from OHLCV data.",
    )
    trigger_train.add_argument("--exchange", default=None)
    trigger_train.add_argument("--symbol", default=None)
    trigger_train.add_argument("--timeframe", default=None)
    trigger_train.add_argument(
        "--input-file",
        default=None,
        help="Optional explicit parquet input file for training.",
    )
    trigger_train.add_argument(
        "--labeling-mode",
        choices=["directional_v1", "triple_barrier_v2"],
        default=None,
        help="Trigger-model labeling mode (default from settings).",
    )
    trigger_train.add_argument(
        "--trade-quality-min-score",
        type=float,
        default=None,
        help="Minimum trade-quality score retained as actionable label evidence.",
    )
    trigger_train.add_argument(
        "--action-confidence-threshold",
        type=float,
        default=None,
        help="Minimum class probability required before actionable buy/sell is emitted.",
    )
    trigger_train.add_argument(
        "--priority2-features-enabled",
        dest="priority2_features_enabled",
        action="store_true",
        help="Enable Priority 2 feature expansion for trigger-model training.",
    )
    trigger_train.add_argument(
        "--no-priority2-features-enabled",
        dest="priority2_features_enabled",
        action="store_false",
        help="Disable Priority 2 feature expansion for trigger-model training.",
    )
    trigger_train.add_argument(
        "--priority2-external-features-path",
        default=None,
        help="Optional external Priority 2 feature file used during training.",
    )
    trigger_train.add_argument(
        "--priority2-feature-columns",
        default=None,
        help=(
            "Comma-separated Priority 2 feature columns to enable "
            "(for example open_interest_feature,participant_positioning_feature). "
            "Use 'stable' for the default stable pair or 'all' for full Priority 2 set."
        ),
    )
    trigger_train.add_argument(
        "--ranked-features-enabled",
        dest="ranked_features_enabled",
        action="store_true",
        help="Enable ranked high-impact feature expansion for trigger-model training.",
    )
    trigger_train.add_argument(
        "--no-ranked-features-enabled",
        dest="ranked_features_enabled",
        action="store_false",
        help="Disable ranked high-impact feature expansion for trigger-model training.",
    )
    trigger_train.add_argument(
        "--ranked-external-features-path",
        default=None,
        help="Optional external ranked feature file used during training.",
    )
    trigger_train.add_argument(
        "--ranked-feature-columns",
        default=None,
        help=(
            "Comma-separated ranked feature columns to enable "
            "(for example flow_signed_volume_imbalance_24,derivatives_basis_z_24). "
            "Use 'stable' for default ranked subset or 'all' for full ranked set."
        ),
    )
    trigger_train.add_argument(
        "--orderbook-features-enabled",
        dest="orderbook_features_enabled",
        action="store_true",
        help="Enable order book feature expansion for trigger-model training.",
    )
    trigger_train.add_argument(
        "--no-orderbook-features-enabled",
        dest="orderbook_features_enabled",
        action="store_false",
        help="Disable order book feature expansion for trigger-model training.",
    )
    trigger_train.add_argument(
        "--orderbook-features-path",
        default=None,
        help=(
            "Optional aligned order book feature parquet path (for example from "
            "retrieve-orderbook-features)."
        ),
    )
    trigger_train.add_argument(
        "--orderbook-feature-columns",
        default=None,
        help=(
            "Comma-separated order book feature columns to enable "
            "(for example orderbook_spread_feature,orderbook_depth_imbalance_feature). "
            "Use 'stable' for the default stable set or 'all' for full order book set."
        ),
    )
    trigger_train.add_argument(
        "--horizon-bars",
        type=int,
        default=None,
        help="Forward bars used for labeling (default from settings).",
    )
    trigger_train.add_argument(
        "--buy-threshold",
        type=float,
        default=None,
        help="Forward return threshold for buy labels (default from settings).",
    )
    trigger_train.add_argument(
        "--sell-threshold",
        type=float,
        default=None,
        help="Absolute forward return threshold for sell labels (default from settings).",
    )
    trigger_train.add_argument(
        "--min-train-samples",
        type=int,
        default=None,
        help="Minimum training rows required after feature/label generation.",
    )
    trigger_train.add_argument(
        "--cost-bps",
        type=float,
        default=None,
        help="One-way cost in bps used for net-expectancy diagnostics.",
    )
    trigger_train.add_argument(
        "--optimize-thresholds",
        dest="optimize_thresholds",
        action="store_true",
        help=(
            "Enable threshold optimization by execution-aligned realized PnL "
            "(with equity return tie-breakers)."
        ),
    )
    trigger_train.add_argument(
        "--no-optimize-thresholds",
        dest="optimize_thresholds",
        action="store_false",
        help="Disable threshold optimization and use provided thresholds directly.",
    )
    trigger_train.set_defaults(optimize_thresholds=None)
    trigger_train.set_defaults(
        priority2_features_enabled=None,
        ranked_features_enabled=None,
        orderbook_features_enabled=None,
    )

    trigger_predict = subparsers.add_parser(
        "predict-trigger",
        help="Generate one explainable buy/sell/hold prediction from latest market data.",
    )
    trigger_predict.add_argument("--exchange", default=None)
    trigger_predict.add_argument("--symbol", default=None)
    trigger_predict.add_argument("--timeframe", default=None)
    trigger_predict.add_argument(
        "--action-confidence-threshold",
        type=float,
        default=None,
        help="Prediction-time actionable confidence threshold override.",
    )
    trigger_predict.add_argument(
        "--priority2-features-enabled",
        dest="priority2_features_enabled",
        action="store_true",
        help="Enable Priority 2 feature expansion for prediction.",
    )
    trigger_predict.add_argument(
        "--no-priority2-features-enabled",
        dest="priority2_features_enabled",
        action="store_false",
        help="Disable Priority 2 feature expansion for prediction.",
    )
    trigger_predict.add_argument(
        "--priority2-external-features-path",
        default=None,
        help="Optional external Priority 2 feature file used during prediction.",
    )
    trigger_predict.add_argument(
        "--priority2-feature-columns",
        default=None,
        help=(
            "Comma-separated Priority 2 feature columns to enable "
            "(for example open_interest_feature,participant_positioning_feature). "
            "Use 'stable' for the default stable pair or 'all' for full Priority 2 set."
        ),
    )
    trigger_predict.add_argument(
        "--ranked-features-enabled",
        dest="ranked_features_enabled",
        action="store_true",
        help="Enable ranked high-impact feature expansion for prediction.",
    )
    trigger_predict.add_argument(
        "--no-ranked-features-enabled",
        dest="ranked_features_enabled",
        action="store_false",
        help="Disable ranked high-impact feature expansion for prediction.",
    )
    trigger_predict.add_argument(
        "--ranked-external-features-path",
        default=None,
        help="Optional external ranked feature file used during prediction.",
    )
    trigger_predict.add_argument(
        "--ranked-feature-columns",
        default=None,
        help=(
            "Comma-separated ranked feature columns to enable "
            "(for example flow_signed_volume_imbalance_24,derivatives_basis_z_24). "
            "Use 'stable' for default ranked subset or 'all' for full ranked set."
        ),
    )
    trigger_predict.add_argument(
        "--orderbook-features-enabled",
        dest="orderbook_features_enabled",
        action="store_true",
        help="Enable order book feature expansion for prediction.",
    )
    trigger_predict.add_argument(
        "--no-orderbook-features-enabled",
        dest="orderbook_features_enabled",
        action="store_false",
        help="Disable order book feature expansion for prediction.",
    )
    trigger_predict.add_argument(
        "--orderbook-features-path",
        default=None,
        help="Optional aligned order book feature parquet path used during prediction.",
    )
    trigger_predict.add_argument(
        "--orderbook-feature-columns",
        default=None,
        help=(
            "Comma-separated order book feature columns to enable "
            "(for example orderbook_spread_feature,orderbook_depth_imbalance_feature). "
            "Use 'stable' for the default stable set or 'all' for full order book set."
        ),
    )
    trigger_predict.add_argument(
        "--model-path",
        default=None,
        help="Optional explicit model.json path. Defaults to latest model for scope.",
    )
    trigger_predict.add_argument(
        "--input-file",
        default=None,
        help="Optional explicit parquet input file for prediction.",
    )
    trigger_predict.set_defaults(
        priority2_features_enabled=None,
        ranked_features_enabled=None,
        orderbook_features_enabled=None,
    )

    trigger_monitor = subparsers.add_parser(
        "monitor-triggers",
        help="Continuously ingest market data, run trigger predictions, and emit alerts.",
    )
    trigger_monitor.add_argument("--exchange", default=None)
    trigger_monitor.add_argument("--symbol", default=None)
    trigger_monitor.add_argument("--timeframe", default=None)
    trigger_monitor.add_argument(
        "--priority2-features-enabled",
        dest="priority2_features_enabled",
        action="store_true",
        help="Enable Priority 2 feature expansion for monitoring predictions.",
    )
    trigger_monitor.add_argument(
        "--no-priority2-features-enabled",
        dest="priority2_features_enabled",
        action="store_false",
        help="Disable Priority 2 feature expansion for monitoring predictions.",
    )
    trigger_monitor.add_argument(
        "--priority2-external-features-path",
        default=None,
        help="Optional external Priority 2 feature file used during monitoring predictions.",
    )
    trigger_monitor.add_argument(
        "--priority2-feature-columns",
        default=None,
        help=(
            "Comma-separated Priority 2 feature columns to enable "
            "(for example open_interest_feature,participant_positioning_feature). "
            "Use 'stable' for the default stable pair or 'all' for full Priority 2 set."
        ),
    )
    trigger_monitor.add_argument(
        "--ranked-features-enabled",
        dest="ranked_features_enabled",
        action="store_true",
        help="Enable ranked high-impact feature expansion for monitoring predictions.",
    )
    trigger_monitor.add_argument(
        "--no-ranked-features-enabled",
        dest="ranked_features_enabled",
        action="store_false",
        help="Disable ranked high-impact feature expansion for monitoring predictions.",
    )
    trigger_monitor.add_argument(
        "--ranked-external-features-path",
        default=None,
        help="Optional external ranked feature file used during monitoring predictions.",
    )
    trigger_monitor.add_argument(
        "--ranked-feature-columns",
        default=None,
        help=(
            "Comma-separated ranked feature columns to enable "
            "(for example flow_signed_volume_imbalance_24,derivatives_basis_z_24). "
            "Use 'stable' for default ranked subset or 'all' for full ranked set."
        ),
    )
    trigger_monitor.add_argument(
        "--orderbook-features-enabled",
        dest="orderbook_features_enabled",
        action="store_true",
        help="Enable order book feature expansion for monitoring predictions.",
    )
    trigger_monitor.add_argument(
        "--no-orderbook-features-enabled",
        dest="orderbook_features_enabled",
        action="store_false",
        help="Disable order book feature expansion for monitoring predictions.",
    )
    trigger_monitor.add_argument(
        "--orderbook-features-path",
        default=None,
        help="Optional aligned order book feature parquet path used during monitoring predictions.",
    )
    trigger_monitor.add_argument(
        "--orderbook-feature-columns",
        default=None,
        help=(
            "Comma-separated order book feature columns to enable "
            "(for example orderbook_spread_feature,orderbook_depth_imbalance_feature). "
            "Use 'stable' for the default stable set or 'all' for full order book set."
        ),
    )
    trigger_monitor.add_argument(
        "--model-path",
        default=None,
        help="Optional explicit model.json path. Defaults to latest model for scope.",
    )
    trigger_monitor.add_argument(
        "--limit",
        type=int,
        default=500,
        help="OHLCV row limit fetched each monitoring cycle.",
    )
    trigger_monitor.add_argument(
        "--poll-seconds",
        type=float,
        default=None,
        help="Seconds between monitor cycles (default from settings).",
    )
    trigger_monitor.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Minimum confidence required for alerts (default from settings).",
    )
    trigger_monitor.add_argument(
        "--webhook-url",
        default=None,
        help="Optional webhook URL for outbound alert delivery.",
    )
    trigger_monitor.add_argument(
        "--notify-on-hold",
        action="store_true",
        help="Also notify for hold predictions (disabled by default).",
    )
    trigger_monitor.add_argument(
        "--paper-trading-enabled",
        dest="paper_trading_enabled",
        action="store_true",
        help="Enable paper-trading execution for actionable trigger alerts.",
    )
    trigger_monitor.add_argument(
        "--no-paper-trading-enabled",
        dest="paper_trading_enabled",
        action="store_false",
        help="Disable paper-trading execution for monitor runs.",
    )
    trigger_monitor.add_argument(
        "--paper-notional-usd",
        type=float,
        default=None,
        help="Paper-trade notional USD for each actionable trigger execution.",
    )
    trigger_monitor.add_argument(
        "--paper-starting-cash-usd",
        type=float,
        default=None,
        help="Starting cash USD used for paper-trading portfolio state initialization.",
    )
    trigger_monitor.add_argument(
        "--paper-fee-bps",
        type=float,
        default=None,
        help="Per-trade paper execution fee in basis points.",
    )
    trigger_monitor.add_argument(
        "--paper-slippage-bps",
        type=float,
        default=None,
        help="Per-trade paper execution slippage in basis points.",
    )
    trigger_monitor.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Optional finite cycle cap for bounded monitor runs.",
    )
    trigger_monitor.set_defaults(
        priority2_features_enabled=None,
        ranked_features_enabled=None,
        orderbook_features_enabled=None,
        paper_trading_enabled=None,
    )

    doctor = subparsers.add_parser(
        "doctor",
        help="Run preflight checks for storage, defaults, and secrets readiness.",
    )
    doctor.add_argument(
        "--require-secrets",
        action="store_true",
        help="Treat missing exchange API secrets as a failure.",
    )

    return parser


def _effective_require_secrets(settings, args: argparse.Namespace) -> bool:
    return settings.require_exchange_secrets or bool(getattr(args, "require_secrets", False))


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    settings = load_settings()
    parser = _base_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        report = run_doctor(settings, require_secrets=_effective_require_secrets(settings, args))
        print(format_doctor_report(report, settings))
        if not report.ok:
            raise SystemExit(1)
        return

    ensure_data_root_ready(
        settings.quant_data_root,
        allow_unmounted=settings.allow_unmounted_data_root,
    )
    ensure_phase1_tree(settings.quant_data_root)

    exchange = getattr(args, "exchange", None) or settings.default_exchange
    symbol = getattr(args, "symbol", None) or settings.default_symbol
    timeframe = getattr(args, "timeframe", None) or settings.default_timeframe

    if args.command == "ingest":
        with tracked_operation(
            settings.quant_data_root,
            operation="ingest",
            dimensions={"exchange": exchange, "symbol": symbol, "timeframe": timeframe},
        ) as metric:
            result = fetch_ohlcv_to_parquet(settings, exchange, symbol, timeframe, limit=args.limit)
            metric["row_count"] = result.row_count
            metric["output_path"] = str(result.output_path)
            metric["data_start"] = str(result.start_timestamp)
            metric["data_end"] = str(result.end_timestamp)
        print(f"Ingested {result.row_count} rows -> {result.output_path}")
        return
    if args.command == "capture-orderbook":
        sample_interval_seconds = (
            args.sample_interval_seconds
            if args.sample_interval_seconds is not None
            else settings.orderbook_capture_sample_interval_seconds
        )
        depth_limit = (
            args.depth_limit
            if args.depth_limit is not None
            else settings.orderbook_capture_depth_limit
        )
        with tracked_operation(
            settings.quant_data_root,
            operation="capture-orderbook",
            dimensions={"exchange": exchange, "symbol": symbol},
        ) as metric:
            result = capture_orderbook_snapshots_to_parquet(
                settings=settings,
                exchange_id=exchange,
                symbol=symbol,
                sample_count=max(1, int(args.sample_count)),
                sample_interval_seconds=max(0.0, float(sample_interval_seconds)),
                depth_limit=max(1, int(depth_limit)),
            )
            metric["row_count"] = result.row_count
            metric["output_path"] = str(result.output_path)
            metric["start_timestamp"] = str(result.start_timestamp)
            metric["end_timestamp"] = str(result.end_timestamp)
            metric["depth_limit"] = result.depth_limit
            metric["sample_interval_seconds"] = result.sample_interval_seconds
        print(
            "Order book capture complete -> "
            f"{result.output_path} "
            f"(rows={result.row_count} depth={result.depth_limit} "
            f"interval={result.sample_interval_seconds:.3f}s)"
        )
        return
    if args.command == "retrieve-orderbook-features":
        source_file = Path(args.input_file).expanduser().resolve() if args.input_file else None
        snapshot_source_path = (
            Path(args.snapshot_source_path).expanduser().resolve()
            if args.snapshot_source_path
            else None
        )
        orderbook_feature_columns = _parse_orderbook_feature_columns(
            args.orderbook_feature_columns,
            default=tuple(settings.orderbook_feature_columns),
        )
        with tracked_operation(
            settings.quant_data_root,
            operation="retrieve-orderbook-features",
            dimensions={
                "exchange": exchange,
                "symbol": symbol,
                "timeframe": timeframe,
            },
        ) as metric:
            result = retrieve_orderbook_features(
                settings=settings,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                source_data_path=source_file,
                snapshot_source_path=snapshot_source_path,
                orderbook_feature_columns=orderbook_feature_columns,
            )
            metric["run_id"] = result.run_id
            metric["source_data_path"] = str(result.source_data_path)
            metric["snapshot_source_path"] = str(result.snapshot_source_path)
            metric["row_count"] = result.row_count
            metric["coverage_ratio"] = result.coverage_ratio
            metric["parquet_path"] = str(result.parquet_path)
            metric["contract_path"] = str(result.contract_path)
            metric["reason_codes"] = list(result.reason_codes)
        print(
            "Order book features retrieved -> "
            f"{result.parquet_path} "
            f"(coverage={result.coverage_ratio:.3f} rows={result.row_count})"
        )
        print(
            "Use this path with --orderbook-features-path: "
            f"{result.parquet_path}"
        )
        return
    if args.command == "retrieve-priority2-features":
        source_file = Path(args.input_file).expanduser().resolve() if args.input_file else None
        local_feature_overrides_path = (
            Path(args.local_feature_overrides_path).expanduser().resolve()
            if args.local_feature_overrides_path
            else (
                Path(settings.priority2_local_feature_overrides_path).expanduser().resolve()
                if settings.priority2_local_feature_overrides_path
                else None
            )
        )
        provider = (
            str(args.provider).strip().lower()
            if args.provider is not None
            else settings.priority2_retrieval_provider
        )
        timeout_seconds = (
            args.timeout_seconds
            if args.timeout_seconds is not None
            else settings.priority2_retrieval_timeout_seconds
        )
        max_points = (
            args.max_points
            if args.max_points is not None
            else settings.priority2_retrieval_max_points
        )
        base_url = args.base_url or settings.priority2_retrieval_base_url
        with tracked_operation(
            settings.quant_data_root,
            operation="retrieve-priority2-features",
            dimensions={
                "exchange": exchange,
                "symbol": symbol,
                "timeframe": timeframe,
                "provider": provider,
            },
        ) as metric:
            result = retrieve_priority2_external_features(
                settings=settings,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                source_data_path=source_file,
                provider=provider,
                timeout_seconds=max(1.0, float(timeout_seconds)),
                max_points_per_request=max(50, int(max_points)),
                base_url=str(base_url),
                local_feature_overrides_path=local_feature_overrides_path,
            )
            metric["run_id"] = result.run_id
            metric["provider"] = result.provider
            metric["source_data_path"] = str(result.source_data_path)
            metric["row_count"] = result.row_count
            metric["coverage_ratio"] = result.coverage_ratio
            metric["parquet_path"] = str(result.parquet_path)
            metric["contract_path"] = str(result.contract_path)
            metric["reason_codes"] = list(result.reason_codes)
        print(
            "Priority 2 external features retrieved -> "
            f"{result.parquet_path} "
            f"(coverage={result.coverage_ratio:.3f} rows={result.row_count})"
        )
        print(
            "Use this path with --priority2-external-features-path: "
            f"{result.parquet_path}"
        )
        return

    if args.command == "backtest":
        source_file = Path(args.input_file).expanduser().resolve() if args.input_file else None
        fee_bps = args.fee_bps if args.fee_bps is not None else settings.backtest_fee_bps
        slippage_bps = args.slippage_bps if args.slippage_bps is not None else settings.backtest_slippage_bps
        with tracked_operation(
            settings.quant_data_root,
            operation="backtest",
            dimensions={"exchange": exchange, "symbol": symbol, "timeframe": timeframe},
        ) as metric:
            result = run_sma_backtest(
                settings=settings,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                fast_window=args.fast_window,
                slow_window=args.slow_window,
                fee_bps=max(0.0, float(fee_bps)),
                slippage_bps=max(0.0, float(slippage_bps)),
                source_data_path=source_file,
                archive_run=args.archive,
            )
            metric["run_dir"] = str(result.run_dir)
            metric["metrics_path"] = str(result.metrics_path)
            metric["manifest_path"] = str(result.manifest_path)
            metric["total_return"] = result.metrics.get("total_return")
            metric["max_drawdown"] = result.metrics.get("max_drawdown")
            metric["source_data_sha256"] = result.source_data_sha256
            if result.archive_path is not None:
                metric["archive_path"] = str(result.archive_path)
        print(f"Backtest complete -> {result.run_dir}")
        return
    if args.command == "train-trigger-model":
        source_file = Path(args.input_file).expanduser().resolve() if args.input_file else None
        labeling_mode = (
            args.labeling_mode
            if args.labeling_mode is not None
            else settings.trigger_model_labeling_mode
        )
        horizon_bars = (
            args.horizon_bars
            if args.horizon_bars is not None
            else settings.trigger_model_horizon_bars
        )
        buy_threshold = (
            args.buy_threshold
            if args.buy_threshold is not None
            else settings.trigger_model_buy_threshold
        )
        sell_threshold = (
            args.sell_threshold
            if args.sell_threshold is not None
            else settings.trigger_model_sell_threshold
        )
        min_train_samples = (
            args.min_train_samples
            if args.min_train_samples is not None
            else settings.trigger_model_min_train_samples
        )
        cost_bps = (
            args.cost_bps
            if args.cost_bps is not None
            else settings.trigger_model_cost_bps
        )
        optimize_thresholds = (
            args.optimize_thresholds
            if args.optimize_thresholds is not None
            else settings.trigger_model_optimize_thresholds
        )
        trade_quality_min_score = (
            args.trade_quality_min_score
            if args.trade_quality_min_score is not None
            else settings.trigger_model_trade_quality_min_score
        )
        action_confidence_threshold = (
            args.action_confidence_threshold
            if args.action_confidence_threshold is not None
            else settings.trigger_model_action_confidence_threshold
        )
        priority2_features_enabled = (
            bool(args.priority2_features_enabled)
            if args.priority2_features_enabled is not None
            else bool(settings.priority2_features_enabled)
        )
        priority2_external_features_path = (
            Path(args.priority2_external_features_path).expanduser().resolve()
            if args.priority2_external_features_path
            else (
                Path(settings.priority2_external_features_path).expanduser().resolve()
                if settings.priority2_external_features_path
                else None
            )
        )
        priority2_feature_columns = _parse_priority2_feature_columns(
            args.priority2_feature_columns,
            default=tuple(settings.priority2_feature_columns),
        )
        ranked_features_enabled = (
            bool(args.ranked_features_enabled)
            if args.ranked_features_enabled is not None
            else bool(settings.ranked_features_enabled)
        )
        ranked_external_features_path = (
            Path(args.ranked_external_features_path).expanduser().resolve()
            if args.ranked_external_features_path
            else (
                Path(settings.ranked_external_features_path).expanduser().resolve()
                if settings.ranked_external_features_path
                else None
            )
        )
        ranked_feature_columns = _parse_ranked_feature_columns(
            args.ranked_feature_columns,
            default=tuple(settings.ranked_feature_columns),
        )
        orderbook_features_enabled = (
            bool(args.orderbook_features_enabled)
            if args.orderbook_features_enabled is not None
            else bool(settings.orderbook_features_enabled)
        )
        orderbook_features_path = (
            Path(args.orderbook_features_path).expanduser().resolve()
            if args.orderbook_features_path
            else (
                Path(settings.orderbook_features_path).expanduser().resolve()
                if settings.orderbook_features_path
                else None
            )
        )
        orderbook_feature_columns = _parse_orderbook_feature_columns(
            args.orderbook_feature_columns,
            default=tuple(settings.orderbook_feature_columns),
        )
        with tracked_operation(
            settings.quant_data_root,
            operation="train-trigger-model",
            dimensions={
                "exchange": exchange,
                "symbol": symbol,
                "timeframe": timeframe,
            },
        ) as metric:
            result = train_trigger_model(
                settings=settings,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                input_file=source_file,
                horizon_bars=max(1, int(horizon_bars)),
                buy_threshold=float(buy_threshold),
                sell_threshold=float(sell_threshold),
                min_train_samples=max(20, int(min_train_samples)),
                cost_bps=max(0.0, float(cost_bps)),
                optimize_thresholds=bool(optimize_thresholds),
                labeling_mode=str(labeling_mode),
                trade_quality_min_score=min(1.0, max(0.0, float(trade_quality_min_score))),
                action_confidence_threshold=min(
                    1.0,
                    max(0.0, float(action_confidence_threshold)),
                ),
                priority2_features_enabled=bool(priority2_features_enabled),
                priority2_external_features_path=priority2_external_features_path,
                priority2_feature_columns=priority2_feature_columns,
                ranked_features_enabled=bool(ranked_features_enabled),
                ranked_external_features_path=ranked_external_features_path,
                ranked_feature_columns=ranked_feature_columns,
                orderbook_features_enabled=bool(orderbook_features_enabled),
                orderbook_features_path=orderbook_features_path,
                orderbook_feature_columns=orderbook_feature_columns,
            )
            metric["model_path"] = str(result.model_path)
            metric["run_dir"] = str(result.run_dir)
            metric["sample_count"] = result.sample_count
            metric["train_count"] = result.train_count
            metric["test_count"] = result.test_count
            metric["accuracy"] = result.accuracy
            metric["selected_buy_threshold"] = result.selected_buy_threshold
            metric["selected_sell_threshold"] = result.selected_sell_threshold
            metric["net_expectancy_per_actionable"] = result.net_expectancy_per_actionable
            metric["execution_backtest_equity_return"] = result.execution_backtest_equity_return
            metric["execution_backtest_realized_pnl_delta_usd"] = (
                result.execution_backtest_realized_pnl_delta_usd
            )
            metric["selected_trade_quality_threshold"] = result.selected_trade_quality_threshold
            metric["selected_action_confidence_threshold"] = result.selected_action_confidence_threshold
        print(
            "Trigger model training complete -> "
            f"{result.model_path} "
            f"(samples={result.sample_count} accuracy={result.accuracy:.3f} "
            f"buy_th={result.selected_buy_threshold:.4f} sell_th={result.selected_sell_threshold:.4f} "
            f"quality_th={result.selected_trade_quality_threshold:.3f} "
            f"conf_th={result.selected_action_confidence_threshold:.3f} "
            f"exec_realized_pnl={result.execution_backtest_realized_pnl_delta_usd:.4f} "
            f"exec_equity_ret={result.execution_backtest_equity_return:.4f})"
        )
        return

    if args.command == "predict-trigger":
        source_file = Path(args.input_file).expanduser().resolve() if args.input_file else None
        model_path = Path(args.model_path).expanduser().resolve() if args.model_path else None
        action_confidence_threshold = (
            args.action_confidence_threshold
            if args.action_confidence_threshold is not None
            else settings.trigger_model_action_confidence_threshold
        )
        priority2_features_enabled = (
            bool(args.priority2_features_enabled)
            if args.priority2_features_enabled is not None
            else bool(settings.priority2_features_enabled)
        )
        priority2_external_features_path = (
            Path(args.priority2_external_features_path).expanduser().resolve()
            if args.priority2_external_features_path
            else (
                Path(settings.priority2_external_features_path).expanduser().resolve()
                if settings.priority2_external_features_path
                else None
            )
        )
        priority2_feature_columns = _parse_priority2_feature_columns(
            args.priority2_feature_columns,
            default=tuple(settings.priority2_feature_columns),
        )
        ranked_features_enabled = (
            bool(args.ranked_features_enabled)
            if args.ranked_features_enabled is not None
            else bool(settings.ranked_features_enabled)
        )
        ranked_external_features_path = (
            Path(args.ranked_external_features_path).expanduser().resolve()
            if args.ranked_external_features_path
            else (
                Path(settings.ranked_external_features_path).expanduser().resolve()
                if settings.ranked_external_features_path
                else None
            )
        )
        ranked_feature_columns = _parse_ranked_feature_columns(
            args.ranked_feature_columns,
            default=tuple(settings.ranked_feature_columns),
        )
        orderbook_features_enabled = (
            bool(args.orderbook_features_enabled)
            if args.orderbook_features_enabled is not None
            else bool(settings.orderbook_features_enabled)
        )
        orderbook_features_path = (
            Path(args.orderbook_features_path).expanduser().resolve()
            if args.orderbook_features_path
            else (
                Path(settings.orderbook_features_path).expanduser().resolve()
                if settings.orderbook_features_path
                else None
            )
        )
        orderbook_feature_columns = _parse_orderbook_feature_columns(
            args.orderbook_feature_columns,
            default=tuple(settings.orderbook_feature_columns),
        )
        with tracked_operation(
            settings.quant_data_root,
            operation="predict-trigger",
            dimensions={
                "exchange": exchange,
                "symbol": symbol,
                "timeframe": timeframe,
            },
        ) as metric:
            result = predict_trigger_signal(
                settings=settings,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                model_path=model_path,
                input_file=source_file,
                write_artifact=True,
                action_confidence_threshold=min(
                    1.0,
                    max(0.0, float(action_confidence_threshold)),
                ),
                priority2_features_enabled=bool(priority2_features_enabled),
                priority2_external_features_path=priority2_external_features_path,
                priority2_feature_columns=priority2_feature_columns,
                ranked_features_enabled=bool(ranked_features_enabled),
                ranked_external_features_path=ranked_external_features_path,
                ranked_feature_columns=ranked_feature_columns,
                orderbook_features_enabled=bool(orderbook_features_enabled),
                orderbook_features_path=orderbook_features_path,
                orderbook_feature_columns=orderbook_feature_columns,
            )
            metric["model_path"] = str(result.model_path)
            metric["source_data_path"] = str(result.source_data_path)
            metric["recommendation"] = result.recommendation
            metric["confidence"] = result.confidence
            metric["prediction_path"] = str(result.prediction_path) if result.prediction_path else None
        print(
            "Trigger prediction -> "
            f"{result.recommendation} "
            f"(confidence={result.confidence:.3f}) "
            f"path={result.prediction_path}"
        )
        for reason in result.top_reasons:
            print(f"- {reason}")
        return

    if args.command == "report":
        with tracked_operation(
            settings.quant_data_root,
            operation="report",
            dimensions={"exchange": exchange, "symbol": symbol, "timeframe": timeframe},
        ) as metric:
            result = generate_daily_report(settings, exchange, symbol, timeframe)
            metric["report_path"] = str(result.report_path)
            metric["strategy_run_dir"] = str(result.strategy_run_dir)
        print(f"Report written -> {result.report_path}")
        return
    if args.command == "archive-backtest":
        strategy_name = args.strategy or STRATEGY_NAME
        if args.run_dir:
            run_dir = Path(args.run_dir).expanduser().resolve()
        else:
            run_dir = latest_backtest_run_dir(settings.quant_data_root, strategy_name)

        with tracked_operation(
            settings.quant_data_root,
            operation="archive-backtest",
            dimensions={"strategy": strategy_name},
        ) as metric:
            archive_path = archive_backtest_run(
                settings.quant_data_root,
                run_dir=run_dir,
                strategy_name=strategy_name,
            )
            metric["run_dir"] = str(run_dir)
            metric["archive_path"] = str(archive_path)
        print(f"Backtest archived -> {archive_path}")
        return

    if args.command == "run-daily":
        with tracked_operation(
            settings.quant_data_root,
            operation="run-daily",
            dimensions={"exchange": exchange, "symbol": symbol, "timeframe": timeframe},
        ) as metric:
            if _effective_require_secrets(settings, args):
                ensure_exchange_secrets_ready(settings)

            logger.info("Running daily pipeline ingest -> backtest -> report")
            ingest_result = fetch_ohlcv_to_parquet(settings, exchange, symbol, timeframe, limit=args.limit)
            logger.info("Ingest complete at %s", ingest_result.output_path)
            metric["ingest"] = {
                "row_count": ingest_result.row_count,
                "output_path": str(ingest_result.output_path),
            }

            backtest_result = run_sma_backtest(
                settings=settings,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                fast_window=args.fast_window,
                slow_window=args.slow_window,
                fee_bps=max(
                    0.0,
                    float(args.fee_bps if args.fee_bps is not None else settings.backtest_fee_bps),
                ),
                slippage_bps=max(
                    0.0,
                    float(
                        args.slippage_bps
                        if args.slippage_bps is not None
                        else settings.backtest_slippage_bps
                    ),
                ),
                archive_run=args.archive_backtest,
            )
            logger.info("Backtest complete at %s", backtest_result.run_dir)
            metric["backtest"] = {
                "run_dir": str(backtest_result.run_dir),
                "total_return": backtest_result.metrics.get("total_return"),
                "max_drawdown": backtest_result.metrics.get("max_drawdown"),
                "manifest_path": str(backtest_result.manifest_path),
            }
            if backtest_result.archive_path is not None:
                metric["backtest"]["archive_path"] = str(backtest_result.archive_path)

            report_result = generate_daily_report(settings, exchange, symbol, timeframe)
            logger.info("Report complete at %s", report_result.report_path)
            metric["report"] = {"report_path": str(report_result.report_path)}
        print(f"Daily pipeline complete -> {report_result.report_path}")
        return

    if args.command == "agent-plane":
        source_file = Path(args.input_file).expanduser().resolve() if args.input_file else None
        thresholds = RiskThresholds(
            min_total_return=(
                args.min_total_return
                if args.min_total_return is not None
                else settings.risk_min_total_return
            ),
            min_sharpe=args.min_sharpe if args.min_sharpe is not None else settings.risk_min_sharpe,
            max_drawdown=args.max_drawdown if args.max_drawdown is not None else settings.risk_max_drawdown,
            max_cost_return_drag=(
                args.max_cost_return_drag
                if args.max_cost_return_drag is not None
                else settings.risk_max_cost_return_drag
            ),
            max_cost_pressure_score=max(
                0.0,
                args.max_cost_pressure_score
                if args.max_cost_pressure_score is not None
                else settings.risk_max_cost_pressure_score,
            ),
            min_signal_confidence=(
                args.min_signal_confidence
                if args.min_signal_confidence is not None
                else settings.risk_min_signal_confidence
            ),
            min_walkforward_quality_score=min(
                1.0,
                max(
                    0.0,
                    (
                        args.min_walkforward_quality_score
                        if args.min_walkforward_quality_score is not None
                        else settings.risk_min_walkforward_quality_score
                    ),
                ),
            ),
            min_regime_confidence=min(
                1.0,
                max(
                    0.0,
                    (
                        args.min_regime_confidence
                        if args.min_regime_confidence is not None
                        else settings.risk_min_regime_confidence
                    ),
                ),
            ),
        )
        priority2_feature_columns = _parse_priority2_feature_columns(
            args.priority2_feature_columns,
            default=tuple(settings.priority2_feature_columns),
        )
        priority2_quality_gate_enabled = (
            bool(args.priority2_quality_gate_enabled)
            if args.priority2_quality_gate_enabled is not None
            else bool(settings.priority2_quality_gate_enabled)
        )
        priority2_quality_min_external_raw_coverage = min(
            1.0,
            max(
                0.0,
                (
                    args.priority2_quality_min_external_raw_coverage
                    if args.priority2_quality_min_external_raw_coverage is not None
                    else settings.priority2_quality_min_external_raw_coverage
                ),
            ),
        )
        priority2_quality_min_non_zero_coverage = min(
            1.0,
            max(
                0.0,
                (
                    args.priority2_quality_min_non_zero_coverage
                    if args.priority2_quality_min_non_zero_coverage is not None
                    else settings.priority2_quality_min_non_zero_coverage
                ),
            ),
        )
        priority2_quality_max_fallback_rate = min(
            1.0,
            max(
                0.0,
                (
                    args.priority2_quality_max_fallback_rate
                    if args.priority2_quality_max_fallback_rate is not None
                    else settings.priority2_quality_max_fallback_rate
                ),
            ),
        )
        priority2_quality_max_staleness_seconds = max(
            0.0,
            (
                args.priority2_quality_max_staleness_seconds
                if args.priority2_quality_max_staleness_seconds is not None
                else settings.priority2_quality_max_staleness_seconds
            ),
        )
        config = AgentPlaneConfig(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            strategy_model=args.strategy_model or settings.ollama_strategy_model,
            ops_model=args.ops_model or settings.ollama_ops_model,
            step_retries=max(
                0,
                args.step_retries if args.step_retries is not None else settings.agent_step_retries,
            ),
            thresholds=thresholds,
            backtest_fee_bps=max(
                0.0,
                args.backtest_fee_bps
                if args.backtest_fee_bps is not None
                else settings.backtest_fee_bps,
            ),
            backtest_slippage_bps=max(
                0.0,
                args.backtest_slippage_bps
                if args.backtest_slippage_bps is not None
                else settings.backtest_slippage_bps,
            ),
            walk_forward_fee_bps=max(
                0.0,
                args.walkforward_fee_bps
                if args.walkforward_fee_bps is not None
                else settings.walk_forward_fee_bps,
            ),
            walk_forward_slippage_bps=max(
                0.0,
                args.walkforward_slippage_bps
                if args.walkforward_slippage_bps is not None
                else settings.walk_forward_slippage_bps,
            ),
            paper_notional_usd=(
                args.paper_notional_usd
                if args.paper_notional_usd is not None
                else settings.paper_trade_notional_usd
            ),
            paper_starting_cash_usd=(
                args.paper_starting_cash_usd
                if args.paper_starting_cash_usd is not None
                else settings.paper_trade_starting_cash_usd
            ),
            paper_fee_bps=(
                args.paper_fee_bps
                if args.paper_fee_bps is not None
                else settings.paper_trade_fee_bps
            ),
            paper_slippage_bps=(
                args.paper_slippage_bps
                if args.paper_slippage_bps is not None
                else settings.paper_trade_slippage_bps
            ),
            paper_sizing_enabled=(
                bool(args.paper_sizing_enabled)
                if args.paper_sizing_enabled is not None
                else bool(settings.paper_sizing_enabled)
            ),
            paper_sizing_target_annual_volatility=max(
                0.01,
                args.paper_sizing_target_annual_volatility
                if args.paper_sizing_target_annual_volatility is not None
                else settings.paper_sizing_target_annual_volatility,
            ),
            paper_sizing_confidence_floor=min(
                1.0,
                max(
                    0.0,
                    args.paper_sizing_confidence_floor
                    if args.paper_sizing_confidence_floor is not None
                    else settings.paper_sizing_confidence_floor,
                ),
            ),
            paper_sizing_confidence_ceiling=min(
                1.0,
                max(
                    0.0,
                    args.paper_sizing_confidence_ceiling
                    if args.paper_sizing_confidence_ceiling is not None
                    else settings.paper_sizing_confidence_ceiling,
                ),
            ),
            paper_sizing_min_fraction=max(
                0.0,
                args.paper_sizing_min_fraction
                if args.paper_sizing_min_fraction is not None
                else settings.paper_sizing_min_fraction,
            ),
            paper_sizing_max_fraction=max(
                0.01,
                args.paper_sizing_max_fraction
                if args.paper_sizing_max_fraction is not None
                else settings.paper_sizing_max_fraction,
            ),
            paper_sizing_drawdown_throttle_start=min(
                0.95,
                max(
                    0.0,
                    args.paper_sizing_drawdown_throttle_start
                    if args.paper_sizing_drawdown_throttle_start is not None
                    else settings.paper_sizing_drawdown_throttle_start,
                ),
            ),
            paper_sizing_drawdown_kill_switch=min(
                0.99,
                max(
                    0.01,
                    args.paper_sizing_drawdown_kill_switch
                    if args.paper_sizing_drawdown_kill_switch is not None
                    else settings.paper_sizing_drawdown_kill_switch,
                ),
            ),
            paper_sizing_fallback_notional_usd=max(
                0.0,
                args.paper_sizing_fallback_notional_usd
                if args.paper_sizing_fallback_notional_usd is not None
                else settings.paper_sizing_fallback_notional_usd,
            ),
            execution_realism_spread_bps=max(
                0.0,
                args.execution_realism_spread_bps
                if args.execution_realism_spread_bps is not None
                else settings.execution_realism_spread_bps,
            ),
            execution_realism_latency_ms=max(
                0.0,
                args.execution_realism_latency_ms
                if args.execution_realism_latency_ms is not None
                else settings.execution_realism_latency_ms,
            ),
            execution_realism_latency_slippage_bps_per_second=max(
                0.0,
                args.execution_realism_latency_slippage_bps_per_second
                if args.execution_realism_latency_slippage_bps_per_second is not None
                else settings.execution_realism_latency_slippage_bps_per_second,
            ),
            execution_realism_liquidity_score=min(
                1.0,
                max(
                    0.0,
                    args.execution_realism_liquidity_score
                    if args.execution_realism_liquidity_score is not None
                    else settings.execution_realism_liquidity_score,
                ),
            ),
            execution_realism_market_depth_notional_usd=max(
                1.0,
                args.execution_realism_market_depth_notional_usd
                if args.execution_realism_market_depth_notional_usd is not None
                else settings.execution_realism_market_depth_notional_usd,
            ),
            execution_realism_notional_impact_coeff=max(
                0.0,
                args.execution_realism_notional_impact_coeff
                if args.execution_realism_notional_impact_coeff is not None
                else settings.execution_realism_notional_impact_coeff,
            ),
            minimum_bars=max(
                10,
                args.minimum_bars if args.minimum_bars is not None else settings.agent_minimum_bars,
            ),
            regime_detector_mode=(
                args.regime_detector_mode
                if args.regime_detector_mode is not None
                else settings.regime_detector_mode
            ),
            regime_enabled=(
                bool(args.regime_enabled)
                if args.regime_enabled is not None
                else bool(settings.regime_enabled)
            ),
            regime_policy_mode=(
                args.regime_policy_mode
                if args.regime_policy_mode is not None
                else settings.regime_policy_mode
            ),
            regime_policy_min_actionable_confidence=min(
                1.0,
                max(
                    0.0,
                    args.regime_policy_min_actionable_confidence
                    if args.regime_policy_min_actionable_confidence is not None
                    else settings.regime_policy_min_actionable_confidence,
                ),
            ),
            regime_policy_transition_confidence=min(
                1.0,
                max(
                    0.0,
                    args.regime_policy_transition_confidence
                    if args.regime_policy_transition_confidence is not None
                    else settings.regime_policy_transition_confidence,
                ),
            ),
            regime_touchpoint_prompting_enabled=(
                bool(args.regime_touchpoint_prompting_enabled)
                if args.regime_touchpoint_prompting_enabled is not None
                else bool(settings.regime_touchpoint_prompting_enabled)
            ),
            regime_touchpoint_calibration_enabled=(
                bool(args.regime_touchpoint_calibration_enabled)
                if args.regime_touchpoint_calibration_enabled is not None
                else bool(settings.regime_touchpoint_calibration_enabled)
            ),
            regime_touchpoint_self_critique_enabled=(
                bool(args.regime_touchpoint_self_critique_enabled)
                if args.regime_touchpoint_self_critique_enabled is not None
                else bool(settings.regime_touchpoint_self_critique_enabled)
            ),
            regime_touchpoint_risk_gate_enabled=(
                bool(args.regime_touchpoint_risk_gate_enabled)
                if args.regime_touchpoint_risk_gate_enabled is not None
                else bool(settings.regime_touchpoint_risk_gate_enabled)
            ),
            regime_volatility_threshold=max(
                0.0001,
                args.regime_volatility_threshold
                if args.regime_volatility_threshold is not None
                else settings.regime_volatility_threshold,
            ),
            regime_trend_spread_threshold=max(
                0.0001,
                args.regime_trend_spread_threshold
                if args.regime_trend_spread_threshold is not None
                else settings.regime_trend_spread_threshold,
            ),
            regime_persistence_bars=max(
                1,
                args.regime_persistence_bars
                if args.regime_persistence_bars is not None
                else settings.regime_persistence_bars,
            ),
            regime_ablation_mode=(
                bool(args.regime_ablation_mode)
                if args.regime_ablation_mode is not None
                else bool(settings.regime_ablation_mode)
            ),
            priority2_features_enabled=(
                bool(args.priority2_features_enabled)
                if args.priority2_features_enabled is not None
                else bool(settings.priority2_features_enabled)
            ),
            priority2_feature_columns=priority2_feature_columns,
            priority2_external_features_path=(
                Path(args.priority2_external_features_path).expanduser().resolve()
                if args.priority2_external_features_path
                else (
                    Path(settings.priority2_external_features_path).expanduser().resolve()
                    if settings.priority2_external_features_path
                    else None
                )
            ),
            priority2_quality_gate_enabled=bool(priority2_quality_gate_enabled),
            priority2_quality_min_external_raw_coverage=float(
                priority2_quality_min_external_raw_coverage
            ),
            priority2_quality_min_non_zero_coverage=float(
                priority2_quality_min_non_zero_coverage
            ),
            priority2_quality_max_fallback_rate=float(priority2_quality_max_fallback_rate),
            priority2_quality_max_staleness_seconds=float(priority2_quality_max_staleness_seconds),
            strategy_fast_window=max(2, int(args.fast_window)) if args.fast_window is not None else None,
            strategy_slow_window=max(3, int(args.slow_window)) if args.slow_window is not None else None,
            walk_forward_train_bars=max(
                50,
                args.walkforward_train_bars
                if args.walkforward_train_bars is not None
                else settings.walk_forward_train_bars,
            ),
            walk_forward_validate_bars=max(
                10,
                args.walkforward_validate_bars
                if args.walkforward_validate_bars is not None
                else settings.walk_forward_validate_bars,
            ),
            walk_forward_step_bars=max(
                10,
                args.walkforward_step_bars
                if args.walkforward_step_bars is not None
                else settings.walk_forward_step_bars,
            ),
            walk_forward_min_windows=max(
                1,
                args.walkforward_min_windows
                if args.walkforward_min_windows is not None
                else settings.walk_forward_min_windows,
            ),
            calibration_min_walkforward_sharpe=(
                args.calibration_min_walkforward_sharpe
                if args.calibration_min_walkforward_sharpe is not None
                else settings.calibration_min_walkforward_sharpe
            ),
            calibration_confidence_floor=(
                args.calibration_confidence_floor
                if args.calibration_confidence_floor is not None
                else settings.calibration_confidence_floor
            ),
            calibration_confidence_ceiling=(
                args.calibration_confidence_ceiling
                if args.calibration_confidence_ceiling is not None
                else settings.calibration_confidence_ceiling
            ),
            calibration_max_contradictions=max(
                0,
                args.calibration_max_contradictions
                if args.calibration_max_contradictions is not None
                else settings.calibration_max_contradictions,
            ),
            calibration_directional_edge_threshold=(
                args.calibration_directional_edge_threshold
                if args.calibration_directional_edge_threshold is not None
                else settings.calibration_directional_edge_threshold
            ),
            calibration_quality_penalty_strength=min(
                1.0,
                max(
                    0.0,
                    args.calibration_quality_penalty_strength
                    if args.calibration_quality_penalty_strength is not None
                    else settings.calibration_quality_penalty_strength,
                ),
            ),
            calibration_directional_contradiction_penalty=min(
                1.0,
                max(
                    0.0,
                    args.calibration_directional_contradiction_penalty
                    if args.calibration_directional_contradiction_penalty is not None
                    else settings.calibration_directional_contradiction_penalty,
                ),
            ),
            calibration_cost_pressure_penalty_strength=min(
                1.0,
                max(
                    0.0,
                    args.calibration_cost_pressure_penalty_strength
                    if args.calibration_cost_pressure_penalty_strength is not None
                    else settings.calibration_cost_pressure_penalty_strength,
                ),
            ),
            self_critique_min_score=min(
                1.0,
                max(
                    0.0,
                    args.self_critique_min_score
                    if args.self_critique_min_score is not None
                    else settings.self_critique_min_score,
                ),
            ),
            self_critique_max_findings=max(
                1,
                args.self_critique_max_findings
                if args.self_critique_max_findings is not None
                else settings.self_critique_max_findings,
            ),
            ops_report_verbosity=(
                args.ops_report_verbosity
                if args.ops_report_verbosity is not None
                else settings.ops_report_verbosity
            ),
            ensemble_mode=(
                args.ensemble_mode
                if args.ensemble_mode is not None
                else settings.ensemble_mode
            ),
            ensemble_enabled_arms=(
                _parse_ensemble_arms(args.ensemble_arms)
                if args.ensemble_arms is not None
                else tuple(settings.ensemble_enabled_arms)
            ),
            ensemble_decay_horizon=max(
                4,
                args.ensemble_decay_horizon
                if args.ensemble_decay_horizon is not None
                else settings.ensemble_decay_horizon,
            ),
            ensemble_exploration_weight=min(
                0.50,
                max(
                    0.0,
                    args.ensemble_exploration_weight
                    if args.ensemble_exploration_weight is not None
                    else settings.ensemble_exploration_weight,
                ),
            ),
            ensemble_turnover_penalty_bps=max(
                0.0,
                args.ensemble_turnover_penalty_bps
                if args.ensemble_turnover_penalty_bps is not None
                else settings.ensemble_turnover_penalty_bps,
            ),
            source_data_path=source_file,
        )
        with tracked_operation(
            settings.quant_data_root,
            operation="agent-plane",
            dimensions={"exchange": exchange, "symbol": symbol, "timeframe": timeframe},
        ) as metric:
            result = run_agent_plane(settings, config)
            metric["run_id"] = result.run_id
            metric["run_dir"] = str(result.run_dir)
            metric["risk_approved"] = result.risk_approved
            metric["intent_status"] = result.intent_status
            metric["paper_trade_execution_status"] = result.paper_trade_execution_status
            metric["paper_trade_execution_path"] = str(result.paper_trade_execution_path)
            metric["ops_report_contract"] = str(result.ops_report_contract_path)
            if result.intent_destination_path is not None:
                metric["intent_destination_path"] = str(result.intent_destination_path)
        print(
            "Agent plane complete -> "
            f"{result.run_dir} "
            f"(risk_approved={result.risk_approved} "
            f"intent={result.intent_status} "
            f"execution={result.paper_trade_execution_status})"
        )
        return

    if args.command == "paper-account-check":
        provider = args.provider or settings.paper_account_provider
        exchange_id = args.exchange or settings.paper_account_exchange
        timeout_seconds = (
            args.timeout_seconds
            if args.timeout_seconds is not None
            else settings.paper_account_timeout_seconds
        )
        sandbox = (
            args.paper_sandbox
            if args.paper_sandbox is not None
            else settings.paper_account_sandbox
        )
        tradingview_base_url = args.tradingview_base_url or settings.tradingview_base_url
        with tracked_operation(
            settings.quant_data_root,
            operation="paper-account-check",
            dimensions={
                "provider": provider,
                "exchange": exchange_id if provider == "ccxt" else "n/a",
            },
        ) as metric:
            result = run_paper_account_probe(
                provider=provider,
                timeout_seconds=max(1.0, float(timeout_seconds)),
                tradingview_base_url=tradingview_base_url,
                exchange_id=exchange_id,
                sandbox=bool(sandbox),
                api_key=settings.paper_account_api_key,
                api_secret=settings.paper_account_api_secret,
                api_passphrase=settings.paper_account_api_passphrase,
            )
            metric["ok"] = result.ok
            metric["provider"] = result.provider
            metric["message"] = result.message
            metric["details"] = result.details

        status = "PASS" if result.ok else "FAIL"
        print(f"{status} provider={result.provider} message={result.message}")
        for key, value in result.details.items():
            print(f"- {key}={value}")
        if not result.ok:
            raise SystemExit(1)
        return

    if args.command == "visualize-run":
        run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
        with tracked_operation(
            settings.quant_data_root,
            operation="visualize-run",
            dimensions={"run_dir": str(run_dir) if run_dir else "latest"},
        ) as metric:
            result = generate_run_visuals(
                quant_data_root=settings.quant_data_root,
                run_dir=run_dir,
                output_dir=output_dir,
            )
            metric["run_dir"] = str(result.run_dir)
            metric["output_dir"] = str(result.output_dir)
            metric["price_signals_path"] = str(result.price_signals_path)
            metric["equity_drawdown_path"] = str(result.equity_drawdown_path)
            metric["returns_diagnostics_path"] = str(result.returns_diagnostics_path)
            metric["buy_trigger_count"] = result.buy_trigger_count
            metric["sell_trigger_count"] = result.sell_trigger_count
        print(
            "Visuals generated -> "
            f"{result.output_dir} "
            f"(buy_triggers={result.buy_trigger_count} sell_triggers={result.sell_trigger_count})"
        )
        return

    if args.command == "monitor-triggers":
        model_path = Path(args.model_path).expanduser().resolve() if args.model_path else None
        poll_seconds = (
            args.poll_seconds
            if args.poll_seconds is not None
            else settings.trigger_monitor_poll_seconds
        )
        priority2_features_enabled = (
            bool(args.priority2_features_enabled)
            if args.priority2_features_enabled is not None
            else bool(settings.priority2_features_enabled)
        )
        priority2_external_features_path = (
            Path(args.priority2_external_features_path).expanduser().resolve()
            if args.priority2_external_features_path
            else (
                Path(settings.priority2_external_features_path).expanduser().resolve()
                if settings.priority2_external_features_path
                else None
            )
        )
        priority2_feature_columns = _parse_priority2_feature_columns(
            args.priority2_feature_columns,
            default=tuple(settings.priority2_feature_columns),
        )
        ranked_features_enabled = (
            bool(args.ranked_features_enabled)
            if args.ranked_features_enabled is not None
            else bool(settings.ranked_features_enabled)
        )
        ranked_external_features_path = (
            Path(args.ranked_external_features_path).expanduser().resolve()
            if args.ranked_external_features_path
            else (
                Path(settings.ranked_external_features_path).expanduser().resolve()
                if settings.ranked_external_features_path
                else None
            )
        )
        ranked_feature_columns = _parse_ranked_feature_columns(
            args.ranked_feature_columns,
            default=tuple(settings.ranked_feature_columns),
        )
        orderbook_features_enabled = (
            bool(args.orderbook_features_enabled)
            if args.orderbook_features_enabled is not None
            else bool(settings.orderbook_features_enabled)
        )
        orderbook_features_path = (
            Path(args.orderbook_features_path).expanduser().resolve()
            if args.orderbook_features_path
            else (
                Path(settings.orderbook_features_path).expanduser().resolve()
                if settings.orderbook_features_path
                else None
            )
        )
        orderbook_feature_columns = _parse_orderbook_feature_columns(
            args.orderbook_feature_columns,
            default=tuple(settings.orderbook_feature_columns),
        )
        confidence_threshold = (
            args.confidence_threshold
            if args.confidence_threshold is not None
            else settings.trigger_monitor_signal_confidence
        )
        webhook_url = args.webhook_url or settings.trigger_monitor_webhook_url
        notify_on_hold = bool(args.notify_on_hold or settings.trigger_monitor_notify_on_hold)
        paper_trading_enabled = (
            bool(args.paper_trading_enabled)
            if args.paper_trading_enabled is not None
            else bool(settings.trigger_monitor_paper_trading_enabled)
        )
        paper_notional_usd = max(
            0.0,
            float(
                args.paper_notional_usd
                if args.paper_notional_usd is not None
                else settings.paper_trade_notional_usd
            ),
        )
        paper_starting_cash_usd = max(
            0.0,
            float(
                args.paper_starting_cash_usd
                if args.paper_starting_cash_usd is not None
                else settings.paper_trade_starting_cash_usd
            ),
        )
        paper_fee_bps = max(
            0.0,
            float(
                args.paper_fee_bps
                if args.paper_fee_bps is not None
                else settings.paper_trade_fee_bps
            ),
        )
        paper_slippage_bps = max(
            0.0,
            float(
                args.paper_slippage_bps
                if args.paper_slippage_bps is not None
                else settings.paper_trade_slippage_bps
            ),
        )
        with tracked_operation(
            settings.quant_data_root,
            operation="monitor-triggers",
            dimensions={
                "exchange": exchange,
                "symbol": symbol,
                "timeframe": timeframe,
            },
        ) as metric:
            result = monitor_trigger_signals(
                settings=settings,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                model_path=model_path,
                limit=max(50, int(args.limit)),
                poll_seconds=max(5.0, float(poll_seconds)),
                confidence_threshold=float(confidence_threshold),
                webhook_url=webhook_url,
                notify_on_hold=notify_on_hold,
                max_cycles=args.max_cycles,
                priority2_features_enabled=bool(priority2_features_enabled),
                priority2_external_features_path=priority2_external_features_path,
                priority2_feature_columns=priority2_feature_columns,
                ranked_features_enabled=bool(ranked_features_enabled),
                ranked_external_features_path=ranked_external_features_path,
                ranked_feature_columns=ranked_feature_columns,
                orderbook_features_enabled=bool(orderbook_features_enabled),
                orderbook_features_path=orderbook_features_path,
                orderbook_feature_columns=orderbook_feature_columns,
                paper_trading_enabled=paper_trading_enabled,
                paper_notional_usd=paper_notional_usd,
                paper_starting_cash_usd=paper_starting_cash_usd,
                paper_fee_bps=paper_fee_bps,
                paper_slippage_bps=paper_slippage_bps,
            )
            metric["cycles_completed"] = result.cycles_completed
            metric["alerts_emitted"] = result.alerts_emitted
            metric["paper_trading_enabled"] = bool(paper_trading_enabled)
            metric["paper_trades_attempted"] = result.paper_trades_attempted
            metric["paper_trades_executed"] = result.paper_trades_executed
            metric["latest_alert_path"] = (
                str(result.latest_alert_path) if result.latest_alert_path else None
            )
            metric["latest_paper_execution_path"] = (
                str(result.latest_paper_execution_path)
                if result.latest_paper_execution_path
                else None
            )
            metric["state_path"] = str(result.state_path)
        print(
            "Trigger monitor complete -> "
            f"cycles={result.cycles_completed} "
            f"alerts={result.alerts_emitted} "
            f"paper_attempted={result.paper_trades_attempted} "
            f"paper_executed={result.paper_trades_executed} "
            f"state={result.state_path}"
        )
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
