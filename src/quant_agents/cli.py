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
from quant_agents.paper_account import run_paper_account_probe
from quant_agents.reporting import generate_daily_report
from quant_agents.storage import ensure_phase1_tree, latest_backtest_run_dir
from quant_agents.trigger_model import (
    monitor_trigger_signals,
    predict_trigger_signal,
    train_trigger_model,
)
from quant_agents.visualization import generate_run_visuals

logger = logging.getLogger(__name__)


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

    backtest = subparsers.add_parser("backtest", help="Run baseline SMA crossover backtest.")
    backtest.add_argument("--exchange", default=None)
    backtest.add_argument("--symbol", default=None)
    backtest.add_argument("--timeframe", default=None)
    backtest.add_argument("--fast-window", type=int, default=20)
    backtest.add_argument("--slow-window", type=int, default=50)
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
    daily.add_argument("--fast-window", type=int, default=20)
    daily.add_argument("--slow-window", type=int, default=50)
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
        "--min-signal-confidence",
        type=float,
        default=None,
        help="Deterministic risk gate: minimum strategy confidence.",
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

    trigger_predict = subparsers.add_parser(
        "predict-trigger",
        help="Generate one explainable buy/sell/hold prediction from latest market data.",
    )
    trigger_predict.add_argument("--exchange", default=None)
    trigger_predict.add_argument("--symbol", default=None)
    trigger_predict.add_argument("--timeframe", default=None)
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

    trigger_monitor = subparsers.add_parser(
        "monitor-triggers",
        help="Continuously ingest market data, run trigger predictions, and emit alerts.",
    )
    trigger_monitor.add_argument("--exchange", default=None)
    trigger_monitor.add_argument("--symbol", default=None)
    trigger_monitor.add_argument("--timeframe", default=None)
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
        "--max-cycles",
        type=int,
        default=None,
        help="Optional finite cycle cap for bounded monitor runs.",
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

    if args.command == "backtest":
        source_file = Path(args.input_file).expanduser().resolve() if args.input_file else None
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
            )
            metric["model_path"] = str(result.model_path)
            metric["run_dir"] = str(result.run_dir)
            metric["sample_count"] = result.sample_count
            metric["train_count"] = result.train_count
            metric["test_count"] = result.test_count
            metric["accuracy"] = result.accuracy
        print(
            "Trigger model training complete -> "
            f"{result.model_path} "
            f"(samples={result.sample_count} accuracy={result.accuracy:.3f})"
        )
        return

    if args.command == "predict-trigger":
        source_file = Path(args.input_file).expanduser().resolve() if args.input_file else None
        model_path = Path(args.model_path).expanduser().resolve() if args.model_path else None
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
            min_signal_confidence=(
                args.min_signal_confidence
                if args.min_signal_confidence is not None
                else settings.risk_min_signal_confidence
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
            minimum_bars=max(
                10,
                args.minimum_bars if args.minimum_bars is not None else settings.agent_minimum_bars,
            ),
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
        confidence_threshold = (
            args.confidence_threshold
            if args.confidence_threshold is not None
            else settings.trigger_monitor_signal_confidence
        )
        webhook_url = args.webhook_url or settings.trigger_monitor_webhook_url
        notify_on_hold = bool(args.notify_on_hold or settings.trigger_monitor_notify_on_hold)
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
            )
            metric["cycles_completed"] = result.cycles_completed
            metric["alerts_emitted"] = result.alerts_emitted
            metric["latest_alert_path"] = (
                str(result.latest_alert_path) if result.latest_alert_path else None
            )
            metric["state_path"] = str(result.state_path)
        print(
            "Trigger monitor complete -> "
            f"cycles={result.cycles_completed} "
            f"alerts={result.alerts_emitted} "
            f"state={result.state_path}"
        )
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
