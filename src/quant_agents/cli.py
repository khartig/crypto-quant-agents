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
from quant_agents.reporting import generate_daily_report
from quant_agents.storage import ensure_phase1_tree, latest_backtest_run_dir
from quant_agents.visualization import generate_run_visuals

logger = logging.getLogger(__name__)


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-phase1",
        description="Phase 1 deterministic crypto quant pipeline.",
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

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
