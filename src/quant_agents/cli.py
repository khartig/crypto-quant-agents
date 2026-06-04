from __future__ import annotations

import argparse
import logging

from quant_agents.backtest import run_sma_backtest
from quant_agents.config import ensure_data_root_ready, load_settings
from quant_agents.ingestion import fetch_ohlcv_to_parquet
from quant_agents.logging_utils import configure_logging
from quant_agents.reporting import generate_daily_report
from quant_agents.storage import ensure_phase1_tree

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

    report = subparsers.add_parser("report", help="Generate daily markdown operations report.")
    report.add_argument("--exchange", default=None)
    report.add_argument("--symbol", default=None)
    report.add_argument("--timeframe", default=None)

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

    return parser


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    settings = load_settings()
    ensure_data_root_ready(settings.quant_data_root)
    ensure_phase1_tree(settings.quant_data_root)

    parser = _base_parser()
    args = parser.parse_args(argv)

    exchange = args.exchange or settings.default_exchange
    symbol = args.symbol or settings.default_symbol
    timeframe = args.timeframe or settings.default_timeframe

    if args.command == "ingest":
        result = fetch_ohlcv_to_parquet(settings, exchange, symbol, timeframe, limit=args.limit)
        print(f"Ingested {result.row_count} rows -> {result.output_path}")
        return

    if args.command == "backtest":
        result = run_sma_backtest(
            settings=settings,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            fast_window=args.fast_window,
            slow_window=args.slow_window,
        )
        print(f"Backtest complete -> {result.run_dir}")
        return

    if args.command == "report":
        result = generate_daily_report(settings, exchange, symbol, timeframe)
        print(f"Report written -> {result.report_path}")
        return

    if args.command == "run-daily":
        logger.info("Running daily pipeline ingest -> backtest -> report")
        ingest_result = fetch_ohlcv_to_parquet(settings, exchange, symbol, timeframe, limit=args.limit)
        logger.info("Ingest complete at %s", ingest_result.output_path)

        backtest_result = run_sma_backtest(
            settings=settings,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            fast_window=args.fast_window,
            slow_window=args.slow_window,
        )
        logger.info("Backtest complete at %s", backtest_result.run_dir)

        report_result = generate_daily_report(settings, exchange, symbol, timeframe)
        logger.info("Report complete at %s", report_result.report_path)
        print(f"Daily pipeline complete -> {report_result.report_path}")
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

