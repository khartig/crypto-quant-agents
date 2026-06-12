#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_agents.agent_plane import AgentPlaneConfig, RiskThresholds, run_agent_plane
from quant_agents.config import load_settings
from quant_agents.storage import symbol_slug

DEFAULT_WINDOW_SPECS: tuple[tuple[str, str, str], ...] = (
    ("uptrend_2025q2", "2025-04-01T00:00:00Z", "2025-08-01T00:00:00Z"),
    ("flat_2025nov_to_2026jan", "2025-11-01T00:00:00Z", "2026-02-01T00:00:00Z"),
    ("drawdown_2026latejan_to_mar", "2026-01-25T00:00:00Z", "2026-04-01T00:00:00Z"),
    ("rebound_2026apr", "2026-04-01T00:00:00Z", "2026-05-01T00:00:00Z"),
    ("decline_2026may_to_now", "2026-05-01T00:00:00Z", "now"),
)

REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class WindowSpec:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


def _parse_timestamp(raw: str, *, now_utc: pd.Timestamp) -> pd.Timestamp:
    value = raw.strip()
    if value.lower() == "now":
        return now_utc
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _parse_window(raw: str, *, now_utc: pd.Timestamp) -> WindowSpec:
    parts = [part.strip() for part in raw.split(",", 2)]
    if len(parts) != 3 or not parts[0] or not parts[1] or not parts[2]:
        raise ValueError(
            "Invalid --window value. Expected format: label,start,end "
            "(example: uptrend,2025-04-01,2025-08-01)."
        )
    start = _parse_timestamp(parts[1], now_utc=now_utc)
    end = _parse_timestamp(parts[2], now_utc=now_utc)
    if not start < end:
        raise ValueError(
            f"Invalid window range for `{parts[0]}`: start must be < end (start={start}, end={end})."
        )
    return WindowSpec(name=parts[0], start=start, end=end)


def _default_windows(now_utc: pd.Timestamp) -> list[WindowSpec]:
    windows: list[WindowSpec] = []
    for name, start_raw, end_raw in DEFAULT_WINDOW_SPECS:
        windows.append(
            WindowSpec(
                name=name,
                start=_parse_timestamp(start_raw, now_utc=now_utc),
                end=_parse_timestamp(end_raw, now_utc=now_utc),
            )
        )
    return windows


def _new_eval_root(root: Path) -> Path:
    now = datetime.now(timezone.utc)
    base = root / "logs" / "analysis" / "regime-window-evals" / f"{now:%Y-%m-%d}"
    base.mkdir(parents=True, exist_ok=True)
    base_id = f"{now:%Y%m%dT%H%M%SZ}"
    for suffix in range(100):
        run_id = base_id if suffix == 0 else f"{base_id}_{suffix:02d}"
        candidate = base / run_id
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise RuntimeError(f"Unable to allocate evaluation output directory under {base}")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_rate(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _delta(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return float(lhs) - float(rhs)


def _mean_optional(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def _collect_scope_parquet_files(root: Path, exchange: str, symbol: str, timeframe: str) -> list[Path]:
    base = (
        root
        / "raw"
        / f"exchange={exchange}"
        / f"symbol={symbol_slug(symbol)}"
        / f"interval={timeframe}"
    )
    if not base.exists():
        raise FileNotFoundError(f"No raw dataset scope found: {base}")
    files = sorted(base.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {base}")
    return files


def _load_market_frame(*, input_file: Path | None, data_root: Path, exchange: str, symbol: str, timeframe: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if input_file is not None:
        candidates = [input_file]
    else:
        candidates = _collect_scope_parquet_files(
            root=data_root,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
        )

    for path in candidates:
        frame = pd.read_parquet(path, columns=list(REQUIRED_COLUMNS))
        frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"])
        frames.append(frame)

    if not frames:
        raise RuntimeError("No market rows were loaded from input parquet source(s).")

    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    merged = merged.reset_index(drop=True)
    return merged


def _slice_window(frame: pd.DataFrame, window: WindowSpec) -> pd.DataFrame:
    scoped = frame.loc[(frame["timestamp"] >= window.start) & (frame["timestamp"] < window.end)].copy()
    scoped = scoped.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    scoped = scoped.reset_index(drop=True)
    return scoped


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run segmented agent-plane evaluation across date-range windows and compare "
            "regime-enabled vs regime-ablated profiles."
        )
    )
    parser.add_argument("--exchange", default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument(
        "--input-file",
        default=None,
        help="Optional explicit parquet file. If omitted, all scope raw parquet files are merged.",
    )
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        help=(
            "Repeatable window in format label,start,end where start/end are ISO timestamps or `now`. "
            "Example: --window uptrend,2025-04-01,2025-08-01"
        ),
    )
    parser.add_argument(
        "--minimum-bars",
        type=int,
        default=None,
        help="Minimum bars required per window; defaults to AGENT_MINIMUM_BARS.",
    )
    parser.add_argument(
        "--step-retries",
        type=int,
        default=0,
        help="Retries per agent-plane step (default: 0 for deterministic segmented evaluation speed).",
    )
    parser.add_argument(
        "--strategy-model",
        default=None,
        help="Strategy model override; defaults to OLLAMA_STRATEGY_MODEL.",
    )
    parser.add_argument(
        "--ops-model",
        default=None,
        help="Ops model override; defaults to OLLAMA_OPS_MODEL.",
    )
    parser.add_argument(
        "--regime-min-confidence",
        type=float,
        default=None,
        help="Regime-enabled profile minimum regime confidence; defaults to RISK_MIN_REGIME_CONFIDENCE.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output summary JSON path; defaults under QUANT_DATA_ROOT/logs/analysis/regime-window-evals.",
    )
    parser.add_argument(
        "--fail-on-insufficient-window",
        action="store_true",
        help="Fail immediately when a window has fewer than minimum bars (default behavior is to skip and continue).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    settings = load_settings()
    exchange = args.exchange or settings.default_exchange
    symbol = args.symbol or settings.default_symbol
    timeframe = args.timeframe or settings.default_timeframe
    now_utc = pd.Timestamp.now(tz="UTC")

    windows = (
        [_parse_window(raw, now_utc=now_utc) for raw in args.window]
        if args.window
        else _default_windows(now_utc=now_utc)
    )
    windows = sorted(windows, key=lambda item: item.start)

    source_file = Path(args.input_file).expanduser().resolve() if args.input_file else None
    market_frame = _load_market_frame(
        input_file=source_file,
        data_root=settings.quant_data_root,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
    )

    evaluation_root = _new_eval_root(settings.quant_data_root)
    input_dir = evaluation_root / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    sandbox_root = evaluation_root / "sandbox"
    sandbox_root.mkdir(parents=True, exist_ok=True)
    eval_settings = replace(
        settings,
        quant_data_root=sandbox_root,
        allow_unmounted_data_root=True,
    )

    minimum_bars = max(10, int(args.minimum_bars if args.minimum_bars is not None else settings.agent_minimum_bars))
    step_retries = max(0, int(args.step_retries))
    regime_enabled_min_conf = float(
        max(
            0.0,
            min(
                1.0,
                args.regime_min_confidence
                if args.regime_min_confidence is not None
                else settings.risk_min_regime_confidence,
            ),
        )
    )
    profiles: tuple[tuple[str, float, bool], ...] = (
        ("regime_enabled", regime_enabled_min_conf, False),
        ("regime_ablated", regime_enabled_min_conf, True),
    )

    window_inputs: dict[str, Path] = {}
    window_stats: dict[str, dict[str, Any]] = {}
    skipped_windows: list[dict[str, Any]] = []
    runnable_windows: list[WindowSpec] = []
    for window in windows:
        scoped = _slice_window(market_frame, window)
        if len(scoped) < minimum_bars:
            message = (
                f"Window `{window.name}` has insufficient bars ({len(scoped)} < {minimum_bars}). "
                "Provide broader ranges or a larger source dataset."
            )
            if args.fail_on_insufficient_window:
                raise RuntimeError(message)
            skipped_windows.append(
                {
                    "name": window.name,
                    "start_utc": window.start.isoformat(),
                    "end_utc": window.end.isoformat(),
                    "available_bars": int(len(scoped)),
                    "required_minimum_bars": minimum_bars,
                    "reason": "insufficient_bars",
                }
            )
            print(
                "WINDOW_SKIPPED "
                + json.dumps(
                    {
                        "window": window.name,
                        "available_bars": int(len(scoped)),
                        "required_minimum_bars": minimum_bars,
                    },
                    sort_keys=True,
                )
            )
            continue
        target = input_dir / f"{window.name}.parquet"
        scoped.to_parquet(target, index=False)
        window_inputs[window.name] = target
        window_stats[window.name] = {
            "bar_count": int(len(scoped)),
            "data_start_utc": pd.Timestamp(scoped["timestamp"].iloc[0]).isoformat(),
            "data_end_utc": pd.Timestamp(scoped["timestamp"].iloc[-1]).isoformat(),
        }
        runnable_windows.append(window)

    if not runnable_windows:
        raise RuntimeError(
            "No runnable windows remain after applying minimum bar requirements. "
            "Provide broader ranges, reduce --minimum-bars, or use a dataset with longer history."
        )

    results: list[dict[str, Any]] = []
    for window in runnable_windows:
        source_path = window_inputs[window.name]
        for profile_name, min_regime_conf, regime_ablation_mode in profiles:
            thresholds = RiskThresholds(
                min_total_return=settings.risk_min_total_return,
                min_sharpe=settings.risk_min_sharpe,
                max_drawdown=settings.risk_max_drawdown,
                max_cost_return_drag=settings.risk_max_cost_return_drag,
                min_signal_confidence=settings.risk_min_signal_confidence,
                min_walkforward_quality_score=settings.risk_min_walkforward_quality_score,
                min_regime_confidence=min_regime_conf,
            )
            config = AgentPlaneConfig(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                strategy_model=args.strategy_model or settings.ollama_strategy_model,
                ops_model=args.ops_model or settings.ollama_ops_model,
                step_retries=step_retries,
                thresholds=thresholds,
                backtest_fee_bps=settings.backtest_fee_bps,
                backtest_slippage_bps=settings.backtest_slippage_bps,
                walk_forward_fee_bps=settings.walk_forward_fee_bps,
                walk_forward_slippage_bps=settings.walk_forward_slippage_bps,
                paper_notional_usd=settings.paper_trade_notional_usd,
                paper_starting_cash_usd=settings.paper_trade_starting_cash_usd,
                paper_fee_bps=settings.paper_trade_fee_bps,
                paper_slippage_bps=settings.paper_trade_slippage_bps,
                minimum_bars=minimum_bars,
                regime_detector_mode=settings.regime_detector_mode,
                regime_volatility_threshold=settings.regime_volatility_threshold,
                regime_trend_spread_threshold=settings.regime_trend_spread_threshold,
                regime_persistence_bars=settings.regime_persistence_bars,
                regime_ablation_mode=bool(regime_ablation_mode),
                walk_forward_train_bars=settings.walk_forward_train_bars,
                walk_forward_validate_bars=settings.walk_forward_validate_bars,
                walk_forward_step_bars=settings.walk_forward_step_bars,
                walk_forward_min_windows=settings.walk_forward_min_windows,
                calibration_min_walkforward_sharpe=settings.calibration_min_walkforward_sharpe,
                calibration_confidence_floor=settings.calibration_confidence_floor,
                calibration_confidence_ceiling=settings.calibration_confidence_ceiling,
                calibration_max_contradictions=settings.calibration_max_contradictions,
                self_critique_min_score=settings.self_critique_min_score,
                self_critique_max_findings=settings.self_critique_max_findings,
                ops_report_verbosity=settings.ops_report_verbosity,
                ensemble_mode=settings.ensemble_mode,
                ensemble_enabled_arms=settings.ensemble_enabled_arms,
                ensemble_decay_horizon=settings.ensemble_decay_horizon,
                ensemble_exploration_weight=settings.ensemble_exploration_weight,
                ensemble_turnover_penalty_bps=settings.ensemble_turnover_penalty_bps,
                source_data_path=source_path,
            )

            run_result = run_agent_plane(eval_settings, config)
            risk = json.loads(run_result.risk_decision_path.read_text(encoding="utf-8"))
            calibration = json.loads(run_result.confidence_calibration_path.read_text(encoding="utf-8"))
            backtest = json.loads(run_result.backtest_evaluation_path.read_text(encoding="utf-8"))

            row = {
                "profile": profile_name,
                "window": window.name,
                "window_start_utc": window.start.isoformat(),
                "window_end_utc": window.end.isoformat(),
                "bar_count": window_stats[window.name]["bar_count"],
                "regime_ablation_mode": bool(regime_ablation_mode),
                "data_start_utc": window_stats[window.name]["data_start_utc"],
                "data_end_utc": window_stats[window.name]["data_end_utc"],
                "risk_approved": bool(risk.get("approved", False)),
                "approval_rate": 1.0 if bool(risk.get("approved", False)) else 0.0,
                "contradiction_detected": bool(calibration.get("contradiction_detected", False)),
                "contradiction_rate": 1.0 if bool(calibration.get("contradiction_detected", False)) else 0.0,
                "reason_codes": [str(code) for code in risk.get("reason_codes", [])],
                "net_total_return": _safe_float(backtest.get("total_return")),
                "sharpe": _safe_float(backtest.get("sharpe")),
                "max_drawdown": _safe_float(backtest.get("max_drawdown")),
                "total_cost_return_drag": _safe_float(backtest.get("total_cost_return_drag")),
                "agent_run_id": run_result.run_id,
                "agent_run_dir": str(run_result.run_dir),
                "artifacts": {
                    "backtest_evaluation": str(run_result.backtest_evaluation_path),
                    "confidence_calibration": str(run_result.confidence_calibration_path),
                    "risk_decision": str(run_result.risk_decision_path),
                    "run_manifest": str(run_result.run_manifest_path),
                },
            }
            results.append(row)

            print(
                "WINDOW_PROFILE_RESULT "
                + json.dumps(
                    {
                        "window": row["window"],
                        "profile": row["profile"],
                        "approval_rate": row["approval_rate"],
                        "contradiction_rate": row["contradiction_rate"],
                        "net_total_return": row["net_total_return"],
                        "sharpe": row["sharpe"],
                        "max_drawdown": row["max_drawdown"],
                        "total_cost_return_drag": row["total_cost_return_drag"],
                    },
                    sort_keys=True,
                )
            )

    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_window: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in results:
        by_profile[row["profile"]].append(row)
        by_window[row["window"]][row["profile"]] = row

    profile_summary: dict[str, dict[str, Any]] = {}
    for profile_name, rows in by_profile.items():
        reason_counts: Counter[str] = Counter()
        for row in rows:
            reason_counts.update(row["reason_codes"])
        profile_summary[profile_name] = {
            "windows_evaluated": len(rows),
            "approval_rate": _safe_rate(sum(1.0 for row in rows if row["risk_approved"]), len(rows)),
            "contradiction_rate": _safe_rate(
                sum(1.0 for row in rows if row["contradiction_detected"]),
                len(rows),
            ),
            "mean_net_total_return": _mean_optional([_safe_float(row["net_total_return"]) for row in rows]),
            "mean_sharpe": _mean_optional([_safe_float(row["sharpe"]) for row in rows]),
            "mean_max_drawdown": _mean_optional([_safe_float(row["max_drawdown"]) for row in rows]),
            "mean_total_cost_return_drag": _mean_optional(
                [_safe_float(row["total_cost_return_drag"]) for row in rows]
            ),
            "reason_code_distribution": dict(
                sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
        }

    window_comparison: list[dict[str, Any]] = []
    for window in runnable_windows:
        reg_row = by_window[window.name].get("regime_enabled")
        abl_row = by_window[window.name].get("regime_ablated")
        if reg_row is None or abl_row is None:
            continue
        window_comparison.append(
            {
                "window": window.name,
                "window_start_utc": window.start.isoformat(),
                "window_end_utc": window.end.isoformat(),
                "bar_count": window_stats[window.name]["bar_count"],
                "regime_enabled": {
                    "approval_rate": reg_row["approval_rate"],
                    "contradiction_rate": reg_row["contradiction_rate"],
                    "reason_codes": reg_row["reason_codes"],
                    "net_total_return": reg_row["net_total_return"],
                    "sharpe": reg_row["sharpe"],
                    "max_drawdown": reg_row["max_drawdown"],
                    "total_cost_return_drag": reg_row["total_cost_return_drag"],
                },
                "regime_ablated": {
                    "approval_rate": abl_row["approval_rate"],
                    "contradiction_rate": abl_row["contradiction_rate"],
                    "reason_codes": abl_row["reason_codes"],
                    "net_total_return": abl_row["net_total_return"],
                    "sharpe": abl_row["sharpe"],
                    "max_drawdown": abl_row["max_drawdown"],
                    "total_cost_return_drag": abl_row["total_cost_return_drag"],
                },
                "delta_regime_minus_ablated": {
                    "approval_rate": _delta(
                        _safe_float(reg_row["approval_rate"]),
                        _safe_float(abl_row["approval_rate"]),
                    ),
                    "contradiction_rate": _delta(
                        _safe_float(reg_row["contradiction_rate"]),
                        _safe_float(abl_row["contradiction_rate"]),
                    ),
                    "net_total_return": _delta(
                        _safe_float(reg_row["net_total_return"]),
                        _safe_float(abl_row["net_total_return"]),
                    ),
                    "sharpe": _delta(_safe_float(reg_row["sharpe"]), _safe_float(abl_row["sharpe"])),
                    "max_drawdown": _delta(
                        _safe_float(reg_row["max_drawdown"]),
                        _safe_float(abl_row["max_drawdown"]),
                    ),
                    "total_cost_return_drag": _delta(
                        _safe_float(reg_row["total_cost_return_drag"]),
                        _safe_float(abl_row["total_cost_return_drag"]),
                    ),
                },
            }
        )

    output = {
        "contract": "segmented_regime_window_evaluation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "input_file": str(source_file) if source_file else None,
            "source_mode": "single_input_file" if source_file else "merged_scope_raw_files",
        },
        "evaluation_root": str(evaluation_root),
        "sandbox_root": str(sandbox_root),
        "profiles": {
            "regime_enabled": {"min_regime_confidence": regime_enabled_min_conf},
            "regime_ablated": {
                "min_regime_confidence": regime_enabled_min_conf,
                "regime_ablation_mode": True,
            },
        },
        "windows": [
            {
                "name": window.name,
                "start_utc": window.start.isoformat(),
                "end_utc": window.end.isoformat(),
                **window_stats[window.name],
            }
            for window in runnable_windows
        ],
        "skipped_windows": skipped_windows,
        "results": results,
        "window_comparison": window_comparison,
        "profile_summary": profile_summary,
    }

    output_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else (evaluation_root / "summary.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("SEGMENTED_REGIME_WINDOW_EVALUATION_STATUS=PASS")
    print(f"SEGMENTED_REGIME_WINDOW_EVALUATION_OUTPUT={output_path}")
    print("PROFILE_SUMMARY")
    print(json.dumps(profile_summary, sort_keys=True))


if __name__ == "__main__":
    main()
