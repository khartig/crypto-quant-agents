#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

QUANT_DATA_ROOT = Path("/mnt/quant-data")
ALERTS_GLOB = QUANT_DATA_ROOT / "logs" / "agents" / "trigger-monitor" / "*" / "alerts.jsonl"
PORTFOLIO_STATE_PATH = QUANT_DATA_ROOT / "paper-trading" / "state" / "portfolio_state.json"
MONITOR_LOG_PATH = Path("/home/kevin/crypto-quant-agents/logs/live-monitor/path_a_monitor.log")
STATE_PATH = Path("/home/kevin/crypto-quant-agents/logs/live-monitor/path_a_reporter_state.json")
SUMMARY_INTERVAL_SECONDS = 2 * 60 * 60
POLL_SECONDS = 5.0

COUNTS = {"buy": 0, "sell": 0, "hold": 0}
SEEN_TRADE_KEYS: set[str] = set()
ALERT_FILE_OFFSETS: dict[str, int] = {}
MONITOR_LOG_OFFSET = 0
CYCLE_RECOMMENDATIONS: dict[int, str] = {}

LAST_CLOSE_PRICE: float | None = None
START_EPOCH = time.time()
NEXT_SUMMARY_EPOCH = START_EPOCH + SUMMARY_INTERVAL_SECONDS
RESET_SESSION = os.getenv("PATH_A_REPORTER_RESET_SESSION", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

NO_ALERT_RECOMMENDATION_RE = re.compile(
    r"cycle=(\d+)\s+no alert recommendation=(buy|sell|hold)",
    re.IGNORECASE,
)
ALERT_RECOMMENDATION_RE = re.compile(r"\[alert\]\s+cycle=(\d+)\s+(BUY|SELL|HOLD)\b")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _write_state(reason: str) -> None:
    payload = {
        "updated_at_utc": _utc_now(),
        "start_epoch": START_EPOCH,
        "start_at_utc": datetime.fromtimestamp(START_EPOCH, tz=timezone.utc).isoformat(),
        "next_summary_epoch": NEXT_SUMMARY_EPOCH,
        "reason": reason,
        "reset_session_mode": RESET_SESSION,
        "counts": dict(COUNTS),
        "tracked_cycles": len(CYCLE_RECOMMENDATIONS),
        "seen_trade_events": len(SEEN_TRADE_KEYS),
        "monitor_log_offset": MONITOR_LOG_OFFSET,
        "alert_file_offsets": dict(ALERT_FILE_OFFSETS),
    }
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        return


def _read_portfolio_stats() -> dict[str, float]:
    state: dict[str, Any] = {}
    if PORTFOLIO_STATE_PATH.exists():
        try:
            parsed = json.loads(PORTFOLIO_STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                state = parsed
        except (json.JSONDecodeError, OSError):
            state = {}

    starting_cash = _to_float(state.get("starting_cash_usd"), 0.0)
    cash = _to_float(state.get("cash_usd"), starting_cash)
    holdings_value = 0.0
    realized_pnl = 0.0
    positions = state.get("positions")
    if isinstance(positions, dict):
        for raw_position in positions.values():
            if not isinstance(raw_position, dict):
                continue
            qty = _to_float(raw_position.get("quantity"), 0.0)
            avg_entry = _to_float(raw_position.get("avg_entry_price"), 0.0)
            realized_pnl += _to_float(raw_position.get("realized_pnl_usd"), 0.0)
            mark_price = LAST_CLOSE_PRICE if LAST_CLOSE_PRICE is not None else avg_entry
            holdings_value += qty * mark_price

    equity = cash + holdings_value
    pnl = equity - starting_cash
    return {
        "starting_cash_usd": starting_cash,
        "cash_usd": cash,
        "realized_pnl_usd": realized_pnl,
        "equity_usd": equity,
        "pnl_usd": pnl,
    }


def _emit_summary() -> None:
    stats = _read_portfolio_stats()
    elapsed_seconds = int(max(0.0, time.time() - START_EPOCH))
    print(
        "[summary] "
        f"ts={_utc_now()} "
        f"elapsed_sec={elapsed_seconds} "
        f"buy={COUNTS['buy']} "
        f"sell={COUNTS['sell']} "
        f"hold={COUNTS['hold']} "
        f"pnl_usd={stats['pnl_usd']:.2f} "
        f"equity_usd={stats['equity_usd']:.2f} "
        f"realized_pnl_usd={stats['realized_pnl_usd']:.2f} "
        f"cash_usd={stats['cash_usd']:.2f}",
        flush=True,
    )
    _write_state(reason="summary")


def _ingest_cycle_recommendation(
    cycle: int,
    recommendation: str,
    *,
    emit_state_update: bool = True,
) -> None:
    normalized = recommendation.strip().lower()
    if normalized not in COUNTS:
        return
    if cycle in CYCLE_RECOMMENDATIONS:
        return
    CYCLE_RECOMMENDATIONS[cycle] = normalized
    COUNTS[normalized] += 1
    if emit_state_update:
        _write_state(reason="cycle_increment")


def _ingest_cycle_from_alert_payload(
    payload: dict[str, Any],
    *,
    emit_state_update: bool = True,
) -> None:
    cycle_raw = payload.get("cycle")
    recommendation_raw = payload.get("recommendation")
    if cycle_raw is None or recommendation_raw is None:
        return
    try:
        cycle = int(cycle_raw)
    except (TypeError, ValueError):
        return
    _ingest_cycle_recommendation(
        cycle=cycle,
        recommendation=str(recommendation_raw),
        emit_state_update=emit_state_update,
    )


def _poll_monitor_log_once() -> None:
    global MONITOR_LOG_OFFSET

    if not MONITOR_LOG_PATH.exists():
        return

    try:
        with MONITOR_LOG_PATH.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(MONITOR_LOG_OFFSET)
            for line in handle:
                no_alert_match = NO_ALERT_RECOMMENDATION_RE.search(line)
                if no_alert_match:
                    cycle = int(no_alert_match.group(1))
                    recommendation = no_alert_match.group(2)
                    _ingest_cycle_recommendation(cycle=cycle, recommendation=recommendation)
                    continue

                alert_match = ALERT_RECOMMENDATION_RE.search(line)
                if alert_match:
                    cycle = int(alert_match.group(1))
                    recommendation = alert_match.group(2)
                    _ingest_cycle_recommendation(cycle=cycle, recommendation=recommendation)
            MONITOR_LOG_OFFSET = handle.tell()
    except OSError:
        return


def _process_alert(payload: dict[str, Any]) -> None:
    global LAST_CLOSE_PRICE
    _ingest_cycle_from_alert_payload(payload)

    close_price = payload.get("close_price")
    if close_price is not None:
        LAST_CLOSE_PRICE = _to_float(close_price, LAST_CLOSE_PRICE or 0.0)

    recommendation = str(payload.get("recommendation", "")).strip().lower()
    execution_status = str(payload.get("paper_trade_execution_status", "")).strip().lower()
    executed_action = str(payload.get("paper_trade_executed_action", recommendation)).strip().lower()
    if execution_status != "executed" or executed_action not in {"buy", "sell"}:
        return

    trade_key = str(
        payload.get("paper_trade_execution_record_path")
        or f"{payload.get('prediction_timestamp_utc')}|{payload.get('cycle')}|{executed_action}"
    )
    if trade_key in SEEN_TRADE_KEYS:
        return
    SEEN_TRADE_KEYS.add(trade_key)

    timestamp = str(payload.get("prediction_timestamp_utc") or payload.get("created_at_utc") or _utc_now())
    notional_usd = _to_float(payload.get("paper_trade_executed_notional_usd"), 0.0)
    confidence = _to_float(payload.get("confidence"), 0.0)
    stats = _read_portfolio_stats()
    print(
        "[trade] "
        f"ts={timestamp} "
        f"action={executed_action.upper()} "
        f"notional_usd={notional_usd:.2f} "
        f"confidence={confidence:.3f} "
        f"pnl_usd={stats['pnl_usd']:.2f} "
        f"equity_usd={stats['equity_usd']:.2f}",
        flush=True,
    )
    _write_state(reason="trade_event")


def _initialize_offsets() -> None:
    global MONITOR_LOG_OFFSET

    for file_path in sorted(glob.glob(str(ALERTS_GLOB))):
        try:
            with Path(file_path).open("r", encoding="utf-8") as handle:
                if not RESET_SESSION:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(payload, dict):
                            _ingest_cycle_from_alert_payload(
                                payload,
                                emit_state_update=False,
                            )
                handle.seek(0, 2)
                ALERT_FILE_OFFSETS[file_path] = handle.tell()
        except OSError:
            continue

    if MONITOR_LOG_PATH.exists():
        try:
            with MONITOR_LOG_PATH.open("r", encoding="utf-8", errors="replace") as handle:
                if not RESET_SESSION:
                    for line in handle:
                        no_alert_match = NO_ALERT_RECOMMENDATION_RE.search(line)
                        if no_alert_match:
                            cycle = int(no_alert_match.group(1))
                            recommendation = no_alert_match.group(2)
                            _ingest_cycle_recommendation(
                                cycle=cycle,
                                recommendation=recommendation,
                                emit_state_update=False,
                            )
                            continue

                        alert_match = ALERT_RECOMMENDATION_RE.search(line)
                        if alert_match:
                            cycle = int(alert_match.group(1))
                            recommendation = alert_match.group(2)
                            _ingest_cycle_recommendation(
                                cycle=cycle,
                                recommendation=recommendation,
                                emit_state_update=False,
                            )
                handle.seek(0, 2)
                MONITOR_LOG_OFFSET = handle.tell()
        except OSError:
            MONITOR_LOG_OFFSET = 0


def _poll_alerts_once() -> None:
    for file_path in sorted(glob.glob(str(ALERTS_GLOB))):
        path = Path(file_path)
        previous_offset = ALERT_FILE_OFFSETS.get(file_path, 0)
        try:
            with path.open("r", encoding="utf-8") as handle:
                handle.seek(previous_offset)
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        _process_alert(payload)
                ALERT_FILE_OFFSETS[file_path] = handle.tell()
        except OSError:
            ALERT_FILE_OFFSETS.pop(file_path, None)


def main() -> None:
    global NEXT_SUMMARY_EPOCH

    print(
        "[reporter] "
        f"started_at={_utc_now()} "
        f"summary_interval_seconds={SUMMARY_INTERVAL_SECONDS} "
        f"event_filter=executed_buy_sell_only reset_session_mode={RESET_SESSION}",
        flush=True,
    )
    _initialize_offsets()
    _write_state(reason="startup")
    _emit_summary()

    while True:
        _poll_monitor_log_once()
        _poll_alerts_once()
        now = time.time()
        if now >= NEXT_SUMMARY_EPOCH:
            _emit_summary()
            while NEXT_SUMMARY_EPOCH <= now:
                NEXT_SUMMARY_EPOCH += SUMMARY_INTERVAL_SECONDS
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
