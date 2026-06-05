from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_agents.agent_contracts import (
    PaperTradeExecution,
    PaperTradeIntent,
    Recommendation,
    write_contract,
)

ROUNDING_DECIMALS = 12


def _round(value: float) -> float:
    return float(round(value, ROUNDING_DECIMALS))


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_state(state_path: Path, *, starting_cash_usd: float, fee_bps: float) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if state_path.exists():
        try:
            parsed = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                payload = parsed
        except (OSError, json.JSONDecodeError):
            payload = {}

    positions_payload = payload.get("positions")
    positions: dict[str, dict[str, float]] = {}
    if isinstance(positions_payload, dict):
        for symbol, raw_position in positions_payload.items():
            if not isinstance(symbol, str) or not isinstance(raw_position, dict):
                continue
            positions[symbol] = {
                "quantity": _coerce_float(raw_position.get("quantity"), 0.0),
                "avg_entry_price": _coerce_float(raw_position.get("avg_entry_price"), 0.0),
                "realized_pnl_usd": _coerce_float(raw_position.get("realized_pnl_usd"), 0.0),
            }

    return {
        "contract": "paper_portfolio_state.v1",
        "updated_at_utc": str(payload.get("updated_at_utc") or datetime.now(timezone.utc).isoformat()),
        "starting_cash_usd": _coerce_float(payload.get("starting_cash_usd"), starting_cash_usd),
        "cash_usd": _coerce_float(payload.get("cash_usd"), starting_cash_usd),
        "fee_bps": _coerce_float(payload.get("fee_bps"), fee_bps),
        "positions": positions,
    }


def _ensure_position(state: dict[str, Any], symbol: str) -> dict[str, float]:
    positions = state.setdefault("positions", {})
    if symbol not in positions or not isinstance(positions[symbol], dict):
        positions[symbol] = {
            "quantity": 0.0,
            "avg_entry_price": 0.0,
            "realized_pnl_usd": 0.0,
        }
    position = positions[symbol]
    return {
        "quantity": _coerce_float(position.get("quantity"), 0.0),
        "avg_entry_price": _coerce_float(position.get("avg_entry_price"), 0.0),
        "realized_pnl_usd": _coerce_float(position.get("realized_pnl_usd"), 0.0),
    }


def _write_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    rounded_state = {
        "contract": str(state.get("contract") or "paper_portfolio_state.v1"),
        "updated_at_utc": str(state.get("updated_at_utc")),
        "starting_cash_usd": _round(_coerce_float(state.get("starting_cash_usd"), 0.0)),
        "cash_usd": _round(_coerce_float(state.get("cash_usd"), 0.0)),
        "fee_bps": _round(_coerce_float(state.get("fee_bps"), 0.0)),
        "positions": {},
    }
    for symbol, raw_position in dict(state.get("positions", {})).items():
        if not isinstance(symbol, str) or not isinstance(raw_position, dict):
            continue
        rounded_state["positions"][symbol] = {
            "quantity": _round(_coerce_float(raw_position.get("quantity"), 0.0)),
            "avg_entry_price": _round(_coerce_float(raw_position.get("avg_entry_price"), 0.0)),
            "realized_pnl_usd": _round(_coerce_float(raw_position.get("realized_pnl_usd"), 0.0)),
        }

    state_path.write_text(json.dumps(rounded_state, indent=2), encoding="utf-8")


def _append_fill_record(fills_log_path: Path, payload: dict[str, Any]) -> None:
    fills_log_path.parent.mkdir(parents=True, exist_ok=True)
    with fills_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def execute_paper_trade_intent(
    *,
    quant_data_root: Path,
    run_id: str,
    created_at_utc: str,
    exchange: str,
    symbol: str,
    timeframe: str,
    intent: PaperTradeIntent,
    mark_price: float | None,
    starting_cash_usd: float,
    fee_bps: float,
) -> PaperTradeExecution:
    try:
        created_at = datetime.fromisoformat(created_at_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        created_at = datetime.now(timezone.utc)

    day_dir = quant_data_root / "paper-trading" / f"{created_at:%Y-%m-%d}"
    state_path = quant_data_root / "paper-trading" / "state" / "portfolio_state.json"
    fills_log_path = day_dir / "fills.jsonl"
    execution_record_path = day_dir / f"paper_trade_execution_{run_id}.json"

    state = _load_state(
        state_path,
        starting_cash_usd=max(0.0, float(starting_cash_usd)),
        fee_bps=max(0.0, float(fee_bps)),
    )
    position = _ensure_position(state, symbol)
    fee_rate = max(0.0, float(fee_bps)) / 10_000.0
    requested_notional_usd = max(0.0, float(intent.notional_usd))

    execution_status = "skipped"
    executed_action: Recommendation = "hold"
    executed_notional_usd = 0.0
    executed_quantity = 0.0
    fee_usd = 0.0
    realized_pnl_delta_usd = 0.0
    reason = "intent_not_emitted"

    valid_price = mark_price is not None and mark_price > 0

    if intent.status != "emitted":
        reason = f"intent_{intent.status}"
    elif intent.action == "buy":
        if not valid_price:
            execution_status = "rejected"
            reason = "invalid_mark_price"
        else:
            cash_before = _coerce_float(state.get("cash_usd"), 0.0)
            max_notional = cash_before / (1.0 + fee_rate)
            executed_notional_usd = min(requested_notional_usd, max(0.0, max_notional))
            if executed_notional_usd <= 0:
                execution_status = "rejected"
                reason = "insufficient_cash"
            else:
                execution_status = "executed"
                executed_action = "buy"
                reason = "executed_buy"
                executed_quantity = executed_notional_usd / float(mark_price)
                fee_usd = executed_notional_usd * fee_rate
                state["cash_usd"] = cash_before - executed_notional_usd - fee_usd
                old_qty = _coerce_float(position.get("quantity"), 0.0)
                old_avg = _coerce_float(position.get("avg_entry_price"), 0.0)
                new_qty = old_qty + executed_quantity
                new_avg = (
                    ((old_qty * old_avg) + (executed_quantity * float(mark_price))) / new_qty
                    if new_qty > 0
                    else 0.0
                )
                position["quantity"] = new_qty
                position["avg_entry_price"] = new_avg
    elif intent.action == "sell":
        if not valid_price:
            execution_status = "rejected"
            reason = "invalid_mark_price"
        else:
            available_qty = max(0.0, _coerce_float(position.get("quantity"), 0.0))
            requested_qty = requested_notional_usd / float(mark_price)
            executed_quantity = min(requested_qty, available_qty)
            if executed_quantity <= 0:
                execution_status = "rejected"
                reason = "no_long_position_to_sell"
            else:
                execution_status = "executed"
                executed_action = "sell"
                reason = "executed_sell"
                executed_notional_usd = executed_quantity * float(mark_price)
                fee_usd = executed_notional_usd * fee_rate
                cash_before = _coerce_float(state.get("cash_usd"), 0.0)
                state["cash_usd"] = cash_before + executed_notional_usd - fee_usd
                avg_entry_price = _coerce_float(position.get("avg_entry_price"), 0.0)
                realized_pnl_delta_usd = (float(mark_price) - avg_entry_price) * executed_quantity
                new_qty = available_qty - executed_quantity
                if new_qty <= 1e-12:
                    new_qty = 0.0
                    position["avg_entry_price"] = 0.0
                position["quantity"] = new_qty
                position["realized_pnl_usd"] = (
                    _coerce_float(position.get("realized_pnl_usd"), 0.0) + realized_pnl_delta_usd
                )
    else:
        reason = "non_actionable_intent"

    state_positions = dict(state.get("positions", {}))
    state_positions[symbol] = {
        "quantity": _coerce_float(position.get("quantity"), 0.0),
        "avg_entry_price": _coerce_float(position.get("avg_entry_price"), 0.0),
        "realized_pnl_usd": _coerce_float(position.get("realized_pnl_usd"), 0.0),
    }
    state["positions"] = state_positions
    state["updated_at_utc"] = created_at_utc
    state["fee_bps"] = max(0.0, float(fee_bps))
    _write_state(state_path, state)

    if execution_status == "executed":
        _append_fill_record(
            fills_log_path,
            {
                "contract": "paper_trade_fill.v1",
                "run_id": run_id,
                "created_at_utc": created_at_utc,
                "exchange": exchange,
                "symbol": symbol,
                "timeframe": timeframe,
                "action": executed_action,
                "quantity": _round(executed_quantity),
                "price": _round(float(mark_price) if mark_price is not None else 0.0),
                "notional_usd": _round(executed_notional_usd),
                "fee_usd": _round(fee_usd),
                "reason": reason,
            },
        )

    position_after = state["positions"].get(symbol, {})
    contract = PaperTradeExecution(
        contract="paper_trade_execution.v1",
        run_id=run_id,
        created_at_utc=created_at_utc,
        mode="paper",
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        intent_status=intent.status,
        intent_action=intent.action,
        execution_status=execution_status,
        executed_action=executed_action,
        requested_notional_usd=_round(requested_notional_usd),
        executed_notional_usd=_round(executed_notional_usd),
        executed_quantity=_round(executed_quantity),
        mark_price=_round(float(mark_price)) if mark_price is not None else None,
        fee_usd=_round(fee_usd),
        cash_after_usd=_round(_coerce_float(state.get("cash_usd"), 0.0)),
        position_qty_after=_round(_coerce_float(position_after.get("quantity"), 0.0)),
        position_avg_entry_after=_round(_coerce_float(position_after.get("avg_entry_price"), 0.0)),
        realized_pnl_delta_usd=_round(realized_pnl_delta_usd),
        reason=reason,
        portfolio_state_path=str(state_path),
        fills_log_path=str(fills_log_path),
        execution_record_path=str(execution_record_path),
    )
    write_contract(execution_record_path, contract)
    return contract
