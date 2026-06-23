from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np

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
        "peak_equity_usd": _coerce_float(payload.get("peak_equity_usd"), starting_cash_usd),
        "max_drawdown_ratio": _coerce_float(payload.get("max_drawdown_ratio"), 0.0),
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
        "peak_equity_usd": _round(_coerce_float(state.get("peak_equity_usd"), 0.0)),
        "max_drawdown_ratio": _round(_coerce_float(state.get("max_drawdown_ratio"), 0.0)),
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


def _mark_to_market_equity(
    *,
    state: dict[str, Any],
    symbol: str,
    mark_price: float | None,
) -> float:
    cash_usd = _coerce_float(state.get("cash_usd"), 0.0)
    position_payload = dict(state.get("positions", {})).get(symbol, {})
    if not isinstance(position_payload, dict):
        position_payload = {}
    quantity = _coerce_float(position_payload.get("quantity"), 0.0)
    entry_price = _coerce_float(position_payload.get("avg_entry_price"), 0.0)
    effective_mark = (
        float(mark_price)
        if mark_price is not None and mark_price > 0.0
        else (entry_price if entry_price > 0.0 else 0.0)
    )
    return cash_usd + (quantity * effective_mark)


def _update_drawdown_metrics(
    *,
    state: dict[str, Any],
    symbol: str,
    mark_price: float | None,
) -> dict[str, float]:
    equity_usd = _mark_to_market_equity(state=state, symbol=symbol, mark_price=mark_price)
    peak_equity_usd = max(
        _coerce_float(state.get("peak_equity_usd"), 0.0),
        equity_usd,
        _coerce_float(state.get("starting_cash_usd"), 0.0),
    )
    drawdown_ratio = (
        max(0.0, (peak_equity_usd - equity_usd) / peak_equity_usd)
        if peak_equity_usd > 0.0
        else 0.0
    )
    max_drawdown_ratio = max(
        _coerce_float(state.get("max_drawdown_ratio"), 0.0),
        drawdown_ratio,
    )
    state["peak_equity_usd"] = peak_equity_usd
    state["max_drawdown_ratio"] = max_drawdown_ratio
    return {
        "equity_usd": equity_usd,
        "peak_equity_usd": peak_equity_usd,
        "drawdown_ratio": drawdown_ratio,
        "max_drawdown_ratio": max_drawdown_ratio,
    }


def summarize_paper_portfolio_risk(
    *,
    quant_data_root: Path,
    symbol: str,
    mark_price: float | None,
    starting_cash_usd: float,
    fee_bps: float,
) -> dict[str, float]:
    state_path = quant_data_root / "paper-trading" / "state" / "portfolio_state.json"
    state = _load_state(
        state_path,
        starting_cash_usd=max(0.0, float(starting_cash_usd)),
        fee_bps=max(0.0, float(fee_bps)),
    )
    summary = _update_drawdown_metrics(
        state=state,
        symbol=symbol,
        mark_price=mark_price,
    )
    summary["cash_usd"] = _coerce_float(state.get("cash_usd"), 0.0)
    return summary

def simulate_paper_trade_execution_step(
    *,
    state: dict[str, Any],
    symbol: str,
    intent_status: str,
    intent_action: Recommendation,
    requested_notional_usd: float,
    mark_price: float | None,
    fee_bps: float,
    slippage_bps: float,
    spread_bps: float = 0.0,
    latency_ms: float = 0.0,
    latency_slippage_bps_per_second: float = 0.0,
    liquidity_score: float = 1.0,
    market_depth_notional_usd: float = 2500.0,
    notional_impact_coeff: float = 2.0,
) -> dict[str, Any]:
    position = _ensure_position(state, symbol)
    fee_rate = max(0.0, float(fee_bps)) / 10_000.0
    requested_notional_usd = max(0.0, float(requested_notional_usd))

    execution_status = "skipped"
    executed_action: Recommendation = "hold"
    executed_notional_usd = 0.0
    executed_quantity = 0.0
    execution_price: float | None = None
    fee_usd = 0.0
    realized_pnl_delta_usd = 0.0
    reason = "intent_not_emitted"

    valid_price = mark_price is not None and mark_price > 0
    base_slippage_bps = max(0.0, float(slippage_bps))
    spread_bps = max(0.0, float(spread_bps))
    latency_ms = max(0.0, float(latency_ms))
    liquidity_score = float(np.clip(float(liquidity_score), 0.0, 1.0))
    market_depth_notional_usd = max(1.0, float(market_depth_notional_usd))
    notional_impact_coeff = max(0.0, float(notional_impact_coeff))
    impact_ratio = (
        requested_notional_usd / market_depth_notional_usd
        if market_depth_notional_usd > 0.0
        else 0.0
    )
    impact_slippage_bps = impact_ratio * notional_impact_coeff
    latency_slippage_bps = max(0.0, float(latency_slippage_bps_per_second)) * (latency_ms / 1000.0)
    effective_slippage_bps = base_slippage_bps + (spread_bps / 2.0) + latency_slippage_bps + impact_slippage_bps
    slippage_rate = max(0.0, float(effective_slippage_bps)) / 10_000.0
    fill_ratio = float(
        np.clip(
            liquidity_score / (1.0 + (impact_ratio * max(0.0, notional_impact_coeff))),
            0.0,
            1.0,
        )
    )
    requested_effective_notional = requested_notional_usd * fill_ratio
    partial_fill = False

    if intent_status != "emitted":
        reason = f"intent_{intent_status}"
    elif intent_action == "buy":
        if not valid_price:
            execution_status = "rejected"
            reason = "invalid_mark_price"
        else:
            execution_price = float(mark_price) * (1.0 + slippage_rate)
            cash_before = _coerce_float(state.get("cash_usd"), 0.0)
            max_notional = cash_before / (1.0 + fee_rate)
            if requested_effective_notional <= 0.0:
                execution_status = "rejected"
                reason = "liquidity_blocked_buy"
            else:
                executed_notional_usd = min(requested_effective_notional, max(0.0, max_notional))
            if executed_notional_usd <= 0:
                execution_status = "rejected"
                if reason == "intent_not_emitted":
                    reason = "insufficient_cash"
            else:
                execution_status = "executed"
                executed_action = "buy"
                partial_fill = executed_notional_usd + 1e-9 < requested_notional_usd
                reason = "executed_buy_partial" if partial_fill else "executed_buy"
                executed_quantity = (
                    executed_notional_usd / execution_price if execution_price and execution_price > 0 else 0.0
                )
                fee_usd = executed_notional_usd * fee_rate
                state["cash_usd"] = cash_before - executed_notional_usd - fee_usd
                old_qty = _coerce_float(position.get("quantity"), 0.0)
                old_avg = _coerce_float(position.get("avg_entry_price"), 0.0)
                new_qty = old_qty + executed_quantity
                existing_cost_basis = old_qty * old_avg
                new_cost_basis = existing_cost_basis + executed_notional_usd + fee_usd
                new_avg = (
                    (new_cost_basis / new_qty)
                    if new_qty > 0
                    else 0.0
                )
                position["quantity"] = new_qty
                position["avg_entry_price"] = new_avg
    elif intent_action == "sell":
        if not valid_price:
            execution_status = "rejected"
            reason = "invalid_mark_price"
        else:
            execution_price = float(mark_price) * (1.0 - slippage_rate)
            if execution_price <= 0:
                execution_status = "rejected"
                reason = "invalid_execution_price_after_slippage"
                execution_price = None
            else:
                available_qty = max(0.0, _coerce_float(position.get("quantity"), 0.0))
                requested_qty = (
                    requested_effective_notional / execution_price
                    if execution_price > 0.0
                    else 0.0
                )
                executed_quantity = min(requested_qty, available_qty)
                if executed_quantity <= 0:
                    execution_status = "rejected"
                    reason = "liquidity_blocked_sell" if requested_effective_notional <= 0.0 else "no_long_position_to_sell"
                else:
                    execution_status = "executed"
                    executed_action = "sell"
                    partial_fill = (
                        executed_quantity + 1e-9
                        < (
                            (requested_notional_usd / execution_price)
                            if execution_price > 0.0
                            else 0.0
                        )
                    )
                    reason = "executed_sell_partial" if partial_fill else "executed_sell"
                    executed_notional_usd = executed_quantity * execution_price
                    fee_usd = executed_notional_usd * fee_rate
                    cash_before = _coerce_float(state.get("cash_usd"), 0.0)
                    state["cash_usd"] = cash_before + executed_notional_usd - fee_usd
                    avg_entry_price = _coerce_float(position.get("avg_entry_price"), 0.0)
                    gross_realized_pnl_delta = (execution_price - avg_entry_price) * executed_quantity
                    realized_pnl_delta_usd = gross_realized_pnl_delta - fee_usd
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
    state["fee_bps"] = max(0.0, float(fee_bps))
    drawdown_snapshot = _update_drawdown_metrics(
        state=state,
        symbol=symbol,
        mark_price=mark_price if mark_price is not None else execution_price,
    )

    position_after = state["positions"].get(symbol, {})
    execution_diagnostics = {
        "base_slippage_bps": base_slippage_bps,
        "spread_bps": spread_bps,
        "latency_ms": latency_ms,
        "latency_slippage_bps": latency_slippage_bps,
        "impact_ratio": impact_ratio,
        "impact_slippage_bps": impact_slippage_bps,
        "effective_slippage_bps": effective_slippage_bps,
        "liquidity_score": liquidity_score,
        "market_depth_notional_usd": market_depth_notional_usd,
        "fill_ratio": fill_ratio,
        "partial_fill": partial_fill,
        "drawdown_snapshot": drawdown_snapshot,
    }
    return {
        "execution_status": execution_status,
        "executed_action": executed_action,
        "requested_notional_usd": requested_notional_usd,
        "executed_notional_usd": executed_notional_usd,
        "executed_quantity": executed_quantity,
        "execution_price": execution_price,
        "fee_usd": fee_usd,
        "realized_pnl_delta_usd": realized_pnl_delta_usd,
        "reason": reason,
        "cash_after_usd": _coerce_float(state.get("cash_usd"), 0.0),
        "position_qty_after": _coerce_float(position_after.get("quantity"), 0.0),
        "position_avg_entry_after": _coerce_float(position_after.get("avg_entry_price"), 0.0),
        "fill_ratio": fill_ratio,
        "spread_bps": spread_bps,
        "latency_ms": latency_ms,
        "liquidity_score": liquidity_score,
        "effective_slippage_bps": effective_slippage_bps,
        "execution_diagnostics": execution_diagnostics,
        "equity_after_usd": drawdown_snapshot.get("equity_usd"),
        "drawdown_ratio": drawdown_snapshot.get("drawdown_ratio"),
        "max_drawdown_ratio": drawdown_snapshot.get("max_drawdown_ratio"),
    }


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
    slippage_bps: float,
    spread_bps: float = 0.0,
    latency_ms: float = 0.0,
    latency_slippage_bps_per_second: float = 0.0,
    liquidity_score: float = 1.0,
    market_depth_notional_usd: float = 2500.0,
    notional_impact_coeff: float = 2.0,
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
    requested_notional_usd = max(0.0, float(intent.notional_usd))
    base_notional_usd = max(0.0, float(intent.base_notional_usd or requested_notional_usd))
    arm_votes = dict(intent.arm_votes)
    arm_weights = {
        str(arm): max(0.0, _coerce_float(weight, 0.0))
        for arm, weight in dict(intent.arm_weights).items()
    }
    selected_arms = [str(arm) for arm in list(intent.selected_arms) if str(arm).strip()]
    ensemble_reason_codes = [str(code) for code in list(intent.ensemble_reason_codes)]
    execution_result = simulate_paper_trade_execution_step(
        state=state,
        symbol=symbol,
        intent_status=str(intent.status),
        intent_action=intent.action,
        requested_notional_usd=requested_notional_usd,
        mark_price=mark_price,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        latency_ms=latency_ms,
        latency_slippage_bps_per_second=latency_slippage_bps_per_second,
        liquidity_score=liquidity_score,
        market_depth_notional_usd=market_depth_notional_usd,
        notional_impact_coeff=notional_impact_coeff,
    )
    execution_status = str(execution_result.get("execution_status", "skipped"))
    executed_action = str(execution_result.get("executed_action", "hold"))
    executed_notional_usd = _coerce_float(execution_result.get("executed_notional_usd"), 0.0)
    executed_quantity = _coerce_float(execution_result.get("executed_quantity"), 0.0)
    execution_price = execution_result.get("execution_price")
    fee_usd = _coerce_float(execution_result.get("fee_usd"), 0.0)
    realized_pnl_delta_usd = _coerce_float(execution_result.get("realized_pnl_delta_usd"), 0.0)
    fill_ratio = float(np.clip(_coerce_float(execution_result.get("fill_ratio"), 0.0), 0.0, 1.0))
    effective_slippage_bps = max(0.0, _coerce_float(execution_result.get("effective_slippage_bps"), 0.0))
    execution_diagnostics = dict(execution_result.get("execution_diagnostics", {}))
    reason = str(execution_result.get("reason", "unknown"))
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
                "mark_price": _round(float(mark_price) if mark_price is not None else 0.0),
                "price": _round(float(execution_price) if execution_price is not None else 0.0),
                "notional_usd": _round(executed_notional_usd),
                "base_notional_usd": _round(base_notional_usd),
                "fee_usd": _round(fee_usd),
                "slippage_bps": _round(max(0.0, float(slippage_bps))),
                "effective_slippage_bps": _round(effective_slippage_bps),
                "spread_bps": _round(max(0.0, float(spread_bps))),
                "latency_ms": _round(max(0.0, float(latency_ms))),
                "liquidity_score": _round(float(np.clip(liquidity_score, 0.0, 1.0))),
                "fill_ratio": _round(fill_ratio),
                "reason": reason,
                "selected_arms": selected_arms,
                "arm_weights": arm_weights,
                "ensemble_reason_codes": ensemble_reason_codes,
                "execution_diagnostics": execution_diagnostics,
            },
        )

    position_after = state.get("positions", {}).get(symbol, {})
    resolved_selected_arms = selected_arms or sorted(arm_weights.keys())
    total_selected_weight = sum(max(0.0, arm_weights.get(arm, 0.0)) for arm in resolved_selected_arms)
    arm_attribution: dict[str, dict[str, Any]] = {}
    for arm in resolved_selected_arms:
        raw_weight = max(0.0, arm_weights.get(arm, 0.0))
        normalized_weight = (
            (raw_weight / total_selected_weight)
            if total_selected_weight > 0
            else (1.0 / len(resolved_selected_arms))
        )
        arm_vote = dict(arm_votes.get(arm, {}))
        arm_attribution[arm] = {
            "weight": _round(normalized_weight),
            "proposed_action": str(arm_vote.get("recommendation", "hold")),
            "selected_action": intent.action,
            "executed_action": executed_action,
            "executed": execution_status == "executed",
            "attributed_notional_usd": _round(executed_notional_usd * normalized_weight),
            "attributed_realized_pnl_delta_usd": _round(realized_pnl_delta_usd * normalized_weight),
        }
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
        execution_price=_round(float(execution_price)) if execution_price is not None else None,
        fee_usd=_round(fee_usd),
        slippage_bps=_round(max(0.0, float(slippage_bps))),
        cash_after_usd=_round(_coerce_float(execution_result.get("cash_after_usd"), 0.0)),
        position_qty_after=_round(_coerce_float(execution_result.get("position_qty_after"), 0.0)),
        position_avg_entry_after=_round(_coerce_float(execution_result.get("position_avg_entry_after"), 0.0)),
        realized_pnl_delta_usd=_round(realized_pnl_delta_usd),
        reason=reason,
        portfolio_state_path=str(state_path),
        fills_log_path=str(fills_log_path),
        execution_record_path=str(execution_record_path),
        base_requested_notional_usd=_round(base_notional_usd),
        fill_ratio=_round(fill_ratio),
        spread_bps=_round(max(0.0, float(spread_bps))),
        latency_ms=_round(max(0.0, float(latency_ms))),
        liquidity_score=_round(float(np.clip(liquidity_score, 0.0, 1.0))),
        effective_slippage_bps=_round(effective_slippage_bps),
        execution_diagnostics=execution_diagnostics,
        arm_votes=arm_votes,
        arm_weights=arm_weights,
        selected_arms=resolved_selected_arms,
        ensemble_reason_codes=ensemble_reason_codes,
        arm_attribution=arm_attribution,
    )
    write_contract(execution_record_path, contract)
    return contract
