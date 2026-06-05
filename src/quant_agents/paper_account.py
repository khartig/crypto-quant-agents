from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal
from urllib.error import URLError
from urllib.request import urlopen

import ccxt

PaperAccountProvider = Literal["tradingview", "ccxt"]


@dataclass(frozen=True)
class PaperAccountProbeResult:
    ok: bool
    provider: PaperAccountProvider
    message: str
    details: dict[str, str | bool | float | int | None]


def _probe_tradingview(timeout_seconds: float, base_url: str) -> PaperAccountProbeResult:
    started = perf_counter()
    status_code: int | None = None
    reachable = False
    network_error: str | None = None

    try:
        with urlopen(base_url, timeout=timeout_seconds) as response:
            status_code = int(getattr(response, "status", 0))
            reachable = 200 <= status_code < 500
    except URLError as exc:
        network_error = str(exc)

    elapsed_ms = round((perf_counter() - started) * 1000, 3)

    return PaperAccountProbeResult(
        ok=False,
        provider="tradingview",
        message=(
            "TradingView website reachability checked, but direct external API connectivity "
            "to TradingView's built-in paper trading account is not supported."
        ),
        details={
            "base_url": base_url,
            "reachable": reachable,
            "status_code": status_code,
            "network_error": network_error,
            "latency_ms": elapsed_ms,
            "recommended_path": "use broker paper API/testnet + TradingView webhook alerts for automation",
        },
    )


def _probe_ccxt(
    *,
    exchange_id: str,
    timeout_seconds: float,
    sandbox: bool,
    api_key: str | None,
    api_secret: str | None,
    api_passphrase: str | None,
) -> PaperAccountProbeResult:
    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        return PaperAccountProbeResult(
            ok=False,
            provider="ccxt",
            message=f"Unsupported exchange id: {exchange_id}",
            details={"exchange": exchange_id, "sandbox": sandbox},
        )

    if not api_key or not api_secret:
        return PaperAccountProbeResult(
            ok=False,
            provider="ccxt",
            message="Missing paper-account credentials (api key/secret).",
            details={"exchange": exchange_id, "sandbox": sandbox},
        )

    exchange = exchange_class(
        {
            "enableRateLimit": True,
            "timeout": int(max(1.0, timeout_seconds) * 1000),
            "apiKey": api_key,
            "secret": api_secret,
            "password": api_passphrase or "",
        }
    )
    if sandbox:
        try:
            exchange.set_sandbox_mode(True)
        except Exception:
            return PaperAccountProbeResult(
                ok=False,
                provider="ccxt",
                message=(
                    f"Exchange `{exchange_id}` does not expose CCXT sandbox mode "
                    "or sandbox initialization failed."
                ),
                details={"exchange": exchange_id, "sandbox": sandbox},
            )

    started = perf_counter()
    try:
        exchange.load_markets()
        has_fetch_balance = bool(getattr(exchange, "has", {}).get("fetchBalance"))
        if has_fetch_balance:
            exchange.fetch_balance()
        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        return PaperAccountProbeResult(
            ok=True,
            provider="ccxt",
            message="Paper account connectivity check succeeded.",
            details={
                "exchange": exchange_id,
                "sandbox": sandbox,
                "fetch_balance_called": has_fetch_balance,
                "latency_ms": elapsed_ms,
            },
        )
    except Exception as exc:
        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        return PaperAccountProbeResult(
            ok=False,
            provider="ccxt",
            message=f"Paper account connectivity check failed: {type(exc).__name__}: {exc}",
            details={
                "exchange": exchange_id,
                "sandbox": sandbox,
                "latency_ms": elapsed_ms,
            },
        )
    finally:
        try:
            exchange.close()
        except Exception:
            pass


def run_paper_account_probe(
    *,
    provider: PaperAccountProvider,
    timeout_seconds: float,
    tradingview_base_url: str,
    exchange_id: str,
    sandbox: bool,
    api_key: str | None,
    api_secret: str | None,
    api_passphrase: str | None,
) -> PaperAccountProbeResult:
    if provider == "tradingview":
        return _probe_tradingview(timeout_seconds=timeout_seconds, base_url=tradingview_base_url)
    return _probe_ccxt(
        exchange_id=exchange_id,
        timeout_seconds=timeout_seconds,
        sandbox=sandbox,
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase,
    )
