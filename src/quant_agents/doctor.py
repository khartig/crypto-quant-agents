from __future__ import annotations

from dataclasses import dataclass

from quant_agents.config import Settings, missing_exchange_secrets
from quant_agents.storage import PHASE1_TREE


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class DoctorReport:
    checks: list[DoctorCheck]
    require_secrets: bool

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)


def run_doctor(settings: Settings, require_secrets: bool = False) -> DoctorReport:
    checks: list[DoctorCheck] = []
    root = settings.quant_data_root
    effective_require_secrets = require_secrets or settings.require_exchange_secrets

    root_exists = root.exists()
    checks.append(
        DoctorCheck(
            name="data_root_exists",
            ok=root_exists,
            detail=f"QUANT_DATA_ROOT={root}",
        )
    )

    mount_ok = root_exists and (settings.allow_unmounted_data_root or root.is_mount())
    checks.append(
        DoctorCheck(
            name="data_root_mount_guard",
            ok=mount_ok,
            detail=(
                f"is_mount={root.is_mount()} allow_unmounted={settings.allow_unmounted_data_root}"
                if root_exists
                else "skipped because data root does not exist"
            ),
        )
    )

    if root_exists:
        missing_dirs = [rel for rel in PHASE1_TREE if not (root / rel).exists()]
        tree_ok = len(missing_dirs) == 0
        detail = "all required phase1 directories exist" if tree_ok else f"missing={missing_dirs}"
    else:
        tree_ok = False
        detail = "skipped because data root does not exist"
    checks.append(DoctorCheck(name="phase1_directory_tree", ok=tree_ok, detail=detail))

    defaults_ok = bool(settings.default_exchange and settings.default_symbol and settings.default_timeframe)
    checks.append(
        DoctorCheck(
            name="default_market_config",
            ok=defaults_ok,
            detail=(
                f"exchange={settings.default_exchange} "
                f"symbol={settings.default_symbol} timeframe={settings.default_timeframe}"
            ),
        )
    )

    missing_secrets = missing_exchange_secrets(settings)
    secrets_ok = len(missing_secrets) == 0 or not effective_require_secrets
    if missing_secrets:
        detail = (
            f"missing required secrets: {missing_secrets}"
            if effective_require_secrets
            else f"optional for current mode; missing={missing_secrets}"
        )
    else:
        detail = "required exchange secrets are configured"
    checks.append(DoctorCheck(name="exchange_secrets", ok=secrets_ok, detail=detail))

    return DoctorReport(checks=checks, require_secrets=effective_require_secrets)


def format_doctor_report(report: DoctorReport, settings: Settings) -> str:
    lines = [
        "quant-agents doctor",
        f"overall={'PASS' if report.ok else 'FAIL'}",
        "",
        "checks:",
    ]
    for check in report.checks:
        lines.append(f"- {'PASS' if check.ok else 'FAIL'} | {check.name} | {check.detail}")

    lines.extend(
        [
            "",
            "config:",
            f"- quant_data_root={settings.quant_data_root}",
            f"- default_exchange={settings.default_exchange}",
            f"- default_symbol={settings.default_symbol}",
            f"- default_timeframe={settings.default_timeframe}",
            f"- require_exchange_secrets={report.require_secrets}",
        ]
    )
    return "\n".join(lines)
