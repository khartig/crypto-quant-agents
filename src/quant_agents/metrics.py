from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator


def _metrics_log_path(root: Path, ts: datetime) -> Path:
    path = root / "logs" / "metrics" / f"{ts:%Y-%m-%d}" / "pipeline_metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _summary_path(root: Path) -> Path:
    path = root / "logs" / "metrics" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return {}


def _update_summary(
    root: Path,
    operation: str,
    status: str,
    finished_at: datetime,
    duration_ms: float,
) -> None:
    summary_file = _summary_path(root)
    summary = _load_summary(summary_file)
    runs = summary.setdefault("runs", {})
    op = runs.setdefault(
        operation,
        {
            "success": 0,
            "failure": 0,
            "last_status": None,
            "last_finished_at": None,
            "last_duration_ms": None,
        },
    )

    if status == "success":
        op["success"] = int(op.get("success", 0)) + 1
    else:
        op["failure"] = int(op.get("failure", 0)) + 1
    op["last_status"] = status
    op["last_finished_at"] = finished_at.isoformat()
    op["last_duration_ms"] = duration_ms
    summary["last_updated_at"] = finished_at.isoformat()
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")


@contextmanager
def tracked_operation(
    root: Path,
    operation: str,
    dimensions: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    started = datetime.now(timezone.utc)
    started_at_perf = perf_counter()
    details: dict[str, Any] = {}
    status = "success"
    error_type: str | None = None
    error_message: str | None = None

    try:
        yield details
    except Exception as exc:
        status = "failure"
        error_type = type(exc).__name__
        error_message = str(exc)
        raise
    finally:
        finished = datetime.now(timezone.utc)
        duration_ms = round((perf_counter() - started_at_perf) * 1000, 3)
        record: dict[str, Any] = {
            "timestamp": finished.isoformat(),
            "operation": operation,
            "status": status,
            "duration_ms": duration_ms,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "dimensions": _json_safe(dimensions or {}),
            "details": _json_safe(details),
        }
        if error_type:
            record["error_type"] = error_type
        if error_message:
            record["error_message"] = error_message

        metrics_file = _metrics_log_path(root, started)
        with metrics_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        _update_summary(root, operation, status, finished, duration_ms)
