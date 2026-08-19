"""FastAPI backend for LocalAIBench.

Serves the JSON API for model discovery, benchmark runs, comparison, and
history, plus a WebSocket endpoint for live run progress. The compiled Astro
frontend is served as static files from ``STATIC_DIR`` when present.
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import inspect
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from local_ai_bench.config import BenchmarkConfig, load_config
from local_ai_bench.domain.events import Event, Events
from local_ai_bench.ollama.client import OllamaClient
from local_ai_bench.ollama.discovery import discover_models
from local_ai_bench.plugins import load_plugins
from local_ai_bench.plugins.base import BenchmarkPlugin
from local_ai_bench.plugins.registry import PluginRegistry
from local_ai_bench.runner.orchestrator import RunOrchestrator
from local_ai_bench.storage.repository import BenchmarkRepository

app = FastAPI(title="LocalAIBench API", version="0.1.0")

# Configurable via environment; Docker Compose overrides these.
_config_path = os.getenv("CONFIG_PATH", "config/default.yaml")
_db_path = os.getenv("DATABASE_URL", "benchmark.db")
_static_dir = os.getenv("STATIC_DIR", "/app/static")
_cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,  # wildcard origins cannot be combined with credentials
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure orchestrator warnings are captured to disk, not only stderr, when
# running under the FastAPI/uvicorn process. Safe to call multiple times.
from local_ai_bench.utils.logging import setup_logging  # noqa: E402

setup_logging()


def _load_cfg() -> BenchmarkConfig:
    try:
        return load_config(_config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"config error: {exc}") from exc


# ---------- Live run progress ----------


class ConnectionManager:
    """Track WebSocket clients per run and broadcast only that run's updates.

    A client subscribes to a specific run via ``/ws?run_id=...``; progress
    events are fanned out only to that run's subscribers, never to everyone.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str | None, set[WebSocket]] = {}

    async def connect(self, ws: WebSocket, run_id: str | None) -> None:
        await ws.accept()
        self._subscribers.setdefault(run_id, set()).add(ws)

    def disconnect(self, ws: WebSocket, run_id: str | None) -> None:
        clients = self._subscribers.get(run_id)
        if clients is None:
            return
        clients.discard(ws)
        if not clients:
            self._subscribers.pop(run_id, None)

    async def broadcast(self, message: dict[str, Any], run_id: str | None) -> None:
        clients = self._subscribers.get(run_id)
        if not clients:
            return
        stale: list[WebSocket] = []
        for ws in list(clients):
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001 - drop dead clients
                stale.append(ws)
        for ws in stale:
            clients.discard(ws)
        if not clients:
            self._subscribers.pop(run_id, None)


class RunStatus(BaseModel):
    """Mutable progress state for one benchmark run, updated by events."""

    run_id: str
    status: str = "pending"
    total: int = 0
    completed: int = 0
    errors: int = 0
    model: str | None = None
    plugin: str | None = None
    case_id: str | None = None
    message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    models: list[str] = Field(default_factory=list)
    plugins: list[str] = Field(default_factory=list)

    @property
    def progress(self) -> float:
        if self.total == 0:
            return 0.0
        return min(1.0, self.completed / self.total)

    def to_message(self, event_type: str = "progress") -> dict[str, Any]:
        return {
            "type": event_type,
            "run_id": self.run_id,
            "status": self.status,
            "progress": self.progress,
            "total": self.total,
            "completed": self.completed,
            "errors": self.errors,
            "model": self.model,
            "plugin": self.plugin,
            "case_id": self.case_id,
            "message": self.message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "models": self.models,
            "plugins": self.plugins,
        }


def _is_terminal(status: str) -> bool:
    return status in ("completed", "failed")


class RunManager:
    """Shared store of live run statuses.

    Active runs are kept for the whole benchmark; terminal (completed/failed)
    statuses are retained briefly so clients can poll a just-finished run, then
    evicted to bound memory growth.
    """

    _TERMINAL_TTL_SECONDS = 3600.0
    _MAX_STATUSES = 200

    def __init__(
        self,
        *,
        terminal_ttl_seconds: float = _TERMINAL_TTL_SECONDS,
        max_statuses: int = _MAX_STATUSES,
    ) -> None:
        self._runs: dict[str, RunStatus] = {}
        self._terminal_ttl_seconds = terminal_ttl_seconds
        self._max_statuses = max_statuses

    def set(self, status: RunStatus) -> None:
        self._sweep()
        self._runs[status.run_id] = status
        self._enforce_cap()

    def get(self, run_id: str) -> RunStatus | None:
        status = self._runs.get(run_id)
        if status is not None and _is_terminal(status.status) and self._is_expired(status):
            self._runs.pop(run_id, None)
            return None
        return status

    def remove(self, run_id: str) -> None:
        self._runs.pop(run_id, None)

    def active(self) -> list[RunStatus]:
        self._sweep()
        active = [s for s in self._runs.values() if not _is_terminal(s.status)]
        active.sort(key=lambda s: s.started_at or "", reverse=True)
        return active

    def _is_expired(self, status: RunStatus) -> bool:
        if status.finished_at is None:
            return False
        try:
            finished = datetime.datetime.fromisoformat(status.finished_at).timestamp()
        except ValueError:
            return False
        return time.time() - finished > self._terminal_ttl_seconds

    def _sweep(self) -> None:
        expired = [
            rid
            for rid, s in self._runs.items()
            if _is_terminal(s.status) and self._is_expired(s)
        ]
        for rid in expired:
            self._runs.pop(rid, None)

    def _enforce_cap(self) -> None:
        while len(self._runs) > self._max_statuses:
            terminal = [s for s in self._runs.values() if _is_terminal(s.status)]
            if not terminal:
                break
            oldest = min(terminal, key=lambda s: s.finished_at or "")
            self._runs.pop(oldest.run_id, None)

    def on_event(self, status: RunStatus, event: Event) -> None:
        """Apply a runner event to the run's progress state."""
        if event.kind == Events.RUN_PLANNED:
            status.total = int(event.data.get("total_cases", 0))
        elif event.kind in (Events.CASE_COMPLETED, Events.CASE_FAILED):
            status.completed += 1
            if event.kind == Events.CASE_FAILED:
                status.errors += 1
        if event.kind == Events.RUN_STARTED:
            status.status = "running"
        if event.host:
            status.message = f"host {event.host}"
        if event.model:
            status.model = event.model
        if event.plugin:
            status.plugin = event.plugin
        if event.case_id:
            status.case_id = event.case_id
        if event.message:
            status.message = event.message


manager = ConnectionManager()
run_manager = RunManager()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, run_id: str | None = None) -> None:
    await manager.connect(ws, run_id)
    try:
        while True:
            data = await ws.receive_text()
            await ws.send_json({"type": "pong", "data": data})
    except WebSocketDisconnect:
        manager.disconnect(ws, run_id)


def _schedule_broadcast(status: RunStatus, event_type: str = "progress") -> None:
    """Schedule an async broadcast from a synchronous event callback."""
    asyncio.get_running_loop().create_task(
        manager.broadcast(status.to_message(event_type), run_id=status.run_id)
    )


# ---------- Models ----------


@app.get("/api/models")
async def list_models() -> dict[str, Any]:
    cfg = _load_cfg()
    all_models: list[dict[str, Any]] = []
    for host in cfg.hosts:
        client = OllamaClient(host.base_url, host.timeout_seconds)
        try:
            models = await discover_models(client, host.name)
            for m in models:
                all_models.append(
                    {
                        "name": m.model_name,
                        "host": m.host_name,
                        "digest": m.digest,
                        "max_context": m.max_context_tokens,
                        "supports_vision": m.supports_vision,
                        "supports_tools": m.supports_tools,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - isolation per host
            all_models.append({"error": str(exc), "host": host.name})
        finally:
            await client.aclose()
    return {"models": all_models, "count": len(all_models)}


# ---------- Plugins ----------


def _effective_plugin_options(cfg: BenchmarkConfig) -> dict[str, dict[str, Any]]:
    """Plugin options = config defaults merged with persisted DB overrides."""
    merged = {pid: dict(opts) for pid, opts in cfg.plugins.options.items()}
    repo = BenchmarkRepository(_db_path)
    try:
        for pid, opts in repo.all_plugin_options().items():
            merged.setdefault(pid, {}).update(opts)
    finally:
        repo.close()
    return merged


def _src_root() -> Path:
    """Repository `src` directory, used to express plugin files as a relative path."""
    pkg = sys.modules.get("local_ai_bench")
    file = getattr(pkg, "__file__", None) if pkg is not None else None
    if file:
        return Path(file).resolve().parent.parent
    return Path("src").resolve()


def _plugin_source_path(cls: type) -> tuple[str | None, Path | None]:
    """Return ``(relative_path, absolute_path)`` to a plugin's source file.

    Resolves the file from the class via :func:`inspect.getfile`, so it works
    for built-in plugins (the source module) and for local plugins loaded from
    disk with ``importlib``. Returns ``(None, None)`` when no source file can be
    resolved.
    """
    try:
        module_file = inspect.getfile(cls)
    except TypeError:
        return None, None
    abs_path = Path(module_file).resolve()
    try:
        rel = abs_path.relative_to(_src_root()).as_posix()
    except ValueError:
        rel = abs_path.as_posix()
    return rel, abs_path


def _plugin_record(pid: str, cls: type, effective: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": pid,
        "name": getattr(cls, "name", pid),
        "description": getattr(cls, "description", ""),
        "category": getattr(cls, "category", "unknown"),
        "version": getattr(cls, "version", "0.0.0"),
        "dataset_version": getattr(cls, "dataset_version", ""),
        "modalities": sorted(m.value for m in getattr(cls, "modalities", set())),
        "options": effective.get(pid, {}),
    }


@app.get("/api/plugins")
async def list_plugins() -> dict[str, Any]:
    cfg = _load_cfg()
    reg = registry()
    effective = _effective_plugin_options(cfg)
    plugins = [_plugin_record(pid, cls, effective) for pid in reg.ids() if (cls := reg.get(pid))]
    return {"plugins": plugins, "compare_default": cfg.plugins.compare_default}


@app.get("/api/plugins/{plugin_id}")
async def get_plugin(plugin_id: str) -> dict[str, Any]:
    """Plugin details incl. its source file (base64) for inspection/download."""
    cfg = _load_cfg()
    reg = registry()
    cls = reg.get(plugin_id)
    if cls is None:
        raise HTTPException(status_code=404, detail=f"Plugin not found: {plugin_id}")
    record = _plugin_record(plugin_id, cls, _effective_plugin_options(cfg))
    rel, abs_path = _plugin_source_path(cls)
    if abs_path is not None and abs_path.is_file():
        raw = abs_path.read_bytes()
        record["source_file"] = rel or abs_path.name
        record["source"] = base64.b64encode(raw).decode("ascii")
    else:
        record["source_file"] = rel
        record["source"] = None
    return {"plugin": record}


class PluginOptionsRequest(BaseModel):
    options: dict[str, Any] = Field(default_factory=dict)


def _validate_option_values(options: dict[str, Any]) -> None:
    """Reject non-JSON-serializable option values with a 422."""
    for key, value in options.items():
        if value is not None and not isinstance(value, (bool, int, float, str)):
            raise HTTPException(
                status_code=422,
                detail=f"Option {key!r} must be a scalar value",
            )


# ---------- Weights ----------


_WEIGHTS_KEY = "weights"


def _effective_weights(cfg: BenchmarkConfig, repo: BenchmarkRepository) -> dict[str, float]:
    """Config weight defaults merged with persisted DB overrides."""
    defaults = cfg.weights.model_dump()
    overrides = repo.get_setting(_WEIGHTS_KEY) or {}
    return {**defaults, **{k: float(v) for k, v in overrides.items()}}


def _weights_payload(cfg: BenchmarkConfig, overrides: dict[str, float]) -> dict[str, Any]:
    defaults = cfg.weights.model_dump()
    effective = {**defaults, **{k: float(v) for k, v in overrides.items()}}
    return {"defaults": defaults, "overrides": overrides, "effective": effective}


@app.get("/api/weights")
async def get_weights() -> dict[str, Any]:
    cfg = _load_cfg()
    repo = BenchmarkRepository(_db_path)
    try:
        overrides = repo.get_setting(_WEIGHTS_KEY) or {}
        return _weights_payload(cfg, {k: float(v) for k, v in overrides.items()})
    finally:
        repo.close()


class WeightsRequest(BaseModel):
    weights: dict[str, float] = Field(default_factory=dict)


@app.put("/api/weights")
async def update_weights(req: WeightsRequest) -> dict[str, Any]:
    cfg = _load_cfg()
    defaults = cfg.weights.model_dump()
    for key, value in req.weights.items():
        if key not in defaults:
            raise HTTPException(status_code=422, detail=f"Unknown weight category: {key}")
        if not math.isfinite(value) or value < 0:
            raise HTTPException(status_code=422, detail=f"Weight {key!r} must be a non-negative number")
    # Prune overrides that equal the config default so resetting is a no-op.
    overrides = {k: v for k, v in req.weights.items() if v != defaults.get(k)}
    repo = BenchmarkRepository(_db_path)
    try:
        repo.set_setting(_WEIGHTS_KEY, overrides)
    finally:
        repo.close()
    return _weights_payload(cfg, overrides)


@app.put("/api/plugins/{plugin_id}/options")
async def update_plugin_options(
    plugin_id: str, req: PluginOptionsRequest
) -> dict[str, Any]:
    reg = registry()
    if reg.get(plugin_id) is None:
        raise HTTPException(status_code=404, detail="Plugin not found")
    _validate_option_values(req.options)
    repo = BenchmarkRepository(_db_path)
    try:
        repo.set_plugin_options(plugin_id, req.options)
        base = _load_cfg().plugins.options.get(plugin_id, {})
        merged = {**base, **req.options}
        return {"plugin_id": plugin_id, "options": merged}
    finally:
        repo.close()


# ---------- Benchmarks ----------


class RunRequest(BaseModel):
    model_names: list[str] = Field(..., min_length=1)
    plugin_ids: list[str] | None = None
    repetitions: int = Field(1, ge=1, le=50)
    warmup_runs: int = Field(0, ge=0, le=10)
    max_retries: int = Field(1, ge=0, le=10)


def registry() -> PluginRegistry:
    reg, _ = load_plugins(_load_cfg().plugins.local_dir)
    return reg


def _build_plugins(
    reg: PluginRegistry, plugin_ids: list[str]
) -> list[BenchmarkPlugin]:
    out: list[BenchmarkPlugin] = []
    for pid in plugin_ids:
        cls = reg.get(pid)
        if cls:
            out.append(cls())
    return out


@app.post("/api/benchmarks/run")
async def start_benchmark(req: RunRequest) -> dict[str, Any]:
    cfg = _load_cfg()
    reg = registry()
    pids = req.plugin_ids or cfg.plugins.enabled
    plugin_instances = _build_plugins(reg, pids)

    # Override runner settings from the request.
    cfg.runner.repetitions = req.repetitions
    cfg.runner.warmup_runs = req.warmup_runs
    cfg.runner.max_retries = req.max_retries

    # Honor UI-set category weights (DB overrides merged over config defaults).
    repo = BenchmarkRepository(_db_path)
    try:
        weight_overrides = repo.get_setting(_WEIGHTS_KEY) or {}
    finally:
        repo.close()
    for key, value in weight_overrides.items():
        if hasattr(cfg.weights, key):
            setattr(cfg.weights, key, float(value))

    run_id = uuid.uuid4().hex[:12]
    status = RunStatus(run_id=run_id, models=req.model_names, plugins=pids)
    run_manager.set(status)

    wanted = set(req.model_names)

    def on_event(event: Event) -> None:
        run_manager.on_event(status, event)
        _schedule_broadcast(status)

    orchestrator = RunOrchestrator(
        cfg,
        plugin_instances,
        event_cb=on_event,
        model_filter=lambda mi: mi.model_name in wanted,
        run_id=run_id,
        plugin_options=_effective_plugin_options(cfg),
    )

    async def _run() -> None:
        try:
            status.status = "running"
            status.started_at = datetime.datetime.now(datetime.UTC).isoformat()
            result = await orchestrator.run()
            repo = BenchmarkRepository(_db_path)
            try:
                repo.save_run(result)
            finally:
                repo.close()
            status.status = "completed"
            status.message = "Benchmark completed"
        except Exception as exc:  # noqa: BLE001 - surface as failed run
            status.status = "failed"
            status.message = str(exc)
        status.finished_at = datetime.datetime.now(datetime.UTC).isoformat()
        _schedule_broadcast(status, "complete")

    asyncio.create_task(_run())
    return {
        "status": "started",
        "run_id": run_id,
        "models": req.model_names,
        "plugins": [p.id for p in plugin_instances],
    }


@app.get("/api/benchmarks")
async def list_benchmarks(
    search: str | None = None,
    model: str | None = None,
    host: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    repo = BenchmarkRepository(_db_path)
    try:
        runs = repo.list_runs(
            search=search, model=model, host=host, date_from=date_from, date_to=date_to
        )
        return {"runs": runs, "filters": repo.distinct_filters()}
    finally:
        repo.close()


@app.get("/api/benchmarks/active")
async def list_active_runs() -> dict[str, Any]:
    return {"runs": [s.to_message("status") for s in run_manager.active()]}


@app.get("/api/benchmarks/{run_id}")
async def get_benchmark(run_id: str) -> dict[str, Any]:
    repo = BenchmarkRepository(_db_path)
    try:
        run = repo.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return {"run": run, "models": repo.compare_models(run_id=run_id)}
    finally:
        repo.close()


def _active_run_ids(run_ids: list[str]) -> list[str]:
    """Return the subset of ids still pending/running in the live manager."""
    return [
        rid
        for rid in run_ids
        if (status := run_manager.get(rid)) is not None and not _is_terminal(status.status)
    ]


class DeleteRunsRequest(BaseModel):
    run_ids: list[str] = Field(..., min_length=1)


@app.api_route("/api/benchmarks/delete", methods=["POST", "DELETE"])
async def delete_benchmarks(req: DeleteRunsRequest) -> dict[str, Any]:
    active = _active_run_ids(req.run_ids)
    if active:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete active run(s): {', '.join(active)}",
        )
    repo = BenchmarkRepository(_db_path)
    try:
        deleted = repo.delete_runs(req.run_ids)
    finally:
        repo.close()
    for rid in req.run_ids:
        run_manager.remove(rid)
    return {"deleted": req.run_ids[:deleted], "count": deleted, "status": "ok"}


@app.delete("/api/benchmarks/{run_id}")
async def delete_benchmark(run_id: str) -> dict[str, Any]:
    if _active_run_ids([run_id]):
        raise HTTPException(status_code=409, detail="Cannot delete an active run")
    repo = BenchmarkRepository(_db_path)
    try:
        if not repo.delete_run(run_id):
            raise HTTPException(status_code=404, detail="Run not found")
    finally:
        repo.close()
    run_manager.remove(run_id)
    return {"deleted": run_id, "status": "ok"}


@app.get("/api/benchmarks/{run_id}/status")
async def get_run_status(run_id: str) -> dict[str, Any]:
    status = run_manager.get(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Run not found or already expired")
    return status.to_message()


@app.get("/api/benchmarks/{run_id}/cases")
async def get_run_cases(run_id: str) -> dict[str, Any]:
    """Return the persisted per-case rows for a completed run (including errors)."""
    repo = BenchmarkRepository(_db_path)
    try:
        rows = repo.cases_for_run(run_id)
        return {"run_id": run_id, "count": len(rows), "cases": rows}
    finally:
        repo.close()


# ---------- Compare ----------


@app.get("/api/compare")
async def compare_models(request: Request) -> dict[str, Any]:
    # Support both ?run=A&run=B (multiple) and ?run=A (single/legacy).
    run_ids = list(request.query_params.getlist("run")) or None
    repo = BenchmarkRepository(_db_path)
    try:
        rows = repo.compare_models(run_ids=run_ids)
        return {"models": rows}
    finally:
        repo.close()


# ---------- History ----------


@app.get("/api/history")
async def get_history(
    search: str | None = None,
    model: str | None = None,
    host: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    repo = BenchmarkRepository(_db_path)
    try:
        runs = repo.list_runs(
            search=search, model=model, host=host, date_from=date_from, date_to=date_to
        )
        return {"runs": runs, "filters": repo.distinct_filters()}
    finally:
        repo.close()


# ---------- Export ----------


@app.get("/api/export/{run_id}.{format}")
async def export_run(run_id: str, format: str) -> dict[str, Any]:
    repo = BenchmarkRepository(_db_path)
    try:
        run = repo.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        models = repo.compare_models(run_id=run_id)

        if format == "json":
            return {"run": run, "models": models}
        if format == "csv":
            import csv
            import io

            fieldnames = [
                "model_name",
                "host_name",
                "overall_score",
                "latency_p50_ms",
                "latency_p95_ms",
                "tokens_per_second",
                "cases_run",
                "errors",
            ]
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for m in models:
                writer.writerow({k: m.get(k) for k in fieldnames})
            return {"csv": output.getvalue()}
        if format == "md":
            return {"markdown": _render_markdown(run, models)}
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
    finally:
        repo.close()


def _render_markdown(run: dict[str, Any], models: list[dict[str, Any]]) -> str:
    lines = [f"# LocalAIBench report · {run['run_id']}", ""]
    lines.append(f"- Timestamp: {run.get('timestamp')}")
    lines.append(f"- App version: {run.get('app_version')}")
    lines.append(f"- Config hash: `{run.get('config_hash')}`")
    hosts = run.get("hosts") or []
    lines.append(f"- Hosts: {', '.join(h.get('name') or h.get('base_url') for h in hosts) if hosts else 'none'}")
    lines.append("")
    lines.append("| Model | Host | Score | p50 (ms) | p95 (ms) | Tokens/s | Cases | Errors |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m in models:
        lines.append(
            f"| {m['model_name']} | {m.get('host_name') or '-'} | {_fmt(m.get('overall_score'), 3)} | "
            f"{_fmt(m.get('latency_p50_ms'))} | {_fmt(m.get('latency_p95_ms'))} | "
            f"{_fmt(m.get('tokens_per_second'))} | {m.get('cases_run', 0)} | "
            f"{m.get('errors', 0)} |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


# ---------- Health ----------


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


# ---------- Frontend static serving ----------


@app.api_route(
    "/api/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    include_in_schema=False,
)
async def api_not_found(path: str) -> JSONResponse:
    return JSONResponse({"detail": "Not found"}, status_code=404)


if Path(_static_dir).is_dir():
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
else:
    @app.get("/{path:path}", include_in_schema=False)
    async def api_only(path: str) -> PlainTextResponse:
        return PlainTextResponse(
            "LocalAIBench API is running. Frontend build not found (set STATIC_DIR).",
            status_code=200,
        )
