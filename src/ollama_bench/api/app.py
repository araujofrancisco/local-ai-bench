"""FastAPI backend for OllamaBench.

Serves the JSON API for model discovery, benchmark runs, comparison, and
history, plus a WebSocket endpoint for live run progress. The compiled Astro
frontend is served as static files from ``STATIC_DIR`` when present.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ollama_bench.config import BenchmarkConfig, load_config
from ollama_bench.domain.events import Event, Events
from ollama_bench.ollama.client import OllamaClient
from ollama_bench.ollama.discovery import discover_models
from ollama_bench.plugins import load_plugins
from ollama_bench.plugins.base import BenchmarkPlugin
from ollama_bench.plugins.registry import PluginRegistry
from ollama_bench.runner.orchestrator import RunOrchestrator
from ollama_bench.storage.repository import BenchmarkRepository

app = FastAPI(title="OllamaBench API", version="0.1.0")

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


def _load_cfg() -> BenchmarkConfig:
    try:
        return load_config(_config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"config error: {exc}") from exc


# ---------- Live run progress ----------


class ConnectionManager:
    """Track connected WebSocket clients and broadcast run updates."""

    def __init__(self) -> None:
        self._active: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._active.discard(ws)

    async def broadcast(self, message: dict[str, Any]) -> None:
        stale: list[WebSocket] = []
        for ws in self._active:
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001 - drop dead clients
                stale.append(ws)
        for ws in stale:
            self._active.discard(ws)


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
        }


class RunManager:
    """Shared store of active run statuses."""

    def __init__(self) -> None:
        self._runs: dict[str, RunStatus] = {}

    def set(self, status: RunStatus) -> None:
        self._runs[status.run_id] = status

    def get(self, run_id: str) -> RunStatus | None:
        return self._runs.get(run_id)

    def remove(self, run_id: str) -> None:
        self._runs.pop(run_id, None)

    def on_event(self, status: RunStatus, event: Event) -> None:
        """Apply a runner event to the run's progress state."""
        if event.kind == Events.CASE_STARTED:
            status.total += 1
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
async def websocket_endpoint(ws: WebSocket) -> None:
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            await ws.send_json({"type": "pong", "data": data})
    except WebSocketDisconnect:
        manager.disconnect(ws)


def _schedule_broadcast(status: RunStatus, event_type: str = "progress") -> None:
    """Schedule an async broadcast from a synchronous event callback."""
    asyncio.get_running_loop().create_task(
        manager.broadcast(status.to_message(event_type))
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
    merged = {pid: dict(opts) for pid, opts in cfg.plugins.options.model_dump().items()}
    repo = BenchmarkRepository(_db_path)
    try:
        for pid, opts in repo.all_plugin_options().items():
            merged.setdefault(pid, {}).update(opts)
    finally:
        repo.close()
    return merged


@app.get("/api/plugins")
async def list_plugins() -> dict[str, Any]:
    cfg = _load_cfg()
    reg = registry()
    effective = _effective_plugin_options(cfg)
    plugins = []
    for pid in reg.ids():
        cls = reg.get(pid)
        if cls:
            plugins.append(
                {
                    "id": pid,
                    "name": getattr(cls, "name", pid),
                    "description": getattr(cls, "description", ""),
                    "category": getattr(cls, "category", "unknown"),
                    "version": getattr(cls, "version", "0.0.0"),
                    "dataset_version": getattr(cls, "dataset_version", ""),
                    "modalities": sorted(m.value for m in getattr(cls, "modalities", set())),
                    "options": effective.get(pid, {}),
                }
            )
    return {"plugins": plugins}


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
        base = _load_cfg().plugins.options.model_dump().get(plugin_id, {})
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

    run_id = uuid.uuid4().hex[:12]
    status = RunStatus(run_id=run_id)
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


@app.delete("/api/benchmarks/{run_id}")
async def delete_benchmark(run_id: str) -> dict[str, Any]:
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


# ---------- Compare ----------


@app.get("/api/compare")
async def compare_models(run_id: str | None = None) -> dict[str, Any]:
    repo = BenchmarkRepository(_db_path)
    try:
        return {"models": repo.compare_models(run_id=run_id)}
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
    lines = [f"# OllamaBench report · {run['run_id']}", ""]
    lines.append(f"- Timestamp: {run.get('timestamp')}")
    lines.append(f"- App version: {run.get('app_version')}")
    lines.append(f"- Config hash: `{run.get('config_hash')}`")
    lines.append("")
    lines.append("| Model | Score | p50 (ms) | p95 (ms) | Tokens/s | Cases | Errors |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in models:
        lines.append(
            f"| {m['model_name']} | {_fmt(m.get('overall_score'), 3)} | "
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
            "OllamaBench API is running. Frontend build not found (set STATIC_DIR).",
            status_code=200,
        )
