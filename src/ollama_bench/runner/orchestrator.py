"""Run orchestrator — drives a benchmark across hosts and models.

Responsibilities:
- Discover models and select the ones to benchmark.
- Run each enabled plugin per model with retries, timeouts, and progress events.
- Isolate failures: a single failed case, plugin, or host must never abort the
  whole run (fail-safely, PLAN §3.1/§13.5).
"""

from __future__ import annotations

import asyncio
import datetime
import statistics
import uuid
from collections.abc import Callable
from typing import Any

import httpx

from ollama_bench import __version__
from ollama_bench.config import BenchmarkConfig, JudgeConfig, RunnerConfig, config_hash
from ollama_bench.context.optimizer import recommend
from ollama_bench.domain.events import Event, Events
from ollama_bench.domain.models import (
    BenchmarkCase,
    BenchmarkCategory,
    CaseResult,
    Evaluation,
    HostConfig,
    ModelBenchmarkResult,
    ModelInfo,
    ModelResponse,
    PluginAggregate,
    RunResult,
    TimingMetrics,
    TokenMetrics,
)
from ollama_bench.judge import Judge
from ollama_bench.ollama.client import OllamaClient
from ollama_bench.ollama.discovery import discover_models
from ollama_bench.plugins.base import BenchmarkPlugin, RunContext
from ollama_bench.selection import filter_models
from ollama_bench.utils.logging import get_logger

log = get_logger("orchestrator")

EventCallback = Callable[[Event], None]
WARMUP_PROMPT = "Hi. Reply with exactly: ready"

_RETRYABLE_STATUS = {408, 429, 502, 503, 504}


class RunOrchestrator:
    def __init__(
        self,
        config: BenchmarkConfig,
        plugins: list[BenchmarkPlugin] | None = None,
        event_cb: EventCallback | None = None,
        model_filter: Callable[[ModelInfo], bool] | None = None,
        run_id: str | None = None,
        plugin_options: dict[str, dict[str, Any]] | None = None,
        client_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.plugins = plugins or []
        self.event_cb = event_cb
        self.model_filter = model_filter
        self.run_id = run_id or _new_run_id()
        self.plugin_options = plugin_options
        self._client_transport = client_transport
        self._events: list[Event] = []

    def emit(self, event: Event) -> None:
        self._events.append(event)
        if self.event_cb:
            self.event_cb(event)

    def _plugin_options_for(self, plugin_id: str) -> dict[str, Any]:
        """Effective config options for a plugin (config defaults + overrides)."""
        if self.plugin_options is not None:
            return self.plugin_options.get(plugin_id, {})
        defaults = self.config.plugins.options
        return dict(defaults.get(plugin_id, {}))

    def event_log(self) -> list[Event]:
        return list(self._events)

    async def run(self) -> RunResult:
        cfg = self.config
        hashes = config_hash(cfg)
        self.emit(Event(Events.RUN_STARTED, data={"config_hash": hashes}))

        result = RunResult(
            run_id=self.run_id,
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            app_version=__version__,
            config_hash=hashes,
            hosts=cfg.hosts,
        )

        if cfg.runner.concurrency > 1:
            result.warnings.append(
                "runner.concurrency > 1 is not implemented; running sequentially"
            )

        for host in cfg.hosts:
            try:
                await self._run_host(host, result)
            except Exception as exc:  # noqa: BLE001 - isolation per host
                msg = f"host {host.name} failed: {exc}"
                log.warning(msg)
                result.errors.append(msg)
                self.emit(Event(Events.HOST_CHECKED, host=host.name, message="failed"))

        self.emit(Event(Events.RUN_COMPLETED, data={"errors": len(result.errors)}))
        return result

    async def _run_host(self, host: HostConfig, result: RunResult) -> None:
        client = OllamaClient(
            host.base_url, host.timeout_seconds, transport=self._client_transport
        )
        try:
            version = await client.health()
            self.emit(Event(Events.HOST_CHECKED, host=host.name, message="ok", data=version))
            all_models = await discover_models(client, host.name)
            selected = _select_models(all_models, self.config.models, self.model_filter)
            for model in selected:
                self.emit(
                    Event(Events.MODEL_DISCOVERED, host=host.name, model=model.model_name)
                )
                try:
                    summary = await _run_model(self, client, host, model)
                    result.models.append(summary)
                except Exception as exc:  # noqa: BLE001 - isolation per model
                    msg = f"model {model.model_name} failed: {exc}"
                    log.warning(msg)
                    result.errors.append(msg)
        finally:
            await client.aclose()


async def _run_model(
    orchestrator: RunOrchestrator,
    client: OllamaClient,
    host: HostConfig,
    model: ModelInfo,
) -> ModelBenchmarkResult:
    summary = ModelBenchmarkResult(
        host_name=host.name, model_name=model.model_name, model_digest=model.digest
    )
    runner = orchestrator.config.runner
    weights = orchestrator.config.weights.model_dump()
    judge = _build_judge(client, orchestrator.config.judge)

    for _ in range(runner.warmup_runs):
        await _warmup(client, model, runner.temperature)

    latencies_ms: list[float] = []
    ttft_ms: list[float] = []
    tps: list[float] = []
    scored: list[tuple[float, float]] = []
    errors = 0
    completion_tokens = 0
    active = [p for p in orchestrator.plugins if p.supports_model(model)]

    for plugin in active:
        ctx = RunContext(
            {**model.model_dump(), **orchestrator._plugin_options_for(plugin.id)}
        )
        ctx.judge = judge
        orchestrator.emit(
            Event(Events.PLUGIN_STARTED, host=host.name, model=model.model_name, plugin=plugin.id)
        )
        case_results: list[CaseResult] = []
        failed_reason: str | None = None
        p_lat: list[float] = []
        p_ttft: list[float] = []
        p_tps: list[float] = []

        try:
            await plugin.prepare(ctx)
            for case in plugin.cases(ctx):
                for attempt in range(1, runner.repetitions + 1):
                    orchestrator.emit(
                        Event(Events.CASE_STARTED, host=host.name, model=model.model_name, case_id=case.id)
                    )
                    resp = await _send_with_retries(client, model, plugin, case, ctx, runner)
                    try:
                        evaluation = await plugin.evaluate(case, resp, ctx)
                    except Exception as exc:  # noqa: BLE001 - isolation per case
                        log.warning("evaluate failed for %s/%s: %s", model.model_name, case.id, exc)
                        evaluation = Evaluation(
                            score=0.0, passed=False, metrics={"error": str(exc)}
                        )
                    case_results.append(
                        CaseResult(
                            case=case, model=model, response=resp, evaluation=evaluation, attempt=attempt
                        )
                    )
                    if resp.error:
                        errors += 1
                        orchestrator.emit(Event(Events.CASE_FAILED, case_id=case.id, message=resp.error))
                        continue
                    if resp.timing.total_ms:
                        latencies_ms.append(resp.timing.total_ms)
                        p_lat.append(resp.timing.total_ms)
                    if resp.timing.time_to_first_token_ms is not None:
                        ttft_ms.append(resp.timing.time_to_first_token_ms)
                        p_ttft.append(resp.timing.time_to_first_token_ms)
                    if resp.tokens.tokens_per_second is not None:
                        tps.append(resp.tokens.tokens_per_second)
                        p_tps.append(resp.tokens.tokens_per_second)
                    completion_tokens += resp.tokens.completion_tokens or 0
                    orchestrator.emit(
                        Event(Events.CASE_COMPLETED, case_id=case.id, data={"ms": resp.timing.total_ms})
                    )
            agg = plugin.aggregate(case_results)
        except Exception as exc:  # noqa: BLE001 - isolation per plugin
            failed_reason = str(exc)
            msg = f"plugin {plugin.id} failed for {model.model_name}: {exc}"
            log.warning(msg)
            summary.warnings.append(msg)
            agg = PluginAggregate(
                plugin_id=plugin.id,
                model_name=model.model_name,
                host_name=host.name,
                total_cases=len(case_results),
                failed_cases=len(case_results),
                metrics={"error": failed_reason},
            )
        finally:
            try:
                await plugin.teardown(ctx)
            except Exception as exc:  # noqa: BLE001 - teardown must not mask results
                log.warning("teardown failed for %s: %s", plugin.id, exc)

        # Per-plugin latency/ttft/tokens aggregates (from successful cases only).
        try:
            agg.latency_p50_ms = _pct(p_lat, 0.5)
            agg.latency_p95_ms = _pct(p_lat, 0.95)
            agg.time_to_first_token_p50_ms = _pct(p_ttft, 0.5)
            agg.tokens_per_second = statistics.mean(p_tps) if p_tps else None
            agg.cases_run = len(p_lat)
        except Exception:  # noqa: BLE001 - never mask orchestration on a stats bug
            pass
        summary.plugins.append(agg)
        summary.cases.extend(case_results)
        if agg.score is not None:
            scored.append((agg.score, _category_weight(weights, plugin.category)))
        orchestrator.emit(
            Event(
                Events.PLUGIN_COMPLETED,
                plugin=plugin.id,
                message=failed_reason,
                data={"score": agg.score},
            )
        )

    summary.cases_run = sum(p.total_cases for p in summary.plugins)
    summary.errors = errors
    summary.completion_tokens_total = completion_tokens
    summary.latency_p50_ms = _pct(latencies_ms, 0.5)
    summary.latency_p95_ms = _pct(latencies_ms, 0.95)
    summary.time_to_first_token_p50_ms = _pct(ttft_ms, 0.5)
    summary.tokens_per_second = statistics.mean(tps) if tps else None
    summary.overall_score = _weighted_mean(scored)

    lc = next((p for p in summary.plugins if p.plugin_id == "long_context"), None)
    if lc and orchestrator.config.context_optimization.enabled:
        raw_per_ctx = (lc.metrics.get("per_context_score") or {})
        per_ctx = {int(size): q for size, q in raw_per_ctx.items() if q is not None}
        summary.context_recommendation = recommend(
            per_ctx,
            orchestrator.config.context_optimization.candidate_sizes,
            quality_threshold=orchestrator.config.context_optimization.quality_threshold,
            latency_budget_ms=orchestrator.config.context_optimization.latency_budget_ms,
        )
    return summary


def _build_judge(client: OllamaClient, cfg: JudgeConfig) -> Judge | None:
    """Create the judge wrapper, or None when disabled/unspecified."""
    if not cfg.enabled or not cfg.model:
        return None
    return Judge(client, cfg.model, cfg.temperature)


def _category_weight(weights: dict[str, float], category: BenchmarkCategory) -> float:
    """Per-category weight, defaulting to 1.0 for unlisted categories."""
    return float(weights.get(category.value, 1.0))


def _weighted_mean(scored: list[tuple[float, float]]) -> float | None:
    """Weighted mean of (score, weight) pairs; None when nothing was scored."""
    total_weight = sum(w for _, w in scored)
    if not total_weight:
        return None
    return round(sum(s * w for s, w in scored) / total_weight, 4)


async def _send_with_retries(
    client: OllamaClient,
    model: ModelInfo,
    plugin: BenchmarkPlugin,
    case: BenchmarkCase,
    ctx: RunContext,
    runner: RunnerConfig,
) -> ModelResponse:
    """Send a chat request, retrying only transient errors (PLAN §13.5).

    Never raises for transport/HTTP failures — it returns a ModelResponse with
    ``error`` set so the case is recorded as failed and the run continues.
    """
    request = plugin.build_request(case, model, ctx)
    last_error: str | None = None

    # `max_retries` is the number of retries AFTER the initial attempt, so the
    # loop always runs at least once (max_retries=0 still sends the request).
    for attempt in range(runner.max_retries + 1):
        try:
            resp = await client.chat(
                model.model_name,
                messages=request["messages"],
                options=request.get("options"),
                stream=request.get("stream", True),
            )
            if resp.error is None:
                return resp
            last_error = resp.error
        except httpx.HTTPStatusError as exc:
            last_error = f"HTTP {exc.response.status_code}: {exc}"
            if not _retryable(exc.response.status_code):
                return _error_response(last_error)
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 - never abort the run on a send error
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < runner.max_retries:
            await asyncio.sleep(runner.retry_backoff_seconds * attempt)

    return _error_response(last_error or "unknown error")


def _retryable(status_code: int) -> bool:
    return status_code in _RETRYABLE_STATUS


def _error_response(message: str) -> ModelResponse:
    return ModelResponse(
        raw={},
        text="",
        timing=TimingMetrics(),
        tokens=TokenMetrics(),
        error=message,
    )


async def _warmup(client: OllamaClient, model: ModelInfo, temperature: float) -> None:
    try:
        await client.chat(
            model.model_name,
            messages=[{"role": "user", "content": WARMUP_PROMPT}],
            options={"temperature": temperature, "num_predict": 8},
            stream=True,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("warmup failed for %s: %s", model.model_name, exc)


def _select_models(
    all_models: list[ModelInfo],
    configured: list[str],
    model_filter: Callable[[ModelInfo], bool] | None,
) -> list[ModelInfo]:
    if model_filter is not None:
        return [m for m in all_models if model_filter(m)]
    return filter_models(all_models, configured or None)


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return round(_nearest_rank(values, q), 1)


def _nearest_rank(values: list[float], q: float) -> float:
    sorted_v = sorted(values)
    idx = max(0, min(len(sorted_v) - 1, round(q * (len(sorted_v) - 1))))
    return sorted_v[idx]


def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]
