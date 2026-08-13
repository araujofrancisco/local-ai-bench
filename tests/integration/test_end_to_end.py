"""End-to-end test: a full benchmark run against a mocked Ollama host.

Covers discovery -> orchestration -> scoring -> report generation -> SQLite
persistence (PLAN §23.5) without any real network I/O.
"""

from __future__ import annotations

import asyncio
import json

from local_ai_bench.config import BenchmarkConfig
from local_ai_bench.domain.events import Event, Events
from local_ai_bench.domain.models import HostConfig
from local_ai_bench.plugins.builtin.smoke import SmokePlugin
from local_ai_bench.reporting.repository import write_report
from local_ai_bench.runner.orchestrator import RunOrchestrator
from local_ai_bench.storage.repository import BenchmarkRepository
from tests.fixtures.mock_ollama import mock_transport


def _orchestrator() -> RunOrchestrator:
    cfg = BenchmarkConfig(
        hosts=[HostConfig(name="mock", base_url="http://mock.ollama")],
        runner={"repetitions": 1, "warmup_runs": 0, "max_retries": 0},
    )
    return RunOrchestrator(
        cfg,
        plugins=[SmokePlugin()],
        run_id="e2e-001",
        client_transport=mock_transport(),
    )


def test_end_to_end_run_scores_and_reports(tmp_path) -> None:
    result = asyncio.run(_orchestrator().run())

    assert result.run_id == "e2e-001"
    assert result.errors == []
    assert len(result.models) == 1

    m = result.models[0]
    assert m.model_name == "e2e-model:latest"
    assert m.cases_run == 2
    assert m.overall_score == 1.0
    assert any(p.plugin_id == "smoke" and p.score == 1.0 for p in m.plugins)

    out = tmp_path / "reports"
    write_report(str(out), result, formats=["json", "markdown", "html"])
    report_json = out / "e2e-001" / "report.json"
    assert report_json.exists()
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["run_id"] == "e2e-001"
    assert (out / "e2e-001" / "report.md").exists()
    assert (out / "e2e-001" / "report.html").exists()


def test_end_to_end_persists_to_sqlite(tmp_path) -> None:
    result = asyncio.run(_orchestrator().run())

    repo = BenchmarkRepository(tmp_path / "benchmark.db")
    try:
        repo.save_run(result)
        runs = repo.list_runs()
        assert any(r["run_id"] == "e2e-001" for r in runs)
        rows = repo.compare_models(run_id="e2e-001")
        assert rows and rows[0]["model_name"] == "e2e-model:latest"
        assert rows[0]["overall_score"] == 1.0
    finally:
        repo.close()


def test_run_planned_reports_exact_total_before_cases_run() -> None:
    """The planned total is announced before the first case completes and equals
    the number of cases actually executed."""
    orch = _orchestrator()
    events: list[Event] = []

    def capture(event: Event) -> None:
        events.append(event)

    orch.event_cb = capture
    result = asyncio.run(orch.run())

    planned = [e for e in events if e.kind == Events.RUN_PLANNED]
    assert planned, "RUN_PLANNED should be emitted"
    assert planned[0].data["total_cases"] == result.models[0].cases_run
    completed = [i for i, e in enumerate(events) if e.kind == Events.CASE_COMPLETED]
    assert completed, "expected case completions"
    assert (
        events.index(planned[0]) < completed[0]
    ), "planned total must be announced before any case completes"


def test_function_calling_stream_captures_tool_calls() -> None:
    """Tool calls emitted mid-stream must survive into ModelResponse and score."""
    from local_ai_bench.plugins.builtin.function_calling import FunctionCallingPlugin

    cfg = BenchmarkConfig(
        hosts=[HostConfig(name="mock", base_url="http://mock.ollama")],
        runner={"repetitions": 1, "warmup_runs": 0, "max_retries": 0},
    )
    orch = RunOrchestrator(
        cfg,
        plugins=[FunctionCallingPlugin()],
        run_id="e2e-fc",
        client_transport=mock_transport(),
    )
    result = asyncio.run(orch.run())

    assert result.errors == []
    assert len(result.models) == 1
    m = result.models[0]
    assert m.model_name == "e2e-model:latest"
    assert m.cases_run == 3
    fc = next(p for p in m.plugins if p.plugin_id == "function_calling")
    assert fc.metrics["tool_call_ratio"] == 1.0
    weather = next(cr for cr in m.cases if cr.case.id == "fc_weather_0001")
    assert (weather.response.tool_calls or [])[0]["function"]["name"] == "get_weather"
    assert weather.evaluation.score == 1.0
