"""End-to-end test: a full benchmark run against a mocked Ollama host.

Covers discovery -> orchestration -> scoring -> report generation -> SQLite
persistence (PLAN §23.5) without any real network I/O.
"""

from __future__ import annotations

import asyncio
import json

import httpx

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


def _recording_transport(recorded: list[dict]) -> httpx.MockTransport:
    """Mock transport that records every /api/chat payload before serving it."""
    inner = mock_transport()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            recorded.append(json.loads(request.content or b"{}"))
        return inner.handler(request)

    return httpx.MockTransport(handler)


def test_multi_turn_forwards_history_to_ollama() -> None:
    """Each turn's request must include the full conversation so the model can
    actually recall earlier turns (regression: only the current prompt was sent)."""
    from local_ai_bench.plugins.builtin.multi_turn import MultiTurnPlugin

    recorded: list[dict] = []
    cfg = BenchmarkConfig(
        hosts=[HostConfig(name="mock", base_url="http://mock.ollama")],
        runner={"repetitions": 1, "warmup_runs": 0, "max_retries": 0},
    )
    orch = RunOrchestrator(
        cfg,
        plugins=[MultiTurnPlugin()],
        run_id="e2e-mt",
        client_transport=_recording_transport(recorded),
    )
    result = asyncio.run(orch.run())

    assert result.errors == []
    assert recorded, "expected at least one /api/chat call"
    second = recorded[1]
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["user", "assistant", "user"], second["messages"]
    assert "KILO-7" in second["messages"][0]["content"]  # turn 0 prompt resent
    assert "token" in second["messages"][2]["content"]  # turn 1 prompt


def test_agent_tool_use_end_to_end_loop() -> None:
    """The agent loop calls a tool, sees its result, and then answers."""
    from local_ai_bench.plugins.builtin.agent_tool_use import AgentToolUsePlugin

    recorded: list[dict] = []

    def agent_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            payload = json.loads(request.content or b"{}")
            recorded.append(payload)
            # Turn 0 (no tool result yet): call the calculate tool.
            # Turn 1 (tool result visible): answer.
            if any(m.get("role") == "tool" for m in payload.get("messages", [])):
                body = {"message": {"role": "assistant", "content": "The result is 420."}, "done": False}
                done = {"done": True, "done_reason": "stop", "total_duration": 500_000_000, "eval_count": 5, "eval_duration": 100_000_000}
            else:
                body = {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "calculate",
                                    "arguments": {"expression": "(15 * 24) + (360 / 12)"},
                                }
                            }
                        ],
                    },
                    "done": False,
                }
                done = {"done": True, "done_reason": "tool_calls", "total_duration": 500_000_000, "eval_count": 5, "eval_duration": 100_000_000}
            lines = [json.dumps(body), json.dumps(done)]
            return httpx.Response(200, content="\n".join(lines) + "\n", headers={"content-type": "application/x-ndjson"})
        return mock_transport().handler(request)

    cfg = BenchmarkConfig(
        hosts=[HostConfig(name="mock", base_url="http://mock.ollama")],
        runner={"repetitions": 1, "warmup_runs": 0, "max_retries": 0},
    )
    orch = RunOrchestrator(
        cfg,
        plugins=[AgentToolUsePlugin()],
        run_id="e2e-agent",
        client_transport=httpx.MockTransport(agent_handler),
    )
    result = asyncio.run(orch.run())

    assert result.errors == []
    # Tool results must be fed back to the model in a later request.
    result_payloads = [p for p in recorded if any(m.get("role") == "tool" for m in p["messages"])]
    assert result_payloads, "expected a request that carries a tool result"
    second = result_payloads[0]
    assert "Result: 390.0" in next(m["content"] for m in second["messages"] if m.get("role") == "tool")
    # The full conversation is forwarded: user prompt + assistant tool_calls + tool result.
    assert second["messages"][0]["role"] == "user"
    assert second["messages"][1]["role"] == "assistant"
    # The agent_math case scores a perfect loop (correct tool + correct answer).
    math = next(cr for cr in result.models[0].cases if cr.case.id == "agent_math_0002")
    assert math.evaluation.score == 1.0
    assert math.evaluation.passed is True
