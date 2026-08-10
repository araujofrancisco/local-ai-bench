"""API boot, repository round-trip, and live-progress logic tests.

Environment variables must be set before ``ollama_bench.api.app`` is imported
so the module-level config/DB defaults point at isolated test artifacts.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TEST_DIR = tempfile.mkdtemp(prefix="ollamabench_api_test_")
_TMP_DB = os.path.join(_TEST_DIR, "benchmark.db")
os.environ["DATABASE_URL"] = _TMP_DB
os.environ["CONFIG_PATH"] = str(Path(__file__).resolve().parents[2] / "config" / "default.yaml")

from fastapi.testclient import TestClient  # noqa: E402

from ollama_bench.api.app import RunManager, RunStatus, app  # noqa: E402
from ollama_bench.domain.events import Event, Events  # noqa: E402
from ollama_bench.domain.models import (  # noqa: E402
    BenchmarkCase,
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
from ollama_bench.storage.repository import BenchmarkRepository  # noqa: E402


def _sample_run() -> RunResult:
    host = HostConfig(name="h1", base_url="http://example.invalid")
    model = ModelInfo(host_name="h1", model_name="m1", digest="d1")
    case = BenchmarkCase(
        id="c1", plugin_id="smoke", dataset_version="v1", input={"prompt": "hi"}
    )
    resp = ModelResponse(
        raw={},
        text="ok",
        timing=TimingMetrics(total_ms=10.0, time_to_first_token_ms=2.0),
        tokens=TokenMetrics(tokens_per_second=50.0, completion_tokens=5),
    )
    ev = Evaluation(score=1.0, passed=True)
    case_result = CaseResult(case=case, model=model, response=resp, evaluation=ev, attempt=1)
    agg = PluginAggregate(
        plugin_id="smoke", model_name="m1", host_name="h1", total_cases=1, successful_cases=1, score=1.0
    )
    mr = ModelBenchmarkResult(
        host_name="h1",
        model_name="m1",
        model_digest="d1",
        plugins=[agg],
        cases=[case_result],
        cases_run=1,
        overall_score=1.0,
    )
    return RunResult(
        run_id="run123",
        timestamp="2026-01-01T00:00:00Z",
        app_version="0.1.0",
        config_hash="abc",
        hosts=[host],
        models=[mr],
    )


def _make_run(run_id: str, model_name: str, host_name: str, timestamp: str) -> RunResult:
    host = HostConfig(name=host_name, base_url="http://example.invalid")
    model = ModelInfo(host_name=host_name, model_name=model_name, digest="d")
    case = BenchmarkCase(
        id="c1", plugin_id="smoke", dataset_version="v1", input={"prompt": "hi"}
    )
    resp = ModelResponse(
        raw={},
        text="ok",
        timing=TimingMetrics(total_ms=10.0, time_to_first_token_ms=2.0),
        tokens=TokenMetrics(tokens_per_second=50.0, completion_tokens=5),
    )
    ev = Evaluation(score=1.0, passed=True)
    case_result = CaseResult(case=case, model=model, response=resp, evaluation=ev, attempt=1)
    agg = PluginAggregate(
        plugin_id="smoke", model_name=model_name, host_name=host_name, total_cases=1, successful_cases=1, score=1.0
    )
    mr = ModelBenchmarkResult(
        host_name=host_name,
        model_name=model_name,
        model_digest="d",
        plugins=[agg],
        cases=[case_result],
        cases_run=1,
        overall_score=1.0,
    )
    return RunResult(
        run_id=run_id,
        timestamp=timestamp,
        app_version="0.1.0",
        config_hash="abc",
        hosts=[host],
        models=[mr],
    )


def test_repository_round_trip_persists_cases() -> None:
    repo = BenchmarkRepository(_TMP_DB)
    repo.save_run(_sample_run())

    runs = repo.list_runs()
    assert runs[0]["run_id"] == "run123"
    assert runs[0]["model_names"] == ["m1"]
    assert runs[0]["hosts"][0]["name"] == "h1"

    run = repo.get_run("run123")
    assert run is not None
    assert run["model_names"] == ["m1"]

    cases = repo._conn.execute("SELECT run_id, model_name, plugin_id, case_id FROM cases").fetchall()
    assert len(cases) == 1
    assert cases[0]["case_id"] == "c1"
    repo.close()


def test_api_health() -> None:
    with TestClient(app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_api_plugins_registered() -> None:
    with TestClient(app) as client:
        resp = client.get("/api/plugins")
        assert resp.status_code == 200
        ids = [p["id"] for p in resp.json()["plugins"]]
        assert "smoke" in ids


def test_api_plugins_include_details_and_options() -> None:
    with TestClient(app) as client:
        resp = client.get("/api/plugins")
        assert resp.status_code == 200
        plugins = {p["id"]: p for p in resp.json()["plugins"]}
        coding = plugins["coding"]
        assert "description" in coding and coding["description"]
        assert coding["dataset_version"]
        assert coding["modalities"] == ["text"]
        assert "execute_code" in coding["options"]
        assert coding["options"]["timeout_seconds"] == 30


def test_api_update_plugin_options_persists() -> None:
    with TestClient(app) as client:
        resp = client.put(
            "/api/plugins/coding/options",
            json={"options": {"execute_code": True, "timeout_seconds": 45}},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["plugin_id"] == "coding"
        assert body["options"]["execute_code"] is True
        assert body["options"]["timeout_seconds"] == 45

        again = client.get("/api/plugins")
        coding = {p["id"]: p for p in again.json()["plugins"]}["coding"]
        assert coding["options"]["execute_code"] is True

        unknown = client.put("/api/plugins/nope/options", json={"options": {}})
        assert unknown.status_code == 404

        bad = client.put(
            "/api/plugins/coding/options",
            json={"options": {"execute_code": {"nested": True}}},
        )
        assert bad.status_code == 422


def test_api_delete_benchmark() -> None:
    repo = BenchmarkRepository(_TMP_DB)
    try:
        repo.save_run(_make_run("todel", "alpha", "hA", "2026-02-10T00:00:00+00:00"))
    finally:
        repo.close()

    with TestClient(app) as client:
        resp = client.delete("/api/benchmarks/todel")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == "todel"

        missing = client.delete("/api/benchmarks/todel")
        assert missing.status_code == 404

        gone = client.get("/api/benchmarks/todel")
        assert gone.status_code == 404


def test_api_history_filters() -> None:
    repo = BenchmarkRepository(_TMP_DB)
    try:
        repo.save_run(_make_run("runalpha", "alpha", "hA", "2026-02-10T00:00:00+00:00"))
        repo.save_run(_make_run("runbeta", "beta", "hB", "2026-03-10T00:00:00+00:00"))
    finally:
        repo.close()

    with TestClient(app) as client:
        def ids(params: str) -> list[str]:
            resp = client.get(f"/api/history?{params}")
            assert resp.status_code == 200
            return [r["run_id"] for r in resp.json()["runs"]]

        assert "runalpha" in ids("search=runalpha")
        assert "runbeta" not in ids("search=runalpha")
        assert "runalpha" in ids("model=alpha")
        assert "runbeta" not in ids("model=alpha")
        assert "runbeta" in ids("host=hB")
        assert "runalpha" not in ids("host=hB")
        assert "runbeta" in ids("date_from=2026-03-01")
        assert "runalpha" not in ids("date_from=2026-03-01")
        assert "runalpha" in ids("date_to=2026-02-28")
        assert "runbeta" not in ids("date_to=2026-02-28")
        assert "runalpha" in ids("model=alpha&date_to=2026-02-28")
        assert "runbeta" not in ids("model=alpha&date_to=2026-02-28")

        filters_resp = client.get("/api/history")
        filters = filters_resp.json()["filters"]
        assert "alpha" in filters["models"]
        assert "hB" in filters["hosts"]


def test_api_benchmarks_returns_run_with_model_names() -> None:
    with TestClient(app) as client:
        resp = client.get("/api/benchmarks")
        assert resp.status_code == 200
        runs = resp.json()["runs"]
        assert any(r["model_names"] == ["m1"] for r in runs)


def test_api_get_benchmark_found_and_missing() -> None:
    with TestClient(app) as client:
        ok = client.get("/api/benchmarks/run123")
        assert ok.status_code == 200
        assert ok.json()["models"][0]["model_name"] == "m1"

        missing = client.get("/api/benchmarks/nope")
        assert missing.status_code == 404


def test_api_export_formats() -> None:
    with TestClient(app) as client:
        for fmt in ("json", "csv", "md"):
            resp = client.get(f"/api/export/run123.{fmt}")
            assert resp.status_code == 200, fmt
        bad = client.get("/api/export/run123.xml")
        assert bad.status_code == 400


def test_api_run_requires_models() -> None:
    with TestClient(app) as client:
        resp = client.post("/api/benchmarks/run", json={"model_names": []})
        assert resp.status_code == 422


def test_run_status_progress_tracking() -> None:
    manager = RunManager()
    status = RunStatus(run_id="r1")
    manager.set(status)

    manager.on_event(status, Event(Events.RUN_STARTED))
    assert status.status == "running"

    manager.on_event(status, Event(Events.CASE_STARTED, model="m1", case_id="a"))
    manager.on_event(status, Event(Events.CASE_STARTED, model="m1", case_id="b"))
    manager.on_event(status, Event(Events.CASE_COMPLETED, model="m1", case_id="a"))
    assert status.completed == 1
    assert status.total == 2
    assert status.progress == 0.5

    manager.on_event(status, Event(Events.CASE_FAILED, model="m1", case_id="b", message="boom"))
    assert status.completed == 2
    assert status.errors == 1
    assert status.progress == 1.0

    msg = status.to_message("complete")
    assert msg["type"] == "complete"
    assert msg["run_id"] == "r1"
