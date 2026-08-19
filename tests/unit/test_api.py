"""API boot, repository round-trip, and live-progress logic tests.

Environment variables must be set before ``local_ai_bench.api.app`` is imported
so the module-level config/DB defaults point at isolated test artifacts.
"""

from __future__ import annotations

import datetime
import os
import tempfile
from pathlib import Path

_TEST_DIR = tempfile.mkdtemp(prefix="ollamabench_api_test_")
_TMP_DB = os.path.join(_TEST_DIR, "benchmark.db")
os.environ["DATABASE_URL"] = _TMP_DB
os.environ["CONFIG_PATH"] = str(Path(__file__).resolve().parents[2] / "config" / "default.yaml")

from fastapi.testclient import TestClient  # noqa: E402

from local_ai_bench.api.app import RunManager, RunStatus, app, run_manager  # noqa: E402
from local_ai_bench.domain.events import Event, Events  # noqa: E402
from local_ai_bench.domain.models import (  # noqa: E402
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
from local_ai_bench.storage.repository import BenchmarkRepository, _uses_host_identity  # noqa: E402


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


def test_get_model_history_includes_host() -> None:
    """Per-model history rows carry the host so duplicate names across servers
    stay distinguishable (powers the `history --model` CLI table)."""
    repo = BenchmarkRepository(_TMP_DB)
    repo.save_run(_sample_run())
    try:
        rows = repo.get_model_history("m1")
        assert rows and rows[0]["host_name"] == "h1"
        assert rows[0]["run_id"] == "run123"
    finally:
        repo.close()


def _two_host_run(run_id: str) -> RunResult:
    """One run where the identical model name ran on two different servers."""
    host_a = HostConfig(name="hA", base_url="http://a.invalid")
    host_b = HostConfig(name="hB", base_url="http://b.invalid")
    models = []
    for host, score in ((host_a, 0.9), (host_b, 0.2)):
        info = ModelInfo(host_name=host.name, model_name="qwen3.5:0.8b", digest="d")
        case = BenchmarkCase(id="c1", plugin_id="smoke", dataset_version="v1", input={"prompt": "hi"})
        resp = ModelResponse(
            raw={},
            text="ok",
            timing=TimingMetrics(total_ms=10.0, time_to_first_token_ms=2.0),
            tokens=TokenMetrics(tokens_per_second=50.0, completion_tokens=5),
        )
        ev = Evaluation(score=score, passed=score > 0.5)
        case_result = CaseResult(case=case, model=info, response=resp, evaluation=ev, attempt=1)
        agg = PluginAggregate(
            plugin_id="smoke", model_name="qwen3.5:0.8b", host_name=host.name,
            total_cases=1, successful_cases=1 if score > 0.5 else 0, score=score,
        )
        models.append(
            ModelBenchmarkResult(
                host_name=host.name, model_name="qwen3.5:0.8b",
                plugins=[agg], cases=[case_result], cases_run=1, overall_score=score,
            )
        )
    return RunResult(
        run_id=run_id,
        timestamp="2026-01-01T00:00:00Z",
        app_version="0.1.0",
        config_hash="x",
        hosts=[host_a, host_b],
        models=models,
    )


def test_repository_same_model_on_two_hosts_stays_distinct() -> None:
    """Saving one run with the same model on two servers must persist BOTH rows —
    the second host must not overwrite the first (regression for host-unaware keys)."""
    repo = BenchmarkRepository(_TMP_DB)
    try:
        repo.save_run(_two_host_run("twohost"))

        rows = repo.compare_models(run_id="twohost")
        by_host = {m["host_name"]: m for m in rows}
        assert sorted(by_host) == ["hA", "hB"]
        # Each host keeps its own score and its own plugin aggregate.
        assert by_host["hA"]["overall_score"] == 0.9
        assert by_host["hB"]["overall_score"] == 0.2
        assert by_host["hA"]["plugins"][0]["score"] == 0.9
        assert by_host["hB"]["plugins"][0]["score"] == 0.2

        models_rows = repo._conn.execute(
            "SELECT host_name, COUNT(*) AS n FROM models WHERE run_id='twohost' GROUP BY host_name"
        ).fetchall()
        assert {r["host_name"] for r in models_rows} == {"hA", "hB"}

        cases = repo.cases_for_run("twohost")
        assert sorted(c["host_name"] for c in cases) == ["hA", "hB"]
    finally:
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
        assert ok.json()["models"][0]["host_name"] == "h1"

        missing = client.get("/api/benchmarks/nope")
        assert missing.status_code == 404


def test_api_benchmark_detail_includes_per_plugin_aggregates() -> None:
    with TestClient(app) as client:
        resp = client.get("/api/benchmarks/run123")
        assert resp.status_code == 200
        model = resp.json()["models"][0]
        assert "plugins" in model
        plugins = model["plugins"]
        assert plugins and plugins[0]["plugin_id"] == "smoke"
        assert "latency_p50_ms" in plugins[0]
        assert "time_to_first_token_p50_ms" in plugins[0]
        assert "tokens_per_second" in plugins[0]


def test_api_compare_multi_run_with_run_column() -> None:
    # Seed a second run, then compare both runs at once.
    repo = BenchmarkRepository(_TMP_DB)
    repo.save_run(_make_run("run-multi", "m2", "h1", "2026-01-02T00:00:00Z"))
    repo.close()
    with TestClient(app) as client:
        resp = client.get("/api/compare?run=run123&run=run-multi")
        assert resp.status_code == 200
        models = resp.json()["models"]
        run_ids = {m["run_id"] for m in models}
        assert {"run123", "run-multi"} <= run_ids
        # Each row carries a run_id and per-plugin aggregates.
        for m in models:
            assert m["plugins"]
            assert "latency_p50_ms" in m["plugins"][0]


def test_api_compare_unscoped_includes_per_plugin_aggregates() -> None:
    # Default compare view (no ?run=) must still surface per-plugin aggregates
    # so weighted/per-plugin columns render. Regression test.
    with TestClient(app) as client:
        resp = client.get("/api/compare")
        assert resp.status_code == 200
        for m in resp.json()["models"]:
            assert m["run_id"] is not None
            assert m["host_name"], "compare rows must carry the host name"
            assert m["plugins"], "unscoped compare rows must carry per-plugin data"
            assert m["plugins"][0]["plugin_id"] == "smoke"
            assert m["plugins"][0]["score"] is not None


def test_api_benchmark_cases_endpoint() -> None:
    with TestClient(app) as client:
        resp = client.get("/api/benchmarks/run123/cases")
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == "run123"
        assert body["count"] == 1
        case = body["cases"][0]
        for field in (
            "model_name", "plugin_id", "case_id", "passed", "score",
            "total_ms", "time_to_first_token_ms", "tokens_per_second", "error",
        ):
            assert field in case
        missing = client.get("/api/benchmarks/does-not-exist/cases")
        assert missing.status_code == 200
        assert missing.json()["count"] == 0


def test_api_plugins_compare_default() -> None:
    with TestClient(app) as client:
        resp = client.get("/api/plugins")
        assert resp.status_code == 200
        body = resp.json()
        assert "compare_default" in body
        assert isinstance(body["compare_default"], list)
        assert any(p["id"] == "smoke" for p in body["plugins"])


def test_api_weights_get_structure() -> None:
    with TestClient(app) as client:
        resp = client.get("/api/weights")
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {"defaults", "overrides", "effective"}
        assert body["defaults"]["coding"] == 1.0
        assert set(body["effective"]) == set(body["defaults"])


def test_api_weights_put_round_trip() -> None:
    with TestClient(app) as client:
        resp = client.put("/api/weights", json={"weights": {"coding": 2.5}})
        assert resp.status_code == 200
        assert resp.json()["effective"]["coding"] == 2.5
        assert resp.json()["overrides"] == {"coding": 2.5}

        persisted = client.get("/api/weights").json()
        assert persisted["effective"]["coding"] == 2.5

        # Setting a category back to its default prunes the override.
        reset = client.put(
            "/api/weights", json={"weights": {"coding": 1.0, "vision": 1.0}}
        )
        assert reset.status_code == 200
        assert reset.json()["overrides"] == {}
        assert reset.json()["effective"]["coding"] == 1.0


def test_api_weights_put_validation() -> None:
    with TestClient(app) as client:
        unknown = client.put("/api/weights", json={"weights": {"nope": 1.0}})
        assert unknown.status_code == 422
        negative = client.put("/api/weights", json={"weights": {"coding": -1}})
        assert negative.status_code == 422


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

    manager.on_event(status, Event(Events.RUN_PLANNED, data={"total_cases": 2}))
    assert status.total == 2

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


def test_run_status_message_includes_status_metadata() -> None:
    status = RunStatus(
        run_id="r2",
        status="running",
        started_at="2026-01-01T00:00:00+00:00",
        models=["m1", "m2"],
        plugins=["smoke"],
    )
    msg = status.to_message("progress")
    assert msg["started_at"] == "2026-01-01T00:00:00+00:00"
    assert msg["models"] == ["m1", "m2"]
    assert msg["plugins"] == ["smoke"]


def test_run_manager_keeps_active_statuses() -> None:
    manager = RunManager(terminal_ttl_seconds=0.0)
    running = RunStatus(run_id="run1", status="running", started_at="2026-01-01T00:00:00Z")
    manager.set(running)
    assert manager.get("run1") is not None
    assert [s.run_id for s in manager.active()] == ["run1"]


def test_run_manager_evicts_expired_terminal_statuses() -> None:
    manager = RunManager(terminal_ttl_seconds=0.0)
    done = RunStatus(run_id="expired1", status="completed", finished_at="2026-01-01T00:00:00Z")
    manager.set(done)
    # TTL of 0s => terminal status is immediately evicted on read-back.
    assert manager.get("expired1") is None
    assert manager.active() == []


def test_run_manager_enforces_cap_evicting_oldest_terminal() -> None:
    now = datetime.datetime.now(datetime.UTC)
    manager = RunManager(terminal_ttl_seconds=3600.0, max_statuses=2)
    manager.set(RunStatus(run_id="a", status="completed", finished_at=(now - datetime.timedelta(seconds=30)).isoformat()))
    manager.set(RunStatus(run_id="b", status="completed", finished_at=(now - datetime.timedelta(seconds=20)).isoformat()))
    manager.set(RunStatus(run_id="c", status="completed", finished_at=(now - datetime.timedelta(seconds=10)).isoformat()))
    assert manager.get("a") is None       # oldest terminal evicted by the cap
    assert manager.get("b") is not None
    assert manager.get("c") is not None


def test_api_batch_delete_benchmarks() -> None:
    repo = BenchmarkRepository(_TMP_DB)
    try:
        repo.save_run(_make_run("b1", "alpha", "hA", "2026-02-10T00:00:00+00:00"))
        repo.save_run(_make_run("b2", "beta", "hB", "2026-02-11T00:00:00+00:00"))
        repo.save_run(_make_run("b3", "gamma", "hC", "2026-02-12T00:00:00+00:00"))
    finally:
        repo.close()

    with TestClient(app) as client:
        resp = client.post("/api/benchmarks/delete", json={"run_ids": ["b1", "b3"]})
        assert resp.status_code == 200
        assert resp.json()["count"] == 2

        assert client.get("/api/benchmarks/b1").status_code == 404
        assert client.get("/api/benchmarks/b3").status_code == 404
        assert client.get("/api/benchmarks/b2").status_code == 200

        # Empty batch is rejected by the request validator.
        assert client.post("/api/benchmarks/delete", json={"run_ids": []}).status_code == 422


def test_api_batch_delete_accepts_delete_method() -> None:
    """The bulk endpoint must accept DELETE so the UI can send a single atomic
    request instead of N concurrent per-run DELETEs (which race on SQLite)."""
    repo = BenchmarkRepository(_TMP_DB)
    try:
        repo.save_run(_make_run("d1", "alpha", "hA", "2026-02-10T00:00:00+00:00"))
        repo.save_run(_make_run("d2", "beta", "hB", "2026-02-11T00:00:00+00:00"))
    finally:
        repo.close()

    with TestClient(app) as client:
        resp = client.request("DELETE", "/api/benchmarks/delete", json={"run_ids": ["d1", "d2"]})
        assert resp.status_code == 200
        assert resp.json()["count"] == 2
        assert client.get("/api/benchmarks/d1").status_code == 404
        assert client.get("/api/benchmarks/d2").status_code == 404


def test_api_cannot_delete_active_run() -> None:
    run_manager.set(RunStatus(run_id="activedel", status="running"))
    try:
        with TestClient(app) as client:
            assert client.delete("/api/benchmarks/activedel").status_code == 409
            batch = client.post(
                "/api/benchmarks/delete",
                json={"run_ids": ["activedel", "other"]},
            )
            assert batch.status_code == 409
    finally:
        run_manager.remove("activedel")


def test_api_active_runs_endpoint() -> None:
    run_manager.set(
        RunStatus(
            run_id="act1",
            status="running",
            started_at="2026-01-01T00:00:00Z",
            models=["m1"],
            plugins=["smoke"],
        )
    )
    run_manager.set(RunStatus(run_id="act2", status="completed", finished_at="2026-01-01T00:00:00Z"))
    try:
        with TestClient(app) as client:
            resp = client.get("/api/benchmarks/active")
            assert resp.status_code == 200
            ids = [r["run_id"] for r in resp.json()["runs"]]
            assert "act1" in ids
            assert "act2" not in ids  # terminal statuses are excluded
    finally:
        run_manager.remove("act1")
        run_manager.remove("act2")


def test_api_get_plugin_returns_source() -> None:
    with TestClient(app) as client:
        resp = client.get("/api/plugins/smoke")
        assert resp.status_code == 200
        body = resp.json()
        plugin = body["plugin"]
        assert plugin["id"] == "smoke"
        assert plugin["name"]
        assert plugin["source_file"]  # e.g. local_ai_bench/plugins/builtin/smoke.py
        assert "source" in plugin and plugin["source"]
        import base64

        src = base64.b64decode(plugin["source"]).decode("utf-8")
        assert "class SmokePlugin" in src

        missing = client.get("/api/plugins/nope")
        assert missing.status_code == 404
def test_repository_migrates_legacy_schema_to_host_identity() -> None:
    """A pre-host DB (keyed by model name only) is rebuilt so plugin/case rows are
    backfilled with the model's host and the new unique key is host-aware."""
    import sqlite3
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE runs (run_id TEXT PRIMARY KEY, timestamp TEXT, app_version TEXT,
                           config_hash TEXT, hosts TEXT);
        CREATE TABLE models (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                             host_name TEXT, model_name TEXT NOT NULL, model_digest TEXT,
                             max_context_tokens INTEGER, completion_tokens_total INTEGER DEFAULT 0,
                             cases_run INTEGER DEFAULT 0, errors INTEGER DEFAULT 0,
                             latency_p50_ms REAL, latency_p95_ms REAL,
                             time_to_first_token_p50_ms REAL, tokens_per_second REAL,
                             overall_score REAL, UNIQUE(run_id, model_name));
        CREATE TABLE plugins (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                              model_name TEXT NOT NULL, plugin_id TEXT NOT NULL,
                              total_cases INTEGER DEFAULT 0, successful_cases INTEGER DEFAULT 0,
                              failed_cases INTEGER DEFAULT 0, skipped_cases INTEGER DEFAULT 0,
                              score REAL, metrics TEXT, UNIQUE(run_id, model_name, plugin_id));
        CREATE TABLE cases (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                            model_name TEXT NOT NULL, plugin_id TEXT NOT NULL, case_id TEXT NOT NULL,
                            passed INTEGER, score REAL, response_text TEXT, error TEXT,
                            total_ms REAL, time_to_first_token_ms REAL,
                            tokens_per_second REAL, prompt_tokens INTEGER, completion_tokens INTEGER,
                            attempt INTEGER DEFAULT 1, raw_response TEXT,
                            UNIQUE(run_id, model_name, plugin_id, case_id, attempt));
        CREATE TABLE plugin_options (plugin_id TEXT PRIMARY KEY, options TEXT);
        CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO runs (run_id, timestamp, app_version, config_hash, hosts)
            VALUES ('oldrun', '2026-01-01T00:00:00Z', '0.1.0', 'abc', '[]');
        INSERT INTO models (run_id, host_name, model_name, overall_score)
            VALUES ('oldrun', 'A', 'qwen3.5:0.8b', 0.7);
        INSERT INTO plugins (run_id, model_name, plugin_id, score)
            VALUES ('oldrun', 'qwen3.5:0.8b', 'smoke', 0.7);
        INSERT INTO cases (run_id, model_name, plugin_id, case_id, passed)
            VALUES ('oldrun', 'qwen3.5:0.8b', 'smoke', 'c1', 1);
        """
    )
    conn.commit()
    conn.close()

    try:
        repo = BenchmarkRepository(path)
        try:
            assert _uses_host_identity(repo._conn) is True
            rows = repo.compare_models(run_id="oldrun")
            assert rows[0]["host_name"] == "A"
            assert {p["plugin_id"] for p in rows[0]["plugins"]} == {"smoke"}
            plug = repo._conn.execute(
                "SELECT host_name FROM plugins WHERE run_id='oldrun'"
            ).fetchone()
            assert plug["host_name"] == "A"
            case = repo._conn.execute(
                "SELECT host_name FROM cases WHERE run_id='oldrun'"
            ).fetchone()
            assert case["host_name"] == "A"
        finally:
            repo.close()
    finally:
        os.unlink(path)
