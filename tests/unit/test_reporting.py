"""Unit tests for report persistence (JSON / Markdown / HTML)."""

import json

from ollama_bench.domain.models import (
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
from ollama_bench.reporting.repository import write_report


def _result(model_name: str = "llama3.2:latest") -> RunResult:
    plugin = PluginAggregate(
        plugin_id="long_context",
        model_name=model_name,
        host_name="h",
        total_cases=5,
        successful_cases=4,
        failed_cases=1,
        score=0.8,
        metrics={"max_context_tokens": 16384, "per_context_score": {1024: 1.0, 4096: 0.5}},
    )
    model = ModelBenchmarkResult(
        host_name="h",
        model_name=model_name,
        plugins=[plugin],
        cases_run=5,
        errors=1,
        latency_p50_ms=123.4,
        tokens_per_second=45.6,
        overall_score=0.8,
        context_recommendation={"recommended_context": 4096, "reason": "stable and fast"},
    )
    return RunResult(
        run_id="abc123",
        timestamp="2026-01-01T00:00:00Z",
        app_version="0.1.0",
        config_hash="deadbeef",
        hosts=[HostConfig(name="h", base_url="http://localhost:11434")],
        models=[model],
        errors=["something failed"],
    )


def test_write_report_writes_all_formats(tmp_path):
    result = _result()
    d = write_report(str(tmp_path), result)

    assert d == str(tmp_path / "abc123")
    for name in ("report.json", "report.md", "report.html"):
        assert (tmp_path / "abc123" / name).is_file()
    assert result.report_dir == d


def test_write_report_respects_formats(tmp_path):
    result = _result()
    write_report(str(tmp_path), result, formats=["json"])

    assert (tmp_path / "abc123" / "report.json").is_file()
    assert not (tmp_path / "abc123" / "report.md").exists()
    assert not (tmp_path / "abc123" / "report.html").exists()


def test_include_raw_cases_false_strips_cases(tmp_path):
    result = _result()
    result.models[0].cases = [
        CaseResult(
            case=BenchmarkCase(id="c1", plugin_id="smoke", dataset_version="v1", input={}),
            model=ModelInfo(host_name="h", model_name="llama3.2:latest"),
            response=ModelResponse(
                raw={},
                text="hello",
                timing=TimingMetrics(total_ms=10.0),
                tokens=TokenMetrics(prompt_tokens=3, completion_tokens=2),
            ),
            evaluation=Evaluation(score=1.0, passed=True),
        )
    ]
    write_report(str(tmp_path), result, include_raw_cases=False)
    data = json.loads((tmp_path / "abc123" / "report.json").read_text(encoding="utf-8"))

    assert "cases" not in data["models"][0]
    assert "latency_p50_ms" in data["models"][0]


def test_html_contains_key_sections(tmp_path):
    result = _result()
    write_report(str(tmp_path), result)
    html = (tmp_path / "abc123" / "report.html").read_text(encoding="utf-8")

    assert "<!doctype html>" in html
    assert "<h1>OllamaBench report</h1>" in html
    assert "llama3.2:latest" in html
    assert "Context-window recommendations" in html
    assert "Context-window performance" in html
    assert "Per-case detail" in html
    assert "something failed" in html
    assert "abc123" in html


def test_html_escapes_model_names(tmp_path):
    result = _result(model_name='<script>alert("x")</script>')
    write_report(str(tmp_path), result)
    html = (tmp_path / "abc123" / "report.html").read_text(encoding="utf-8")

    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_html_empty_run_still_renders(tmp_path):
    result = RunResult(
        run_id="empty",
        timestamp="2026-01-01T00:00:00Z",
        app_version="0.1.0",
        config_hash="x",
        models=[],
    )
    write_report(str(tmp_path), result)
    html = (tmp_path / "empty" / "report.html").read_text(encoding="utf-8")

    assert "No models benchmarked" in html
