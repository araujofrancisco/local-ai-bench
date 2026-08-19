"""Unit tests for report persistence (JSON / Markdown / HTML)."""

import json

from local_ai_bench.domain.models import (
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
from local_ai_bench.reporting.repository import write_report


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
    assert "<h1>LocalAIBench report</h1>" in html
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


def test_report_distinguishes_identical_model_across_hosts(tmp_path):
    """With the same model on two servers, each host's row is reported distinctly."""
    host_a = HostConfig(name="hA", base_url="http://a.invalid")
    host_b = HostConfig(name="hB", base_url="http://b.invalid")
    plug_a = PluginAggregate(
        plugin_id="smoke", model_name="qwen3.5:0.8b", host_name="hA",
        total_cases=1, successful_cases=1, score=0.9,
    )
    plug_b = PluginAggregate(
        plugin_id="smoke", model_name="qwen3.5:0.8b", host_name="hB",
        total_cases=1, successful_cases=0, score=0.2,
    )
    model_a = ModelBenchmarkResult(
        host_name="hA", model_name="qwen3.5:0.8b", plugins=[plug_a],
        cases_run=1, overall_score=0.9,
    )
    model_b = ModelBenchmarkResult(
        host_name="hB", model_name="qwen3.5:0.8b", plugins=[plug_b],
        cases_run=1, overall_score=0.2,
    )
    result = RunResult(
        run_id="twohost",
        timestamp="2026-01-01T00:00:00Z",
        app_version="0.1.0",
        config_hash="x",
        hosts=[host_a, host_b],
        models=[model_a, model_b],
    )
    write_report(str(tmp_path), result)

    md = (tmp_path / "twohost" / "report.md").read_text(encoding="utf-8")
    html = (tmp_path / "twohost" / "report.html").read_text(encoding="utf-8")

    # Model summary shows a Host column with each server's row.
    assert "| qwen3.5:0.8b | hA |" in md
    assert "| qwen3.5:0.8b | hB |" in md
    # HTML renders both server rows.
    assert ">qwen3.5:0.8b</td><td>hA</td>" in html
    assert ">qwen3.5:0.8b</td><td>hB</td>" in html
