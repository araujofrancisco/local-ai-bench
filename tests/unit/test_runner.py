"""Unit tests for runner helpers."""

import asyncio
from typing import Any

import httpx

from local_ai_bench.config import BenchmarkConfig, RunnerConfig
from local_ai_bench.domain.models import (
    BenchmarkCase,
    ModelInfo,
    ModelResponse,
    TimingMetrics,
    TokenMetrics,
)
from local_ai_bench.runner.orchestrator import (
    RunOrchestrator,
    _nearest_rank,
    _pct,
    _select_models,
    _send_with_retries,
)


def _model(name: str) -> ModelInfo:
    return ModelInfo(host_name="h", model_name=name)


def test_pct_on_small_samples():
    values = [100.0, 200.0, 300.0, 400.0, 500.0]
    assert _pct(values, 0.5) == 300.0
    assert _pct(values, 0.95) == 500.0


def test_pct_empty():
    assert _pct([], 0.5) is None


def test_nearest_rank_median():
    assert _nearest_rank([1, 3, 2], 0.5) == 2


def test_select_models_without_config_and_filter_returns_all():
    all_m = [_model("a"), _model("b")]
    assert _select_models(all_m, [], None) == all_m


def test_select_models_config_exact_or_glob():
    all_m = [_model("a"), _model("b"), _model("ab")]
    assert [m.model_name for m in _select_models(all_m, ["ab"], None)] == ["ab"]


def test_select_models_config_glob():
    all_m = [_model("llama3.2:latest"), _model("qwen:0.8b"), _model("llama3.1:8b")]
    assert [m.model_name for m in _select_models(all_m, ["llama*"], None)] == [
        "llama3.2:latest",
        "llama3.1:8b",
    ]


def test_select_models_prefers_filter_over_config():
    all_m = [_model("a"), _model("b")]
    keep_a = lambda m: m.model_name == "a"  # noqa: E731
    assert [m.model_name for m in _select_models(all_m, ["b"], keep_a)] == ["a"]


def test_orchestrator_respects_provided_run_id():
    cfg = BenchmarkConfig(hosts=[])
    result = asyncio.run(RunOrchestrator(cfg, run_id="abc123").run())
    assert result.run_id == "abc123"


def test_orchestrator_generates_run_id_when_omitted():
    cfg = BenchmarkConfig(hosts=[])
    result = asyncio.run(RunOrchestrator(cfg).run())
    assert result.run_id != ""


def test_orchestrator_plugin_options_overrides() -> None:
    cfg = BenchmarkConfig(hosts=[])
    orch = RunOrchestrator(
        cfg, plugin_options={"coding": {"execute_code": True, "timeout_seconds": 99}}
    )
    assert orch._plugin_options_for("coding") == {
        "execute_code": True,
        "timeout_seconds": 99,
    }
    assert orch._plugin_options_for("smoke") == {}


def test_orchestrator_plugin_options_defaults_to_config() -> None:
    from local_ai_bench.config import PluginConfig

    cfg = BenchmarkConfig(
        hosts=[], plugins=PluginConfig(options={"coding": {"timeout_seconds": 7}})
    )
    orch = RunOrchestrator(cfg)
    assert orch._plugin_options_for("coding")["timeout_seconds"] == 7


def test_count_case_runs_multiplies_by_repetitions() -> None:
    cfg = BenchmarkConfig(hosts=[], runner=RunnerConfig(repetitions=4))
    orch = RunOrchestrator(cfg)

    class _P:
        id = "counting"

        def cases(self, ctx: Any):  # noqa: ANN001
            for i in range(3):
                yield BenchmarkCase(id=f"c{i}", plugin_id="counting", dataset_version="v1", input={})

    assert orch._count_case_runs(_MODEL, _P()) == 12


# --- Retry / error-isolation behavior (PLAN §13.5) ---


class _FakeClient:
    """Stub Ollama client that fails N times then succeeds."""

    def __init__(self, fail_count: int = 0, exc: Exception | None = None) -> None:
        self.calls = 0
        self.fail_count = fail_count
        self.exc = exc or httpx.ConnectError("boom")

    async def chat(self, *args: Any, **kwargs: Any) -> ModelResponse:
        self.calls += 1
        if self.calls <= self.fail_count:
            raise self.exc
        return ModelResponse(
            raw={},
            text="ok",
            timing=TimingMetrics(total_ms=1.0),
            tokens=TokenMetrics(completion_tokens=1, tokens_per_second=100.0),
        )


class _FakePlugin:
    id = "fake"

    def build_request(self, case: BenchmarkCase, model: ModelInfo, ctx: Any) -> dict[str, Any]:
        return {"messages": [], "options": {}, "stream": True}


_CASE = BenchmarkCase(
    id="c1", plugin_id="fake", dataset_version="v1", input={"prompt": "hi"}
)
_MODEL = ModelInfo(host_name="h", model_name="m")


def _runner(max_retries: int = 3) -> RunnerConfig:
    return RunnerConfig(max_retries=max_retries, retry_backoff_seconds=0.01)


async def test_send_retries_transient_error_then_succeeds():
    client = _FakeClient(fail_count=2)
    resp = await _send_with_retries(client, _MODEL, _FakePlugin(), _CASE, {}, _runner())
    assert resp.error is None
    assert resp.text == "ok"
    assert client.calls == 3


async def test_send_retries_exhausted_returns_error_response():
    client = _FakeClient(fail_count=99)
    resp = await _send_with_retries(client, _MODEL, _FakePlugin(), _CASE, {}, _runner())
    assert resp.error is not None
    assert resp.text == ""
    # max_retries=3 => 1 initial attempt + 3 retries.
    assert client.calls == 4


async def test_send_does_not_retry_non_retryable_status():
    client = _FakeClient(fail_count=99, exc=httpx.HTTPStatusError("400", request=httpx.Request("POST", "/"), response=httpx.Response(400, request=httpx.Request("POST", "/"))))
    resp = await _send_with_retries(client, _MODEL, _FakePlugin(), _CASE, {}, _runner(max_retries=5))
    assert resp.error is not None
    assert "400" in resp.error
    assert client.calls == 1


async def test_send_retries_once_when_max_retries_is_one():
    client = _FakeClient(fail_count=99)
    resp = await _send_with_retries(client, _MODEL, _FakePlugin(), _CASE, {}, _runner(max_retries=1))
    assert resp.error is not None
    assert client.calls == 2


async def test_send_still_attempts_once_when_max_retries_is_zero():
    # max_retries=0 means "no retries", not "no request" — the initial attempt
    # must still be sent (regression: the old range(1, max_retries+1) skipped it).
    client = _FakeClient(fail_count=99)
    resp = await _send_with_retries(client, _MODEL, _FakePlugin(), _CASE, {}, _runner(max_retries=0))
    assert resp.error is not None
    assert client.calls == 1

    ok = _FakeClient(fail_count=0)
    good = await _send_with_retries(ok, _MODEL, _FakePlugin(), _CASE, {}, _runner(max_retries=0))
    assert good.error is None
    assert good.text == "ok"
    assert ok.calls == 1


# --- Weighted overall scoring ---

from local_ai_bench.domain.models import BenchmarkCategory  # noqa: E402
from local_ai_bench.runner.orchestrator import _category_weight, _weighted_mean  # noqa: E402


def test_weighted_mean_plain():
    assert _weighted_mean([(1.0, 1.0), (0.5, 1.0)]) == 0.75


def test_weighted_mean_respects_weights():
    assert _weighted_mean([(1.0, 3.0), (0.0, 1.0)]) == 0.75


def test_weighted_mean_empty():
    assert _weighted_mean([]) is None


def test_category_weight_uses_category_value():
    weights = {"reasoning": 2.0, "coding": 1.5}
    assert _category_weight(weights, BenchmarkCategory.REASONING) == 2.0
    assert _category_weight(weights, BenchmarkCategory.VISION) == 1.0