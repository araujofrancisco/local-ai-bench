"""Unit tests for the plugin registry and local plugin loading."""

import base64

from local_ai_bench.domain.models import ModelResponse, TimingMetrics, TokenMetrics
from local_ai_bench.plugins import load_plugins
from local_ai_bench.plugins.base import RunContext
from local_ai_bench.plugins.registry import PluginRegistry

VALID_PLUGIN = '''\
from typing import Any, ClassVar
from local_ai_bench.plugins.base import BenchmarkPlugin
from local_ai_bench.domain.models import BenchmarkCase, BenchmarkCategory, Evaluation, Modality, ModelInfo, ModelResponse
from local_ai_bench.plugins.base import RunContext
from collections.abc import Iterable

class MyBench(BenchmarkPlugin):
    id: ClassVar[str] = "my_bench"
    name: ClassVar[str] = "My benchmark"
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.REASONING

    def cases(self, ctx: RunContext) -> Iterable[BenchmarkCase]:
        return []

    def build_request(self, case: BenchmarkCase, model: ModelInfo, ctx: RunContext) -> dict[str, Any]:
        return {}

    async def evaluate(self, case: BenchmarkCase, response: ModelResponse, ctx: RunContext) -> Evaluation:
        return Evaluation(score=1.0, passed=True)
'''

BROKEN_PLUGIN = 'this is not valid python :('


def test_load_dir_registers_local_plugin(tmp_path):
    (tmp_path / "my_bench.py").write_text(VALID_PLUGIN, encoding="utf-8")
    reg = PluginRegistry()
    errors = reg.load_dir(tmp_path)

    assert errors == []
    assert reg.get("my_bench") is not None
    assert "my_bench" in reg.ids()


def test_load_dir_skips_missing_directory(tmp_path):
    reg = PluginRegistry()
    assert reg.load_dir(tmp_path / "nope") == []


def test_load_dir_reports_broken_files_but_keeps_valid(tmp_path):
    (tmp_path / "good.py").write_text(VALID_PLUGIN, encoding="utf-8")
    (tmp_path / "broken.py").write_text(BROKEN_PLUGIN, encoding="utf-8")
    reg = PluginRegistry()
    errors = reg.load_dir(tmp_path)

    assert reg.get("my_bench") is not None
    assert any("broken.py" in e for e in errors)


def test_load_plugins_includes_builtins(tmp_path):
    (tmp_path / "my_bench.py").write_text(VALID_PLUGIN, encoding="utf-8")
    reg, errors = load_plugins(str(tmp_path))

    assert errors == []
    assert "smoke" in reg.ids()
    assert "my_bench" in reg.ids()


def test_duplicate_plugin_id_rejected():
    reg = PluginRegistry()
    from local_ai_bench.plugins.builtin.smoke import SmokePlugin

    reg.register(SmokePlugin)
    try:
        reg.register(SmokePlugin)
        raise AssertionError("expected ValueError for duplicate id")
    except ValueError:
        pass


# --- Plugin option wiring ---


def _resp(text: str) -> ModelResponse:
    return ModelResponse(
        raw={},
        text=text,
        timing=TimingMetrics(total_ms=1.0),
        tokens=TokenMetrics(tokens_per_second=100.0),
    )


def _first_case(plugin, ctx):
    return next(iter(plugin.cases(ctx)))


async def test_coding_execute_code_false_is_static_only() -> None:
    from local_ai_bench.plugins.builtin.coding import CodingPlugin

    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": False})
    case = _first_case(plugin, ctx)
    ev = await plugin.evaluate(
        case,
        _resp("def reverse_string(s):\n    return s[::-1]\n"),
        ctx,
    )
    assert ev.score == 1.0
    assert ev.passed is True
    assert ev.metrics["execute_code"] is False
    assert ev.metrics["tests_total"] == 0


async def test_coding_execute_code_true_runs_tests() -> None:
    from local_ai_bench.plugins.builtin.coding import CodingPlugin

    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": True, "timeout_seconds": 10})
    case = _first_case(plugin, ctx)
    ev = await plugin.evaluate(
        case,
        _resp("def reverse_string(s):\n    return s[::-1]\n"),
        ctx,
    )
    assert ev.passed is True
    assert ev.metrics["tests_total"] == len(case.expected["tests"])
    assert ev.metrics["tests_passed"] == len(case.expected["tests"])


async def test_coding_timeout_seconds_is_honored() -> None:
    from local_ai_bench.plugins.builtin.coding import CodingPlugin

    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": True, "timeout_seconds": 1})
    case = _first_case(plugin, ctx)
    sleepy = "import time\n\ndef reverse_string(s):\n    time.sleep(60)\n    return s\n"
    ev = await plugin.evaluate(case, _resp(sleepy), ctx)
    assert ev.passed is False


async def test_coding_execute_code_defaults_to_true() -> None:
    from local_ai_bench.plugins.builtin.coding import CodingPlugin

    plugin = CodingPlugin()
    ctx = RunContext({})
    ev = await plugin.evaluate(
        _first_case(plugin, ctx),
        _resp("def reverse_string(s):\n    return s[::-1]\n"),
        ctx,
    )
    assert ev.metrics["execute_code"] is True
    assert ev.passed is True


def test_coding_dataset_v3_and_expanded_cases() -> None:
    from local_ai_bench.plugins.builtin.coding import CodingPlugin

    assert CodingPlugin.dataset_version == "v3"
    plugin = CodingPlugin()
    cases = list(plugin.cases(RunContext({})))
    assert len(cases) >= 30
    ids = {c.id for c in cases}
    assert {
        "code_lru_cache_0011",
        "code_trie_0012",
        "code_edit_distance_0013",
        "code_n_queens_0014",
        "code_two_sum_0006",
        "code_has_cycle_0024",
        "code_product_except_self_0028",
        "code_parse_ints_0033",
        "code_min_stack_0031",
        "code_valid_bst_0032",
    } <= ids


async def test_coding_class_based_solution_lru() -> None:
    from local_ai_bench.plugins.builtin.coding import CodingPlugin

    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": True, "timeout_seconds": 15})
    case = next(c for c in plugin.cases(ctx) if c.id == "code_lru_cache_0011")
    solution = (
        "class LRUCache:\n"
        "    def __init__(self, capacity):\n"
        "        self.cap = capacity\n"
        "        self.d = {}\n"
        "        self.order = []\n"
        "    def _touch(self, key):\n"
        "        try:\n"
        "            self.order.remove(key)\n"
        "        except ValueError:\n"
        "            pass\n"
        "        self.order.append(key)\n"
        "    def get(self, key):\n"
        "        if key not in self.d: return -1\n"
        "        self._touch(key)\n"
        "        return self.d[key]\n"
        "    def put(self, key, value):\n"
        "        if key not in self.d and len(self.d) >= self.cap:\n"
        "            old = self.order.pop(0)\n"
        "            del self.d[old]\n"
        "        self.d[key] = value\n"
        "        self._touch(key)\n"
    )
    ev = await plugin.evaluate(case, _resp(solution), ctx)
    assert ev.passed is True, ev
    assert ev.score == 1.0


async def test_coding_partial_credit_on_failed_assertions() -> None:
    from local_ai_bench.plugins.builtin.coding import CodingPlugin

    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": True, "timeout_seconds": 15})
    case = next(c for c in plugin.cases(ctx) if c.id == "code_max_subarray_0009")
    # Overconfident Kadane that always restarts from 0 -> wrong on all-negative input.
    wrong = (
        "def max_subarray(nums):\n"
        "    best = cur = 0\n"
        "    for n in nums:\n"
        "        cur = max(0, cur + n)\n"
        "        best = max(best, cur)\n"
        "    return best\n"
    )
    ev = await plugin.evaluate(case, _resp(wrong), ctx)
    assert 0.0 < ev.score < 1.0
    assert ev.passed is False


async def test_coding_extracts_fenced_code_from_prose() -> None:
    from local_ai_bench.plugins.builtin.coding import CodingPlugin

    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": True, "timeout_seconds": 10})
    reply = "Here's my solution:\n```python\ndef reverse_string(s):\n    return s[::-1]\n```\nHope this helps!"
    ev = await plugin.evaluate(_first_case(plugin, ctx), _resp(reply), ctx)
    assert ev.passed is True


def test_vision_max_image_dimension_caps_size() -> None:
    from local_ai_bench.plugins.builtin.vision import VisionPlugin

    plugin = VisionPlugin()
    small = _first_case(plugin, RunContext({"max_image_dimension": 8}))
    large = _first_case(plugin, RunContext({"max_image_dimension": 768}))

    def dims(case) -> tuple[int, int]:
        data = base64.b64decode(case.input["image_b64"])
        width, height = data[16:20], data[20:24]
        return int.from_bytes(width, "big"), int.from_bytes(height, "big")

    assert dims(small) == (8, 8)
    assert dims(large) == (32, 32)


def test_coding_has_description() -> None:
    from local_ai_bench.plugins.builtin.coding import CodingPlugin

    assert CodingPlugin.description
    assert "description" in CodingPlugin.__dict__


# --- Multi-context plugin ---


def _multi_ctx(sizes=None, max_ctx=None, expected="paris", prompt="What is the capital of France?"):
    from local_ai_bench.plugins.builtin.multi_context import MultiContextPlugin

    plugin = MultiContextPlugin()
    opts: dict = {"expected": expected, "prompt": prompt, "contains": True}
    if sizes is not None:
        opts["context_sizes"] = sizes
    if max_ctx is not None:
        opts["max_context_tokens"] = max_ctx
    return plugin, RunContext(opts)


def test_multicontext_cases_cover_each_size_and_double_num_ctx() -> None:
    from local_ai_bench.domain.models import ModelInfo

    plugin, ctx = _multi_ctx(sizes=[512, 1024, 4096])
    model = ModelInfo(host_name="h", model_name="m", max_context_tokens=None)
    cases = list(plugin.cases(ctx))
    assert [c.input["target_context"] for c in cases] == [512, 1024, 4096]
    assert all(c.id.startswith("multictx_ctx_") for c in cases)
    # num_ctx is double the target context (prompt+completion window convention).
    assert plugin.build_request(cases[0], model, ctx)["options"]["num_ctx"] == 1024


def test_multicontext_prunes_sizes_above_model_window() -> None:
    plugin, ctx = _multi_ctx(sizes=[512, 1024, 4096, 16384], max_ctx=5000)
    cases = list(plugin.cases(ctx))
    assert [c.input["target_context"] for c in cases] == [512, 1024, 4096]


async def test_multicontext_evaluate_contains_mode() -> None:
    plugin, ctx = _multi_ctx(sizes=[512], expected="paris")
    case = _first_case(plugin, ctx)
    good = _resp("The capital of France is Paris.")
    bad = _resp("The capital of France is Lyon.")
    assert (await plugin.evaluate(case, good, ctx)).score == 1.0
    assert (await plugin.evaluate(case, bad, ctx)).score == 0.0


async def test_multicontext_evaluate_numeric_mode() -> None:
    plugin, ctx = _multi_ctx(sizes=[512], expected="42", prompt="pick a number")
    ctx.options["contains"] = False
    case = _first_case(plugin, ctx)
    ev = await plugin.evaluate(case, _resp("The number is 42, clearly."), ctx)
    assert ev.score == 1.0


def test_multicontext_aggregate_reports_per_context_score() -> None:
    from local_ai_bench.domain.models import (
        BenchmarkCase,
        CaseResult,
        Evaluation,
        ModelInfo,
        ModelResponse,
        TimingMetrics,
        TokenMetrics,
    )

    plugin, _ctx = _multi_ctx(sizes=[512, 1024])
    model = ModelInfo(host_name="h", model_name="m", max_context_tokens=None)
    results = [
        CaseResult(
            case=BenchmarkCase(id="a", plugin_id="multi_context", dataset_version="v1", input={"target_context": 512}),
            model=model,
            response=ModelResponse(raw={}, text="paris", timing=TimingMetrics(total_ms=1.0), tokens=TokenMetrics()),
            evaluation=Evaluation(score=1.0, passed=True),
            attempt=1,
        ),
        CaseResult(
            case=BenchmarkCase(id="b", plugin_id="multi_context", dataset_version="v1", input={"target_context": 1024}),
            model=model,
            response=ModelResponse(raw={}, text="nope", timing=TimingMetrics(total_ms=1.0), tokens=TokenMetrics()),
            evaluation=Evaluation(score=0.0, passed=False),
            attempt=1,
        ),
    ]
    out = plugin.aggregate(results)
    assert out.metrics["per_context_score"] == {512: 1.0, 1024: 0.0}
