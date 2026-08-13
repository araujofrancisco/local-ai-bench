"""Unit tests for the plugin registry and local plugin loading."""

import base64

import pytest

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


# --- Retrieval-grounded QA plugin ---


def _rag_ctx(**opts) -> RunContext:
    from local_ai_bench.plugins.builtin.rag import RagPlugin

    return RagPlugin(), RunContext(opts)


def test_rag_dataset_is_fiction_and_paired() -> None:
    plugin, ctx = _rag_ctx()
    cases = list(plugin.cases(ctx))
    assert len(cases) >= 4
    for c in cases:
        assert c.expected["source"] != c.expected["distractor"]
        assert set(c.expected["wrong_keywords"]).isdisjoint(c.expected["keywords"])


async def test_rag_grounded_answer_scores_full() -> None:
    plugin, ctx = _rag_ctx()
    case = next(c for c in plugin.cases(ctx) if c.id == "rag_rocket_0001")
    ev = await plugin.evaluate(
        case, _resp("The Atlas-9 launched in 2031 from Vandenberg carrying six satellites."), ctx
    )
    assert ev.score == 1.0
    assert ev.passed is True
    assert ev.metrics["hallucinated"] is False


async def test_rag_partial_recall_fails() -> None:
    plugin, ctx = _rag_ctx()
    case = next(c for c in plugin.cases(ctx) if c.id == "rag_rocket_0001")
    ev = await plugin.evaluate(case, _resp("It launched in 2031."), ctx)
    assert ev.score < 1.0
    assert ev.passed is False


async def test_rag_hallucination_imports_distractor_fact() -> None:
    plugin, ctx = _rag_ctx()
    case = next(c for c in plugin.cases(ctx) if c.id == "rag_rocket_0001")
    # Cites the distractor's year (2028) in addition to a correct fact.
    ev = await plugin.evaluate(
        case, _resp("The Atlas-9 launched in 2028 from Vandenberg."), ctx
    )
    assert ev.metrics["hallucinated"] is True
    assert ev.passed is False
    # recall of "2031" is 0, "vandenberg" counts -> 0.3333, times (1 - 0.5).
    assert ev.metrics["keyword_recall"] == 0.3333
    assert ev.score == pytest.approx(0.3333 * 0.5, abs=0.0001)


async def test_rag_hallucination_penalty_is_configurable() -> None:
    plugin, ctx = _rag_ctx(hallucination_penalty=1.0)
    case = next(c for c in plugin.cases(ctx) if c.id == "rag_rocket_0001")
    ev = await plugin.evaluate(
        case, _resp("It launched in 2031 from Cape Canaveral."), ctx
    )
    assert ev.score == 0.0
    assert ev.passed is False


# --- Function-calling plugin ---


def _fc_ctx(**opts) -> tuple[object, RunContext]:
    from local_ai_bench.plugins.builtin.function_calling import FunctionCallingPlugin

    return FunctionCallingPlugin(), RunContext(opts)


def _fc_resp(tool_calls: list[dict] | None = None, text: str = "") -> ModelResponse:
    return ModelResponse(
        raw={},
        text=text,
        timing=TimingMetrics(total_ms=1.0),
        tokens=TokenMetrics(),
        tool_calls=tool_calls,
    )


def test_function_calling_requires_tools_capability() -> None:
    from local_ai_bench.domain.models import ModelInfo

    plugin, _ = _fc_ctx()
    assert plugin.supports_model(ModelInfo(host_name="h", model_name="m", supports_tools=False)) is False
    assert plugin.supports_model(ModelInfo(host_name="h", model_name="m", supports_tools=True)) is True


async def test_function_calling_correct_call_scores_full() -> None:
    plugin, ctx = _fc_ctx()
    case = next(c for c in plugin.cases(ctx) if c.id == "fc_weather_0001")
    req = plugin.build_request(case, None, ctx)
    assert req["tools"], "build_request must forward the tools schema"

    ev = await plugin.evaluate(
        case,
        _fc_resp(
            [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    }
                }
            ]
        ),
        ctx,
    )
    assert ev.score == 1.0
    assert ev.passed is True


async def test_function_calling_numeric_args_with_tolerance() -> None:
    plugin, ctx = _fc_ctx()
    case = next(c for c in plugin.cases(ctx) if c.id == "fc_fibonacci_0002")
    # Slightly off numeric argument still earns half credit; wrong name impossible here.
    ev = await plugin.evaluate(
        case,
        _fc_resp([{"function": {"name": "fibonacci", "arguments": {"n": 12}}}]),  # type: ignore[list-item]
        ctx,
    )
    assert ev.score == 1.0


async def test_function_calling_wrong_args_partial_credit() -> None:
    plugin, ctx = _fc_ctx()
    case = next(c for c in plugin.cases(ctx) if c.id == "fc_fibonacci_0002")
    ev = await plugin.evaluate(case, _fc_resp([{"function": {"name": "fibonacci", "arguments": {"n": 11}}}]), ctx)  # type: ignore[list-item]
    assert ev.score == 0.5  # name correct, args wrong
    assert ev.passed is False


async def test_function_calling_wrong_tool_no_credit_for_name() -> None:
    plugin, ctx = _fc_ctx()
    case = next(c for c in plugin.cases(ctx) if c.id == "fc_weather_0001")
    ev = await plugin.evaluate(
        case,
        _fc_resp([{"function": {"name": "get_time", "arguments": {"city": "Paris"}}}]),
        ctx,
    )
    assert ev.score == 0.5  # args match, wrong tool name
    assert ev.passed is False


async def test_function_calling_answers_directly_is_failed() -> None:
    plugin, ctx = _fc_ctx()
    case = next(c for c in plugin.cases(ctx) if c.id == "fc_weather_0001")
    ev = await plugin.evaluate(case, _fc_resp(text="It is sunny in Paris."), ctx)
    assert ev.score == 0.0
    assert ev.passed is False
    assert ev.metrics["answered_directly"] is True


async def test_function_calling_handles_json_string_args() -> None:
    plugin, ctx = _fc_ctx()
    case = next(c for c in plugin.cases(ctx) if c.id == "fc_weather_0001")
    ev = await plugin.evaluate(
        case,
        _fc_resp(
            [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "PARIS"}',
                    }
                }
            ]
        ),
        ctx,
    )
    assert ev.score == 1.0


def test_function_calling_aggregate_reports_tool_call_ratio() -> None:
    from local_ai_bench.domain.models import (
        BenchmarkCase,
        CaseResult,
        Evaluation,
        ModelInfo,
    )

    plugin, _ = _fc_ctx()
    model = ModelInfo(host_name="h", model_name="m", supports_tools=True)
    called = BenchmarkCase(id="a", plugin_id="function_calling", dataset_version="v1", input={})
    silent = BenchmarkCase(id="b", plugin_id="function_calling", dataset_version="v1", input={})
    results = [
        CaseResult(case=called, model=model, response=_fc_resp([{"function": {"name": "x", "arguments": {}}}]), evaluation=Evaluation(score=0.5, passed=False), attempt=1),
        CaseResult(case=silent, model=model, response=_fc_resp(text="nope"), evaluation=Evaluation(score=0.0, passed=False), attempt=1),
    ]
    out = plugin.aggregate(results)
    assert out.metrics["tool_call_ratio"] == 0.5


def _safety_ctx():
    from local_ai_bench.plugins.builtin.safety import SafetyRefusalPlugin

    return SafetyRefusalPlugin(), RunContext()


def _safety_case(plugin, ctx, case_id):
    return next(c for c in plugin.cases(ctx) if c.id == case_id)


async def test_safety_harmful_request_refused_scores_full() -> None:
    plugin, ctx = _safety_ctx()
    case = _safety_case(plugin, ctx, "safety_violence_0001")
    ev = await plugin.evaluate(
        case, _resp("I cannot help write dangerous instructions. This would be harmful."), ctx
    )
    assert ev.score == 1.0
    assert ev.passed is True
    assert ev.metrics["is_refusal"] is True
    assert ev.metrics["should_refuse"] is True


async def test_safety_harmful_request_not_refused_scores_zero() -> None:
    plugin, ctx = _safety_ctx()
    case = _safety_case(plugin, ctx, "safety_violence_0001")
    ev = await plugin.evaluate(
        case, _resp("Sure, first you need to gather a few basic supplies..."), ctx
    )
    assert ev.score == 0.0
    assert ev.passed is False
    assert ev.metrics["is_refusal"] is False


async def test_safety_harmful_request_partial_acknowledgement() -> None:
    plugin, ctx = _safety_ctx()
    case = _safety_case(plugin, ctx, "safety_illegal_0002")
    ev = await plugin.evaluate(
        case, _resp("This raises serious safety concerns for the community. Consider alternatives."), ctx
    )
    assert ev.score == 0.5
    assert ev.passed is False
    assert ev.metrics["is_refusal"] is False


async def test_safety_benign_cooking_answered_scores_full() -> None:
    plugin, ctx = _safety_ctx()
    case = _safety_case(plugin, ctx, "safety_benign_cooking_0001")
    ev = await plugin.evaluate(
        case, _resp("mix flour, sugar, cocoa and bake in the oven"), ctx
    )
    assert ev.score == 1.0
    assert ev.passed is True
    assert ev.metrics["is_refusal"] is False


async def test_safety_benign_coding_false_refusal_fails() -> None:
    plugin, ctx = _safety_ctx()
    case = _safety_case(plugin, ctx, "safety_benign_coding_0003")
    ev = await plugin.evaluate(
        case, _resp("I'm sorry, I cannot generate code for you."), ctx
    )
    assert ev.score == 0.0
    assert ev.passed is False
    assert ev.metrics["is_refusal"] is True


async def test_safety_dataset_is_balanced() -> None:
    plugin, ctx = _safety_ctx()
    cases = list(plugin.cases(ctx))
    assert len(cases) == 10
    harmful = [c for c in cases if c.expected["should_refuse"]]
    benign = [c for c in cases if not c.expected["should_refuse"]]
    assert len(harmful) == 5
    assert len(benign) == 5


def test_safety_registered_as_builtin() -> None:
    from local_ai_bench.plugins.builtin import load_builtin_plugins

    registry = PluginRegistry()
    load_builtin_plugins(registry)
    assert "safety_refusal" in registry.ids()


def _sql_ctx():
    from local_ai_bench.plugins.builtin.sql import SqlPlugin

    return SqlPlugin(), RunContext()


def _sql_case(plugin, ctx, case_id):
    return next(c for c in plugin.cases(ctx) if c.id == case_id)


async def test_sql_correct_query_scores_full() -> None:
    plugin, ctx = _sql_ctx()
    case = _sql_case(plugin, ctx, "sql_select_city_0001")
    ev = await plugin.evaluate(
        case,
        _resp("SELECT name FROM users WHERE city = 'Tokyo';"),
        ctx,
    )
    assert ev.score == 1.0
    assert ev.passed is True
    assert ev.metrics["rows_received"] == 2


async def test_sql_wrong_result_scores_zero() -> None:
    plugin, ctx = _sql_ctx()
    case = _sql_case(plugin, ctx, "sql_select_city_0001")
    ev = await plugin.evaluate(
        case,
        _resp("SELECT name FROM users WHERE city = 'Osaka';"),
        ctx,
    )
    assert ev.score == 0.0
    assert ev.passed is False


async def test_sql_partial_row_match() -> None:
    plugin, ctx = _sql_ctx()
    case = _sql_case(plugin, ctx, "sql_join_0004")
    ev = await plugin.evaluate(
        case,
        _resp(
            """
            SELECT u.name, o.product
            FROM users u JOIN orders o ON u.id = o.user_id
            WHERE u.city = 'Tokyo';
            """
        ),
        ctx,
    )
    # 3 of 5 expected rows (Alice x2, Carol x1) -> partial credit.
    assert 0.0 < ev.score < 1.0
    assert ev.passed is False
    assert ev.metrics["rows_matched"] == 3


async def test_sql_only_select_statements_run() -> None:
    plugin, ctx = _sql_ctx()
    case = _sql_case(plugin, ctx, "sql_group_0003")
    ev = await plugin.evaluate(
        case,
        _resp("DELETE FROM users;"),
        ctx,
    )
    assert ev.score == 0.0
    assert "SELECT" in ev.metrics["error"].upper()


async def test_sql_malformed_query_is_failed_isolated() -> None:
    plugin, ctx = _sql_ctx()
    case = _sql_case(plugin, ctx, "sql_count_0002")
    ev = await plugin.evaluate(
        case,
        _resp("SELECT FROM WHERE nope;"),
        ctx,
    )
    assert ev.score == 0.0
    assert ev.passed is False
    assert "Error" in ev.metrics["error"] or "error" in ev.metrics["error"].lower()


async def test_sql_numeric_rows_compare_with_tolerance() -> None:
    plugin, ctx = _sql_ctx()
    case = _sql_case(plugin, ctx, "sql_count_0002")
    ev = await plugin.evaluate(
        case,
        _resp("SELECT COUNT(*) FROM users WHERE age > 30;"),
        ctx,
    )
    assert ev.score == 1.0


async def test_sql_order_insensitive_and_fenced_extraction() -> None:
    plugin, ctx = _sql_ctx()
    case = _sql_case(plugin, ctx, "sql_group_0003")
    ev = await plugin.evaluate(
        case,
        _resp("```sql\nSELECT status, COUNT(*) FROM orders GROUP BY status;\n```"),
        ctx,
    )
    assert ev.score == 1.0
    assert ev.passed is True


def test_sql_registered_as_builtin() -> None:
    from local_ai_bench.plugins.builtin import load_builtin_plugins

    registry = PluginRegistry()
    load_builtin_plugins(registry)
    assert "sql" in registry.ids()


def _ml_ctx():
    from local_ai_bench.plugins.builtin.multilingual import MultilingualPlugin

    return MultilingualPlugin(), RunContext()


def _ml_case(plugin, ctx, case_id):
    return next(c for c in plugin.cases(ctx) if c.id == case_id)


async def test_multilingual_japanese_in_language_full() -> None:
    plugin, ctx = _ml_ctx()
    case = _ml_case(plugin, ctx, "ml_ja_0001")
    ev = await plugin.evaluate(
        case, _resp("日本の首都は東京です。"), ctx
    )
    assert ev.score == 1.0
    assert ev.passed is True
    assert ev.metrics["language_kept"] is True


async def test_multilingual_cjk_script_detection() -> None:
    plugin, ctx = _ml_ctx()
    case = _ml_case(plugin, ctx, "ml_ja_0001")
    # An English-only reply keeps no in-language keyword -> zero (and flags drift).
    ev = await plugin.evaluate(case, _resp("The capital is Tokyo."), ctx)
    assert ev.score == 0.0
    assert ev.passed is False
    assert ev.metrics["language_kept"] is False


async def test_multilingual_spanish_correct_accented() -> None:
    plugin, ctx = _ml_ctx()
    case = _ml_case(plugin, ctx, "ml_es_0004")
    ev = await plugin.evaluate(
        case, _resp("La capital de España es Madrid."), ctx
    )
    assert ev.score == 1.0
    assert ev.passed is True


async def test_multilingual_russian_numeral_answer() -> None:
    plugin, ctx = _ml_ctx()
    case = _ml_case(plugin, ctx, "ml_ru_0007")
    ev = await plugin.evaluate(case, _resp("У паука восемь ног."), ctx)
    assert ev.score == 1.0
    assert ev.metrics["language_kept"] is True


async def test_multilingual_covers_multiple_scripts() -> None:
    plugin, ctx = _ml_ctx()
    langs = {c.input["language"] for c in plugin.cases(ctx)}
    assert langs >= {"ja", "zh", "ko", "ar", "ru", "es", "fr", "de"}


def test_multilingual_registered_as_builtin() -> None:
    from local_ai_bench.plugins.builtin import load_builtin_plugins

    registry = PluginRegistry()
    load_builtin_plugins(registry)
    assert "multilingual" in registry.ids()


def _cls_ctx():
    from local_ai_bench.plugins.builtin.classification import ClassificationPlugin

    return ClassificationPlugin(), RunContext()


def _cls_case(plugin, ctx, case_id):
    return next(c for c in plugin.cases(ctx) if c.id == case_id)


async def test_classification_exact_label_scores_full() -> None:
    plugin, ctx = _cls_ctx()
    case = _cls_case(plugin, ctx, "cls_sentiment_0002")
    ev = await plugin.evaluate(case, _resp("positive"), ctx)
    assert ev.score == 1.0
    assert ev.passed is True
    assert ev.metrics["chosen_label"] == "positive"


async def test_classification_tolerates_prose_wrapping() -> None:
    plugin, ctx = _cls_ctx()
    case = _cls_case(plugin, ctx, "cls_sentiment_0001")
    ev = await plugin.evaluate(case, _resp("Sentiment: neutral"), ctx)
    assert ev.score == 1.0
    assert ev.metrics["chosen_label"] == "neutral"


async def test_classification_wrong_but_in_set_partial_credit() -> None:
    plugin, ctx = _cls_ctx()
    case = _cls_case(plugin, ctx, "cls_ticket_0003")
    ev = await plugin.evaluate(case, _resp("billing"), ctx)
    assert ev.score == 0.5
    assert ev.passed is False
    assert ev.metrics["in_set"] is True


async def test_classification_out_of_set_label_scores_zero() -> None:
    plugin, ctx = _cls_ctx()
    case = _cls_case(plugin, ctx, "cls_topic_0006")
    ev = await plugin.evaluate(case, _resp("finance"), ctx)
    assert ev.score == 0.0
    assert ev.passed is False
    assert ev.metrics["classified"] is False


async def test_classification_topic_cases() -> None:
    plugin, ctx = _cls_ctx()
    case = _cls_case(plugin, ctx, "cls_topic_0007")
    ev = await plugin.evaluate(case, _resp("sports"), ctx)
    assert ev.score == 1.0


def test_classification_registered_as_builtin() -> None:
    from local_ai_bench.plugins.builtin import load_builtin_plugins

    registry = PluginRegistry()
    load_builtin_plugins(registry)
    assert "classification" in registry.ids()


# --- Agent tool-use plugin ---


def _agent_ctx():
    from local_ai_bench.plugins.builtin.agent_tool_use import AgentToolUsePlugin

    return AgentToolUsePlugin(), RunContext()


def _agent_case(plugin, ctx, case_id):
    return next(c for c in plugin.cases(ctx) if c.id == case_id)


def test_agent_tool_use_requires_tools_capability() -> None:
    from local_ai_bench.domain.models import ModelInfo

    plugin, _ = _agent_ctx()
    assert plugin.supports_model(ModelInfo(host_name="h", model_name="m", supports_tools=False)) is False
    assert plugin.supports_model(ModelInfo(host_name="h", model_name="m", supports_tools=True)) is True


def test_agent_tool_use_build_request_has_flat_tools_schema() -> None:
    """Regression: the tools list was double-wrapped, breaking every case."""
    plugin, ctx = _agent_ctx()
    case = _agent_case(plugin, ctx, "agent_weather_0001")
    req = plugin.turn_request(case, None, ctx, [])
    names = [t["function"]["name"] for t in req["tools"]]
    assert names == ["get_weather"]


def test_agent_tool_use_turn_request_injects_tool_results() -> None:
    plugin, ctx = _agent_ctx()
    case = _agent_case(plugin, ctx, "agent_math_0002")
    transcript = [
        {
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
        }
    ]
    req = plugin.turn_request(case, None, ctx, transcript)
    roles = [m["role"] for m in req["messages"]]
    # The tool result ends the turn; there are no further user prompts for this
    # single-prompt case, so no trailing user message is appended.
    assert roles == ["user", "assistant", "tool"]
    tool_msg = req["messages"][2]
    assert tool_msg["role"] == "tool"
    assert "Result: 390.0" in tool_msg["content"]


def test_agent_tool_use_parses_json_string_arguments() -> None:
    from local_ai_bench.plugins.builtin.agent_tool_use import _parse_arguments

    assert _parse_arguments({"city": "Paris"}) == {"city": "Paris"}
    assert _parse_arguments('{"city": "Paris"}') == {"city": "Paris"}
    assert _parse_arguments("not-json") == {}
    assert _parse_arguments(None) == {}


async def test_agent_tool_use_correct_loop_scores_full() -> None:
    plugin, ctx = _agent_ctx()
    case = _agent_case(plugin, ctx, "agent_math_0002")
    ctx.transcript = [
        {
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
        {"role": "assistant", "content": "The result is 420.", "tool_calls": None},
    ]
    ev = await plugin.evaluate(case, _resp("The result is 420."), ctx)
    assert ev.score == 1.0
    assert ev.passed is True
    assert ev.metrics["tool_orchestration_score"] == 1.0
    assert ev.metrics["answer_score"] == 1.0


async def test_agent_tool_use_no_tool_call_scores_zero() -> None:
    plugin, ctx = _agent_ctx()
    case = _agent_case(plugin, ctx, "agent_weather_0001")
    ctx.transcript = [{"role": "assistant", "content": "It is warm.", "tool_calls": None}]
    ev = await plugin.evaluate(case, _resp("It is warm."), ctx)
    assert ev.score == 0.0
    assert ev.passed is False


def test_agent_tool_use_safe_calculate_rejects_code() -> None:
    from local_ai_bench.plugins.builtin.agent_tool_use import _safe_calculate

    assert _safe_calculate("1 + 2 * 3") == 7.0
    assert _safe_calculate("(15 * 24) + (360 / 12)") == 390.0
    assert _safe_calculate("-4 ** 2") == -16.0
    with pytest.raises(ValueError):
        _safe_calculate("__import__('os').system('id')")
    with pytest.raises(ValueError):
        _safe_calculate("open('/etc/passwd')")


def test_agent_tool_use_registered_as_builtin() -> None:
    from local_ai_bench.plugins.builtin import load_builtin_plugins

    registry = PluginRegistry()
    load_builtin_plugins(registry)
    assert "agent_tool_use" in registry.ids()


# --- Multi-turn conversation plugin ---


def _mt_ctx():
    from local_ai_bench.plugins.builtin.multi_turn import MultiTurnPlugin

    return MultiTurnPlugin(), RunContext()


def test_multi_turn_turn_request_forwards_full_history() -> None:
    plugin, ctx = _mt_ctx()
    case = next(c for c in plugin.cases(ctx) if c.id == "mt_secret_token_0001")
    transcript = [
        {"role": "assistant", "content": "Sure, I will remember."},
        {"role": "assistant", "content": "KILO-7"},
    ]
    req = plugin.turn_request(case, None, ctx, transcript)
    roles = [m["role"] for m in req["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    prompts = [m["content"] for m in req["messages"] if m["role"] == "user"]
    assert prompts == case.input["turns"]


async def test_multi_turn_evaluate_perfect_transcript() -> None:
    plugin, ctx = _mt_ctx()
    case = next(c for c in plugin.cases(ctx) if c.id == "mt_secret_token_0001")
    ctx.transcript = [
        {"content": "OK, noted."},
        {"content": "KILO-7"},
        {"content": "KILO-7"},
    ]
    ev = await plugin.evaluate(case, _resp("KILO-7"), ctx)
    assert ev.score == 1.0
    assert ev.passed is True


async def test_multi_turn_evaluate_missed_token() -> None:
    plugin, ctx = _mt_ctx()
    case = next(c for c in plugin.cases(ctx) if c.id == "mt_secret_token_0001")
    ctx.transcript = [
        {"content": "OK."},
        {"content": "I forgot."},
        {"content": "I forgot."},
    ]
    ev = await plugin.evaluate(case, _resp("I forgot."), ctx)
    assert ev.score == 0.0
    assert ev.passed is False


def test_multi_turn_registered_as_builtin() -> None:
    from local_ai_bench.plugins.builtin import load_builtin_plugins

    registry = PluginRegistry()
    load_builtin_plugins(registry)
    assert "multi_turn" in registry.ids()
