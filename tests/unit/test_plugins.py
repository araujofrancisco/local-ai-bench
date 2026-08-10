"""Unit tests for the plugin registry and local plugin loading."""

import base64

from ollama_bench.domain.models import ModelResponse, TimingMetrics, TokenMetrics
from ollama_bench.plugins import load_plugins
from ollama_bench.plugins.base import RunContext
from ollama_bench.plugins.registry import PluginRegistry

VALID_PLUGIN = '''\
from typing import Any, ClassVar
from ollama_bench.plugins.base import BenchmarkPlugin
from ollama_bench.domain.models import BenchmarkCase, BenchmarkCategory, Evaluation, Modality, ModelInfo, ModelResponse
from ollama_bench.plugins.base import RunContext
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
    from ollama_bench.plugins.builtin.smoke import SmokePlugin

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
    from ollama_bench.plugins.builtin.coding import CodingPlugin

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
    from ollama_bench.plugins.builtin.coding import CodingPlugin

    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": True, "timeout_seconds": 10})
    case = _first_case(plugin, ctx)
    ev = await plugin.evaluate(
        case,
        _resp("def reverse_string(s):\n    return s[::-1]\n"),
        ctx,
    )
    assert ev.passed is True
    assert ev.metrics["tests_total"] == 3
    assert ev.metrics["tests_passed"] == 3


async def test_coding_timeout_seconds_is_honored() -> None:
    from ollama_bench.plugins.builtin.coding import CodingPlugin

    plugin = CodingPlugin()
    ctx = RunContext({"execute_code": True, "timeout_seconds": 1})
    case = _first_case(plugin, ctx)
    sleepy = "import time\n\ndef reverse_string(s):\n    time.sleep(60)\n    return s\n"
    ev = await plugin.evaluate(case, _resp(sleepy), ctx)
    assert ev.passed is False


def test_vision_max_image_dimension_caps_size() -> None:
    from ollama_bench.plugins.builtin.vision import VisionPlugin

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
    from ollama_bench.plugins.builtin.coding import CodingPlugin

    assert CodingPlugin.description
    assert "description" in CodingPlugin.__dict__
