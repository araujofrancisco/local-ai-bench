"""Function-calling benchmark — the model must invoke the right tool with the right args.

Each case declares an Ollama ``tools`` schema and a prompt that *requires* a tool
call. The model fails the case if it answers directly instead of calling the
tool, calls the wrong tool, or passes wrong arguments. Argument values are
compared with the same tolerance rules as other numeric plugins, so floating
points and integer inputs are graded fairly.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any, ClassVar

from local_ai_bench.domain.models import (
    BenchmarkCase,
    BenchmarkCategory,
    CaseResult,
    Evaluation,
    Modality,
    ModelInfo,
    PluginAggregate,
)
from local_ai_bench.plugins.builtin._base import BaseTextPlugin, mean_score
from local_ai_bench.plugins.score import normalize_text

_CASES = [
    {
        "id": "fc_weather_0001",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a given city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name."}
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "prompt": "What is the weather in Paris today?",
        "expected": {"tool": "get_weather", "args": {"city": "paris"}},
    },
    {
        "id": "fc_fibonacci_0002",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "fibonacci",
                    "description": "Compute the n-th Fibonacci number (1-indexed).",
                    "parameters": {
                        "type": "object",
                        "properties": {"n": {"type": "integer", "description": "Position."}},
                        "required": ["n"],
                    },
                },
            }
        ],
        "prompt": "What is the 12th Fibonacci number?",
        "expected": {"tool": "fibonacci", "args": {"n": 12}},
    },
    {
        "id": "fc_user_lookup_0003",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "fetch_user_profile",
                    "description": "Fetch the profile for a user id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "Account id."}
                        },
                        "required": ["user_id"],
                    },
                },
            }
        ],
        "prompt": "Show me the profile of the customer with id u-42.",
        "expected": {"tool": "fetch_user_profile", "args": {"user_id": "u-42"}},
    },
]


def _as_dict(value: Any) -> dict[str, Any]:
    """Accept dict args or a JSON-string encoding of them."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


class FunctionCallingPlugin(BaseTextPlugin):
    id: ClassVar[str] = "function_calling"
    name: ClassVar[str] = "Function Calling"
    description: ClassVar[str] = "Model must call the right tool with the right arguments."
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.FUNCTION_CALLING
    version: ClassVar[str] = "0.1.0"
    dataset_version: ClassVar[str] = "v1"
    modalities: ClassVar[set[Modality]] = {Modality.TEXT}

    def supports_model(self, model: ModelInfo) -> bool:
        return bool(model.supports_tools)

    def cases(self, ctx) -> Iterable[BenchmarkCase]:  # noqa: ANN001
        for spec in _CASES:
            yield BenchmarkCase(
                id=spec["id"],
                plugin_id=self.id,
                dataset_version=self.dataset_version,
                input={"prompt": spec["prompt"]},
                expected={"tools": spec["tools"], **spec["expected"]},
            )

    def build_request(self, case, model, ctx) -> dict[str, Any]:  # noqa: ANN001
        return {
            "messages": [{"role": "user", "content": case.input["prompt"]}],
            "options": {"temperature": 0.0, "num_predict": 128},
            "tools": (case.expected or {}).get("tools", []),
        }

    async def evaluate(self, case, response, ctx) -> Evaluation:  # noqa: ANN001
        expected = case.expected or {}
        metrics: dict[str, Any] = {"tool_calls": len(response.tool_calls or [])}
        if response.error:
            metrics["error"] = response.error
            return Evaluation(score=0.0, passed=False, metrics=metrics)

        calls = response.tool_calls or []
        if not calls:
            return Evaluation(
                score=0.0,
                passed=False,
                metrics={**metrics, "answered_directly": bool(response.text.strip())},
            )

        call = calls[0].get("function") or {}
        got_name = call.get("name")
        got_args = _as_dict(call.get("arguments"))
        name_ok = got_name == expected.get("tool")
        metrics["tool_name"] = got_name
        metrics["tool_args"] = got_args
        metrics["name_ok"] = name_ok

        args_fraction, args_metrics = self._args_fraction(
            got_args, expected.get("args", {}), ctx
        )
        metrics.update(args_metrics)
        score = round((0.5 if name_ok else 0.0) + 0.5 * args_fraction, 4)
        return Evaluation(
            score=score,
            passed=score == 1.0,
            metrics=metrics,
        )

    def _args_fraction(
        self, got: dict[str, Any], expected: dict[str, Any], ctx: Any
    ) -> tuple[float, dict[str, Any]]:
        tolerance = float(ctx.options.get("arg_tolerance", 0.5))
        matched = 0
        per_key: dict[str, bool] = {}
        for key, want in expected.items():
            got_val = got.get(key)
            if _values_close(want, got_val, tolerance):
                matched += 1
                per_key[key] = True
            else:
                per_key[key] = False
        total = len(expected) or 1
        return round(matched / total, 4), {"args_matched": f"{matched}/{total}"}

    def aggregate(self, results: Sequence[CaseResult]) -> PluginAggregate:
        agg = super().aggregate(results)
        called = [r for r in results if (r.response.tool_calls or [])]
        agg.metrics["tool_call_ratio"] = round(
            len(called) / len(results), 4
        ) if results else 0.0
        agg.metrics["name_match_ratio"] = round(
            sum(1 for r in results if r.evaluation.metrics.get("name_ok"))
            / len(results),
            4,
        ) if results else 0.0
        agg.score = mean_score(results)
        return agg


def _values_close(want: Any, got: Any, tolerance: float) -> bool:
    """Compare a tool argument to the expected value.

    Numeric values are compared within ``tolerance``; strings use the shared
    normalized (lowercased, punctuation-stripped) comparison so casing and small
    formatting differences do not fail a case.
    """
    if want is None or got is None:
        return want is None and got is None
    try:
        return abs(float(want) - float(got)) <= tolerance
    except (TypeError, ValueError):
        return normalize_text(str(want)) == normalize_text(str(got))