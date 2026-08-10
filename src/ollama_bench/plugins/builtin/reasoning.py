"""Reasoning benchmark — deterministic math/logic problems."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

from ollama_bench.domain.models import (
    BenchmarkCase,
    BenchmarkCategory,
    Evaluation,
    Modality,
)
from ollama_bench.plugins.builtin._base import BaseTextPlugin
from ollama_bench.plugins.score import normalize_answer, numeric_close

_CASES = [
    {"id": "reason_math_pct_0001", "prompt": "What is 15% of 240?", "expected": "36"},
    {
        "id": "reason_math_speed_0002",
        "prompt": "If a train travels 300 miles in 5 hours, what is its average speed in mph?",
        "expected": "60",
    },
    {
        "id": "reason_logic_days_0003",
        "prompt": "If today is Wednesday, what day of the week will it be in 100 days?",
        "expected": "friday",
    },
    {
        "id": "reason_math_sum_0004",
        "prompt": "What is the sum of the numbers from 1 to 10?",
        "expected": "55",
    },
]


class ReasoningPlugin(BaseTextPlugin):
    id: ClassVar[str] = "reasoning"
    name: ClassVar[str] = "Reasoning"
    description: ClassVar[str] = "Multi-step logical and arithmetic reasoning problems."
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.REASONING
    version: ClassVar[str] = "0.1.0"
    dataset_version: ClassVar[str] = "v1"
    modalities: ClassVar[set[Modality]] = {Modality.TEXT}

    def cases(self, ctx) -> Iterable[BenchmarkCase]:
        for spec in _CASES:
            yield BenchmarkCase(
                id=spec["id"],
                plugin_id=self.id,
                dataset_version=self.dataset_version,
                input={"prompt": spec["prompt"]},
                expected={"answer": spec["expected"]},
            )

    def build_request(self, case, model, ctx) -> dict[str, Any]:  # noqa: ANN001
        return {
            "messages": [{"role": "user", "content": case.input["prompt"]}],
            "options": {"temperature": 0.0, "num_predict": 256},
        }

    async def evaluate(self, case, response, ctx) -> Evaluation:  # noqa: ANN001
        expected = case.expected["answer"]
        text = response.text
        score = 1.0 if numeric_close(text, expected) else 0.0
        return Evaluation(
            score=score,
            passed=bool(score),
            metrics={"answer": normalize_answer(text), "expected": expected},
        )