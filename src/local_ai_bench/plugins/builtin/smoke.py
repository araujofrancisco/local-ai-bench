"""Smoke/generation benchmark — the default health & latency workload.

Produces a real, reproducible generation task per model so a fresh install can
produce a useful report before any Milestone 5 plugins are implemented.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, ClassVar

from local_ai_bench.domain.models import (
    BenchmarkCase,
    BenchmarkCategory,
    CaseResult,
    Evaluation,
    Modality,
    ModelInfo,
    ModelResponse,
    PluginAggregate,
)
from local_ai_bench.plugins.base import BenchmarkPlugin, RunContext

_CASES = [
    {
        "id": "smoke_sum_it_0001",
        "input": {
            "prompt": (
                "List the numbers from 1 to 10, one per line. Do not add anything else."
            ),
        },
        "expected": {"min_lines": 10},
    },
    {
        "id": "smoke_triangle_0002",
        "input": {
            "prompt": "What is 15% of 240? Answer with a single number only.",
        },
        "expected": {"answer": "36"},
    },
]


class SmokePlugin(BenchmarkPlugin):
    id: ClassVar[str] = "smoke"
    name: ClassVar[str] = "Smoke / generation"
    description: ClassVar[str] = "Fast sanity check: basic text generation completes with valid output."
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.REASONING
    version: ClassVar[str] = "0.1.0"
    dataset_version: ClassVar[str] = "v1"
    modalities: ClassVar[set[Modality]] = {Modality.TEXT}

    def cases(self, ctx: RunContext) -> Iterable[BenchmarkCase]:
        for spec in _CASES:
            yield BenchmarkCase(
                id=spec["id"],
                plugin_id=self.id,
                dataset_version=self.dataset_version,
                input=spec["input"],
                expected=spec.get("expected"),
            )

    def build_request(
        self, case: BenchmarkCase, model: ModelInfo, ctx: RunContext
    ) -> dict[str, Any]:
        return {
            "messages": [{"role": "user", "content": case.input["prompt"]}],
            "options": {"temperature": 0.0, "num_predict": 64},
        }

    async def evaluate(
        self, case: BenchmarkCase, response: ModelResponse, ctx: RunContext
    ) -> Evaluation:
        text = response.text.strip()
        return Evaluation(score=0.0 if not text else 1.0, passed=text != "", metrics={"chars": len(text)})

    def aggregate(self, results: Sequence[CaseResult]) -> PluginAggregate:
        passed = [r for r in results if r.evaluation.passed]
        return PluginAggregate(
            plugin_id=self.id,
            model_name=results[0].model.model_name if results else "",
            host_name=results[0].model.host_name if results else "",
            total_cases=len(results),
            successful_cases=len(passed),
            failed_cases=len(results) - len(passed),
            score=round(len(passed) / len(results), 4) if results else None,
        )