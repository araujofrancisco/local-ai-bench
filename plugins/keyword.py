"""Example local plugin: a simple keyword-presence benchmark.

Drop additional `.py` files like this one into the directory named by
`plugins.local_dir` in your config; `local-ai-bench plugins list` picks them up
automatically with no changes to the core runner.
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
        "id": "kw_hello_0001",
        "input": {"prompt": "Say exactly: hello world"},
        "expected": {"keywords": ["hello", "world"]},
    },
    {
        "id": "kw_llama_0002",
        "input": {"prompt": "What model family is Llama?"},
        "expected": {"keywords": ["llama"]},
    },
]


class KeywordPlugin(BenchmarkPlugin):
    id: ClassVar[str] = "keyword"
    name: ClassVar[str] = "Keyword presence"
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
        text = response.text.lower()
        needed = (case.expected or {}).get("keywords") or []
        hits = [kw for kw in needed if kw in text]
        score = len(hits) / len(needed) if needed else 0.0
        return Evaluation(
            score=score,
            passed=score == 1.0,
            metrics={"keywords_found": hits, "keywords_required": needed},
        )

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
