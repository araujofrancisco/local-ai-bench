"""Long-context benchmark — measures how performance degrades with context size.

Tests the model across increasing context lengths and reports a relative
score so users can see at what context size a model starts to break down,
plus how the model reports its own context-window size.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, ClassVar

from local_ai_bench.domain.models import (
    BenchmarkCase,
    BenchmarkCategory,
    Evaluation,
    Modality,
    PluginAggregate,
)
from local_ai_bench.plugins.builtin._base import BaseTextPlugin
from local_ai_bench.plugins.score import numeric_close

# Repeated factual question; we append filler tokens to simulate growing context.
_QUESTION = "How many continents are there on Earth?"
_EXPECTED = "7"
_CONTEXT_SIZES = [256, 1024, 4096, 16384]


def _filler(n: int) -> str:
    """Deterministic placeholder text to inflate the prompt to ~n tokens."""
    chunk = "The quick brown fox jumps over the lazy dog. "
    repeats = max(1, n // 9)
    return (chunk * repeats)[: n * 4]


class LongContextPlugin(BaseTextPlugin):
    id: ClassVar[str] = "long_context"
    name: ClassVar[str] = "Long Context"
    description: ClassVar[str] = "Fact recall accuracy as context length grows."
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.LONG_CONTEXT
    version: ClassVar[str] = "0.1.0"
    dataset_version: ClassVar[str] = "v1"
    modalities: ClassVar[set[Modality]] = {Modality.TEXT}

    def supports_model(self, model) -> bool:  # noqa: ANN001
        # Skip the 16k case if the model advertises a smaller context window.
        max_ctx = getattr(model, "max_context_tokens", None)
        if max_ctx is None:
            return True
        return max_ctx >= min(_CONTEXT_SIZES)

    def cases(self, ctx) -> Iterable[BenchmarkCase]:
        max_ctx = ctx.options.get("max_context_tokens") if hasattr(ctx, "options") else None
        for i, tokens in enumerate(_CONTEXT_SIZES):
            if max_ctx is not None and tokens > max_ctx:
                continue
            prompt = f"{_filler(tokens)}\n\nFinal question: {_QUESTION}"
            yield BenchmarkCase(
                id=f"longctx_ctx_{tokens}_{i:04d}",
                plugin_id=self.id,
                dataset_version=self.dataset_version,
                input={"prompt": prompt, "target_context": tokens},
                expected={"answer": _EXPECTED},
            )

    def build_request(self, case, model, ctx) -> dict[str, Any]:  # noqa: ANN001
        return {
            "messages": [{"role": "user", "content": case.input["prompt"]}],
            "options": {
                "temperature": 0.0,
                "num_predict": 64,
                "num_ctx": case.input["target_context"] * 2,
            },
        }

    async def evaluate(self, case, response, ctx) -> Evaluation:  # noqa: ANN001
        expected = case.expected["answer"]
        text = response.text
        ok = numeric_close(text, expected)
        target = case.input["target_context"]
        return Evaluation(
            score=1.0 if ok else 0.0,
            passed=bool(ok),
            metrics={"target_context": target, "truncated": response.truncated},
        )

    def aggregate(self, results: Sequence) -> PluginAggregate:  # type: ignore[override]
        agg = super().aggregate(results)
        per_ctx = {
            r.case.input.get("target_context"): r.evaluation.score
            for r in results
        }
        agg.metrics["per_context_score"] = per_ctx
        if hasattr(agg, "model_name") and results:
            agg.metrics["max_context_tokens"] = (
                getattr(results[0].model, "max_context_tokens", None)
            )
        return agg