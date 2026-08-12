"""Generalised context-window benchmark.

Runs a single configurable prompt at several ``num_ctx`` (context window) sizes
and measures how accuracy/latency degrades as the context grows. This is more
general than the built-in ``long_context`` plugin, which is hard-coded to the
"continents" fact-recall question: here the question, expected answer and the
sizes to probe are all configurable via plugin options.

Each size yields a distinct ``BenchmarkCase`` whose ``input.target_context`` is
used by ``build_request`` to set Ollama's ``num_ctx`` (doubled, matching the
``long_context`` convention, since ``num_ctx`` counts both prompt and
completion). Filler text is prepended so the prompt actually *fills* the larger
windows, exercising the model under realistic context pressure.

Options (set under ``plugins.options.multi_context``):

| key            | default                                | meaning |
| -------------- | -------------------------------------- | ------- |
| ``prompt``     | "What is the capital of France?"       | question sent to the model |
| ``expected``   | "paris"                                  | expected substring (case-insensitive contain) |
| ``context_sizes`` | config candidate sizes, else `[512,1024,4096,8192,16384]` | window sizes to probe |
| ``filler``     | a pangram sentence                     | pad text used to fill the context |
| ``contains``   | `true`                                  | if true, score via substring; else numeric compare on `expected` |

A model with a known ``max_context_tokens`` prunes any size beyond its limit.
"""

from __future__ import annotations

import math
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
from local_ai_bench.plugins.base import RunContext
from local_ai_bench.plugins.builtin._base import BaseTextPlugin
from local_ai_bench.plugins.score import normalize_answer, numeric_close

_DEFAULT_SIZES = [512, 1024, 4096, 8192, 16384]
_DEFAULT_FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
)
_DEFAULT_PROMPT = "What is the capital of France?"
_DEFAULT_EXPECTED = "paris"

# Rough token-to-character ratio used to approximate fill length.
_CHARS_PER_TOKEN = 4


def _filler_text(target_tokens: int, filler: str) -> str:
    """Return filler text worth roughly ``target_tokens`` tokens."""
    if target_tokens <= 0:
        return ""
    unit_len = max(1, len(filler))
    repeats = max(1, math.ceil((target_tokens * _CHARS_PER_TOKEN) / unit_len))
    return (filler * repeats)[: target_tokens * _CHARS_PER_TOKEN]


def _resolve_sizes(ctx: RunContext) -> list[int]:
    sizes = ctx.options.get("context_sizes")
    if isinstance(sizes, list) and sizes:
        return [int(s) for s in sizes]
    return list(_DEFAULT_SIZES)


class MultiContextPlugin(BaseTextPlugin):
    """Sweep accuracy/latency across multiple context-window sizes."""

    id: ClassVar[str] = "multi_context"
    name: ClassVar[str] = "Context Window"
    description: ClassVar[
        str
    ] = "Measures correctness and latency at several context-window sizes."
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.LONG_CONTEXT
    version: ClassVar[str] = "0.1.0"
    dataset_version: ClassVar[str] = "v1"
    modalities: ClassVar[set[Modality]] = {Modality.TEXT}

    def supports_model(self, model: ModelInfo) -> bool:
        # Only meaningful for models whose advertised window can cover the
        # smallest candidate size; otherwise let it run and prune in `cases`.
        return True

    def cases(self, ctx: RunContext) -> Iterable[BenchmarkCase]:
        max_ctx = ctx.options.get("max_context_tokens")
        prompt = ctx.options.get("prompt") or _DEFAULT_PROMPT
        expected = ctx.options.get("expected")
        filler = ctx.options.get("filler") or _DEFAULT_FILLER
        for idx, size in enumerate(_resolve_sizes(ctx)):
            if isinstance(max_ctx, int) and size > max_ctx:
                continue
            padded = _filler_text(size, filler)
            content = f"{padded}\n\n{prompt}" if padded else prompt
            yield BenchmarkCase(
                id=f"multictx_ctx_{size}_{idx:04d}",
                plugin_id=self.id,
                dataset_version=self.dataset_version,
                input={"prompt": content, "target_context": size, "base_prompt": prompt},
                expected={"expected": expected} if expected else None,
                metadata={"target_context": size},
            )

    def build_request(self, case: BenchmarkCase, model: ModelInfo, ctx: RunContext) -> dict[str, Any]:
        target = case.input.get("target_context", 4096)
        return {
            "messages": [{"role": "user", "content": case.input["prompt"]}],
            "options": {
                "temperature": 0.0,
                "num_predict": 64,
                "num_ctx": target * 2,
            },
        }

    async def evaluate(self, case: BenchmarkCase, response: ModelResponse, ctx: RunContext) -> Evaluation:
        expected = case.expected.get("expected") if case.expected else ctx.options.get("expected")
        text = response.text or ""
        use_contains = ctx.options.get("contains", True)
        if expected:
            if use_contains:
                passed = normalize_answer(text).lower().find(str(expected).lower()) != -1
                score = 1.0 if passed else 0.0
            else:
                passed = bool(numeric_close(text, str(expected)))
                score = 1.0 if passed else 0.0
        else:
            passed = bool(text.strip()) and response.error is None
            score = 1.0 if passed else 0.0
        return Evaluation(
            score=score,
            passed=passed,
            metrics={
                "target_context": case.input.get("target_context"),
                "chars": len(text),
                "error": response.error,
            },
        )

    def aggregate(self, results: Sequence[CaseResult]) -> PluginAggregate:
        agg = super().aggregate(results)
        per_context: dict[int, float | None] = {}
        for r in results:
            size = r.case.input.get("target_context")
            if size is not None:
                per_context[int(size)] = r.evaluation.score
        agg.metrics["per_context_score"] = per_context
        agg.metrics["max_context_tokens"] = (
            results[0].model.max_context_tokens if results else None
        )
        return agg
