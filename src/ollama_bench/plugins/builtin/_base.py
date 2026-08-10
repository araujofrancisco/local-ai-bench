"""Shared base for built-in benchmark plugins (aggregation helper)."""

from __future__ import annotations

from collections.abc import Sequence

from ollama_bench.domain.models import (
    CaseResult,
    PluginAggregate,
)
from ollama_bench.plugins.base import BenchmarkPlugin


def mean_score(results: Sequence[CaseResult]) -> float | None:
    scores = [r.evaluation.score for r in results if r.evaluation.score is not None]
    return round(sum(scores) / len(scores), 4) if scores else None


class BaseTextPlugin(BenchmarkPlugin):
    """Default aggregation: mean case score + success/fail counts."""

    def aggregate(self, results: Sequence[CaseResult]) -> PluginAggregate:
        passed = [r for r in results if r.evaluation.passed]
        return PluginAggregate(
            plugin_id=self.id,
            model_name=results[0].model.model_name if results else "",
            host_name=results[0].model.host_name if results else "",
            total_cases=len(results),
            successful_cases=len(passed),
            failed_cases=len(results) - len(passed),
            score=mean_score(results),
        )