"""Benchmark plugin contract (Milestone 3)."""

from __future__ import annotations

from abc import ABC, abstractmethod
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


class RunContext:
    """Mutable context shared across a run (config + live state)."""

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self.options: dict[str, Any] = dict(options or {})
        self.tmp_dir: str | None = None
        self.judge: Any = None


class BenchmarkPlugin(ABC):
    id: ClassVar[str] = "base"
    name: ClassVar[str] = "Base plugin"
    description: ClassVar[str] = ""
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.REASONING
    version: ClassVar[str] = "0.1.0"
    dataset_version: ClassVar[str] = "v1"
    modalities: ClassVar[set[Modality]] = {Modality.TEXT}

    def supports_model(self, model: ModelInfo) -> bool:
        return True

    async def prepare(self, ctx: RunContext) -> None:
        return None

    @abstractmethod
    def cases(self, ctx: RunContext) -> Iterable[BenchmarkCase]:
        raise NotImplementedError

    @abstractmethod
    def build_request(
        self,
        case: BenchmarkCase,
        model: ModelInfo,
        ctx: RunContext,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def evaluate(
        self,
        case: BenchmarkCase,
        response: ModelResponse,
        ctx: RunContext,
    ) -> Evaluation:
        raise NotImplementedError

    def aggregate(self, results: Sequence[CaseResult]) -> PluginAggregate:
        passed = [r for r in results if r.evaluation.passed]
        return PluginAggregate(
            plugin_id=self.id,
            model_name=results[0].model.model_name if results else "",
            host_name=results[0].model.host_name if results else "",
            total_cases=len(results),
            successful_cases=len(passed),
            failed_cases=len(results) - len(passed),
        )

    async def teardown(self, ctx: RunContext) -> None:
        return None