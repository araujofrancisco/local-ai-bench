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
        self.transcript: list[dict[str, Any]] = []
        self.turn_count: int = 0


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


class MultiTurnPlugin(BenchmarkPlugin):
    """Optional capability: a case is a multi-turn conversation (PLAN §14.3).

    The runner drives up to ``max_turns`` Ollama calls per case attempt. For
    each turn it calls :meth:`turn_request` (receiving the transcript so far),
    sends the request, appends the assistant reply to ``ctx.transcript``, and
    keeps going until the model errors or :meth:`should_stop` returns True.
    :meth:`evaluate` then scores the run — it can grade the whole conversation
    through ``ctx.transcript`` (list of ``{"role": "assistant", ...}`` dicts)
    in addition to the final :class:`ModelResponse`.

    The runner still owns the request/retry/event plumbing, so multi-turn
    plugins get the same fail-safe isolation as single-shot ones. A case counts
    once for progress regardless of turn count.
    """

    max_turns: ClassVar[int] = 5

    @abstractmethod
    def turn_request(
        self,
        case: BenchmarkCase,
        model: ModelInfo,
        ctx: RunContext,
        transcript: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the request payload for the next user turn.

        Returns the same shape as :meth:`BenchmarkPlugin.build_request`
        (``messages``/``options``/``tools``). The ``messages`` list must
        contain the **full conversation** so far: the user prompt for each
        completed turn, the prior assistant replies (including any
        ``tool_calls``), and any tool results (role ``"tool"``) — the
        orchestrator sends ``request["messages"]`` verbatim and never rebuilds
        history itself. ``transcript`` holds the turns already completed
        (assistant replies); the orchestrator appends each reply after sending,
        so the plugin never reconstructs state.
        """
        raise NotImplementedError

    def should_stop(self, case: BenchmarkCase, response: ModelResponse, ctx: RunContext, turn: int) -> bool:
        """Stop the conversation after this turn? ``turn`` is 0-based."""
        return turn + 1 >= self.max_turns