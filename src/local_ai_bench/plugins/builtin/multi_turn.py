"""Multi-turn conversation benchmark — memory and consistency across turns.

The model is led through a short conversation and scored on whether it retains
information given in earlier turns. Because the runner owns the request/reply
loop (see :class:`MultiTurnPlugin`), this plugin only declares the user prompts
and scores the resulting transcript — it never talks to Ollama directly.

Each case provides the prompts for each turn and an ``expected`` mapping that is
matched against the conversation: ``turn_keywords`` (keywords that must appear in
a given assistant reply, 1-indexed) and the optional ``turn_exact`` (a single
token a reply must equal, normalized). The case score is the mean of per-turn
scores.
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
    PluginAggregate,
)
from local_ai_bench.plugins.base import MultiTurnPlugin as MultiTurnCapability
from local_ai_bench.plugins.base import RunContext
from local_ai_bench.plugins.builtin._base import BaseTextPlugin
from local_ai_bench.plugins.score import normalize_text

_CASES = [
    {
        "id": "mt_secret_token_0001",
        "turns": [
            "Remember this secret token: KILO-7. I will ask for it later.",
            "What is the secret token I told you to remember? Reply with just the token.",
            "Repeat it one more time, exactly. Nothing else.",
        ],
        # 0-indexed -> required tokens; turn 1 merely stores the token.
        "turn_exact": {1: "KILO-7", 2: "KILO-7"},
    },
    {
        "id": "mt_preference_0002",
        "turns": [
            "I prefer answers in Celsius. Note that.",
            "What is the boiling point of water? Give just the number.",
            "In which temperature scale did I ask you to answer? One word.",
        ],
        # turn 1 should acknowledge celsius; turn 2 should give ~100; turn 3 celsius.
        "turn_keywords": {1: ["celsius"], 2: ["100"], 3: ["celsius"]},
    },
    {
        "id": "mt_entity_0003",
        "turns": [
            "My account id is AC-9042. Don't forget it.",
            "What is my account id? Reply with just the id.",
            "Confirm by repeating the account id one final time.",
        ],
        "turn_exact": {1: "ac-9042", 2: "ac-9042"},
    },
]


class MultiTurnPlugin(BaseTextPlugin, MultiTurnCapability):
    id: ClassVar[str] = "multi_turn"
    name: ClassVar[str] = "Multi-Turn Conversation"
    description: ClassVar[str] = "Memory and instruction-following across turns."
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.MULTI_TURN
    version: ClassVar[str] = "0.1.0"
    dataset_version: ClassVar[str] = "v1"
    modalities: ClassVar[set[Modality]] = {Modality.TEXT}
    max_turns: ClassVar[int] = 3

    def supports_model(self, model: ModelInfo) -> bool:
        return True

    def build_request(self, case: BenchmarkCase, model: ModelInfo, ctx: RunContext) -> dict[str, Any]:  # noqa: ANN001
        # Single-shot fallback equivalent to the first turn. The orchestrator
        # uses turn_request for multi-turn plugins; this satisfies the base
        # contract for doctor/validation paths that build one request.
        return self.turn_request(case, model, ctx, [])

    def cases(self, ctx) -> Iterable[BenchmarkCase]:  # noqa: ANN001
        for spec in _CASES:
            yield BenchmarkCase(
                id=spec["id"],
                plugin_id=self.id,
                dataset_version=self.dataset_version,
                input={"turns": spec["turns"]},
                expected={
                    "turn_exact": spec.get("turn_exact", {}),
                    "turn_keywords": spec.get("turn_keywords", {}),
                },
            )

    def turn_request(
        self,
        case: BenchmarkCase,
        model: ModelInfo,
        ctx: RunContext,  # noqa: ANN001
        transcript: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompts: list[str] = case.input["turns"]
        idx = len(transcript)  # number of assistant replies already received
        if idx >= len(prompts):
            idx = len(prompts) - 1
        return {
            "messages": [{"role": "user", "content": prompts[idx]}],
            "options": {"temperature": 0.0, "num_predict": 64},
        }

    async def evaluate(self, case, response, ctx) -> Evaluation:  # noqa: ANN001
        expected = case.expected or {}
        turn_exact: dict[int, str] = expected.get("turn_exact", {})
        turn_keywords: dict[int, list[str]] = expected.get("turn_keywords", {})
        transcript = ctx.transcript or []

        per_turn: list[float] = []
        metrics: dict[str, Any] = {"turns": len(transcript), "per_turn": []}
        for i, msg in enumerate(transcript):
            text = msg.get("content") or ""
            score = None
            if i in turn_exact:
                want = turn_exact[i]
                score = 1.0 if normalize_text(text) == normalize_text(want) else 0.0
            elif i in turn_keywords:
                norm = normalize_text(text)
                hits = sum(1 for kw in turn_keywords[i] if normalize_text(kw) in norm)
                score = round(hits / len(turn_keywords[i]), 4) if turn_keywords[i] else 0.0
            if score is not None:
                per_turn.append(score)
                metrics["per_turn"].append({"turn": i + 1, "score": score})

        score = round(sum(per_turn) / len(per_turn), 4) if per_turn else 0.0
        metrics["error"] = response.error
        return Evaluation(score=score, passed=score == 1.0 and response.error is None, metrics=metrics)

    def aggregate(self, results: Sequence[CaseResult]) -> PluginAggregate:
        agg = super().aggregate(results)
        turn_counts = [r.evaluation.metrics.get("turns", 0) for r in results]
        agg.metrics["avg_turns"] = round(sum(turn_counts) / len(results), 4) if results else 0.0
        return agg