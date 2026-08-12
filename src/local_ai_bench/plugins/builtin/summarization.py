"""Summarization benchmark — recall of key facts from the source text."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

from local_ai_bench.domain.models import (
    BenchmarkCase,
    BenchmarkCategory,
    Evaluation,
    Modality,
)
from local_ai_bench.judge import judge_evaluation
from local_ai_bench.plugins.builtin._base import BaseTextPlugin
from local_ai_bench.plugins.score import keyword_recall

_TEXT = (
    "The International Space Station (ISS) is a habitable artificial satellite "
    "in low Earth orbit. It was launched in 1998 and has been continuously occupied "
    "since November 2000. The ISS orbits at an altitude of approximately 418 km "
    "and travels at a speed of about 27,600 km/h. It is the largest human-made "
    "object in space and cost roughly $150 billion to build. The station hosts a "
    "variety of scientific experiments in microgravity and is serviced by regular "
    "crew visits from NASA, Roscosmos, SpaceX, and other space agencies."
)

_CASES = [
    {
        "id": "summarize_iss_0001",
        "prompt": f"Summarize the following text in 2-3 sentences:\n\n{_TEXT}",
        "keywords": ["international space station", "1998", "418", "27,600", "150 billion"],
    },
]


class SummarizationPlugin(BaseTextPlugin):
    id: ClassVar[str] = "summarization"
    name: ClassVar[str] = "Summarization"
    description: ClassVar[str] = "Faithfulness of summaries against source documents."
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.SUMMARIZATION
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
                expected={"keywords": spec["keywords"]},
            )

    def build_request(self, case, model, ctx) -> dict[str, Any]:  # noqa: ANN001
        return {
            "messages": [{"role": "user", "content": case.input["prompt"]}],
            "options": {"temperature": 0.0, "num_predict": 160},
        }

    async def evaluate(self, case, response, ctx) -> Evaluation:  # noqa: ANN001
        expected_kw = case.expected["keywords"]
        recall = keyword_recall(response.text, expected_kw)
        return await judge_evaluation(
            ctx,
            case,
            response,
            rubric=(
                "Summary must faithfully cover the key facts of the source text "
                "without hallucinating and stay concise."
            ),
            deterministic_score=recall,
            passed=recall >= 0.4,
            pass_threshold=0.4,
            metrics={"keyword_recall": recall, "summary": response.text[:120]},
        )