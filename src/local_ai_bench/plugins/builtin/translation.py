"""Translation benchmark — semantic correctness, not string-exact equality."""

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

_CASES = [
    {
        "id": "translate_en_to_fr_0001",
        "prompt": "Translate to French: 'Good morning, how are you?'",
        "keywords": ["bonjour", "vous"],
        "expected_lang": "fr",
    },
    {
        "id": "translate_en_to_es_0002",
        "prompt": "Translate to Spanish: 'I would like a coffee, please.'",
        "keywords": ["café", "me gustaría"],
        "expected_lang": "es",
    },
    {
        "id": "translate_en_to_de_0003",
        "prompt": "Translate to German: 'Where is the nearest train station?'",
        "keywords": ["bahnhof", "naher"],
        "expected_lang": "de",
    },
]


class TranslationPlugin(BaseTextPlugin):
    id: ClassVar[str] = "translation"
    name: ClassVar[str] = "Translation"
    description: ClassVar[str] = "Translation quality between language pairs."
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.TRANSLATION
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
            "options": {"temperature": 0.0, "num_predict": 128},
        }

    async def evaluate(self, case, response, ctx) -> Evaluation:  # noqa: ANN001
        expected_kw = case.expected["keywords"]
        recall = keyword_recall(response.text, expected_kw)
        return await judge_evaluation(
            ctx,
            case,
            response,
            rubric=(
                "Translation must preserve the meaning of the source text and "
                "read fluently in the target language."
            ),
            deterministic_score=recall,
            passed=recall >= 0.5,
            pass_threshold=0.5,
            metrics={"keyword_recall": recall, "text": response.text[:120]},
        )