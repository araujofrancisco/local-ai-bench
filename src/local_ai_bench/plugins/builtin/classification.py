"""Classification benchmark — the model must put text into a known label set.

This is a common real-world task (triage, tagging, sentiment, routing). Each
case describes an ``input`` text, a small set of allowed ``labels``, and the
expected ``label``. The reply is scored on:

1. **Exact label** — the output tokens match the expected label after
   normalization (handles a bare word, or "label: X" / "sentiment: positive"
   framing).
2. **In-set** — even when the chosen label differs from the expected one, the
   model still scores partial credit if it picked any allowed label (vs.
   inventing an out-of-set label or refusing).

Scoring is deterministic (normalized substring + set membership), with no judge
required.
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
    PluginAggregate,
)
from local_ai_bench.plugins.builtin._base import BaseTextPlugin
from local_ai_bench.plugins.score import normalize_text

_CASES = [
    {
        "id": "cls_sentiment_0001",
        "input": "The train was late again, but the refund process was painless.",
        "labels": ["positive", "negative", "neutral"],
        "expected": "neutral",
    },
    {
        "id": "cls_sentiment_0002",
        "input": "Absolutely loved it — best purchase all year!",
        "labels": ["positive", "negative", "neutral"],
        "expected": "positive",
    },
    {
        "id": "cls_ticket_0003",
        "input": "My order #4823 never arrived and billing charged me twice.",
        "labels": ["billing", "shipping", "refund", "cancellation"],
        "expected": "shipping",
    },
    {
        "id": "cls_ticket_0004",
        "input": "I'd like a full refund for the damaged headphones.",
        "labels": ["billing", "shipping", "refund", "cancellation"],
        "expected": "refund",
    },
    {
        "id": "cls_urgency_0005",
        "input": "Server is down across the whole office — respond immediately!",
        "labels": ["low", "medium", "high"],
        "expected": "high",
    },
    {
        "id": "cls_topic_0006",
        "input": "The new CUDA kernel shaved 40% off our inference latency.",
        "labels": ["sports", "politics", "technology", "weather"],
        "expected": "technology",
    },
    {
        "id": "cls_topic_0007",
        "input": "The referee's call decided the championship in extra time.",
        "labels": ["sports", "politics", "technology", "weather"],
        "expected": "sports",
    },
]


def _extract_label(text: str, labels: list[str]) -> str | None:
    """Find which allowed label appears in the response (normalized)."""
    norm = normalize_text(text)
    for label in labels:
        if normalize_text(label) in norm:
            return label
    return None


class ClassificationPlugin(BaseTextPlugin):
    id: ClassVar[str] = "classification"
    name: ClassVar[str] = "Classification"
    description: ClassVar[str] = (
        "Model must assign input text to one of a given set of labels."
    )
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.CLASSIFICATION
    version: ClassVar[str] = "0.1.0"
    dataset_version: ClassVar[str] = "v1"
    modalities: ClassVar[set[Modality]] = {Modality.TEXT}

    def supports_model(self, model) -> bool:  # noqa: ANN001
        return True

    def cases(self, ctx) -> Iterable[BenchmarkCase]:  # noqa: ANN001
        for spec in _CASES:
            yield BenchmarkCase(
                id=spec["id"],
                plugin_id=self.id,
                dataset_version=self.dataset_version,
                input={
                    "text": spec["input"],
                    "labels": spec["labels"],
                },
                expected={
                    "label": spec["expected"],
                    "labels": spec["labels"],
                },
            )

    def build_request(self, case, model, ctx) -> dict[str, Any]:  # noqa: ANN001
        return {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Classify the text into one of the given labels. Reply "
                        "with the single label only.\n"
                        f"Labels: {', '.join(case.input['labels'])}"
                    ),
                },
                {"role": "user", "content": case.input["text"]},
            ],
            "options": {"temperature": 0.0, "num_predict": 16},
        }

    async def evaluate(self, case, response, ctx) -> Evaluation:  # noqa: ANN001
        labels: list[str] = case.expected["labels"]
        expected_label: str = case.expected["label"]
        chosen = _extract_label(response.text, labels)

        metrics: dict[str, Any] = {
            "expected_label": expected_label,
            "chosen_label": chosen,
        }
        if chosen is None:
            return Evaluation(
                score=0.0,
                passed=False,
                metrics={**metrics, "in_set": False, "classified": False},
            )

        if chosen == expected_label:
            return Evaluation(
                score=1.0,
                passed=True,
                metrics={**metrics, "in_set": True, "classified": True},
            )

        # Picked a valid but wrong label: half credit for staying in the set.
        return Evaluation(
            score=0.5,
            passed=False,
            metrics={**metrics, "in_set": True, "classified": True},
        )

    def aggregate(self, results: Sequence[CaseResult]) -> PluginAggregate:
        base = super().aggregate(results)
        in_set = [
            r
            for r in results
            if r.evaluation.metrics and r.evaluation.metrics.get("in_set")
        ]
        base.metrics["in_set_ratio"] = (
            round(len(in_set) / len(results), 4) if results else None
        )
        return base