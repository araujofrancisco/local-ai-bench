"""Retrieval-grounded QA benchmark — answers must draw on the source passage.

Each case presents two "retrieved" documents to the model: a *source* passage
containing the answer facts and a *distractor* passage containing similar but
wrong facts (as real RAG pipelines return imperfect results). The score rewards
recall of the source facts while penalizing hallucination — importing any
distractor-only fact.

All facts are fictional so the model cannot answer from world knowledge; it must
actually ground its answer in the provided source document.
"""

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
from local_ai_bench.plugins.score import keyword_recall, normalize_text

_CASES = [
    {
        "id": "rag_rocket_0001",
        "document": (
            "Acme Space's Atlas-9 rocket launched successfully on 5 March 2031 "
            "from the Vandenberg site. It carries a payload of six satellites "
            "for the Halley-on-orbit network."
        ),
        "distractor": (
            "Acme Space's Atlas-9 rocket launched successfully on 5 March 2028 "
            "from the Cape Canaveral site. It carries a payload of twelve "
            "satellites for the Halley-on-orbit network."
        ),
        "question": "When did Acme Space's Atlas-9 launch, from where, and how many satellites did it carry?",
        "keywords": ["2031", "vandenberg", "six"],
        "wrong_keywords": ["2028", "cape canaveral", "twelve"],
    },
    {
        "id": "rag_account_0002",
        "document": (
            "The Nimbus account renewal policy, effective 1 July 2025, requires "
            "a monthly fee of $14 and auto-renews unless the customer cancels "
            "by email at least ten days before renewal."
        ),
        "distractor": (
            "The Nimbus account renewal policy, effective 1 January 2024, "
            "requires a monthly fee of $7 and auto-renews unless the customer "
            "cancels by phone at least three days before renewal."
        ),
        "question": "Per the current Nimbus policy, what is the monthly fee and how many days before renewal must a customer cancel?",
        "keywords": ["14", "ten"],
        "wrong_keywords": ["7", "three"],
    },
    {
        "id": "rag_medication_0003",
        "document": (
            "Corvacor 200 mg tablets are taken twice daily with food. The most "
            "common side effect is mild nausea, and patients should not consume "
            "grapefruit while on the treatment."
        ),
        "distractor": (
            "Corvacor 100 mg tablets are taken once daily on an empty stomach. "
            "The most common side effect is headache, and patients should not "
            "consume caffeine while on the treatment."
        ),
        "question": "How often are Corvacor 200 mg tablets taken, and what must patients avoid eating?",
        "keywords": ["twice", "grapefruit"],
        "wrong_keywords": ["once", "empty stomach", "caffeine"],
    },
    {
        "id": "rag_inventory_0004",
        "document": (
            "Warehouse Sector 7 currently holds 42 crates of blue widgets, 13 "
            "crates of green widgets, and zero red widgets pending a delivery "
            "scheduled for next Tuesday."
        ),
        "distractor": (
            "Warehouse Sector 7 currently holds 40 crates of blue widgets, 30 "
            "crates of green widgets, and 5 red widgets pending a delivery "
            "scheduled for next Friday."
        ),
        "question": "What quantity of green widgets does Sector 7 hold today?",
        "keywords": ["13"],
        "wrong_keywords": ["30", "40", "5"],
    },
]


class RagPlugin(BaseTextPlugin):
    id: ClassVar[str] = "rag"
    name: ClassVar[str] = "Retrieval-Grounded QA"
    description: ClassVar[str] = "Answers must be grounded in the source passage, not the distractor."
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.RETRIEVAL
    version: ClassVar[str] = "0.1.0"
    dataset_version: ClassVar[str] = "v1"
    modalities: ClassVar[set[Modality]] = {Modality.TEXT}

    def cases(self, ctx) -> Iterable[BenchmarkCase]:  # noqa: ANN001
        for spec in _CASES:
            yield BenchmarkCase(
                id=spec["id"],
                plugin_id=self.id,
                dataset_version=self.dataset_version,
                input={"question": spec["question"]},
                expected={
                    "keywords": spec["keywords"],
                    "wrong_keywords": spec["wrong_keywords"],
                    "source": spec["document"],
                    "distractor": spec["distractor"],
                },
            )

    def build_request(self, case, model, ctx) -> dict[str, Any]:  # noqa: ANN001
        expected = case.expected or {}
        prompt = (
            "Answer the question using ONLY the provided documents. If the "
            "information is not in the documents, say so rather than guessing.\n\n"
            f"Document 1:\n{expected['source']}\n\n"
            f"Document 2:\n{expected['distractor']}\n\n"
            f"Question: {case.input['question']}"
        )
        return {
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.0, "num_predict": 200},
        }

    async def evaluate(self, case, response, ctx) -> Evaluation:  # noqa: ANN001
        expected = case.expected or {}
        recall = keyword_recall(response.text, expected.get("keywords", []))
        norm = normalize_text(response.text)
        hallucinated = [
            kw for kw in expected.get("wrong_keywords", []) if normalize_text(kw) in norm
        ]
        penalty = float(ctx.options.get("hallucination_penalty", 0.5))
        score = round(recall * (1.0 - penalty) if hallucinated else recall, 4)
        metrics = {
            "keyword_recall": recall,
            "hallucinated": bool(hallucinated),
            "wrong_keywords_hit": hallucinated,
            "text": response.text[:120],
        }
        return await judge_evaluation(
            ctx,
            case,
            response,
            rubric=(
                "The answer must be grounded solely in the source document: it "
                "must contain the correct facts and must not repeat facts that "
                "appear only in the distractor document."
            ),
            deterministic_score=score,
            passed=(recall >= 0.6 and not hallucinated),
            pass_threshold=0.6,
            metrics=metrics,
        )