"""Structured-output (JSON) benchmark — schema/schema compliance."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

from local_ai_bench.domain.models import (
    BenchmarkCase,
    BenchmarkCategory,
    Evaluation,
    Modality,
)
from local_ai_bench.plugins.builtin._base import BaseTextPlugin
from local_ai_bench.plugins.score import valid_json_with_fields

_CASES = [
    {
        "id": "json_product_0001",
        "prompt": (
            'Output valid JSON for a product with fields: '
            'name, price, in_stock (no other text). Product: "Widget Pro", $19.99, in stock.'
        ),
        "required_fields": ["name", "price", "in_stock"],
    },
    {
        "id": "json_recipe_0002",
        "prompt": (
            'Output valid JSON for a recipe with fields: title, servings, ingredients '
            '(array of strings), steps (array of strings). Make a simple pancake recipe.'
        ),
        "required_fields": ["title", "servings", "ingredients", "steps"],
    },
]


class StructuredOutputPlugin(BaseTextPlugin):
    id: ClassVar[str] = "structured_output"
    name: ClassVar[str] = "Structured Output"
    description: ClassVar[str] = "Validity and accuracy of JSON/structured responses."
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.STRUCTURED_OUTPUT
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
                expected={"required_fields": spec["required_fields"]},
            )

    def build_request(self, case, model, ctx) -> dict[str, Any]:  # noqa: ANN001
        return {
            "messages": [{"role": "user", "content": case.input["prompt"]}],
            "options": {"temperature": 0.0, "num_predict": 512},
        }

    async def evaluate(self, case, response, ctx) -> Evaluation:  # noqa: ANN001
        required = case.expected["required_fields"]
        score, metrics = valid_json_with_fields(response.text, required)
        return Evaluation(score=score, passed=score >= 1.0, metrics=metrics)
