"""Vision benchmark — image-description / caption-style understanding.

Tests whether the model can describe or reason about a small synthetic PNG image.
The image is a 32x32 checkerboard generated deterministically so no external
asset is required. Correctness is judged by keyword recall against expected
color/pattern terms.
"""

from __future__ import annotations

import base64
import struct
import zlib
from collections.abc import Iterable
from typing import Any, ClassVar

from ollama_bench.domain.models import (
    BenchmarkCase,
    BenchmarkCategory,
    Evaluation,
    Modality,
)
from ollama_bench.plugins.builtin._base import BaseTextPlugin
from ollama_bench.plugins.score import keyword_recall


def _generate_checkerboard_png(size: int = 32) -> str:
    """Minimal 8-bit grayscale PNG of a size x size checkerboard, returned as base64."""

    def chunk(typ: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + typ
            + data
            + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
        )

    raw = bytearray()
    for y in range(size):
        raw.append(0)  # filter type byte per scanline
        for x in range(size):
            raw.append(255 if (x + y) % 2 == 0 else 64)
    compressed = zlib.compress(bytes(raw))
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0)  # 8-bit grayscale
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", ihdr)
    png += chunk(b"IDAT", compressed)
    png += chunk(b"IEND", b"")
    return base64.b64encode(png).decode("ascii")


_CASES = [
    {
        "id": "vision_checkerboard_0001",
        "prompt": "What colors do you see in this image? Describe the pattern.",
        "keywords": ["black", "white", "checkerboard", "square"],
    },
]


class VisionPlugin(BaseTextPlugin):
    id: ClassVar[str] = "vision"
    name: ClassVar[str] = "Vision"
    description: ClassVar[str] = "Image understanding of a synthetic checkerboard pattern."
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.VISION
    version: ClassVar[str] = "0.1.0"
    dataset_version: ClassVar[str] = "v1"
    modalities: ClassVar[set[Modality]] = {Modality.IMAGE}

    def supports_model(self, model) -> bool:  # noqa: ANN001
        return getattr(model, "supports_vision", False)

    def cases(self, ctx) -> Iterable[BenchmarkCase]:
        max_dim = int(ctx.options.get("max_image_dimension", 768))
        size = max(2, min(max_dim, 32))
        image_b64 = _generate_checkerboard_png(size)
        for spec in _CASES:
            yield BenchmarkCase(
                id=spec["id"],
                plugin_id=self.id,
                dataset_version=self.dataset_version,
                input={"prompt": spec["prompt"], "image_b64": image_b64},
                expected={"keywords": spec["keywords"]},
            )

    def build_request(self, case, model, ctx) -> dict[str, Any]:  # noqa: ANN001
        return {
            "messages": [
                {
                    "role": "user",
                    "content": case.input["prompt"],
                    "images": [case.input["image_b64"]],
                }
            ],
            "options": {"temperature": 0.0, "num_predict": 128},
        }

    async def evaluate(self, case, response, ctx) -> Evaluation:  # noqa: ANN001
        expected_kw = case.expected["keywords"]
        recall = keyword_recall(response.text, expected_kw)
        return Evaluation(
            score=recall,
            passed=recall >= 0.25,
            metrics={"keyword_recall": recall},
        )