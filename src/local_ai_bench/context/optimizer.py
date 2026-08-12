"""Context-window recommendation (PLAN §16).

Given per-context-size quality scores from the long-context plugin, pick a
practical recommendation that balances quality, stability, and latency.

Heuristic (PLAN §16.6):
1. Take the maximum quality across stable sizes.
2. Keep sizes whose quality is within ``quality_threshold`` of that maximum.
3. Apply the latency budget if configured.
4. Recommend the smallest size among the survivors.
5. If nothing survives, recommend the largest stable size with a warning.
"""

from __future__ import annotations

from typing import Any


def recommend(
    per_context_score: dict[int, float | None],
    candidate_sizes: list[int],
    *,
    quality_threshold: float = 0.90,
    latency_budget_ms: float | None = None,
    per_context_latency: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Compute a context-window recommendation dict."""
    per_context_latency = per_context_latency or {}
    valid = {
        size: q
        for size, q in per_context_score.items()
        if q is not None and size in candidate_sizes
    }
    if not valid:
        return {
            "recommended_context": None,
            "reason": "no stable context results",
            "curve": [],
        }

    max_quality = max(valid.values())
    minimum_quality = quality_threshold * max_quality
    acceptable = {size: q for size, q in valid.items() if q >= minimum_quality}

    if latency_budget_ms is not None:
        acceptable = {
            size: q
            for size, q in acceptable.items()
            if per_context_latency.get(size, 0.0) <= latency_budget_ms
        }

    curve = [
        {
            "context_size": size,
            "quality": q,
            "latency_ms": per_context_latency.get(size),
            "stable": True,
        }
        for size, q in sorted(valid.items())
    ]

    if not acceptable:
        largest = max(valid)
        return {
            "recommended_context": largest,
            "reason": "no size met the quality threshold; recommending the largest stable size",
            "curve": curve,
        }

    recommended = min(acceptable)
    return {
        "recommended_context": recommended,
        "reason": "smallest stable size meeting the quality threshold",
        "curve": curve,
    }
