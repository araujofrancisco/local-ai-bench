"""Model selection — CLI flags and interactive picking over autodetected models."""

from __future__ import annotations

import fnmatch

from local_ai_bench.domain.models import ModelInfo

DEFAULT_HOST_NAME = "local"
DEFAULT_HOST_URL = "http://127.0.0.1:11434"


def split_patterns(raw: str | None) -> list[str]:
    """Split a CLI value like 'llama*,qwen*' into patterns (comma or space separated)."""
    if not raw:
        return []
    return [p.strip() for p in raw.replace(",", " ").split() if p.strip()]


def matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def filter_models(
    models: list[ModelInfo],
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[ModelInfo]:
    """Keep autodetected models matching at least one include pattern and no exclude pattern.

    With no include patterns, all models are kept. Patterns use shell-style globs.
    """
    include = include or []
    exclude = exclude or []
    return [
        m
        for m in models
        if (not include or matches_any(m.model_name, include))
        and not matches_any(m.model_name, exclude)
    ]


def pick_interactive(models: list[ModelInfo]) -> list[ModelInfo]:
    """Ask the user to choose models by index/range from a numbered list."""
    if not models:
        return []
    print("Available models:")
    for i, m in enumerate(models):
        print(f"  [{i}] {m.model_name}")
    print("  (a) all / (q) quit")
    raw = input("Select models (indices or ranges, e.g. 0 2-4): ").strip().lower()
    if raw in {"", "q"}:
        return []
    if raw == "a":
        return list(models)
    return _resolve_indices(models, raw)


def _resolve_indices(models: list[ModelInfo], raw: str) -> list[ModelInfo]:
    """Return the models selected by a spec like '0 2-4' (indices/ranges)."""
    selected: set[int] = set()
    for part in raw.replace(",", " ").split():
        if "-" in part:
            try:
                start, end = (int(x) for x in part.split("-", 1))
            except ValueError:
                continue
            selected.update(range(start, end + 1))
        else:
            try:
                selected.add(int(part))
            except ValueError:
                continue
    return [m for i, m in enumerate(models) if i in selected]