"""Selection — CLI flags and interactive picking over hosts and models."""

from __future__ import annotations

import fnmatch
from collections.abc import Callable

from local_ai_bench.domain.models import HostConfig, ModelInfo

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


def pick_hosts(hosts: list[HostConfig]) -> list[HostConfig]:
    """Ask the user to choose hosts by index/range from a numbered list."""
    return _pick_from(
        hosts,
        title="Available hosts",
        label=lambda h: f"{h.name} ({h.base_url})",
        prompt="Select hosts (indices or ranges, e.g. 0 2-4)",
    )


def pick_interactive(models: list[ModelInfo]) -> list[ModelInfo]:
    """Ask the user to choose models by index/range from a numbered list."""
    return _pick_from(
        models,
        title="Available models",
        label=lambda m: m.model_name,
        prompt="Select models (indices or ranges, e.g. 0 2-4)",
    )


def _pick_from[T](
    items: list[T],
    *,
    title: str,
    label: Callable[[T], str],
    prompt: str,
) -> list[T]:
    """Shared numbered picker: print ``items``, then resolve an index/range spec.

    ``(a)`` selects everything, ``(q)`` / empty input selects nothing. The label
    callable renders each item in the numbered listing.
    """
    if not items:
        return []
    print(title + ":")
    for i, item in enumerate(items):
        print(f"  [{i}] {label(item)}")
    print("  (a) all / (q) quit")
    raw = input(f"{prompt}: ").strip().lower()
    if raw in {"", "q"}:
        return []
    if raw == "a":
        return list(items)
    return _resolve_indices(items, raw)


def _resolve_indices[T](items: list[T], raw: str) -> list[T]:
    """Return the items selected by a spec like '0 2-4' (indices/ranges)."""
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
    return [item for i, item in enumerate(items) if i in selected]