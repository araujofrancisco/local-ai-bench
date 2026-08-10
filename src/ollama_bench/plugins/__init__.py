"""Plugin package (Milestone 3+)."""

from ollama_bench.plugins.base import BenchmarkPlugin, RunContext
from ollama_bench.plugins.builtin import load_builtin_plugins
from ollama_bench.plugins.registry import PluginRegistry

__all__ = ["BenchmarkPlugin", "PluginRegistry", "RunContext", "load_builtin_plugins", "load_plugins"]


def load_plugins(local_dir: str | None = None) -> tuple[PluginRegistry, list[str]]:
    """Build a registry with built-ins plus any local ``.py`` plugins.

    Returns ``(registry, errors)`` where ``errors`` lists any plugin files that
    failed to import. Missing local directories are not errors.
    """
    reg = PluginRegistry()
    load_builtin_plugins(reg)
    errors: list[str] = []
    if local_dir:
        errors = reg.load_dir(local_dir)
    return reg, errors