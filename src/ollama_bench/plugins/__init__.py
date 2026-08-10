"""Plugin package (Milestone 3+)."""

from pathlib import Path

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
    # Host-runnable fallback: when the configured absolute (container) path like
    # /plugins does not exist on a plain host checkout, also scan ./plugins so
    # local plugins such as keyword remain discoverable outside Docker.
    local_path = Path(local_dir) if local_dir else None
    if local_path and not local_path.is_dir():
        here = Path.cwd() / "plugins"
        if here.is_dir():
            errors += reg.load_dir(str(here))
    return reg, errors