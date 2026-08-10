"""Plugin registry — loads built-in and local plugins."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from ollama_bench.plugins.base import BenchmarkPlugin


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, type[BenchmarkPlugin]] = {}

    def register(self, plugin_cls: type[BenchmarkPlugin], allow_overwrite: bool = False) -> None:
        pid = plugin_cls.id
        if pid in self._plugins and not allow_overwrite:
            raise ValueError(f"duplicate plugin id: {pid}")
        self._plugins[pid] = plugin_cls

    def get(self, plugin_id: str) -> type[BenchmarkPlugin] | None:
        return self._plugins.get(plugin_id)

    def ids(self) -> list[str]:
        return sorted(self._plugins)

    def load_module(self, module: str) -> None:
        """Register all BenchmarkPlugin subclasses exported from `module`."""
        import importlib
        import inspect

        mod = importlib.import_module(module)
        prefix = f"{module}."
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if (
                inspect.isclass(obj)
                and obj is not BenchmarkPlugin
                and issubclass(obj, BenchmarkPlugin)
                and obj.__module__.startswith(prefix)
            ):
                self.register(obj)

    def load_file(self, path: str | Path) -> None:
        """Register all BenchmarkPlugin subclasses defined in a Python file."""
        import inspect

        p = Path(path)
        spec = importlib.util.spec_from_file_location(f"ollama_bench_plugin_{p.stem}", p)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load plugin file: {p}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for _name, obj in inspect.getmembers(mod, inspect.isclass):
            if obj is not BenchmarkPlugin and issubclass(obj, BenchmarkPlugin):
                self.register(obj)

    def load_dir(self, directory: str | Path) -> list[str]:
        """Load every local plugin ``.py`` file in a directory.

        Returns a list of error messages for files that failed to import.
        A missing or empty directory yields no errors.
        """
        d = Path(directory)
        errors: list[str] = []
        if not d.is_dir():
            return errors
        for path in sorted(d.glob("*.py")):
            try:
                self.load_file(path)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path.name}: {exc}")
        return errors