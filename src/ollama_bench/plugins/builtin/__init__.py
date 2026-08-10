"""Built-in plugins package."""

import contextlib

from ollama_bench.plugins.registry import PluginRegistry


def load_builtin_plugins(registry: PluginRegistry) -> None:
    """Register all built-in plugins that are currently implemented."""
    from ollama_bench.plugins.builtin.coding import CodingPlugin
    from ollama_bench.plugins.builtin.long_context import LongContextPlugin
    from ollama_bench.plugins.builtin.reasoning import ReasoningPlugin
    from ollama_bench.plugins.builtin.smoke import SmokePlugin
    from ollama_bench.plugins.builtin.structured_output import StructuredOutputPlugin
    from ollama_bench.plugins.builtin.summarization import SummarizationPlugin
    from ollama_bench.plugins.builtin.translation import TranslationPlugin
    from ollama_bench.plugins.builtin.vision import VisionPlugin

    for cls in (
        SmokePlugin,
        ReasoningPlugin,
        TranslationPlugin,
        SummarizationPlugin,
        StructuredOutputPlugin,
        CodingPlugin,
        VisionPlugin,
        LongContextPlugin,
    ):
        with contextlib.suppress(ValueError):
            registry.register(cls)
