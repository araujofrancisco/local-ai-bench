"""Built-in plugins package."""

import contextlib

from local_ai_bench.plugins.registry import PluginRegistry


def load_builtin_plugins(registry: PluginRegistry) -> None:
    """Register all built-in plugins that are currently implemented."""
    from local_ai_bench.plugins.builtin.coding import CodingPlugin
    from local_ai_bench.plugins.builtin.long_context import LongContextPlugin
    from local_ai_bench.plugins.builtin.multi_context import MultiContextPlugin
    from local_ai_bench.plugins.builtin.reasoning import ReasoningPlugin
    from local_ai_bench.plugins.builtin.smoke import SmokePlugin
    from local_ai_bench.plugins.builtin.structured_output import StructuredOutputPlugin
    from local_ai_bench.plugins.builtin.summarization import SummarizationPlugin
    from local_ai_bench.plugins.builtin.translation import TranslationPlugin
    from local_ai_bench.plugins.builtin.vision import VisionPlugin

    for cls in (
        SmokePlugin,
        ReasoningPlugin,
        TranslationPlugin,
        SummarizationPlugin,
        StructuredOutputPlugin,
        CodingPlugin,
        VisionPlugin,
        LongContextPlugin,
        MultiContextPlugin,
    ):
        with contextlib.suppress(ValueError):
            registry.register(cls)
