"""Built-in plugins package."""

import contextlib

from local_ai_bench.plugins.registry import PluginRegistry


def load_builtin_plugins(registry: PluginRegistry) -> None:
    """Register all built-in plugins that are currently implemented."""
    from local_ai_bench.plugins.builtin.agent_tool_use import AgentToolUsePlugin
    from local_ai_bench.plugins.builtin.classification import ClassificationPlugin
    from local_ai_bench.plugins.builtin.coding import CodingPlugin
    from local_ai_bench.plugins.builtin.function_calling import FunctionCallingPlugin
    from local_ai_bench.plugins.builtin.long_context import LongContextPlugin
    from local_ai_bench.plugins.builtin.multi_context import MultiContextPlugin
    from local_ai_bench.plugins.builtin.multi_turn import MultiTurnPlugin
    from local_ai_bench.plugins.builtin.multilingual import MultilingualPlugin
    from local_ai_bench.plugins.builtin.rag import RagPlugin
    from local_ai_bench.plugins.builtin.reasoning import ReasoningPlugin
    from local_ai_bench.plugins.builtin.safety import SafetyRefusalPlugin
    from local_ai_bench.plugins.builtin.smoke import SmokePlugin
    from local_ai_bench.plugins.builtin.sql import SqlPlugin
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
        RagPlugin,
        FunctionCallingPlugin,
        AgentToolUsePlugin,
        MultiTurnPlugin,
        SafetyRefusalPlugin,
        SqlPlugin,
        MultilingualPlugin,
        ClassificationPlugin,
    ):
        with contextlib.suppress(ValueError):
            registry.register(cls)
