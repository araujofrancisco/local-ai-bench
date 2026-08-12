"""Ollama package."""

from local_ai_bench.ollama.client import OllamaClient
from local_ai_bench.ollama.discovery import discover_models

__all__ = ["OllamaClient", "discover_models"]