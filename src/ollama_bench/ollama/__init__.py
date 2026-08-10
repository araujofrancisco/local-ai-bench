"""Ollama package."""

from ollama_bench.ollama.client import OllamaClient
from ollama_bench.ollama.discovery import discover_models

__all__ = ["OllamaClient", "discover_models"]