"""Model discovery for an Ollama host."""

from __future__ import annotations

from typing import Any

from local_ai_bench.domain.models import ModelInfo
from local_ai_bench.ollama.client import OllamaClient


def _capabilities(entry: dict[str, Any]) -> list[str]:
    caps = entry.get("capabilities") or []
    if not caps:
        details = entry.get("details") or {}
        caps = details.get("capabilities")
    return list(caps or [])


def model_info_from_entry(host_name: str, entry: dict[str, Any]) -> ModelInfo:
    details = entry.get("details") or {}
    caps = _capabilities(entry)
    context_length = details.get("context_length")
    return ModelInfo(
        host_name=host_name,
        model_name=entry.get("name") or entry.get("model") or "",
        digest=entry.get("digest"),
        max_context_tokens=context_length if isinstance(context_length, int) else None,
        supports_vision="vision" in caps,
        supports_tools="tools" in caps,
        supports_json_mode="json" in caps or "tools" in caps,
        quantized_level=details.get("quantization_level"),
        parameter_size=details.get("parameter_size"),
        raw_metadata=entry,
    )


async def discover_models(client: OllamaClient, host_name: str) -> list[ModelInfo]:
    """List all models available on the host as ModelInfo objects."""
    entries = await client.list_models()
    return [model_info_from_entry(host_name, e) for e in entries]