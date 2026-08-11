"""Async client for the Ollama HTTP API.

Supports health checks, model listing/metadata, and streaming chat requests
with timing/token metrics, as described in PLAN.md §13.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from ollama_bench.domain.models import ModelResponse
from ollama_bench.ollama.metrics import timing_from_done, tokens_from_done

_DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: int = 300,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        timeout = (
            httpx.Timeout(timeout_seconds, connect=10.0)
            if timeout_seconds
            else _DEFAULT_TIMEOUT
        )
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, transport=transport
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict[str, Any]:
        """GET /api/version — raises on unreachable host."""
        resp = await self._client.get("/api/version")
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data

    async def list_models(self) -> list[dict[str, Any]]:
        """GET /api/tags — raw model list."""
        resp = await self._client.get("/api/tags")
        resp.raise_for_status()
        models: list[dict[str, Any]] = resp.json().get("models") or []
        return models

    async def show_model(self, model: str) -> dict[str, Any]:
        """GET /api/show — model metadata."""
        resp = await self._client.post("/api/show", json={"model": model})
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
        stream: bool = True,
    ) -> ModelResponse:
        """POST /api/chat. Returns a ModelResponse with timing/token metrics."""
        payload = _build_payload(model, messages, options, stream)
        started_at = time.monotonic()
        raw: dict[str, Any] = {}

        if not stream:
            resp = await self._client.post("/api/chat", json={**payload, "stream": False})
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            wall_ms = (time.monotonic() - started_at) * 1000.0
            raw = data
            text = (data.get("message") or {}).get("content") or ""
            done = data
        else:
            text, done, wall_ms = await self._stream_chat(payload, started_at)

        timing = timing_from_done(done, started_at, wall_ms)
        tokens = tokens_from_done(done)
        done_reason = done.get("done_reason")
        return ModelResponse(
            raw=raw,
            text=text,
            timing=timing,
            tokens=tokens,
            done_reason=done_reason,
            truncated=done_reason == "length",
        )

    async def _stream_chat(
        self, payload: dict[str, Any], started_at: float
    ) -> tuple[str, dict[str, Any], float]:
        parts: list[str] = []
        done_chunk: dict[str, Any] = {}
        first_token_at: float | None = None
        wall_ms = 0.0
        async with self._client.stream(
            "POST", "/api/chat", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = _parse_json(line)
                if not chunk:
                    continue
                delta = chunk.get("message", {}).get("content")
                if delta:
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    parts.append(delta)
                if chunk.get("done"):
                    done_chunk = chunk
                    wall_ms = (time.monotonic() - started_at) * 1000.0
        return "".join(parts), done_chunk, wall_ms


def _build_payload(
    model: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any] | None,
    stream: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "options": options or {},
        "stream": stream,
    }
    return payload


def _parse_json(line: str) -> dict[str, Any] | None:
    import json

    try:
        parsed: dict[str, Any] = json.loads(line)
        return parsed
    except json.JSONDecodeError:
        return None