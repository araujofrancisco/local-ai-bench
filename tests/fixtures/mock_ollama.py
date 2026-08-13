"""Fake Ollama transport for integration tests.

Implements just enough of the Ollama API (version, tags, chat) to drive a full
benchmark run through ``httpx.MockTransport`` with no real network I/O.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

MODELS = [
    {
        "name": "e2e-model:latest",
        "digest": "deadbeef",
        "details": {"context_length": 8192},
        "capabilities": ["tools"],
    }
]


def _chat_body(payload: dict[str, Any]) -> str:
    """Streaming SSE body (Ollama /api/chat). Canned answer scores 1.0.

    When the payload declares ``tools`` the model "calls" get_weather(city=Paris)
    to exercise the tool-call capture path in the client.
    """
    if payload.get("tools"):
        lines = [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
                    ],
                },
                "done": False,
            },
            {
                "done": True,
                "done_reason": "tool_calls",
                "total_duration": 500_000_000,
                "load_duration": 100_000_000,
                "prompt_eval_count": 20,
                "prompt_eval_duration": 40_000_000,
                "eval_count": 5,
                "eval_duration": 100_000_000,
            },
        ]
    else:
        lines = [
            {"message": {"role": "assistant", "content": "36"}, "done": False},
            {
                "done": True,
                "done_reason": "stop",
                "total_duration": 500_000_000,
                "load_duration": 100_000_000,
                "prompt_eval_count": 20,
                "prompt_eval_duration": 40_000_000,
                "eval_count": 5,
                "eval_duration": 100_000_000,
            },
        ]
    return "\n".join(json.dumps(line) for line in lines) + "\n"


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/version":
        return httpx.Response(200, json={"version": "0.3.2"})
    if path == "/api/tags":
        return httpx.Response(200, json={"models": MODELS})
    if path == "/api/show":
        return httpx.Response(200, json={"model_info": {"context_length": 8192}})
    if path == "/api/chat":
        payload = json.loads(request.content or b"{}")
        return httpx.Response(
            200,
            content=_chat_body(payload),
            headers={"content-type": "application/x-ndjson"},
        )
    return httpx.Response(404, json={"error": f"unexpected path {path}"})


def mock_transport() -> httpx.MockTransport:
    return httpx.MockTransport(_handler)
