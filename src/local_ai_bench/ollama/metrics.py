"""Compute timing/token metrics from Ollama response fields.

Ollama's streaming NDJSON ends with a `done` chunk carrying the key
duration/count fields. Durations are in nanoseconds and named:
  total_duration, load_duration, prompt_eval_duration, eval_duration.
"""

from __future__ import annotations

from typing import Any

from local_ai_bench.domain.models import TimingMetrics, TokenMetrics

_NS_TO_MS = 1e-6


def timing_from_done(done: dict[str, Any], started_at: float, wall_ms: float) -> TimingMetrics:
    total_ns = done.get("total_duration")
    load_ns = done.get("load_duration")
    ttft_ms: float | None = None
    ttft_ns = done.get("prompt_eval_duration")
    if ttft_ns is not None:
        ttft_ms = ttft_ns * _NS_TO_MS

    generation_ms: float | None = None
    eval_ns = done.get("eval_duration")
    if eval_ns is not None:
        generation_ms = eval_ns * _NS_TO_MS

    return TimingMetrics(
        started_at=started_at,
        first_token_at=started_at + (ttft_ms / 1000.0) if ttft_ms is not None else None,
        finished_at=started_at + (wall_ms / 1000.0),
        total_ms=float(total_ns * _NS_TO_MS) if total_ns is not None else wall_ms,
        time_to_first_token_ms=ttft_ms,
        generation_ms=generation_ms,
        load_ms=float(load_ns * _NS_TO_MS) if load_ns is not None else None,
    )


def tokens_from_done(done: dict[str, Any]) -> TokenMetrics:
    prompt_count = done.get("prompt_eval_count")
    eval_count = done.get("eval_count")
    eval_ns = done.get("eval_duration")

    tps: float | None = None
    if eval_count is not None and eval_ns:
        tps = float(eval_count) / (float(eval_ns) / 1e9)

    prompt_tps: float | None = None
    prompt_ns = done.get("prompt_eval_duration")
    if prompt_count is not None and prompt_ns:
        prompt_tps = float(prompt_count) / (float(prompt_ns) / 1e9)

    return TokenMetrics(
        prompt_tokens=prompt_count,
        completion_tokens=eval_count,
        tokens_per_second=tps,
        prompt_eval_tokens_per_second=prompt_tps,
    )