"""Core domain models for LocalAIBench."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Modality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    CODE = "code"
    JSON = "json"


class BenchmarkCategory(StrEnum):
    TRANSLATION = "translation"
    CODING = "coding"
    VISION = "vision"
    SUMMARIZATION = "summarization"
    REASONING = "reasoning"
    STRUCTURED_OUTPUT = "structured_output"
    LONG_CONTEXT = "long_context"
    RETRIEVAL = "retrieval"
    FUNCTION_CALLING = "function_calling"
    MULTI_TURN = "multi_turn"
    SAFETY = "safety"
    SQL = "sql"
    MULTILINGUAL = "multilingual"
    CLASSIFICATION = "classification"


class HostConfig(BaseModel):
    name: str
    base_url: str
    timeout_seconds: int = 300


class ModelInfo(BaseModel):
    host_name: str
    model_name: str
    digest: str | None = None
    max_context_tokens: int | None = None
    supports_vision: bool = False
    supports_tools: bool = False
    supports_json_mode: bool = False
    quantized_level: str | None = None
    parameter_size: str | None = None
    raw_metadata: dict[str, Any] | None = None


class BenchmarkCase(BaseModel):
    id: str
    plugin_id: str
    dataset_version: str
    input: dict[str, Any]
    expected: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TimingMetrics(BaseModel):
    started_at: float = 0.0
    first_token_at: float | None = None
    finished_at: float = 0.0
    total_ms: float = 0.0
    time_to_first_token_ms: float | None = None
    generation_ms: float | None = None
    load_ms: float | None = None


class TokenMetrics(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_per_second: float | None = None
    prompt_eval_tokens_per_second: float | None = None


class ModelResponse(BaseModel):
    raw: dict[str, Any]
    text: str
    timing: TimingMetrics
    tokens: TokenMetrics
    error: str | None = None
    done_reason: str | None = None
    truncated: bool = False
    tool_calls: list[dict[str, Any]] | None = None


class Evaluation(BaseModel):
    score: float | None = None
    passed: bool | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None
    judge_model: str | None = None


class CaseResult(BaseModel):
    case: BenchmarkCase
    model: ModelInfo
    response: ModelResponse
    evaluation: Evaluation
    attempt: int = 1


class PluginAggregate(BaseModel):
    plugin_id: str
    model_name: str
    host_name: str
    total_cases: int = 0
    successful_cases: int = 0
    failed_cases: int = 0
    skipped_cases: int = 0
    score: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    time_to_first_token_p50_ms: float | None = None
    tokens_per_second: float | None = None
    cases_run: int = 0


class ModelBenchmarkResult(BaseModel):
    """Aggregated benchmark metrics for a single model across all plugins."""

    host_name: str
    model_name: str
    model_digest: str | None = None
    plugins: list[PluginAggregate] = Field(default_factory=list)
    cases: list[CaseResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    context_recommendation: dict[str, Any] | None = None
    completion_tokens_total: int = 0
    cases_run: int = 0
    errors: int = 0
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    time_to_first_token_p50_ms: float | None = None
    tokens_per_second: float | None = None
    overall_score: float | None = None


class RunResult(BaseModel):
    """The full outcome of one benchmark run."""

    run_id: str
    timestamp: str
    app_version: str
    config_hash: str
    hosts: list[HostConfig] = Field(default_factory=list)
    models: list[ModelBenchmarkResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    report_dir: str | None = None