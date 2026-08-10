"""Configuration loading and validation (Milestone 1).

Follows the schema described in PLAN.md §11.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, field_validator, model_validator

from ollama_bench.domain.models import HostConfig
from ollama_bench.selection import DEFAULT_HOST_NAME, DEFAULT_HOST_URL


class PluginOptions(BaseModel):
    translation: dict[str, Any] = Field(default_factory=dict)
    coding: dict[str, Any] = Field(default_factory=dict)
    vision: dict[str, Any] = Field(default_factory=dict)


class PluginConfig(BaseModel):
    enabled: list[str] = Field(default_factory=list)
    local_dir: str = "./plugins"
    options: PluginOptions = Field(default_factory=PluginOptions)


class RunnerConfig(BaseModel):
    repetitions: int = 3
    warmup_runs: int = 1
    concurrency: int = 1
    temperature: float = 0.0
    seed: int = 42
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0


class JudgeConfig(BaseModel):
    enabled: bool = True
    model: str | None = None
    temperature: float = 0.0


class ContextOptimizationConfig(BaseModel):
    enabled: bool = True
    candidate_sizes: list[int] = Field(default_factory=lambda: [512, 1024, 2048, 4096, 8192, 16384])
    max_candidate_size: int = 32768
    quality_threshold: float = 0.90
    latency_budget_ms: float | None = None
    needle_positions: list[float] = Field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])


class ReportingConfig(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["json", "markdown", "html"])
    include_raw_cases: bool = True


class WeightsConfig(BaseModel):
    translation: float = 1.0
    coding: float = 1.0
    vision: float = 1.0
    summarization: float = 1.0
    reasoning: float = 1.0
    structured_output: float = 1.0
    long_context: float = 1.0


class AppConfig(BaseModel):
    name: str = "OllamaBench"
    output_dir: str = "./reports"
    log_level: str = "info"


class BenchmarkConfig(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    hosts: list[HostConfig] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    context_optimization: ContextOptimizationConfig = Field(default_factory=ContextOptimizationConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    weights: WeightsConfig = Field(default_factory=WeightsConfig)

    @field_validator("hosts")
    @classmethod
    def _unique_host_names(cls, hosts: list[HostConfig]) -> list[HostConfig]:
        names = [h.name for h in hosts]
        if len(names) != len(set(names)):
            raise ValueError("host names must be unique")
        return hosts

    @field_validator("models")
    @classmethod
    def _nonempty_strings(cls, models: list[str]) -> list[str]:
        if any(not m.strip() for m in models):
            raise ValueError("model names must be non-empty strings")
        return models

    @model_validator(mode="after")
    def _default_host(self) -> BenchmarkConfig:
        """If no host is configured, fall back to the localhost default.

        The only truly required configuration is how to connect to Ollama; an
        omitted host simply means "the default local Ollama".
        """
        if not self.hosts:
            self.hosts = [HostConfig(name=DEFAULT_HOST_NAME, base_url=DEFAULT_HOST_URL)]
        return self


DEFAULT_CONFIG_TEXT = """\
# OllamaBench configuration.
# The only required setting is how to connect to Ollama. If you omit `hosts`
# entirely, the default local Ollama (http://127.0.0.1:11434) is used.
#
# Models are NOT configured here: they are auto-detected from each host.
# Choose which ones to benchmark at run time, e.g.:
#   ollama-bench run --models 'qwen*'
#   ollama-bench run --models llama3.2:latest,qwen2.5-coder:14b
#   ollama-bench run --exclude '*:0.8b'
#   ollama-bench run --interactive

app:
  name: OllamaBench
  output_dir: ./reports
  log_level: info

hosts:
  - name: local
    base_url: http://127.0.0.1:11434
    timeout_seconds: 300

plugins:
  enabled:
    - smoke
  # Directory scanned for extra local plugins (.py files, one plugin class each).
  local_dir: ./plugins

runner:
  repetitions: 3
  warmup_runs: 1
  concurrency: 1
  temperature: 0.0
  seed: 42

judge:
  enabled: false
  model: null

reporting:
  formats:
    - json
    - markdown
    - html
  include_raw_cases: true
"""


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate a YAML configuration file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    try:
        return BenchmarkConfig.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError
        raise ValueError(f"invalid config in {p}: {exc}") from exc


def config_hash(config: BenchmarkConfig) -> str:
    """Stable hash of the meaningful config, for reproducibility reporting."""
    payload = config.model_dump(mode="json", exclude_none=True)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_default_config(path: str | Path) -> Path:
    """Create a starter config file for `ollama-bench init`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(f"config already exists: {p}")
    p.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")
    return p