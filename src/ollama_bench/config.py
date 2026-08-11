"""Configuration loading and validation (Milestone 1).

Follows the schema described in PLAN.md §11.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ollama_bench.domain.models import HostConfig
from ollama_bench.selection import DEFAULT_HOST_NAME, DEFAULT_HOST_URL


class PluginConfig(BaseModel):
    """Plugin selection and per-plugin option defaults.

    ``options`` is a free-form ``{plugin_id: {key: value}}`` map so any plugin
    (built-in or local) can declare YAML option defaults without touching the
    schema — the runner reads them from ``ctx.options`` at run time.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: list[str] = Field(default_factory=list)
    local_dir: str = "./plugins"
    options: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Plugin ids whose per-plugin score column should appear by default on the
    # Compare page. Empty => all run plugins' score columns are shown by default.
    compare_default: list[str] = Field(default_factory=list)


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
    multi_context: float = 1.0


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
        """If no host is configured, fall back to the local Ollama.

        The only truly required configuration is how to connect to Ollama; an
        omitted host simply means "the default local Ollama" — or ``$OLLAMA_HOST``
        when set (Docker Compose sets it to the host machine automatically).
        """
        if not self.hosts:
            base_url = os.getenv("OLLAMA_HOST", DEFAULT_HOST_URL)
            self.hosts = [HostConfig(name=DEFAULT_HOST_NAME, base_url=base_url)]
        return self


# The starter config shipped with the package. Kept in sync with
# `config/default.yaml`; when that file is present in the checkout it is copied
# verbatim (single source of truth), otherwise this text is the fallback.
DEFAULT_CONFIG_TEXT = """\
# OllamaBench configuration.
# The only required setting is how to connect to Ollama. If you omit `hosts`
# entirely (the recommended starting point), the default local Ollama
# (http://127.0.0.1:11434) is used — or $OLLAMA_HOST if that environment
# variable is set. Docker Compose sets OLLAMA_HOST to the host machine's
# Ollama automatically.
#
# Benchmark other hosts too by listing them here, e.g.:
#   hosts:
#     - name: lab-server
#       base_url: http://192.168.10.108:11434
#       timeout_seconds: 300
#
# Models are NOT listed here — they are auto-detected from each host by
# `GET /api/tags`. Choose which ones to benchmark at run time, e.g.:
#   ollama-bench run --models 'qwen*'
#   ollama-bench run --models llama3.2:latest,qwen2.5-coder:14b
#   ollama-bench run --exclude '*:0.8b'
#   ollama-bench run --interactive
#
# With no --models/--exclude/--interactive flags, every autodetected model is
# benchmarked. There is no "models" section in this file.

app:
  name: OllamaBench
  output_dir: ./reports
  log_level: info

plugins:
  enabled:
    - smoke
    - reasoning
    - translation
    - summarization
    - structured_output
    - coding
    - vision
    - keyword
    - long_context
    - multi_context
  # Directory scanned for extra local plugins (.py files, e.g. ./plugins/keyword.py).
  # Inside the Docker container this is the mounted host directory ./plugins -> /app/plugins.
  local_dir: ./plugins
  # Plugin ids whose per-plugin score column appears by default on the Compare
  # page. Leave empty ([]) to show a score column for every enabled plugin that
  # ran; list specific ids to limit which columns appear by default.
  compare_default:
    - translation
    - coding
    - vision
    - summarization
    - reasoning
    - structured_output
    - long_context
  options:
    coding:
      execute_code: false
      timeout_seconds: 30
    vision:
      max_image_dimension: 768
    multi_context:
      prompt: "What is the capital of France?"
      expected: "paris"
      context_sizes: [512, 1024, 4096, 8192, 16384]
      contains: true

runner:
  repetitions: 3
  warmup_runs: 1
  concurrency: 1
  temperature: 0.0
  seed: 42
  max_retries: 2
  retry_backoff_seconds: 2

# Judge used for subjective scoring by plugins that support it.
judge:
  enabled: false
  model: llama3.2:latest
  temperature: 0.0

context_optimization:
  enabled: true
  candidate_sizes:
    - 512
    - 1024
    - 2048
    - 4096
    - 8192
    - 16384
  max_candidate_size: 32768
  quality_threshold: 0.90
  latency_budget_ms: null
  needle_positions:
    - 0.1
    - 0.3
    - 0.5
    - 0.7
    - 0.9

reporting:
  formats:
    - json
    - markdown
    - html
  include_raw_cases: true

weights:
  translation: 1.0
  coding: 1.0
  vision: 1.0
  summarization: 1.0
  reasoning: 1.0
  structured_output: 1.0
  long_context: 1.0
"""


def _packaged_default_config() -> str | None:
    """Return the shipped ``config/default.yaml`` if present in the checkout.

    Keeps `ollama-bench init` output identical to the versioned default so the
    two can never drift. Falls back to :data:`DEFAULT_CONFIG_TEXT` when the
    package is installed without the repo's config directory.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / "config" / "default.yaml",
        Path.cwd() / "config" / "default.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return None


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
    content = _packaged_default_config() or DEFAULT_CONFIG_TEXT
    p.write_text(content, encoding="utf-8")
    return p