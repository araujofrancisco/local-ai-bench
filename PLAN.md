# PLAN.md — Ollama Local Network LLM Benchmark Suite

## 1. Project Name

Working name: **LocalAIBench**

Alternative names if desired:

- `local-ai-bench`
- `local-llm-bench`
- `lan-llm-eval`

The final CLI command should be simple:

```bash
local-ai-bench
```

---

## 2. Mission

Build a local-first, plugin-based benchmarking application for evaluating LLMs served by Ollama instances on a local network.

The system must:

1. Discover or accept configured Ollama hosts.
2. List available models.
3. Benchmark models across multiple practical categories:
   - Translation
   - Coding
   - Vision
   - Summarization
   - Reasoning
   - Structured output / JSON compliance
   - Retrieval-grounded answering / RAG-style tasks
   - Long-context retention
4. Support future benchmark categories through a plugin system.
5. Find an optimal or recommended context-window size per model and/or per workload.
6. Save reproducible reports in JSON, Markdown, and HTML.
7. Gracefully handle failures, unsupported models, and partial benchmark runs.

---

## 3. Guiding Principles

### 3.1 Architecture

- **Plugin-first**: new benchmark categories should be addable without changing the core.
- **SOLID**:
  - Single Responsibility: Ollama client, runner, plugins, reporting, and CLI are separate.
  - Open/Closed: core runner is closed for modification but extensible through plugins.
  - Liskov Substitution: any benchmark plugin should be usable through the same plugin interface.
  - Interface Segregation: plugins should not be forced to implement capabilities they do not support.
  - Dependency Inversion: runner depends on abstract interfaces, not concrete benchmark implementations.
- **KISS**: avoid databases, queues, or distributed workers until the benchmark engine is stable.
- **DRY**: shared prompt rendering, metric collection, scoring, and report generation should be centralized.
- **Fail safely**: one failing plugin, model, or host should not destroy the entire run.

### 3.2 Benchmarking

- Benchmarks must be reproducible.
- Datasets must be versioned.
- Prompts must be stable.
- Randomness should be controlled.
- Model settings should be explicit.
- Raw results should be saved even if scoring or reporting fails.

### 3.3 UX

- The user should be able to run a useful benchmark with one command.
- Progress must be visible.
- Errors must be actionable.
- Empty states must guide the user.
- Reports must be understandable without reading source code.

---

## 4. Scope

### 4.1 In scope for v1

- Local configuration file.
- Ollama host health checks.
- Ollama model discovery.
- Benchmark plugin framework.
- Benchmark runner with retries, timeouts, and progress events.
- Built-in benchmark plugins:
  - Translation
  - Coding
  - Vision
  - Summarization
  - Reasoning
  - Structured output
  - Long-context / context-window probe
- Context-window optimizer.
- JSON, Markdown, and HTML reports.
- CLI for running benchmarks and viewing reports.
- Local file-based artifact storage.
- Basic web report serving is optional.

### 4.2 Out of scope for v1

- Multi-user authentication.
- Cloud provider benchmarking.
- Fine-tuning.
- Model hosting or model download management.
- Distributed benchmark orchestration.
- Paid evaluation metrics.
- Heavy database-backed historical analytics.
- Automatic internet dataset downloads.

---

## 5. Target Users

### Primary user

A developer or ML engineer who wants to compare local Ollama models for practical workloads.

### Secondary user

A team lead or architect who wants reproducible benchmark reports to choose a model for an application.

### User needs

- Easy setup.
- Clear model comparison.
- Category-specific scores.
- Context-window recommendations.
- Exportable reports.
- Confidence that results are reproducible.

---

## 6. Product UX Requirements

### 6.1 Core CLI commands

The CLI should provide:

```bash
local-ai-bench init
local-ai-bench doctor
local-ai-bench models
local-ai-bench plugins list
local-ai-bench run
local-ai-bench run --config config.yaml
local-ai-bench report list
local-ai-bench report view latest
local-ai-bench report open latest
```

### 6.2 `init`

Creates:

- `config.yaml`
- `datasets/`
- `plugins/`
- `reports/`

Should show a friendly next-steps message.

### 6.3 `doctor`

Validates:

- Ollama hosts reachable.
- Models available.
- Required plugins load successfully.
- Datasets exist.
- Report directory writable.
- Optional judge model available.
- Optional code execution sandbox available.

Should output clear pass/fail items.

Example UX:

```text
✔ Host local: http://127.0.0.1:11434 reachable
✔ Found 4 models
✖ Judge model not found: llama3.2:3b
  Hint: set judge_model in config.yaml or disable judge-based scoring
```

### 6.4 `run`

Should show:

- Current host.
- Current model.
- Current plugin.
- Current case.
- Elapsed time.
- Number of completed cases.
- Number of failures.
- Estimated time remaining if possible.

Example:

```text
Running translation benchmark
Host: local
Model: qwen2.5:7b
Case: en_to_es_004
Progress: 12/40
Failures: 0
Elapsed: 00:01:23
```

### 6.5 Loading states

CLI:

- Spinners for host/model discovery.
- Progress bars for benchmark cases.
- Timers for long-running cases.

Optional web dashboard:

- Skeleton loaders while fetching reports.
- Empty-state cards when no runs exist.
- Error banners when API calls fail.

### 6.6 Error states

Errors must be visible but should not always abort the run.

Examples:

- Host unreachable: mark host unavailable and continue with other hosts if configured.
- Model unsupported by plugin: skip model/plugin combination and record reason.
- Case timeout: record failed case and continue.
- Context size unsupported: record failure and stop increasing context size for that model.
- Plugin crash: record plugin failure and continue with other plugins.

### 6.7 Empty states

If no hosts are configured:

```text
No Ollama hosts configured.
Add hosts to config.yaml or run: local-ai-bench init
```

If no models are found:

```text
No models found on http://127.0.0.1:11434.
Install a model with: ollama pull llama3.2:3b
```

If no reports exist:

```text
No benchmark reports found.
Run your first benchmark with: local-ai-bench run
```

---

## 7. Recommended Technical Stack

### Primary stack

- Python 3.12+
- `httpx` for async HTTP requests to Ollama
- `pydantic` for configuration and result schemas
- `typer` for CLI
- `rich` for terminal UX
- `jinja2` for HTML/Markdown report templates
- `pyyaml` for configuration parsing
- `pytest` and `pytest-asyncio` for testing
- `ruff` and `mypy` for code quality

### Optional later stack

- FastAPI for local report browsing
- SQLite for historical run indexing
- Docker for sandboxed code execution
- Playwright only if UI testing becomes necessary

### Why this stack?

- Python is ideal for benchmark orchestration and evaluation.
- Async HTTP allows efficient calls while still limiting concurrency.
- Pydantic gives strong validation for config and results.
- File-based reports keep deployment simple.
- CLI-first UX avoids unnecessary frontend complexity in v1.

---

## 8. High-Level Architecture

```text
                 +----------------------+
                 |        CLI / UI      |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |     Run Orchestrator |
                 +----+--------+--------+
                      |        |
        +-------------+        +-------------+
        v                                    v
+------------------+                 +------------------+
| Plugin Registry  |                 | Context Optimizer|
+--------+---------+                 +--------+---------+
         |                                    |
         v                                    v
+------------------+                 +------------------+
| Benchmark Plugins|                 | Long-Context Probe|
+--------+---------+                 +--------+---------+
         |                                    |
         +------------------+-----------------+
                            v
                 +----------------------+
                 |    Ollama Gateway    |
                 +----+------------+----+
                      |            |
                      v            v
              +----------+    +----------+
              | Host A   |    | Host B   |
              +----------+    +----------+
                            |
                            v
                 +----------------------+
                 |  Result Repository   |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |   Report Generator   |
                 +----------------------+
```

---

## 9. Module Responsibilities

### 9.1 `local_ai_bench.config`

Responsible for:

- Loading YAML config.
- Validating config with Pydantic.
- Resolving enabled hosts, models, plugins, and options.
- Computing config hash for reproducibility.

### 9.2 `local_ai_bench.ollama`

Responsible for:

- Health checks.
- Listing models.
- Fetching model metadata.
- Sending chat requests.
- Streaming responses.
- Measuring timing metrics.
- Mapping Ollama-specific errors.

### 9.3 `local_ai_bench.plugins`

Responsible for:

- Defining benchmark plugin contracts.
- Discovering built-in plugins.
- Discovering local file plugins.
- Discovering installed entry-point plugins.
- Validating plugin metadata.

### 9.4 `local_ai_bench.runner`

Responsible for:

- Creating benchmark cases.
- Scheduling model/plugin/case execution.
- Applying retries and timeouts.
- Emitting progress events.
- Collecting case results.
- Handling partial failures.

### 9.5 `local_ai_bench.context`

Responsible for:

- Generating context-size candidates.
- Running long-context probes.
- Estimating optimal context window.
- Recording context-quality curves.

### 9.6 `local_ai_bench.reporting`

Responsible for:

- Aggregating results.
- Writing JSON, JSONL, Markdown, and HTML reports.
- Rendering comparison tables.
- Rendering warnings and skipped cases.
- Preserving raw artifacts.

### 9.7 `local_ai_bench.cli`

Responsible for:

- Command parsing.
- User-facing progress display.
- Launching runs.
- Opening reports.

---

## 10. Repository Structure

```text
local-ai-bench/
├── PLAN.md
├── README.md
├── pyproject.toml
├── config/
│   └── default.yaml
├── datasets/
│   ├── translation/
│   ├── coding/
│   ├── vision/
│   ├── summarization/
│   ├── reasoning/
│   ├── structured_output/
│   └── long_context/
├── plugins/
│   └── example_plugin.py
├── reports/
├── src/
│   └── local_ai_bench/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── domain/
│       │   ├── __init__.py
│       │   ├── models.py
│       │   ├── results.py
│       │   └── events.py
│       ├── ollama/
│       │   ├── __init__.py
│       │   ├── client.py
│       │   ├── discovery.py
│       │   └── metrics.py
│       ├── plugins/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── registry.py
│       │   └── builtin/
│       │       ├── translation.py
│       │       ├── coding.py
│       │       ├── vision.py
│       │       ├── summarization.py
│       │       ├── reasoning.py
│       │       ├── structured_output.py
│       │       └── long_context.py
│       ├── runner/
│       │   ├── __init__.py
│       │   ├── orchestrator.py
│       │   ├── scheduler.py
│       │   └── events.py
│       ├── context/
│       │   ├── __init__.py
│       │   ├── optimizer.py
│       │   └── probes.py
│       ├── reporting/
│       │   ├── __init__.py
│       │   ├── repository.py
│       │   ├── aggregator.py
│       │   ├── markdown.py
│       │   ├── html.py
│       │   └── templates/
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           ├── timing.py
│           └── hashing.py
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

---

## 11. Configuration Design

The configuration file should be simple, explicit, and validated.

### 11.1 Example `config.yaml`

```yaml
app:
  name: LocalAIBench
  output_dir: ./reports
  log_level: info

hosts:
  - name: local
    base_url: http://127.0.0.1:11434
    timeout_seconds: 300
  # - name: workstation
  #   base_url: http://192.168.1.50:11434
  #   timeout_seconds: 300

models:
  - llama3.2:3b
  - qwen2.5-coder:7b
  # - llava:7b

plugins:
  enabled:
    - translation
    - coding
    - vision
    - summarization
    - reasoning
    - structured_output
    - long_context
  options:
    translation:
      language_pairs:
        - en->es
        - en->fr
    coding:
      execute_code: false
      timeout_seconds: 30
    vision:
      max_image_dimension: 768

runner:
  repetitions: 3
  warmup_runs: 1
  concurrency: 1
  temperature: 0.0
  seed: 42
  max_retries: 2
  retry_backoff_seconds: 2

judge:
  enabled: true
  model: llama3.2:3b
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
```

### 11.2 Configuration rules

- Hosts must have unique names.
- Base URLs must be valid HTTP URLs.
- Enabled plugins must exist.
- Model list can be empty, meaning all discovered models.
- If judge is enabled but judge model is missing, warn and fall back to deterministic scoring where possible.
- If `execute_code` is true, require explicit user confirmation or a `--unsafe-execute-code` flag.

---

## 12. Domain Model

Use Pydantic models for all major structures.

### 12.1 Core concepts

```python
from enum import Enum
from pydantic import BaseModel


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    CODE = "code"
    JSON = "json"


class BenchmarkCategory(str, Enum):
    TRANSLATION = "translation"
    CODING = "coding"
    VISION = "vision"
    SUMMARIZATION = "summarization"
    REASONING = "reasoning"
    STRUCTURED_OUTPUT = "structured_output"
    LONG_CONTEXT = "long_context"


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
    raw_metadata: dict | None = None


class BenchmarkCase(BaseModel):
    id: str
    plugin_id: str
    dataset_version: str
    input: dict
    expected: dict | None = None
    metadata: dict = {}


class TimingMetrics(BaseModel):
    started_at: float
    first_token_at: float | None = None
    finished_at: float
    total_ms: float
    time_to_first_token_ms: float | None = None
    generation_ms: float | None = None


class TokenMetrics(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_per_second: float | None = None


class ModelResponse(BaseModel):
    raw: dict
    text: str
    timing: TimingMetrics
    tokens: TokenMetrics
    error: str | None = None


class Evaluation(BaseModel):
    score: float | None = None
    passed: bool | None = None
    metrics: dict = {}
    rationale: str | None = None
    judge_model: str | None = None


class CaseResult(BaseModel):
    case: BenchmarkCase
    model: ModelInfo
    response: ModelResponse
    evaluation: Evaluation
    attempt: int = 1
```

### 12.2 Aggregated result

```python
class PluginAggregate(BaseModel):
    plugin_id: str
    model_name: str
    host_name: str
    total_cases: int
    successful_cases: int
    failed_cases: int
    skipped_cases: int
    score: float | None = None
    metrics: dict = {}


class ContextWindowResult(BaseModel):
    model_name: str
    host_name: str
    context_size: int
    quality_score: float
    average_latency_ms: float
    p95_latency_ms: float
    tokens_per_second: float | None = None
    error_rate: float
    stable: bool
    recommended: bool = False
```

---

## 13. Ollama Integration Requirements

### 13.1 Supported endpoints

The Ollama gateway should use:

- `GET /api/tags`
  - List available models.
- `GET /api/show`
  - Fetch model metadata when available.
- `POST /api/chat`
  - Primary benchmark endpoint.
  - Supports multimodal messages where applicable.
- `POST /api/generate`
  - Optional fallback for simple completion-style benchmarks.

### 13.2 Chat request baseline

Default request options:

```json
{
  "model": "model-name",
  "messages": [],
  "stream": true,
  "options": {
    "temperature": 0.0,
    "num_ctx": 4096
  }
}
```

### 13.3 Metrics to collect

For each request:

- Total request duration.
- Time to first token.
- Generation time.
- Prompt tokens, if provided by Ollama.
- Completion tokens, if provided by Ollama.
- Tokens per second.
- HTTP status.
- Error type.
- Whether response was truncated.
- Whether response was empty.

### 13.4 Streaming

Use streaming where possible to measure time to first token.

If streaming is unavailable or unstable:

- Fall back to non-streaming.
- Mark `time_to_first_token_ms` as null.
- Record fallback reason.

### 13.5 Retries

Retry only transient network errors:

- Connection refused.
- DNS failure.
- Timeout during connection.
- HTTP 502/503/504.

Do not retry:

- Invalid model.
- Malformed benchmark output.
- Evaluation failures.
- Context-size failures unless configured to retry with smaller context.

### 13.6 Host discovery

For v1, use explicit configuration.

Optional convenience behavior:

- If no hosts are configured, try `http://127.0.0.1:11434`.
- Respect `OLLAMA_HOST` environment variable if present.
- Do not perform aggressive network scanning by default.

Alternative future enhancement:

- mDNS/DNS-SD discovery.
- Manual CIDR scan only if explicitly requested.

---

## 14. Plugin Architecture

The plugin system is the core extensibility mechanism.

### 14.1 Plugin goals

A benchmark plugin should be able to:

- Declare its category and supported modalities.
- Provide dataset cases.
- Build prompts or chat messages.
- Define model compatibility rules.
- Evaluate model outputs.
- Aggregate case results.
- Participate in context-window testing if relevant.

### 14.2 Abstract plugin contract

```python
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import ClassVar


class BenchmarkPlugin(ABC):
    id: ClassVar[str]
    name: ClassVar[str]
    category: ClassVar[BenchmarkCategory]
    version: ClassVar[str]
    dataset_version: ClassVar[str]
    modalities: ClassVar[set[Modality]]

    @abstractmethod
    def supports_model(self, model: ModelInfo) -> bool:
        """Return True if this plugin can benchmark the given model."""

    @abstractmethod
    async def prepare(self, ctx: "RunContext") -> None:
        """Load datasets, warm caches, validate dependencies."""

    @abstractmethod
    def cases(self, ctx: "RunContext") -> Iterable[BenchmarkCase]:
        """Yield benchmark cases for this plugin."""

    @abstractmethod
    def build_request(
        self,
        case: BenchmarkCase,
        model: ModelInfo,
        ctx: "RunContext",
    ) -> dict:
        """Build the Ollama API request payload."""

    @abstractmethod
    async def evaluate(
        self,
        case: BenchmarkCase,
        response: ModelResponse,
        ctx: "RunContext",
    ) -> Evaluation:
        """Score the model response."""

    def aggregate(self, results: Sequence[CaseResult]) -> PluginAggregate:
        """Aggregate case results into plugin-level metrics."""

    async def teardown(self, ctx: "RunContext") -> None:
        """Clean up temporary files or subprocesses."""
```

### 14.3 Optional plugin capabilities

Use capability mixins or optional methods.

Example:

```python
class ContextAwarePlugin:
    def context_probe_cases(self, ctx: "RunContext") -> Iterable[BenchmarkCase]:
        raise NotImplementedError
```

This keeps the base interface small.

### 14.4 Plugin discovery

Support three discovery mechanisms:

1. Built-in plugins inside `local_ai_bench.plugins.builtin`.
2. Local plugins in `./plugins/*.py`.
3. Installed plugins via Python entry points:

```toml
[project.entry-points."local_ai_bench.plugins"]
my_benchmark = "my_package.plugin:MyBenchmarkPlugin"
```

### 14.5 Plugin validation

The registry should validate:

- Unique plugin ID.
- Valid category.
- Non-empty dataset version.
- Plugin class can be instantiated.
- Plugin does not raise during `supports_model`.

### 14.6 Plugin failure isolation

If a plugin fails:

- Record plugin error.
- Continue with other plugins.
- Include error in report.
- Do not corrupt already collected results.

---

## 15. Built-in Benchmark Plugins

Each plugin should be small, deterministic, and easy to inspect.

---

## 15.1 Translation Benchmark

Plugin ID: `translation`

### Purpose

Evaluate practical translation quality between language pairs.

### Dataset

Small local dataset in `datasets/translation/v1/`.

Example case:

```json
{
  "id": "en_to_es_0001",
  "source_language": "en",
  "target_language": "es",
  "text": "The meeting was postponed because of bad weather.",
  "expected_keywords": ["reunión", "pospuesta", "clima"],
  "domain": "general"
}
```

### Prompt strategy

Use a strict instruction:

```text
Translate the following text from English to Spanish.
Return only the translation, no explanation.

Text:
{input_text}
```

### Evaluation

Primary metrics:

- Keyword recall.
- Length ratio sanity.
- Optional LLM-as-judge score for fluency and fidelity.

Optional future metrics:

- BLEU if reference translations are available.
- COMET if local model support is added.

### Score

Composite:

- 40% keyword recall
- 30% length sanity
- 30% judge score if available

If judge unavailable:

- 70% keyword recall
- 30% length sanity

### Edge cases

- Empty translation.
- Wrong language.
- Model adds explanation.
- Model repeats prompt.

---

## 15.2 Coding Benchmark

Plugin ID: `coding`

### Purpose

Evaluate code generation quality.

### Dataset

Small Python coding tasks in `datasets/coding/v1/`.

Example:

```json
{
  "id": "py_list_chunk_0001",
  "language": "python",
  "prompt": "Write a function chunk_list(items, size) that splits a list into chunks.",
  "function_name": "chunk_list",
  "tests": [
    "assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]",
    "assert chunk_list([], 2) == []"
  ]
}
```

### Execution modes

#### Safe mode, default

Do not execute generated code.

Evaluate:

- Syntax validity.
- Presence of required function.
- Static heuristic checks.
- Optional judge score.

#### Unsafe execution mode

Only when explicitly enabled.

Run code in temporary directory with:

- Timeout.
- Memory limit where supported.
- No network access where supported.
- Separate subprocess.

### Metrics

- Syntax valid rate.
- Function extraction rate.
- Unit test pass rate.
- Pass@1.
- Latency.
- Tokens per second.

### Edge cases

- Markdown fences around code.
- Missing function.
- Infinite loop.
- Import attempts.
- Malformed code.
- Test timeout.

---

## 15.3 Vision Benchmark

Plugin ID: `vision`

### Purpose

Evaluate multimodal models on image understanding.

### Dataset

Local images and JSON metadata in `datasets/vision/v1/`.

Example:

```json
{
  "id": "vision_street_sign_0001",
  "image": "images/street_sign_0001.png",
  "question": "What does the sign say?",
  "expected_keywords": ["stop"],
  "task": "ocr"
}
```

### Model compatibility

Only run if:

```python
model.supports_vision is True
```

Otherwise mark skipped.

### Request construction

Use Ollama chat message format with base64 image data where supported.

Example conceptual payload:

```json
{
  "model": "llava:7b",
  "messages": [
    {
      "role": "user",
      "content": "What does the sign say?",
      "images": ["BASE64_IMAGE"]
    }
  ]
}
```

### Evaluation

- Keyword recall.
- Optional judge score.
- Exact match for OCR-like tasks.
- Bounding sanity checks if metadata provides constraints.

### Edge cases

- Model ignores image.
- Image too large.
- Base64 encoding failure.
- Model claims it cannot see images.
- Unsupported image format.

---

## 15.4 Summarization Benchmark

Plugin ID: `summarization`

### Purpose

Evaluate abstractive summarization quality.

### Dataset

Local articles or synthetic documents.

Example:

```json
{
  "id": "summarization_business_0001",
  "text": "...",
  "must_mention": ["revenue", "acquisition"],
  "max_words": 80
}
```

### Prompt

```text
Summarize the following text in 3 sentences.
Do not include information not present in the text.

Text:
{input_text}
```

### Evaluation

- Keyword coverage.
- Length compliance.
- Optional judge score for faithfulness and conciseness.
- Penalty for hallucinated keywords if known.

### Edge cases

- Summary longer than limit.
- Copying full text.
- Empty summary.
- Hallucinated facts.

---

## 15.5 Reasoning Benchmark

Plugin ID: `reasoning`

### Purpose

Evaluate basic math, logic, and structured reasoning.

### Dataset

Short deterministic problems.

Example:

```json
{
  "id": "math_percentage_0001",
  "question": "What is 15% of 240?",
  "expected_answer": "36",
  "answer_format": "numeric"
}
```

### Evaluation

- Exact match after normalization.
- Numeric tolerance where appropriate.
- Optional judge for free-form reasoning.

### Metrics

- Accuracy.
- Answer extraction rate.
- Latency.

### Edge cases

- Model explains too much.
- Model outputs multiple answers.
- Units missing.
- Minor formatting differences.

---

## 15.6 Structured Output Benchmark

Plugin ID: `structured_output`

### Purpose

Evaluate JSON/schema compliance.

### Dataset

Tasks requiring JSON output.

Example:

```json
{
  "id": "structured_invoice_0001",
  "instruction": "Extract company name and total amount.",
  "text": "Invoice from Acme Corp. Total due: $250.00",
  "expected_schema": {
    "type": "object",
    "required": ["company_name", "total_amount"]
  }
}
```

### Evaluation

- Valid JSON rate.
- Schema compliance rate.
- Field correctness.
- Repairability score if JSON can be fixed by simple cleanup.

### Edge cases

- Markdown code fences.
- Trailing prose.
- Invalid JSON.
- Missing fields.
- Wrong types.

---

## 15.7 Long Context Benchmark

Plugin ID: `long_context`

### Purpose

Measure long-context retention and help identify optimal context window size.

### Probe type

Needle-in-a-haystack style.

### Dataset generation

Generate deterministic filler text with a hidden fact.

Example hidden fact:

```text
The secret project code is BLUE FALCON.
```

Place the needle at a configured position:

- 10%
- 30%
- 50%
- 70%
- 90%

Then ask:

```text
What is the secret project code?
Answer with only the code.
```

### Metrics

- Retrieval accuracy.
- Answer exact match.
- Latency.
- Tokens per second.
- Failure rate.
- Stability across repetitions.

### Edge cases

- Model hallucinates.
- Model truncates.
- Request fails due to memory.
- Context size exceeds model support.

---

## 16. Context Window Optimization

This is a first-class feature.

### 16.1 Goal

Find a practical context-window recommendation for each model.

The recommendation should balance:

- Quality.
- Latency.
- Throughput.
- Stability.
- Resource usage.

### 16.2 Candidate sizes

Candidate sizes come from:

1. Explicit config list.
2. Model metadata if available.
3. Safe default powers of two.

Example:

```text
512, 1024, 2048, 4096, 8192, 16384
```

Constraints:

- Do not exceed `max_candidate_size`.
- Do not exceed known model max context if available.
- Stop increasing after repeated hard failures.

### 16.3 Probe design

For each candidate context size:

1. Generate haystack with hidden needle.
2. Test multiple needle positions.
3. Run configured repetitions.
4. Measure:
   - Retrieval accuracy.
   - Total latency.
   - Time to first token.
   - Tokens per second.
   - Error rate.

### 16.4 Quality score

Example:

```text
quality_score = correct_needle_retrievals / total_needle_retrievals
```

### 16.5 Stability score

```text
stability_score = 1 - error_rate
```

If a context size produces OOM, timeout, or API failure, mark it unstable.

### 16.6 Recommended context selection

Default heuristic:

1. Find maximum quality score across stable sizes.
2. Keep sizes where:

```text
quality_score >= quality_threshold * max_quality_score
```

3. Among those, choose the smallest context size unless latency budget favors a smaller one.
4. If latency budget is configured, exclude sizes where p95 latency exceeds budget.
5. If no stable size meets threshold, recommend largest stable size with a warning.

Example:

```text
quality_threshold = 0.90
max_quality_score = 0.98
minimum_accepted_quality = 0.882
```

Choose smallest stable context with quality >= 0.882 and acceptable latency.

### 16.7 Context report output

For each model:

```json
{
  "model": "qwen2.5:7b",
  "recommended_context": 8192,
  "reason": "Highest stable quality with acceptable latency",
  "curve": [
    {
      "context_size": 512,
      "quality": 0.6,
      "p95_latency_ms": 800,
      "tokens_per_second": 42.1,
      "stable": true
    },
    {
      "context_size": 8192,
      "quality": 0.97,
      "p95_latency_ms": 3100,
      "tokens_per_second": 31.8,
      "stable": true
    }
  ]
}
```

### 16.8 Context optimization edge cases

- Model supports large context but quality collapses.
- Model becomes slower non-linearly.
- Context request fails only on remote host due to network timeout.
- Ollama does not expose true max context.
- `num_ctx` option is ignored by model.
- Large context causes host memory pressure.

The report should distinguish:

- Unsupported.
- Unstable.
- Low quality.
- Too slow.

---

## 17. Scoring and Aggregation

### 17.1 Case-level score

Each plugin returns a normalized score between 0 and 1 where possible.

If binary pass/fail:

```text
passed = 1.0
failed = 0.0
```

If rubric-based:

```text
score = weighted_metric_sum
```

### 17.2 Plugin-level score

Aggregate by model:

- Mean score.
- Median score.
- Success rate.
- Error rate.
- p50 latency.
- p95 latency.
- tokens per second.

### 17.3 Overall model score

Use configurable category weights.

Default:

```yaml
weights:
  translation: 1.0
  coding: 1.0
  vision: 1.0
  summarization: 1.0
  reasoning: 1.0
  structured_output: 1.0
  long_context: 1.0
```

Overall score:

```text
overall = sum(plugin_score * weight) / sum(weights for plugins executed)
```

Do not include skipped plugins in denominator.

### 17.4 Report comparability

Reports must clearly state:

- App version.
- Config hash.
- Dataset versions.
- Plugin versions.
- Ollama host URLs.
- Model digests.
- Timestamp.
- Repetitions.
- Temperature.
- Context sizes used.

---

## 18. Report Requirements

Every run must create a timestamped report directory.

Example:

```text
reports/
└── 2026-06-16T14-30-00_local-ai-bench/
    ├── report.json
    ├── report.md
    ├── index.html
    ├── assets/
    │   ├── styles.css
    │   └── charts.js
    ├── raw/
    │   ├── cases.jsonl
    │   ├── events.jsonl
    │   └── errors.jsonl
    └── media/
```

### 18.1 `report.json`

Machine-readable full result.

Should include:

- Run metadata.
- Config.
- Hosts.
- Models.
- Plugin results.
- Case results.
- Context optimization results.
- Warnings.
- Skips.
- Errors.

### 18.2 `report.md`

Human-readable summary.

Sections:

1. Executive summary.
2. Environment.
3. Overall model ranking.
4. Category results.
5. Context-window recommendations.
6. Failures and warnings.
7. Reproduction instructions.

### 18.3 `index.html`

Static HTML report.

Features:

- Sortable tables.
- Category tabs or sections.
- Context chart if possible.
- Warnings section.
- Raw JSON download link.

No external CDN dependencies by default.

### 18.4 Executive summary

Example:

```markdown
## Executive Summary

Best overall model: qwen2.5:7b
Best coding model: qwen2.5-coder:7b
Best translation model: llama3.2:3b
Best vision model: llava:7b
Recommended context for qwen2.5:7b: 8192 tokens
```

### 18.5 Empty or partial reports

If a benchmark run is partial:

- Show which plugins completed.
- Show which plugins failed.
- Show reasons.
- Still include available scores.

Never silently drop failures.

---

## 19. Runner Behavior

### 19.1 Execution order

Recommended default order:

1. Load config.
2. Run host health checks.
3. Discover models.
4. Load plugins.
5. Validate plugin/model compatibility.
6. For each host:
   1. For each model:
      1. Warmup request.
      2. Run enabled plugins.
      3. Run context optimization if enabled.
7. Aggregate results.
8. Generate reports.

### 19.2 Concurrency

Default concurrency:

```yaml
concurrency: 1
```

Why:

- Local GPUs can be easily overloaded.
- Benchmark stability is more important than throughput.
- Sequential execution produces cleaner latency measurements.

Optional later:

- Per-host concurrency.
- Per-model concurrency.
- Benchmark-only parallelism where safe.

### 19.3 Warmup

Each model should receive at least one warmup request before scoring.

Warmup should:

- Use a tiny harmless prompt.
- Not be included in scores.
- Verify that model is loaded.
- Detect immediate failures.

### 19.4 Repetitions

For each benchmark case:

- Run configured repetitions.
- Use median for latency metrics.
- Use mean for quality metrics unless otherwise specified.

Default:

```yaml
repetitions: 3
```

### 19.5 Timeouts

Each request must have:

- Connect timeout.
- Read timeout.
- Overall benchmark case timeout.

If timeout occurs:

- Record error.
- Mark case failed.
- Continue.

### 19.6 Event stream

The runner should emit structured events:

```python
RunStarted
RunPlanned        # data.total_cases: cumulative planned case-run count
HostChecked
ModelDiscovered
PluginStarted
CaseStarted
CaseCompleted
CaseFailed
PluginCompleted
ContextProbeStarted
ContextProbeCompleted
RunCompleted
ReportGenerated
```

`RunPlanned` carries the cumulative planned number of case-runs (Σ cases ×
repetitions over selected models and supported plugins), emitted per host after
it is discovered but before any case runs. It lets progress layers display a
real total up front instead of accruing it from `CaseStarted`.

These events should power:

- CLI progress.
- JSONL event logs.
- Future live dashboard.

---

## 20. LLM-as-Judge Design

Some benchmarks benefit from a local judge model.

### 20.1 Judge purpose

Use judge for subjective scoring:

- Translation fluency.
- Summarization faithfulness.
- Vision answer correctness.
- Coding explanation quality, if needed.

### 20.2 Judge constraints

- Judge must be local via Ollama.
- Judge should use temperature 0.
- Judge prompt must request strict JSON.
- Judge failures should not crash benchmark.
- Judge should be optional.

### 20.3 Judge prompt pattern

```text
You are an evaluation judge.
Evaluate the candidate response using the rubric.
Return only valid JSON.

Rubric:
{rubric}

Input:
{input}

Expected:
{expected}

Candidate response:
{candidate}

Return JSON:
{
  "score": 0.0 to 1.0,
  "passed": true or false,
  "rationale": "short explanation"
}
```

### 20.4 Judge fallback

If judge unavailable:

- Use deterministic metrics.
- Mark report with warning:

```text
Judge scoring unavailable. Deterministic fallback used.
```

---

## 21. Security Requirements

### 21.1 Local-first

Default behavior should avoid internet access.

Do not:

- Upload prompts.
- Upload images.
- Call external scoring APIs.
- Send telemetry.

### 21.2 Code execution

Code execution must be opt-in.

If enabled:

- Use temp directories.
- Use subprocess timeout.
- Use resource limits where supported.
- Avoid shell execution.
- Do not persist generated code outside report artifacts unless configured.

CLI flag:

```bash
local-ai-bench run --unsafe-execute-code
```

### 21.3 Report sanitization

If HTML reports render model output:

- Escape HTML.
- Do not execute scripts.
- Treat model output as untrusted.

### 21.4 Secrets

If API keys are ever added:

- Do not write secrets into reports.
- Do not log secrets.
- Support environment variable expansion.

---

## 22. Error Handling Matrix

| Situation | Behavior |
|---|---|
| Host unreachable | Mark host failed, continue other hosts if any |
| Model not found | Skip model, record warning |
| Plugin unsupported model | Skip combination, record reason |
| Vision plugin on text-only model | Skip with reason `vision_not_supported` |
| Empty model response | Score zero, record `empty_response` |
| Malformed JSON from structured output | Score low or zero, record `invalid_json` |
| Code syntax error | Score syntax metric zero, continue |
| Code execution timeout | Mark failed, continue |
| Context size fails | Mark unstable, try smaller/larger based on policy |
| Judge model unavailable | Fall back to deterministic scoring |
| Report generation fails | Preserve raw JSONL artifacts and show CLI error |
| Config invalid | Fail fast with validation errors |
| Dataset missing | Fail plugin, not entire run, if other plugins can continue |

---

## 23. Testing Strategy

### 23.1 Unit tests

Cover:

- Config validation.
- Plugin registry.
- Scoring functions.
- Context recommendation logic.
- Metric calculations.
- Report aggregation.

### 23.2 Integration tests

Use mock Ollama server.

Cover:

- `/api/tags`
- `/api/show`
- `/api/chat`
- Streaming behavior.
- Timeouts.
- Error responses.

### 23.3 Golden tests

For deterministic plugins:

- Save expected report fragments.
- Compare against known outputs.

### 23.4 Plugin tests

Each built-in plugin should have:

- Fixture cases.
- Known good response tests.
- Known bad response tests.
- Unsupported model tests.

### 23.5 End-to-end test

Use tiny fake models and small datasets to run:

```bash
local-ai-bench run --config tests/fixtures/config.yaml
```

Assert:

- Report directory created.
- JSON report valid.
- Markdown report exists.
- HTML report exists.
- Errors are recorded where expected.

---

## 24. Implementation Milestones

### Milestone Status

Updated 2026-08-11. All v1 milestones are implemented and tested (94→99 unit
tests passing; ruff + mypy clean). Features beyond the original plan are noted.

| Milestone | Status | Notes |
|---|---|---|
| M0 Project Scaffold | ✅ Done | Package, pyproject, ruff/mypy/pytest, CLI entry point |
| M1 Configuration System | ✅ Done | Pydantic models, YAML load, config hash, `init` writes packaged default verbatim; default omits `hosts` and honors `OLLAMA_HOST` |
| M2 Ollama Gateway | ✅ Done | Health, tags, show, streaming chat, timing/token metrics |
| M3 Plugin Framework | ✅ Done | Registry, built-in + local discovery; extras: plugin source API (`/api/plugins/{id}`) |
| M4 Benchmark Runner | ✅ Done | Orchestrator, event stream, retries/timeouts, JSONL raw persistence, active-run status restore |
| M5 Built-in Text Benchmarks | ✅ Done | translation, summarization, reasoning, structured_output |
| M6 Coding & Vision | ✅ Done | coding safe/unsafe modes, vision skips non-vision models |
| M7 Context Optimizer | ✅ Done | Needle-in-haystack, candidate selection, recommendation heuristic; extra: multi_context plugin |
| M8 Reporting | ✅ Done | JSON/Markdown/HTML, comparison tables, warnings/skips; context-window table now dynamic to probed sizes |
| M9 CLI Polish | ✅ Done | doctor, models, plugins list, report list/view/open, history, compare, run-single |
| M10 Web Report Viewer | ✅ Done | FastAPI + Astro; history/compare/plugins/run pages, delete UI, live active-run status, SQLite storage |

Extras added beyond the v1 plan:

- `multi_context` benchmark plugin (fixed context sizes + keyword matching).
- SQLite run history (`benchmark.db`) with repository access layer.
- Web UI: compare page with sortable/column-picker tables, multi-run compare,
  per-plugin score/latency columns and `compare_default` config, plugin option
  editing, bulk + per-page delete.
- CLI `run-single` and `compare` commands.
- Docker deployment (Dockerfile, docker-compose.yml) with a mounted `plugins/` dir.
- Per-plugin YAML option defaults via a free-form `plugins.options` map.

### Open / Next

- Commit the in-flight changes: free-form `plugins.options`, `compare_default`,
  dynamic context-window report table, weighted-plugin fallback in compare page.
- Category weights are now editable from the Compare page (`/api/weights`), with
  a live "Weighted" overall-score column and DB-persisted overrides that apply
  to future runs. Stored per-run overall scores remain unchanged (recomputed
  live in the UI).
- Recommended default config is now zero-config for new users: `hosts` is
  omitted (falls back to local `127.0.0.1:11434` or `$OLLAMA_HOST`), `local_dir`
  is `./plugins` (Docker mounts it at `/app/plugins`), and `local-ai-bench init`
  writes this working starter config.
- E2E integration tests added (mock Ollama via `httpx.MockTransport`): a full
  discovery → run → report → SQLite-persist path. This surfaced and fixed a
  `max_retries=0` bug in `_send_with_retries` where the initial request was
  never sent; `max_retries` now correctly means "retries after the first
  attempt". `OllamaClient`/`RunOrchestrator` accept an optional `transport`
  for testability.

---

## Milestone 0: Project Scaffold

Tasks:

- Create Python package structure.
- Add `pyproject.toml`.
- Configure `ruff`, `mypy`, and `pytest`.
- Add CLI entry point.
- Add basic logging.

Acceptance criteria:

```bash
local-ai-bench --help
```

works.

---

## Milestone 1: Configuration System

Tasks:

- Define Pydantic settings.
- Load YAML config.
- Validate hosts, models, plugins, runner options.
- Compute config hash.

Acceptance criteria:

- Invalid config fails with readable errors.
- Default config can be generated.
- Config hash changes when meaningful options change.

---

## Milestone 2: Ollama Gateway

Tasks:

- Implement health check.
- Implement model listing.
- Implement model metadata fetch.
- Implement chat request with streaming.
- Collect timing/token metrics.

Acceptance criteria:

- Can list models from local Ollama.
- Can send a simple chat request.
- Can record latency and token metrics.
- Handles unreachable host gracefully.

---

## Milestone 3: Plugin Framework

Tasks:

- Implement `BenchmarkPlugin`.
- Implement plugin registry.
- Implement local plugin loading.
- Implement entry-point plugin loading.
- Create example plugin.

Acceptance criteria:

- Adding a new local plugin file makes it visible in:

```bash
local-ai-bench plugins list
```

without modifying core runner.

---

## Milestone 4: Benchmark Runner

Tasks:

- Implement run orchestration.
- Implement event stream.
- Implement retries/timeouts.
- Implement case result collection.
- Implement raw JSONL persistence.

Acceptance criteria:

- A fake plugin can run end-to-end.
- Progress is displayed.
- Failed cases do not stop entire run.
- Raw results are saved.

---

## Milestone 5: Built-in Text Benchmarks

Tasks:

- Implement translation plugin.
- Implement summarization plugin.
- Implement reasoning plugin.
- Implement structured output plugin.

Acceptance criteria:

- Each plugin has dataset fixtures.
- Each plugin produces normalized score.
- Each plugin handles unsupported/empty outputs.

---

## Milestone 6: Coding and Vision Benchmarks

Tasks:

- Implement coding plugin safe mode.
- Implement optional unsafe execution mode.
- Implement vision plugin.
- Add image handling utilities.

Acceptance criteria:

- Coding benchmark runs safely by default.
- Vision benchmark skips non-vision models.
- Images are correctly encoded for Ollama-compatible models.

---

## Milestone 7: Context Optimizer

Tasks:

- Implement needle-in-haystack generator.
- Implement context candidate selection.
- Implement probe runner.
- Implement recommendation heuristic.
- Store context curve results.

Acceptance criteria:

- Report includes recommended context size.
- Context curve includes quality and latency metrics.
- Unstable sizes are marked.

---

## Milestone 8: Reporting

Tasks:

- Implement JSON report.
- Implement Markdown report.
- Implement HTML report.
- Implement comparison tables.
- Implement warnings/skips section.

Acceptance criteria:

- A completed run generates all report formats.
- Reports include enough metadata to reproduce run.
- Partial runs still generate useful reports.

---

## Milestone 9: CLI Polish

Tasks:

- Add `doctor`.
- Add `models`.
- Add `plugins list`.
- Add `report list/view/open`.
- Improve errors and hints.

Acceptance criteria:

- New user can go from install to first report with minimal friction.

---

## Milestone 10: Optional Web Report Viewer

Tasks:

- Add FastAPI app.
- Serve report list.
- Serve static generated reports.
- Add simple run detail page.

Acceptance criteria:

- Reports can be browsed locally.
- No external internet assets required.

---

## 25. Definition of Done for v1

The project is done when:

1. `local-ai-bench init` creates a working config.
2. `local-ai-bench doctor` validates environment.
3. `local-ai-bench run` benchmarks configured models.
4. Built-in plugins run successfully where supported.
5. Unsupported model/plugin combinations are skipped cleanly.
6. Context-window optimization produces a recommendation.
7. Reports are saved as JSON, Markdown, and HTML.
8. Raw case results are preserved.
9. Failures are visible and actionable.
10. Tests pass.
11. README explains how to run benchmarks.
12. Plugin developer guide explains how to add a new benchmark.

---

## 26. Plugin Developer Experience

Adding a new benchmark should be simple.

Example future plugin:

```text
plugins/
└── medical_qa.py
```

The plugin should only need to define:

- ID.
- Category.
- Dataset.
- Prompt builder.
- Evaluator.
- Aggregator.

Then:

```bash
local-ai-bench plugins list
```

should show it.

No core runner changes should be required.

---

## 27. Extensibility Ideas

Future plugins may include:

- ~~Agent/tool-use benchmark~~ ✅ Built-in `agent_tool_use` (v1: multi-turn tool loops with deterministic tool execution, gated on model tool capability).
- ~~Function-calling benchmark~~ ✅ Built-in `function_calling` (v1: name + args accuracy, gated on model tool capability).
- ~~Retrieval-augmented generation benchmark~~ ✅ Built-in `rag` (v1: grounded QA with distractor-passage hallucination penalty).
- ~~Multilingual benchmark~~ ✅ Built-in `multilingual` (v1: in-language comprehension + Unicode-script fidelity across 8 languages, deterministic).
- ~~Safety/refusal benchmark~~ ✅ Built-in `safety_refusal` (v1: refusal of harmful requests + false-positive refusal checks on benign prompts).
- Long-document QA benchmark.
- ~~SQL generation benchmark~~ ✅ Built-in `sql` (v1: execution-based row comparison against an in-memory SQLite schema, induces correct SELECT generation).
- Regex generation benchmark.
- ~~Classification benchmark~~ ✅ Built-in `classification` (v1: label-set matching for sentiment/triage/urgency/topic, deterministic).
- Embedding benchmark, if Ollama embedding endpoints are used.
- ~~Multi-turn conversation benchmark~~ ✅ Built-in `multi_turn` (v1: conversational memory and follow-up comprehension, capped at 4 turns).
- Robustness benchmark against typos and noisy input.

The plugin architecture must support these without redesign.

---

## 28. Technical Debt Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Plugin interface too rigid | Hard to add new benchmark types | Use optional capability mixins |
| Ollama metadata incomplete | Wrong context assumptions | Allow manual model overrides |
| LLM judge unstable | Inconsistent scores | Prefer deterministic metrics where possible |
| Vision payload incompatibility | Plugin failures | Detect capability and skip |
| Code execution unsafe | Security risk | Disabled by default, sandbox later |
| Context search too slow | Poor UX | Limit candidates, cache probes, allow subset runs |
| Reports too large | Hard to inspect | Separate summary report from raw JSONL |
| Latency metrics noisy | Misleading results | Warmup, repetitions, median/p95 metrics |
| Host overload | Crashes/timeouts | Default concurrency 1, timeouts, backoff |

---

## 29. Observability Requirements

Every run should write:

- Structured logs.
- Event JSONL.
- Error JSONL.
- Raw case result JSONL.

Log fields should include:

- Run ID.
- Host.
- Model.
- Plugin.
- Case ID.
- Attempt number.
- Duration.
- Error type.

---

## 30. Coding Agent Instructions

The coding agent should implement the system in milestone order.

Do not start with the web dashboard.

Do not add a database until file-based reporting is stable.

Do not add cloud providers.

Do not add telemetry.

Prefer:

- Small modules.
- Strong typing.
- Pydantic validation.
- Explicit errors.
- Testable pure functions.
- Clear docstrings for complex logic.

When making design decisions, prefer:

1. Correct benchmark results.
2. Reproducibility.
3. Simplicity.
4. Plugin extensibility.
5. UX polish.

If a requirement is ambiguous, choose the option that preserves benchmark validity and plugin extensibility.