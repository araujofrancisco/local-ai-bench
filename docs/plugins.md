# Plugins

This is the dedicated reference for LocalAIBench plugins. It covers the plugin
architecture, how options and the judge are wired in, every built-in plugin
(with its dataset, evaluation, and configurable options), and how to author a
local plugin. For a quick pointer on how the plugin *pages* of the UI work,
see [usage.md → Plugins page](usage.md#plugins-page).

---

## 1. What is a plugin?

A plugin is a small Python class that implements the
[`BenchmarkPlugin`](src/local_ai_bench/plugins/base.py) contract. It owns three
things: a set of benchmark **cases**, a way to turn each case into an Ollama
chat **request**, and a way to **score** the resulting response. The runner
discovers plugins, instantiates one model per plugin, and drives the lifecycle
below for every host/model combination.

A plugin is *not* responsible for talking to Ollama directly — the runner's
`OllamaClient` does that. The plugin only says *what to send* and *what it
means*.

### Lifecycle (in run order)

```
supports_model(model)   → skip model if False
prepare(ctx)            → load datasets / warm caches / validate deps (optional)
cases(ctx)              → yield BenchmarkCase fixtures for this plugin
for case in cases:
    for attempt in range(repetitions):
        build_request(case, model, ctx)        → dict payload for Ollama /api/chat
        (runner sends the request, times it)   → ModelResponse
        evaluate(case, response, ctx)          → Evaluation(score, passed, metrics, …)
aggregate(case_results)    → PluginAggregate (summary score + metrics)
teardown(ctx)            → clean up subprocesses/files (optional)
```

Key types (all in [`domain/models.py`](src/local_ai_bench/domain/models.py)):

| Type | Holds |
| --- | --- |
| `BenchmarkCase` | `id`, `plugin_id`, `dataset_version`, `input` (sent to the model), `expected` (what good looks like), `metadata` |
| `ModelResponse` | `text`, `error`, `timing` (total/TTFT/load ms), `tokens` (prompt/completion/tokens-per-second), `truncated`, `done_reason` |
| `Evaluation` | `score` (0–1), `passed`, per-case `metrics` dict, optional `rationale`/`judge_model` |
| `PluginAggregate` | `score`, pass/fail/skipped counts, latency p50/p95, TTFT p50, tokens/s, and a `metrics` dict of plugin-specific rollups |

---

## 2. Discovery & registration

There are two ways plugins enter the registry ([`registry.py`](src/local_ai_bench/plugins/registry.py)):

1. **Built-ins** — `load_builtin_plugins()` registers every plugin in
   [`plugins/builtin/`](src/local_ai_bench/plugins/builtin/).
2. **Local files** — every `.py` in `plugins.local_dir` (default `./plugins`,
   mounted at `/app/plugins` in Docker) is scanned and any
   `BenchmarkPlugin` subclass it defines is registered.

The registry **rejects duplicate ids** (unless explicitly overwritten), so a
local plugin that reuses a built-in id (e.g. `coding`) is rejected and
reported — never a silent foot-gun. Local import errors are collected and shown
by `local-ai-bench doctor`.

### Model compatibility

`supports_model(model)` gates whether a plugin runs for a given model. Built-ins
that override it:

| Plugin | Rule |
| --- | --- |
| `vision` | only models `model.supports_vision == True`; otherwise skipped |
| `long_context` | skipped if the model's advertised `max_context_tokens` is below the smallest probe size |
| `multi_context` | always eligible; sizes are pruned per-case from `max_context_tokens` |
| others | all models (default `True`) |

Unsupported model/plugin combinations are recorded as *skipped*, never errors.

---

## 3. Options & configuration

Options flow through two layers that are merged at run time and are read from
`ctx.options` inside a plugin:

```
plugins.options.<id>   (YAML config defaults)   ──┐
                         +                         ├── ► ctx.options  (api/app.py:_effective_plugin_options)
SQLite plugin_options  (UI/PUT overrides)      ──┘
```

- `plugins.enabled` — which plugin ids the runner will consider.
- `plugins.local_dir` — where extra `.py` plugins are scanned.
- `plugins.compare_default` — the per-plugin *score* columns shown by default
  on the Compare page (leave `[]` to show a column for every enabled plugin).
- `plugins.options.<id>` — default option values used unless overridden.

Overrides saved from the Plugins page are persisted in the SQLite
`plugin_options` table and merged **over** the YAML defaults; the YAML file is
never rewritten. The same effective options are used by both the CLI and the Web
UI.

---

## 4. Shared scoring helpers

Most built-ins reuse deterministic helpers from
[`plugins/score.py`](src/local_ai_bench/plugins/score.py) rather than rolling
their own:

| Helper | Used by | What it does |
| --- | --- | --- |
| `keyword_recall(text, keywords)` | translation, summarization, vision | fraction of expected keywords matched (normalized substring match) |
| `normalize_answer(text)` / `numeric_close(text, expected, tol)` | reasoning, long_context, multi_context | extract a bare number, compare within tolerance or exact string |
| `valid_json_with_fields(raw, required)` | structured_output | parses JSON (with light fence/repair) and scores field coverage 0–1 |
| `python_syntax_ok` / `extract_python` / `symbol_defined` | coding | extract runnable code, check syntax, check a function/class exists |
| `detect_inefficient(source, checks)` | coding | AST anti-pattern scans (`nested_loops`, `in_param_scan_loop`, `linear_list_op_in_loop`) |
| `blend_scores(deterministic, judge, judge_weight)` | coding, translation, summarization | weighted blend of deterministic vs. judge score |

---

## 5. LLM-as-judge

Subjective scoring (fluency, faithfulness, code quality) is delegated to an
optional judge model. The judge is *opt-in*:

- Configure `judge.enabled`/`judge.model`/`judge.temperature` in config.
- Plugins call `judge_evaluation(ctx, case, response, rubric=..., deterministic_score=..., passed=..., pass_threshold=..., metrics=..., judge_weight=...)` (in `judge.py`).
- Composite score = `(1 - judge_weight) * deterministic + judge_weight * judge`.
- **Judge failures never break a run**: a timeout, parse error, or missing judge
  falls back to the deterministic score, and the run continues.

Per-plugin default `judge_weight`:

| Plugin | Default `judge_weight` |
| --- | --- |
| translation | `0.4` |
| summarization | `0.4` |
| coding | `0.0` (disabled by default — set under `plugins.options.coding`) |
| vision | `0.0` (no judge integration; deterministic keyword recall only) |
| others | `0.0` (deterministic only) |

---

## 6. Built-in plugin reference

### 6.1 `smoke`
**Category** `reasoning` · **Modality** `text` · **Dataset** `v1` (2 cases, `smoke_sum_it_0001`, `smoke_triangle_0002`)

Purpose: fast sanity check that a model produces non-empty, well-formed output.

Request: `temperature: 0.0`, `num_predict: 64`.

Evaluation: binary — `1.0` if the response is non-blank, else `0.0`. No judge.

Aggregation: mean of case scores (via the custom `SmokePlugin.aggregate`);
metric `chars` per case.

Options: none.

### 6.2 `reasoning`
**Category** `reasoning` · **Modality** `text` · **Dataset** `v1` (4 cases)

Purpose: basic arithmetic & logic facts (percentages, speed, weekday math, sum
1..10).

Request: `temperature: 0.0`, `num_predict: 256`.

Evaluation: deterministic exact match via `numeric_close` (tolerance 0.5) or
normalized string equality; `score ∈ {0.0, 1.0}`. No judge.

Aggregation: mean score (`BaseTextPlugin`).

Options: none.

### 6.3 `translation`
**Category** `translation` · **Modality** `text` · **Dataset** `v1` (3 cases)

Three fixed language pairs: `en→fr`, `en→es`, `en→de`, each judged by keyword
recall against target-language terms.

Request: `temperature: 0.0`, `num_predict: 128`.

Evaluation: `score = keyword_recall` (0–1); `passed = recall >= 0.5`. When a
judge is configured it is blended at `judge_weight=0.4` with a fluency/faithfulness
rubric.

Aggregation: mean score.

Options: none.

### 6.4 `summarization`
**Category** `summarization` · **Modality** `text` · **Dataset** `v1` (1 case, 1 source doc)

Purpose: recall of key facts from a source document.

Request: `temperature: 0.0`, `num_predict: 160`.

Evaluation: `score = keyword_recall`; `passed = recall >= 0.4`. Judge blended
at `judge_weight=0.4` (faithfulness/conciseness rubric) when configured.

Aggregation: mean score.

Options: none.

### 6.5 `structured_output`
**Category** `structured_output` · **Modality** `text` · **Dataset** `v1` (2 cases)

Purpose: JSON/schema compliance.

Request: `temperature: 0.0`, `num_predict: 512`.

Evaluation: `valid_json_with_fields` strips markdown fences, extracts the first
balanced JSON object/array, and scores the fraction of `required_fields`
present. `passed = score >= 1.0` (all required fields present and valid JSON).
No judge.

Aggregation: mean score.

Options: none.

### 6.6 `vision`
**Category** `vision` · **Modality** `image` · **Dataset** `v1` (1 case)

Purpose: image understanding of a deterministically generated 8-bit grayscale
checkerboard PNG (no external assets).

- `supports_model`: only models reporting `supports_vision == True`.
- Request sends the base64 image in a chat `images` message; `temperature: 0.0`,
  `num_predict: 128`.

Evaluation: `score = keyword_recall` of color/pattern terms; `passed = recall >= 0.25`.
No judge.

Aggregation: mean score.

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `max_image_dimension` | `768` | Cap on the synthesized checkerboard dimension. The image is generated as `min(max_image_dimension, 32)` — values above 32 have no visual effect (the checkerboard is 32×32). |

### 6.7 `coding`
**Category** `coding` · **Modality** `text` · **Dataset** `v3` (33 cases) · **Version** `0.2.0`

Purpose: correctness plus algorithmic complexity of generated Python — the
largest and most discriminating plugin. The `v3` dataset separates strong from
weak coders across strings, arrays, hashing, bit manipulation, trees, linked
lists, graphs, DP, robustness, and stateful classes.

#### How cases are scored

Each case declares an `expected` payload of:

- `function_name` — the symbol that must be defined.
- `tests` — executable assertions, one per item. An item is
  `(code, should_pass)` or `(code, should_pass, opts)`, where `opts` may carry:
  - `kind` — `"assert"` (default) or `"raises"` (the code is expected to raise).
  - `exc` — for `raises` items, the exception class name that *must* be raised
    (e.g. `ValueError`); any other outcome fails the check.
  - `should_pass` — for `assert` items, `True` means the code *should* run
    successfully, `False` means it should fail.
  - `weight` — partial-credit weight (default `1.0`).
- `perf` *(optional)* — relative big-O scale checks. Each defines a `small` and
  `large` probe expression plus a `ratio` bound; the harness times the solution
  on both (warmup + measure) inside one isolated subprocess and requires
  `large_ms / small_ms < ratio`. An O(n²) answer that passes the small
  assertions is still penalized.
- `approach` *(optional)* — names of static AST anti-pattern checks (see below)
  applied to the generated source; used to penalize brute-force solutions that
  nonetheless pass.
- `harness` *(optional)* — auxiliary fixture source (e.g. `TreeNode`/`ListNode`
  class definitions) prepended so tree/linked-list cases don't depend on the
  model defining them.

The case score is the **weight-fraction** of passing items:

```
score = passed_weight / total_weight
```

A `1.0` score is required for `passed=True` (used for `pass@1`).

#### Static approach penalty

Run only when the case carries an `approach` list and the solution passes every
executable check. `detect_inefficient` flags:

| Detector | Flags |
| --- | --- |
| `nested_loops` | an O(n²)-style loop nested inside another loop |
| `in_param_scan_loop` | a loop that membership-scans a *parameter* (`x in some_arg`), i.e. linear scan of raw input |
| `linear_list_op_in_loop` | O(n) list method calls (`index`/`count`/`remove`/`find`/`pop(0)`) inside a loop |

If any detector fires, `approach_penalty` (default `0.1`) is deducted from the
case score and `approach_penalty_applied` is set in metrics. Correct hash-based
solutions are never flagged (only loops scanning the raw parameter).

#### Execution safety

Generated code runs in isolated subprocesses (`python -I`, no user site-packages,
no env overrides) in a temp directory under a per-case time budget
(`timeout_seconds`). The perf probes *only* ever evaluate the case's own literal
probe expressions — never arbitrary model output — via `eval` of known literals.

#### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `execute_code` | `true` | When `false`, evaluation is static only (syntax + required function defined); no tests run. |
| `timeout_seconds` | `30` | Per-case (and per-perf-probe) subprocess timeout. |
| `enable_perf` | `true` | Run the relative big-O perf checks for cases that declare `perf`. |
| `perf_ratio_default` | `6.0` | Default `large/small` ratio bound used by perf checks (a per-check `ratio` overrides). |
| `approach_penalty` | `0.1` | Score deducted when a passing solution is flagged by an `approach` detector. |
| `judge_weight` | `0.0` | LLM-as-judge code-quality blend weight (`0` = disabled). |

When `judge_weight > 0` and `judge.enabled` is set, the judge blends a
code-quality score (correctness + efficient algorithm) into each case; see §5.

#### Metrics (what you see)

Per-case metrics on `Evaluation`:

`execute_code`, `syntax_ok`, `syntax_error`, `symbol_defined`, `symbol_not_found`,
`tests_passed`, `tests_total`, `stderr` (tail when a snippet errors),
`perf_checked`, `perf_ok`, `worst_perf_ratio`, `approach_flagged`,
`approach_penalty_applied`, and (when a judge runs) `judge_score`/`judge_rationale`/`judge_weight`.

Aggregate metrics, surfaced in the Compare page score-column tooltip:

| Metric | Meaning |
| --- | --- |
| `pass_at_1` | fraction of cases at score 1.0 |
| `syntax_ok_ratio` | cases with valid syntax |
| `tests_passed_total` / `tests_total` | aggregate test pass count / total |
| `tests_pass_ratio` | `passed/total` |
| `perf_checked_cases` | cases that ran perf checks |
| `complexity_ok_ratio` | perf checks that passed |
| `approach_penalized_cases` | cases where the approach penalty fired |
| `judge_scored_cases` | cases with a non-null judge score |

### 6.8 `long_context`
**Category** `long_context` · **Modality** `text` · **Dataset** `v1` (4 context sizes: 256, 1024, 4096, 16384)

Purpose: fact recall (continents = 7) degraded as filler context grows.

Request: `temperature: 0.0`, `num_predict: 64`; `num_ctx = target_context * 2`.

Evaluation: exact match via `numeric_close`; `score ∈ {0, 1}` with `truncated`
flag in per-case metrics.

Aggregation: mean score, plus `per_context_score` (`{context_size: score}`) and
`max_context_tokens` (model-derived).

Options: none (reads model `max_context_tokens` from the merged options).

### 6.9 `multi_context`
**Category** `long_context` · **Modality** `text` · **Dataset** `v1` (5 configurable sizes)

A more general context-sweep than `long_context`: the question, expected answer,
and probe sizes are all configurable. Filler text is prepended so the prompt
actually fills the window.

Request: `temperature: 0.0`, `num_predict: 64`; `num_ctx = target_context * 2`.

Evaluation: substring (`contains=true`) or numeric match; `score ∈ {0, 1}`.

Aggregation: mean score, plus `per_context_score` and `max_context_tokens`.

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `prompt` | `"What is the capital of France?"` | question sent to the model |
| `expected` | `"paris"` | expected substring (or numeric, see `contains`) |
| `context_sizes` | `[512, 1024, 4096, 8192, 16384]` | window sizes to probe |
| `contains` | `true` | `true` = substring match; `false` = numeric compare on `expected` |

Sizes beyond the model's `max_context_tokens` are pruned per-case.

---

## 7. Example local plugin

Drop a `.py` file into `plugins.local_dir` and it is picked up automatically.
See [`plugins/keyword.py`](plugins/keyword.py) for a complete working example —
it is also registered by the default config so you can inspect its behavior end
to end.

Minimal template:

```python
from collections.abc import Iterable
from typing import Any, ClassVar

from local_ai_bench.domain.models import (
    BenchmarkCase, BenchmarkCategory, Evaluation, Modality, ModelInfo, ModelResponse,
)
from local_ai_bench.plugins.base import BenchmarkPlugin, RunContext
from local_ai_bench.plugins.score import keyword_recall


class KeywordPlugin(BenchmarkPlugin):
    id: ClassVar[str] = "keyword"
    name: ClassVar[str] = "Keyword presence"
    description: ClassVar[str] = "Outputs that include all required keywords."
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.REASONING
    dataset_version: ClassVar[str] = "v1"
    modalities: ClassVar[set[Modality]] = {Modality.TEXT}

    def supports_model(self, model: ModelInfo) -> bool:
        return True

    def cases(self, ctx: RunContext) -> Iterable[BenchmarkCase]:
        yield BenchmarkCase(
            id="kw_hello_0001", plugin_id=self.id, dataset_version=self.dataset_version,
            input={"prompt": "Say exactly: hello world"},
            expected={"keywords": ["hello", "world"]},
        )

    def build_request(self, case, model, ctx) -> dict[str, Any]:
        return {
            "messages": [{"role": "user", "content": case.input["prompt"]}],
            "options": {"temperature": 0.0, "num_predict": 64},
        }

    async def evaluate(self, case, response, ctx) -> Evaluation:
        recall = keyword_recall(response.text, case.expected["keywords"])
        return Evaluation(score=recall, passed=recall == 1.0,
                          metrics={"keyword_recall": recall})
```

Guidelines for local plugins:

- `id` must be unique — clashing with a built-in id is rejected and reported by
  `local-ai-bench doctor`.
- Read options from `ctx.options` (they're the merged config + DB overrides).
- Use the helpers in `plugins.score` instead of re-implementing fuzzy/numeric match.
- Call `judge_evaluation(...)` if subjective scoring helps; it degrades safely.
- Failures inside `evaluate`/`build_request` are isolated: the case is recorded
  as failed and the run continues (see `troubleshooting.md`).
- `cases()` may be a generator; the runner consumes it eagerly.
- Prefer deterministic, small datasets so reports stay reproducible.

---

## 8. Enabling / disabling plugins

```yaml
plugins:
  enabled: [smoke, coding, vision]   # only these run
  local_dir: ./plugins
  compare_default: [coding, vision] # which score columns show by default on Compare
```

`local-ai-bench plugins` (CLI) or the Plugins page (Web UI) lists what is
discovered. The Plugins page also lets you edit a plugin's options in place —
those edits persist in SQLite and override `plugins.options` for the next run.
