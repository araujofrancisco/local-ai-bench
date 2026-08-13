# Usage

LocalAIBench has two interfaces:

- **CLI** (`local-ai-bench`) — run benchmarks from the terminal, generate reports, inspect history.
- **Web UI** — dashboard, run/compare/history/plugins pages served by the FastAPI backend.

Both use the same configuration file and the same SQLite results database.

---

## Command-line interface

Install the package (see [deployment](deployment.md)), then:

```bash
local-ai-bench --help
```

### `init`

Write a starter `config.yaml` into the current directory.

```bash
local-ai-bench init --config config.yaml
```

### `doctor`

Validate the configuration, host reachability, model discovery, output
directory, and plugin loading. Run this first when something is wrong.

```bash
local-ai-bench doctor
local-ai-bench doctor --config config.yaml   # validate a specific config
```

### `models`

List models discovered on the configured hosts.

```bash
local-ai-bench models
local-ai-bench models --config config.yaml
```

### `plugins`

List the benchmark plugins that are available (built-in + local).

```bash
local-ai-bench plugins
local-ai-bench plugins --config config.yaml
```

### `run`

Run a benchmark against the configured hosts with the enabled plugins. Models
are auto-discovered; select them with flags:

```bash
local-ai-bench run                                 # all discovered models
local-ai-bench run --config config.yaml            # use a specific config
local-ai-bench run --models 'qwen*'                # glob match
local-ai-bench run --models llama3.2:latest,qwen2.5-coder:14b
local-ai-bench run --exclude '*:0.8b'              # skip some models
local-ai-bench run --interactive                   # pick interactively
local-ai-bench run --db benchmark.db               # save results to SQLite
```

### `run-single`

Run **all enabled plugins** for a single model and save results to SQLite and
files. Useful for deep-diving one model. The model name is a required positional
argument; it must match a discovered model exactly.

```bash
local-ai-bench run-single llama3.2:latest
local-ai-bench run-single llama3.2:latest --db benchmark.db
local-ai-bench run-single llama3.2:latest --config config.yaml
```

### `report`

List, view, or open generated reports (JSON / Markdown / HTML per the
`reporting.formats` config). The optional `action` argument is one of
`list` (default), `view`, or `open`.

```bash
local-ai-bench report                 # list newest run
local-ai-bench report list            # list all runs
local-ai-bench report view            # view the newest report
local-ai-bench report open --run <run_id>   # open a specific report
local-ai-bench report --config config.yaml
```

### `history`

Show benchmark history from SQLite.

```bash
local-ai-bench history --db benchmark.db
local-ai-bench history --db benchmark.db --model llama3.2:latest
```

### `compare`

Compare all models across runs (optionally limited to one run).

```bash
local-ai-bench compare --db benchmark.db
local-ai-bench compare --db benchmark.db --run <run_id>
```

### `version`

Print the installed version.

---

## Web UI

The UI is served at the backend root (default `http://localhost:8000`).

| Page | URL | Purpose |
| --- | --- | --- |
| Dashboard | `/` | Model overview, recent runs (with delete), and a live **Active Runs** panel |
| Run | `/run` | Select models + plugins and start a benchmark; live progress + resume |
| Compare | `/compare` | Side-by-side multi-run comparison. Pick runs from History → **Compare selected**, or visit `/compare?run=A&run=B`. Sort any column, show/hide columns, and expand per-plugin score/latency columns. |
| History | `/history` | All runs, with multi-criteria filtering, running-status chips, and single/bulk deletion |
| Plugins | `/plugins` | Plugin details, source viewer, and option editing |

### Run page

1. Pick one or more models (Ctrl/Cmd-click to multi-select).
2. Tick the plugins you want to run.
3. Set repetitions, warmup runs, and max retries, then **Start Benchmark**.

The benchmark runs **in the background** on the server (a detached task) so you
can navigate away freely. Progress is streamed over a per-run WebSocket
(`/ws?run_id=<id>`) with a status-polling fallback. The progress bar shows the
**planned total** number of cases up front — once the host is discovered and
the run plan is computed, it displays `Completed X of Y cases` where `Y` stays
fixed for the rest of the run (no more "total" climbing alongside completions).
The run also appears in the
Dashboard **Active Runs** panel and History (with a running/queued status chip)
from any page.

If you return to `/run`, the active run is **resumed automatically** — the page
reconnects and keeps showing progress, or offers a "View results" link if it
already finished. You can also open a specific run directly via `/run?run=<id>`
(e.g. the "Track →" link on the dashboard). Resume relies on `localStorage` and
the in-memory run status: a container restart clears in-memory state, but
completed runs remain in the database.

### Dashboard

- **Active Runs** shows every run that is currently pending/running, polled
  live from the server. Use **Track →** to open the run page for any of them.
- **Recent Benchmarks** lists the latest persisted runs; use **View** to open the
  comparison or **Delete** to remove a run (and all its case data) after a
  confirmation modal.

### History page

- **Filter** runs by free-text run-ID search, model, host, and a from/to date
  range. Filters combine with AND logic.
- **Status chips**: runs that are still running or queued (sourced from the live
  Active Runs list) appear at the top with a **Track** link; persisted runs show
  **Compare** and **Delete**.
- **Delete a single run** with its row's Delete button (confirmation modal).
- **Bulk delete**: tick the checkbox per row (or the header checkbox to select
  all visible), then **Delete Selected**. Active runs cannot be selected.
- A toast confirms each successful deletion; errors are shown inline as a toast.

To compare runs, select one or more completed rows with the checkboxes and click
**Compare selected**. This opens `/compare?run=<r1>&run=<r2>…` (active runs are
excluded from selection).

### Compare page

`/compare` shows a sortable, customizable table:

- **Sort** — click any numeric column header to sort ascending/descending (the
  active column shows `▲/▼`). Sort choice is remembered between visits.
- **Columns** — click the **Columns** panel to toggle individual columns on/off.
  Columns include general metrics (Model, Score, p50, p95, TTFT, Tokens/s, Cases,
  Errors) and **a score column per plugin** that ran in the selected runs. Toggle
  extra per-plugin latency/TTFT/throughput columns (e.g. `smoke p50`,
  `translation Tokens/s`). Your column choices persist across reloads.
- **Per-plugin columns by default** — which plugin score columns appear by default
  is controlled by `plugins.compare_default` in `config/default.yaml`. Leave it
  empty (`[]`) to show a score column for every enabled plugin; list specific ids
  to limit the default set (e.g. `compare_default: [translation, coding]`).
- **Per-case errors** — click **Show per-case errors** to expand a table listing
  each failed case (model, plugin, case id, and the error message) for the
  selected run(s). The same detail is persisted in SQLite and available via
  `GET /api/benchmarks/{run_id}/cases`. Tool/transport failures and
  `evaluate()` exceptions both populate the `error` field.
- **Category weights** — click the **⚖ Weights** button to open the category
  weights editor. Adjust how much each category contributes to the overall
  score: the **Weighted** column recomputes every model's overall score live
  from its per-plugin scores. **Save weights** persists overrides (via
  `PUT /api/weights`) so future runs use them; **Reset to defaults** clears
  overrides. Weights default to `1.0` for every category. Categories are
  normalized so that `agent_tool_use` shares the `function_calling` weight and
  `multi_context` shares the `long_context` weight; unlisted categories fall
  back to `1.0`.

### Plugins page

Each plugin card shows:

- Name, category, version, dataset version, and modality chips
- Description (built-ins include one)
- A **Code** button that views the plugin's source file in a modal
  (with a `.py` download link)
- An **Options** editor (expandable) where you can change the plugin's
  configurable settings and press **Save**.

Saved options are stored in the SQLite database (`plugin_options` table) and
**merged over** the values in `config/default.yaml` at run time — your YAML
file is never modified. Deleting the database (e.g. `docker compose down -v`)
resets all option overrides.

#### Plugin options that affect behavior

Per-plugin option defaults and their effects are documented in
[plugins.md](plugins.md). Built-ins that read options at run time include
`coding` (`execute_code`, `timeout_seconds`, `enable_perf`, `perf_ratio_default`,
`approach_penalty`, `judge_weight`), `vision` (`max_image_dimension`),
`multi_context` (`prompt`, `expected`, `context_sizes`, `contains`, `filler`),
`long_context` (`max_context_tokens`), `rag` (`hallucination_penalty`),
`function_calling` (`arg_tolerance`), and `agent_tool_use` (`followup_prompts`);
local plugins can read any value from `ctx.options`.

---

## Writing a local plugin

This section is a quick overview; see the full
[plugins.md](plugins.md) guide for the complete template and authoring
guidelines.

Drop a `.py` file into the directory named by `plugins.local_dir` (default
`./plugins`) and it is picked up automatically — no core changes required. Each
file defines a class subclassing `BenchmarkPlugin`:

```python
from local_ai_bench.plugins.base import BenchmarkPlugin, RunContext
from local_ai_bench.domain.models import (
    BenchmarkCase, Evaluation, ModelInfo, ModelResponse,
)

class KeywordPlugin(BenchmarkPlugin):
    id = "keyword"             # unique; referenced in plugins.enabled
    name = "Keyword presence"  # shown in UIs/reports

    def cases(self, ctx: RunContext):          # yield BenchmarkCase fixtures
        ...

    def build_request(self, case, model, ctx): # dict sent to /api/chat
        ...

    async def evaluate(self, case, response, ctx):  # return Evaluation(score, passed)
        ...
```

Then enable it:

```yaml
plugins:
  enabled: [smoke, keyword]
  local_dir: ./plugins
```

See `plugins/keyword.py` in the repository for a complete working example.
Plugins that register a duplicate `id` are rejected and reported.

---

## API endpoints

Base URL: `http://<host>:8000`. Interactive OpenAPI docs at `/docs`.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/models` | Discovered models |
| GET | `/api/plugins` | Plugins with description, dataset version, modalities, effective options, and `compare_default` (ids shown by default on Compare) |
| GET | `/api/plugins/{id}` | Plugin detail incl. `source_file` and base64-encoded `source` |
| PUT | `/api/plugins/{id}/options` | Persist option overrides for a plugin |
| GET | `/api/weights` | Category weights: `defaults`, persisted `overrides`, merged `effective` |
| PUT | `/api/weights` | Persist weight overrides (body `{"weights":{"coding":2.5}}`); values equal to defaults are pruned |
| POST | `/api/benchmarks/run` | Start a benchmark run (runs in the background) |
| GET | `/api/benchmarks` | List persisted runs (`search`, `model`, `host`, `date_from`, `date_to`) |
| GET | `/api/benchmarks/active` | Live pending/running runs (in-memory; not persisted) |
| GET | `/api/benchmarks/{run_id}` | Run metadata + per-model results |
| GET | `/api/benchmarks/{run_id}/status` | Live progress/status (retained briefly after completion) |
| GET | `/api/benchmarks/{run_id}/cases` | Per-case rows for a run, including error text (transport + evaluate failures) |
| DELETE | `/api/benchmarks/{run_id}` | Delete a run (cascades all its data). **409** if the run is still active |
| POST, DELETE | `/api/benchmarks/delete` | Batch delete (body `{"run_ids":[...]}`). **409** if any id is still active |
| GET | `/api/compare` | Compare models (`?run=<id>` optional) |
| GET | `/api/history` | History with the same filters + `filters` meta for dropdowns |
| GET | `/api/export/{run_id}.{json,csv,md}` | Export a run |
| GET | `/api/health` | Health check |
| WS | `/ws?run_id=<id>` | Live progress broadcasts scoped to one run |

Live run status is held **in-memory** only. It is exposed via
`GET /api/benchmarks/active` and `GET /api/benchmarks/{run_id}/status`, and
pushed over the WebSocket. Terminal (completed/failed) statuses are retained
briefly (default ~1 hour, max ~200 entries) to let clients resume a
just-finished run, then evicted to bound memory usage. A container restart
clears all in-memory state; persisted run results are unaffected.
