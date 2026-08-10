# Usage

OllamaBench has two interfaces:

- **CLI** (`ollama-bench`) — run benchmarks from the terminal, generate reports, inspect history.
- **Web UI** — dashboard, run/compare/history/plugins pages served by the FastAPI backend.

Both use the same configuration file and the same SQLite results database.

---

## Command-line interface

Install the package (see [deployment](deployment.md)), then:

```bash
ollama-bench --help
```

### `init`

Write a starter `config.yaml` into the current directory.

```bash
ollama-bench init --config config.yaml
```

### `doctor`

Validate the configuration, host reachability, model discovery, output
directory, and plugin loading. Run this first when something is wrong.

```bash
ollama-bench doctor
```

### `models`

List models discovered on the configured hosts.

```bash
ollama-bench models
```

### `plugins`

List the benchmark plugins that are available (built-in + local).

```bash
ollama-bench plugins
```

### `run`

Run a benchmark against the configured hosts with the enabled plugins. Models
are auto-discovered; select them with flags:

```bash
ollama-bench run                                 # all discovered models
ollama-bench run --models 'qwen*'                # glob match
ollama-bench run --models llama3.2:latest,qwen2.5-coder:14b
ollama-bench run --exclude '*:0.8b'              # skip some models
ollama-bench run --interactive                   # pick interactively
ollama-bench run --db benchmark.db               # save results to SQLite
```

### `run-single`

Run **all enabled plugins** for a single model and save results to SQLite and
files. Useful for deep-diving one model.

```bash
ollama-bench run-single --db benchmark.db
```

### `report`

List, view, or open generated reports (JSON / Markdown / HTML per the
`reporting.formats` config).

```bash
ollama-bench report               # newest run
ollama-bench report --run <run_id>
```

### `history`

Show benchmark history from SQLite.

```bash
ollama-bench history --db benchmark.db
ollama-bench history --db benchmark.db --model llama3.2:latest
```

### `compare`

Compare all models across runs (optionally limited to one run).

```bash
ollama-bench compare --db benchmark.db
ollama-bench compare --db benchmark.db --run <run_id>
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
| Compare | `/compare` | Side-by-side model comparison (`?run=<id>` for one run, with a delete action) |
| History | `/history` | All runs, with multi-criteria filtering, running-status chips, and single/bulk deletion |
| Plugins | `/plugins` | Plugin details and option editing |

### Run page

1. Pick one or more models (Ctrl/Cmd-click to multi-select).
2. Tick the plugins you want to run.
3. Set repetitions, warmup runs, and max retries, then **Start Benchmark**.

The benchmark runs **in the background** on the server (a detached task) so you
can navigate away freely. Progress is streamed over a per-run WebSocket
(`/ws?run_id=<id>`) with a status-polling fallback. The run also appears in the
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

| Plugin | Option | Default | Effect |
| --- | --- | --- | --- |
| `coding` | `execute_code` | `false` | When `true`, generated code is actually executed in a subprocess against the unit tests. When `false`, evaluation is static only (syntax + function-defined). |
| `coding` | `timeout_seconds` | `30` | Subprocess timeout used when `execute_code` is enabled. |
| `vision` | `max_image_dimension` | `768` | Caps the size of the synthetic checkerboard image sent to the model. |

Any local plugin can read these per-run values from `ctx.options`.

---

## Writing a local plugin

Drop a `.py` file into the directory named by `plugins.local_dir` (default
`./plugins`) and it is picked up automatically — no core changes required.
Each file should define a class subclassing `BenchmarkPlugin`:

```python
from ollama_bench.plugins.base import BenchmarkPlugin, RunContext
from ollama_bench.domain.models import (
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
| GET | `/api/plugins` | Plugins with description, dataset version, modalities, effective options |
| GET | `/api/plugins/{id}` | Plugin detail incl. `source_file` and base64-encoded `source` |
| PUT | `/api/plugins/{id}/options` | Persist option overrides for a plugin |
| POST | `/api/benchmarks/run` | Start a benchmark run (runs in the background) |
| GET | `/api/benchmarks` | List persisted runs (`search`, `model`, `host`, `date_from`, `date_to`) |
| GET | `/api/benchmarks/active` | Live pending/running runs (in-memory; not persisted) |
| GET | `/api/benchmarks/{run_id}` | Run metadata + per-model results |
| GET | `/api/benchmarks/{run_id}/status` | Live progress/status (retained briefly after completion) |
| DELETE | `/api/benchmarks/{run_id}` | Delete a run (cascades all its data). **409** if the run is still active |
| POST | `/api/benchmarks/delete` | Batch delete (body `{"run_ids":[...]}`). **409** if any id is still active |
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
