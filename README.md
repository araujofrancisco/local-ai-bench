# OllamaBench

Local-first, plugin-based LLM benchmarking for [Ollama](https://ollama.com)
hosts on your own network.

OllamaBench discovers models from one or more Ollama hosts, runs configurable
benchmark plugins against them, scores the results, and stores everything in
SQLite for comparison. It ships with both a terminal CLI and a web UI
(FastAPI + Astro) that you can deploy in a single Docker container.

> **Looking for the design notes / roadmap?** See [`PLAN.md`](PLAN.md).

---

## Features

- **Plugin-based benchmarks** — smoke, reasoning, translation, summarization,
  structured output, coding, vision, and long-context out of the box, plus
  drop-in local plugins (`.py` files, no core changes).
- **Model discovery** — models are auto-detected from each configured host and
  selected with globs or interactively.
- **Web UI** — dashboard (with a live **Active Runs** panel), run (with live
  WebSocket progress), comparison, history, and plugin management.
- **Background runs** — benchmarks run as server-side detached tasks; see their
  live status from any page via the dashboard Active Runs panel and the History
  running/queued chips (`GET /api/benchmarks/active`).
- **Run resume** — navigate away from the Run page mid-benchmark and come back;
  the active run's status is restored automatically. Open `/run?run=<id>` to
  track a specific run from anywhere.
- **Plugin options** — view plugin details and edit their options in the UI;
  settings persist in SQLite and are merged over config defaults at run time.
- **View plugin source** — each plugin card has a **Code** button that fetches the
  plugin's source file and shows it in a modal, with a download link for the `.py`
  (via `GET /api/plugins/{id}`).
- **History filters + deletion** — filter runs by model, host, date range, and
  run-ID search; delete single runs or select multiple for **bulk delete**
  (active runs are protected with a 409). See running runs with status chips.
- **Reports & exports** — JSON/Markdown/HTML reports and per-run
  `json|csv|md` exports.
- **Per-case error logging** — failed cases (transport + evaluation) are captured
  in SQLite with full error text; view them via the Compare page's "Show per-case
  errors" panel or `GET /api/benchmarks/{id}/cases`.
- **Multi-run comparison** — select multiple runs from History → **Compare
  selected**; the Compare page shows a unified table with a Run column, per-plugin
  score columns, click-to-sort, and column show/hide toggles.
- **Configurable default plugin columns** — `plugins.compare_default` controls
  which per-plugin score columns appear by default.
- **Retries and error isolation** — a failed case, plugin, or host never aborts
  the whole run.

---

## Quick start (Docker)

```bash
git clone <repo-url> ollama-bench
cd ollama-bench
# point config/default.yaml at your Ollama host first
docker compose up -d --build
```

Open <http://localhost:8000>.

See [docs/deployment.md](docs/deployment.md) for details, including the
`host.docker.internal` / Linux-host note.

---

## Quick start (CLI)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
ollama-bench init
ollama-bench doctor          # verify host + plugins
ollama-bench run --models 'qwen*'
```

`ollama-bench --help` lists every command. Full walkthroughs:
[docs/usage.md](docs/usage.md).

---

## Architecture

```
┌──────────────────────────  Single container  ─────────────────────────┐
│                                                                        │
│   Browser ──▶ FastAPI ──▶ Astro static assets (compiled at build)      │
│                 │  ├─ /api/* (JSON, incl. /api/benchmarks/active)      │
│                 │  └─ /ws?run_id=<id>   (live progress, per-run)      │
│                 ▼                                                      │
│        SQLite (/data/benchmark.db)          Ollama host(s)             │
│        + plugin options overrides          (HTTP /api/chat, /api/tags) │
└────────────────────────────────────────────────────────────────────────┘
```

- **Backend**: FastAPI (`src/ollama_bench/api/app.py`) — API, WebSocket
  progress, static serving.
- **Runner**: `RunOrchestrator` drives hosts/models/plugins with events,
  retries, and weighted scoring (`src/ollama_bench/runner/orchestrator.py`).
- **Plugins**: registry + built-ins (`src/ollama_bench/plugins/`), local
  plugins scanned from `plugins/`.
- **Storage**: SQLite via `BenchmarkRepository` (`src/ollama_bench/storage/repository.py`).
- **Frontend**: Astro pages in `web/src/pages/`, client-side rendering that
  talks to the API.

---

## Configuration

Single YAML file (see `config/default.yaml`; `ollama-bench init` writes a
copy). The only required setting is the Ollama host:

```yaml
hosts:
  - name: lab-server
    base_url: http://host.docker.internal:11434
    timeout_seconds: 300
```

Other sections:

| Section | What it controls |
| --- | --- |
| `app` | Name, report output dir, log level |
| `hosts` | Ollama endpoints to benchmark |
| `plugins` | `enabled` ids, `local_dir` for local plugins, `options` defaults |
| `runner` | Repetitions, warmups, temperature, seed, retries, concurrency |
| `judge` | Optional judge model for subjective scoring |
| `context_optimization` | Candidate context sizes for long-context tuning |
| `reporting` | Report formats, whether raw cases are included |
| `weights` | Per-category weight when computing overall scores |

Models are **not** configured here — they are discovered from each host and
selected at run time.

---

## Web UI pages

| Page | URL | Notes |
| --- | --- | --- |
| Dashboard | `/` | Model cards, stats, recent runs (delete here too), live Active Runs panel |
| Run | `/run` | Start benchmarks; live progress; resumes active runs (`?run=<id>` to track) |
| Compare | `/compare` | Side-by-side performance (`?run=<id>` for one run, with a delete action) |
| History | `/history` | Filter by model/host/date/search; running-status chips; single + bulk delete |
| Plugins | `/plugins` | Details, modalities, and an options editor |

---

## Documentation

- **[docs/usage.md](docs/usage.md)** — CLI commands, Web UI walkthrough,
  writing local plugins, API reference.
- **[docs/deployment.md](docs/deployment.md)** — Docker compose, environment
  variables, volumes, operations, local development.
- **[docs/troubleshooting.md](docs/troubleshooting.md)** — common issues and
  fixes.

Interactive API docs are served at `http://localhost:8000/docs`.

---

## Development

```bash
pip install -e ".[dev]"          # backend deps + ruff/mypy/pytest
uvicorn ollama_bench.api.app:app --reload --host 0.0.0.0 --port 8000

cd web && npm install            # frontend
npm run dev                      # Astro dev server on http://localhost:4321
```

Quality gates:

```bash
ruff check src/ tests/
mypy src/
pytest
cd web && npm run build && npm run test:pages   # build + jsdom page smoke tests
```

See [docs/deployment.md#development](docs/deployment.md#development-run-without-docker).

