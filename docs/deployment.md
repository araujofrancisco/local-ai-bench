# Deployment

LocalAIBench runs as a **single container** that serves both the compiled Astro
frontend and the FastAPI backend. SQLite is stored on a volume so results
survive restarts.

---

## Prerequisites

- Docker Engine ≥ 24
- Docker Compose ≥ 2
- An Ollama host reachable over the network (e.g. `http://host.docker.internal:11434` from the container, or a LAN IP)

---

## Quick start

```bash
git clone <repo-url> local-ai-bench
cd local-ai-bench
docker compose up -d --build
```

Open <http://localhost:8000>.

> Point the config at your Ollama server first — see
> [Configuration](#configuration) below.

---

## docker-compose.yml

```yaml
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=${OLLAMA_HOST:-http://host.docker.internal:11434}
      - DATABASE_URL=/data/benchmark.db
      - CONFIG_PATH=/config/default.yaml
    volumes:
      - ./config:/config
      - ./data:/data
      - ./plugins:/app/plugins
    restart: unless-stopped
```

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `CONFIG_PATH` | `config/default.yaml` | Path to the YAML configuration inside the container. |
| `DATABASE_URL` | `benchmark.db` | Path to the SQLite database file. |
| `STATIC_DIR` | `/app/static` | Where the compiled frontend assets live. |
| `CORS_ORIGINS` | `*` | Comma-separated CORS allow-list. Lock this down in production. |
| `OLLAMA_HOST` | `http://host.docker.internal:11434` | Ollama base URL used when the config file omits the `hosts` section (the recommended default). Point this at the host machine's Ollama. |

### Volumes

| Mount | Contents |
| --- | --- |
| `./config:/config` | Configuration file (`default.yaml`). Mounted read-write; the app only reads it. |
| `./data:/data` | SQLite database (`benchmark.db`) — benchmark results **and** plugin option overrides. |
| `./plugins:/app/plugins` | Local plugin `.py` files. Mounted at the app workdir so the default `local_dir: ./plugins` resolves to it. |

---

## Configuration

The container uses `config/default.yaml` mounted from the host directory.

The recommended starting point **omits** `hosts` entirely — the app then uses
`$OLLAMA_HOST` (set by docker-compose to the host machine's Ollama) or falls
back to the local `http://127.0.0.1:11434`:

```yaml
# hosts omitted -> uses $OLLAMA_HOST / 127.0.0.1:11434
```

To benchmark a specific machine (for example another host on your LAN), list
it explicitly:

```yaml
hosts:
  - name: lab-server
    base_url: http://host.docker.internal:11434   # from inside Docker
    timeout_seconds: 300
```

On Docker Desktop (Windows/macOS) `host.docker.internal` resolves to the host
machine automatically. On Linux Docker Engine add
`extra_hosts: ["host.docker.internal:host-gateway"]` to the service if you need
that name, or simply use your machine's LAN IP.

Models are **not** listed in the config — they are auto-discovered from each
host at run time and selected in the UI or via `--models` flags.

Full reference: see the comments in `config/default.yaml` and the
[config section of the README](../README.md#configuration).

---

## Operations

```bash
docker compose build          # rebuild image (also rebuilds the frontend)
docker compose up -d          # start detached
docker compose down           # stop
docker compose down -v        # stop AND delete the data volume (wipes results + plugin option overrides)
docker compose logs -f app    # follow logs
docker compose restart app    # restart the service
```

Or via the included Makefile:

```bash
make build up logs down clean
```

To update after pulling new code:

```bash
git pull
docker compose up -d --build
```

### Health check

```bash
curl http://localhost:8000/api/health
# {"status":"ok","version":"0.1.0"}
```

### Exposing to other machines

`ports: "8000:8000"` already binds to all interfaces, so other machines on the
LAN can reach the UI at `http://<your-host-ip>:8000`. The frontend calls the
API with relative URLs, so it works regardless of the host address you use to
open it.

For production, consider:

- Restricting `CORS_ORIGINS` to your actual domain.
- Putting a reverse proxy (nginx/caddy) in front for TLS.
- Ensuring the Ollama host is reachable from where the container runs.

---

## Development (run without Docker)

Backend (Python ≥ 3.12):

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
uvicorn local_ai_bench.api.app:app --reload --host 0.0.0.0 --port 8000
```

Frontend (Astro):

```bash
cd web
npm install
npm run dev        # dev server on http://localhost:4321
npm run build      # static build to web/dist
npm run test:pages # jsdom smoke tests against the built pages
```

Note: in local development the backend serves the API only, unless `STATIC_DIR`
points at a built frontend. Use `npm run build` and set
`STATIC_DIR=web/dist` to serve the UI from FastAPI locally:

```bash
STATIC_DIR=web/dist uvicorn local_ai_bench.api.app:app --host 0.0.0.0 --port 8000
```

### Tests

```bash
ruff check src/ tests/
mypy src/
pytest
```
