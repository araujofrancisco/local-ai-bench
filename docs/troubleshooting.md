# Troubleshooting

> First stop: `ollama-bench doctor` (CLI) or `GET /api/health` (web) — they
> surface the most common problems (host reachability, model discovery, plugin
> loading).

---

## Cannot connect to Ollama

**Symptom:** dashboard shows "No models discovered" / the run page is empty; `ollama-bench models` errors.

Checks, in order:

1. Verify the host from **inside the container**:
   ```bash
   curl -fsS http://host.docker.internal:11434/api/version
   ```
2. The connection comes from `hosts` in `config/default.yaml`, **not** the
   `OLLAMA_HOST` env var (that is only a reference). Confirm the `base_url` is
   reachable from the container's network, not just from your host shell.
3. On Linux Docker Engine, `host.docker.internal` may not resolve. Add to the
   service:
   ```yaml
   extra_hosts:
     - "host.docker.internal:host-gateway"
   ```
   or use a LAN IP directly: `base_url: http://192.168.x.x:11434`.
4. Ollama must be listening on `0.0.0.0` (not `127.0.0.1`) to accept remote
   connections: `OLLAMA_HOST=0.0.0.0 ollama serve`.
5. Firewalls / AP isolation on the network can block the port — test with the
   bundled helper:
   ```bash
   scripts/host-check.sh http://<host>:11434 <model>
   ```

## Old / blank pages after an update

**Symptom:** you rebuilt the container but the UI still shows old behavior or
blank error boxes.

The frontend is client-side rendered; browsers cache the old static assets.

- Hard refresh: **Ctrl/Cmd + Shift + R**.
- Or restart the container to be sure the new build is served:
  ```bash
  docker compose up -d --build
  ```

## Frontend build fails during `docker compose build`

**Symptom:** the Node builder stage errors, usually `esbuild`-related.

esbuild needs native build tools in the builder image. The Dockerfile installs
them:

```dockerfile
RUN apk add --no-cache python3 make g++
```

If you removed or changed that line, restore it, or build the frontend
separately (`cd web && npm install && npm run build`) and copy `web/dist`.

## Database is not persisting / resets

- Make sure the `./data` directory exists and is writable by the container.
- `docker compose down -v` **deletes the volume** — that intentionally wipes
  the SQLite DB (benchmark results *and* plugin option overrides).
- If the DB path changed, confirm `DATABASE_URL` and the `./data:/data` mount
  agree.

## Plugin option edits have no effect / lost

- Option overrides live in SQLite (`plugin_options` table), not the YAML.
  Saving via the Plugins page writes to the DB, merged over config defaults at
  run time.
- Deleting the DB (`down -v`) resets all overrides.
- Only `coding` (`execute_code`, `timeout_seconds`) and `vision`
  (`max_image_dimension`) currently use options. Options for other plugins are
  stored but have no effect unless the plugin reads `ctx.options`.

## Plugin does not show up

- Confirm the file is in `plugins.local_dir` and enabled in `plugins.enabled`.
- The registry rejects plugins with a **duplicate `id`** — check for an id
  clash with a built-in (e.g. don't name a local plugin `coding`).
- Local plugins run under the same Python environment; an import error in the
  file is reported (see `ollama-bench doctor` output) and the file is skipped.

## Run status disappears after a container restart

Live run status is kept **in memory**. A restart clears it, so the Run page
can no longer resume a run that was in progress. Completed runs are written to
SQLite and remain visible in History/Compare.

## Live progress updates don't stream (no WebSocket)

The Run page falls back to polling `/api/benchmarks/{id}/status` every ~2.5 s,
so progress still shows — just less granular. Common causes:

- A reverse proxy in front of the app not forwarding WS upgrades (`/ws`).
- The browser blocking `ws://` (unlikely on `http://` sites; make sure the page
  is not opened over `https://` while the API is plain `http://`).

## The coding plugin "executes code"

With `coding.execute_code: true`, generated code is executed in a subprocess
against the unit tests. This is inherent to a coding benchmark, but it does run
model output as code:

- It runs in a throwaway temp file with a timeout (`timeout_seconds`).
- Prefer `execute_code: false` (the default) when benchmarking untrusted
  models, or run the container in a sandboxed/isolated environment.

## Port conflicts

If port 8000 is taken:

```bash
docker compose down          # free the previous instance
# or change the mapping in docker-compose.yml:
#   - "8080:8000"
```

## CORS errors

The backend allows all origins by default (`CORS_ORIGINS=*`) for simplicity.
If you serve the UI from a different origin in production, set:

```yaml
environment:
  - CORS_ORIGINS=https://bench.example.com
```

## Benchmark is very slow

- Lower `runner.repetitions` (default 3) and `runner.warmup_runs` (default 1).
- `context_optimization` runs many candidate context sizes — disable it
  (`context_optimization.enabled: false`) if you don't need context tuning.
- Each plugin adds cases; deselect plugins you don't need on the Run page.
- First call to a model may pull/load it into memory (see `scripts/host-check.sh`).

## Get more logs

```bash
docker compose logs -f app
```

Or reproduce with the CLI for clearer output:

```bash
ollama-bench doctor
ollama-bench run --models '<one-model>' --interactive
```
