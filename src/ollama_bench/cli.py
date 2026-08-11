"""CLI entry point for ollama-bench (Milestone 0)."""

from __future__ import annotations

import asyncio
import json
import sys
import webbrowser
from collections.abc import Callable
from pathlib import Path

import rich
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ollama_bench import __version__
from ollama_bench.config import BenchmarkConfig, config_hash, load_config, write_default_config
from ollama_bench.domain.events import Event
from ollama_bench.domain.models import ModelInfo
from ollama_bench.ollama.client import OllamaClient
from ollama_bench.ollama.discovery import discover_models
from ollama_bench.plugins import load_plugins
from ollama_bench.plugins.base import BenchmarkPlugin
from ollama_bench.plugins.registry import PluginRegistry
from ollama_bench.reporting.repository import write_report
from ollama_bench.runner.orchestrator import RunOrchestrator
from ollama_bench.selection import filter_models, pick_interactive, split_patterns
from ollama_bench.storage.repository import BenchmarkRepository
from ollama_bench.utils.logging import setup_logging

app = typer.Typer(
    name="ollama-bench",
    help="Local-first, plugin-based LLM benchmarking for Ollama hosts.",
    no_args_is_help=True,
)
console = Console()


def _config_path(path: str | None) -> str:
    """Resolve the config path, falling back to common locations.

    `ollama-bench init` writes `config.yaml`; the repository ships
    `config/default.yaml`. Accepting both keeps the CLI usable out of the box.
    """
    if path:
        return path
    candidates = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "config" / "config.yaml",
        Path.cwd() / "config" / "default.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.as_posix()
    return candidates[0].as_posix()


def _load(path: str | None) -> BenchmarkConfig:
    try:
        return load_config(_config_path(path))
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None


def _plugins(registry: PluginRegistry, plugins: list[str]) -> list[BenchmarkPlugin]:
    out: list[BenchmarkPlugin] = []
    for pid in plugins:
        cls = registry.get(pid)
        if cls is None:
            console.print(f"[yellow]plugin not found (not implemented yet): {pid}[/yellow]")
            continue
        out.append(cls())
    return out


def _warn_plugin_errors(errors: list[str]) -> None:
    for err in errors:
        console.print(f"[yellow]⚠ local plugin load failed: {err}[/yellow]")


def _load_plugin_instances(cfg: BenchmarkConfig) -> list[BenchmarkPlugin]:
    """Instantiate the enabled plugins, falling back to the smoke plugin."""
    registry, errors = load_plugins(cfg.plugins.local_dir)
    _warn_plugin_errors(errors)
    plugin_instances = _plugins(registry, cfg.plugins.enabled)
    if not plugin_instances:
        smoke_cls = registry.get("smoke")
        if smoke_cls is not None:
            plugin_instances = [smoke_cls()]
    return plugin_instances


@app.command()
def version() -> None:
    """Show version."""
    console.print(f"ollama-bench {__version__}")


@app.command()
def init(config: str | None = typer.Option(None, "--config", help="Path to write")) -> None:
    """Create a starter configuration for the current directory."""
    dest = Path(_config_path(config))
    for sub in ("datasets", "plugins", "reports"):
        (dest.parent / sub).mkdir(parents=True, exist_ok=True)
    try:
        write_default_config(dest)
    except FileExistsError:
        console.print(f"[yellow]config already exists: {dest}[/yellow]")
        raise typer.Exit(1) from None
    console.print(f"[green]Created[/green] {dest}")
    console.print("Next: run: ollama-bench doctor")
    console.print("The default host is your local Ollama (http://127.0.0.1:11434, or $OLLAMA_HOST); add other hosts to the config if needed.")


@app.command()
def doctor(config: str | None = typer.Option(None, "--config")) -> None:
    """Validate hosts, discovery, output dir, and available plugins."""
    cfg = _load(config)
    registry, errors = load_plugins(cfg.plugins.local_dir)

    ok = True
    if errors:
        console.print("[yellow]local plugin load issues:[/yellow]")
        for err in errors:
            console.print(f"[yellow]  ✖ {err}[/yellow]")
    console.print(f"[dim]plugins dir: {cfg.plugins.local_dir}[/dim]")
    report_dir = Path(cfg.app.output_dir)
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✔[/green] output dir writable: {report_dir}")
    except OSError as exc:
        console.print(f"[red]✖[/red] output dir not writable: {exc}")
        ok = False

    for pid in cfg.plugins.enabled:
        if registry.get(pid) is None:
            console.print(f"[yellow]✖[/yellow] plugin not implemented yet: {pid}")

    async def _check() -> bool:
        good = True
        for host in cfg.hosts:
            client = OllamaClient(host.base_url, host.timeout_seconds)
            try:
                ver = await client.health()
                models_list = await discover_models(client, host.name)
                console.print(
                    f"[green]✔[/green] host {host.name} reachable ({ver.get('version')}), "
                    f"{len(models_list)} models"
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]✖[/red] host {host.name} unreachable: {exc}")
                good = False
            finally:
                await client.aclose()
        return good

    try:
        ok = asyncio.run(_check()) and ok
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✖[/red] doctor failed: {exc}")
        ok = False

    if ok:
        console.print("[green]All checks passed.[/green]")
    else:
        sys.exit(1)


@app.command()
def models(config: str | None = typer.Option(None, "--config")) -> None:
    """List models discovered on configured hosts."""
    cfg = _load(config)
    if not cfg.hosts:
        console.print("[yellow]No hosts configured.[/yellow]")
        raise typer.Exit(1)

    async def _print() -> None:
        for host in cfg.hosts:
            client = OllamaClient(host.base_url, host.timeout_seconds)
            try:
                await client.health()
                discovered = await discover_models(client, host.name)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]host {host.name}: {exc}[/red]")
                continue
            finally:
                await client.aclose()
            table = Table(title=f"{host.name} models")
            table.add_column("Model")
            table.add_column("Vision")
            table.add_column("Tools")
            table.add_column("Ctx")
            for m in discovered:
                table.add_row(
                    m.model_name,
                    "yes" if m.supports_vision else "",
                    "yes" if m.supports_tools else "",
                    str(m.max_context_tokens or ""),
                )
            console.print(table)

    asyncio.run(_print())


@app.command()
def plugins(config: str | None = typer.Option(None, "--config")) -> None:
    """List available benchmark plugins."""
    cfg = _load(config)
    registry, errors = load_plugins(cfg.plugins.local_dir)
    for err in errors:
        console.print(f"[yellow]⚠ local plugin load failed: {err}[/yellow]")
    table = Table(title=f"Plugins (local dir: {cfg.plugins.local_dir})")
    table.add_column("ID")
    table.add_column("Name")
    for pid in registry.ids():
        cls = registry.get(pid)
        table.add_row(pid, cls.name if cls else "")
    console.print(table)


def _discover_all(cfg: BenchmarkConfig) -> list[ModelInfo]:
    """Best-effort discovery across all hosts for selection/resolution."""

    async def _discover() -> list[ModelInfo]:
        out: list[ModelInfo] = []
        for host in cfg.hosts:
            client = OllamaClient(host.base_url, host.timeout_seconds)
            try:
                await client.health()
                out.extend(await discover_models(client, host.name))
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]⚠ discovery for {host.name} failed: {exc}[/yellow]")
            finally:
                await client.aclose()
        return out

    return asyncio.run(_discover())


@app.command()
def run(
    config: str | None = typer.Option(None, "--config"),
    models: str | None = typer.Option(
        None, "--models", help="Which autodetected models to benchmark (glob, comma/space separated)."
    ),
    exclude: str | None = typer.Option(
        None, "--exclude", help="Skip autodetected models matching these globs."
    ),
    interactive: bool = typer.Option(
        False, "--interactive", help="Pick models interactively from the autodetected list."
    ),
    db: str | None = typer.Option(
        None, "--db", help="SQLite DB path to save results to (e.g. benchmark.db)."
    ),
) -> None:
    """Run a benchmark against configured hosts; models are auto-detected and selected by flags."""
    _run_benchmark(config=config, models=models, exclude=exclude, interactive=interactive, db=db)


@app.command("run-single")
def run_single(
    model: str = typer.Argument(..., help="Exact model name to benchmark."),
    config: str | None = typer.Option(None, "--config"),
    db: str | None = typer.Option(
        None, "--db", help="SQLite DB path to save results to (e.g. benchmark.db)."
    ),
) -> None:
    """Run ALL enabled plugins for a single model and save to SQLite (and files)."""
    cfg = _load(config)
    plugin_instances = _load_plugin_instances(cfg)

    discovered = _discover_all(cfg)
    match = next((m for m in discovered if m.model_name == model), None)
    if match is None:
        console.print(f"[red]Model not found: {model}[/red]")
        console.print("Available models:")
        for m in sorted(discovered, key=lambda x: x.model_name):
            console.print(f"  • {m.model_name}")
        raise typer.Exit(1)

    console.print(f"[green]Benchmarking single model:[/green] {model}")
    console.print(f"[dim]Plugins: {', '.join(p.id for p in plugin_instances)}[/dim]")
    console.print(f"[dim]config hash: {config_hash(cfg)}[/dim]")

    _execute_run(
        cfg,
        plugin_instances,
        model_filter=lambda mi: mi.model_name == model,  # noqa: E731
        db=db,
        detail=True,
    )


def _run_benchmark(
    config: str | None = None,
    models: str | None = None,
    exclude: str | None = None,
    interactive: bool = False,
    db: str | None = None,
) -> None:
    cfg = _load(config)
    plugin_instances = _load_plugin_instances(cfg)

    discovered = _discover_all(cfg)
    selected_names: set[str] = set()
    model_filter = None
    if discovered:
        include = split_patterns(models) or list(cfg.models)
        exclude_pats = split_patterns(exclude)
        pool = filter_models(discovered, include, exclude_pats)
        if interactive:
            pool = pick_interactive(pool)
        if models and not pool:
            console.print("[red]No models matched your --models pattern(s).[/red]")
            raise typer.Exit(1)
        selected_names = {m.model_name for m in pool}
        console.print(f"[dim]Autodetected[/dim] {len(discovered)} models; [green]selected {len(pool)}[/green]")
        for m in sorted(pool, key=lambda m: m.model_name):
            console.print(f"  • {m.model_name}")
        if selected_names:
            model_filter = lambda mi: mi.model_name in selected_names  # noqa: E731

    console.print(f"[dim]config hash: {config_hash(cfg)}[/dim]")
    _execute_run(cfg, plugin_instances, model_filter=model_filter, db=db, detail=False)


def _execute_run(
    cfg: BenchmarkConfig,
    plugin_instances: list[BenchmarkPlugin],
    *,
    model_filter: Callable[[ModelInfo], bool] | None,
    db: str | None,
    detail: bool,
) -> None:
    """Shared runner: orchestrate, write the report, persist to SQLite, summarize."""
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
    ) as progress:
        progress.add_task(description="Running benchmarks...", total=None)

        def on_event(event: Event) -> None:
            console.log(
                f"[{event.kind}] {event.model or ''} {event.plugin or ''} {event.case_id or ''}".strip()
            )

        orchestrator = RunOrchestrator(cfg, plugin_instances, event_cb=on_event, model_filter=model_filter)
        result = asyncio.run(orchestrator.run())

    report_dir = write_report(
        cfg.app.output_dir,
        result,
        formats=cfg.reporting.formats,
        include_raw_cases=cfg.reporting.include_raw_cases,
    )

    if db:
        repo = BenchmarkRepository(db)
        try:
            repo.save_run(result)
            console.print(f"[green]Saved to SQLite:[/green] {db}")
        finally:
            repo.close()

    summary = Table(title="Summary")
    if detail:
        summary.add_column("Plugin")
        summary.add_column("Cases")
        summary.add_column("Passed")
        summary.add_column("Score")
        for mr in sorted(result.models, key=lambda x: x.model_name):
            for p in mr.plugins:
                summary.add_row(
                    f"{mr.model_name} / {p.plugin_id}",
                    str(p.total_cases),
                    str(p.successful_cases),
                    _num(p.score, 3),
                )
    else:
        summary.add_column("Model")
        summary.add_column("p50 ms")
        summary.add_column("p95 ms")
        summary.add_column("Tokens/s")
        summary.add_column("Score")
        for mr in sorted(result.models, key=lambda x: (x.overall_score or 0), reverse=True):
            summary.add_row(
                mr.model_name,
                _num(mr.latency_p50_ms),
                _num(mr.latency_p95_ms),
                _num(mr.tokens_per_second),
                _num(mr.overall_score, 3),
            )
    console.print(summary)

    for err in result.errors:
        console.print(f"[red]✖ {err}[/red]")
    console.print(f"[green]Report written to[/green] {report_dir}")


def _num(x: float | None, digits: int = 1) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


@app.command()
def report(
    action: str = typer.Argument("list", help="list | view | open"),
    run_id: str | None = typer.Option(None, "--run", help="Run ID; default newest"),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """List, view, or open generated reports."""
    cfg = _load(config)
    base = Path(cfg.app.output_dir)
    if action == "list":
        dirs = sorted([p for p in base.iterdir() if p.is_dir()]) if base.exists() else []
        if not dirs:
            console.print("[yellow]No benchmark reports found. Run: ollama-bench run[/yellow]")
            return
        for d in reversed(dirs):
            console.print(d.name)
        return

    resolved = _resolve_report_dir(base, run_id)
    if resolved is None:
        console.print("[yellow]No reports found.[/yellow]")
        raise typer.Exit(1)

    if action == "view":
        md = resolved / "report.md"
        if md.exists():
            rich.print(md.read_text(encoding="utf-8"))
        else:
            console.print(json.dumps(json.loads((resolved / "report.json").read_text()), indent=2))
        return
    if action == "open":
        target = resolved / "report.html"
        if not target.exists():
            target = resolved / "report.md"
        if not target.exists():
            console.print("[yellow]No HTML or Markdown report in this run.[/yellow]")
            raise typer.Exit(1)
        webbrowser.open(target.resolve().as_uri())
        console.print(f"[green]Opened[/green] {target}")
        return
    console.print(f"[red]unknown action: {action}[/red] (use list | view | open)")
    raise typer.Exit(2)


def _resolve_report_dir(base: Path, run_id: str | None) -> Path | None:
    """Resolve a report directory from a run id, or the newest run."""
    if run_id:
        d = base / run_id
        return d if d.is_dir() else None
    matches = sorted([p for p in base.iterdir() if p.is_dir()]) if base.exists() else []
    return matches[-1] if matches else None


@app.command()
def history(
    db: str = typer.Option("benchmark.db", "--db", help="SQLite DB path."),
    model: str | None = typer.Option(None, "--model", help="Filter by model name."),
) -> None:
    """Show benchmark history from SQLite."""
    repo = BenchmarkRepository(db)
    try:
        if model:
            rows = repo.get_model_history(model)
            if not rows:
                console.print(f"[yellow]No history for model: {model}[/yellow]")
                return
            table = Table(title=f"History for {model}")
            table.add_column("Timestamp")
            table.add_column("Run ID")
            table.add_column("Score")
            table.add_column("p50 ms")
            table.add_column("p95 ms")
            table.add_column("Tokens/s")
            table.add_column("Cases")
            table.add_column("Errors")
            for r in rows:
                table.add_row(
                    r["timestamp"][:19],
                    r["run_id"],
                    _num(r["overall_score"], 3),
                    _num(r["latency_p50_ms"]),
                    _num(r["latency_p95_ms"]),
                    _num(r["tokens_per_second"]),
                    str(r["cases_run"]),
                    str(r["errors"]),
                )
            console.print(table)
        else:
            runs = repo.list_runs()
            if not runs:
                console.print("[yellow]No runs in database. Run: ollama-bench run --db benchmark.db[/yellow]")
                return
            table = Table(title="Benchmark history")
            table.add_column("Run ID")
            table.add_column("Timestamp")
            table.add_column("App")
            table.add_column("Hosts")
            for r in runs:
                hosts = r["hosts"] or []
                table.add_row(
                    r["run_id"],
                    r["timestamp"][:19],
                    r["app_version"],
                    ", ".join(h.get("name", "?") for h in hosts),
                )
            console.print(table)
    finally:
        repo.close()


@app.command()
def compare(
    db: str = typer.Option("benchmark.db", "--db", help="SQLite DB path."),
    run_id: str | None = typer.Option(None, "--run", help="Limit to a specific run."),
) -> None:
    """Compare all models across runs."""
    repo = BenchmarkRepository(db)
    try:
        rows = repo.compare_models(run_id=run_id)
        if not rows:
            console.print("[yellow]No data in database.[/yellow]")
            return
        table = Table(title="Model comparison")
        table.add_column("Model")
        table.add_column("Score")
        table.add_column("p50 ms")
        table.add_column("p95 ms")
        table.add_column("TTFT p50")
        table.add_column("Tokens/s")
        table.add_column("Cases")
        table.add_column("Errors")
        for r in rows:
            table.add_row(
                r["model_name"],
                _num(r["overall_score"], 3),
                _num(r["latency_p50_ms"]),
                _num(r["latency_p95_ms"]),
                _num(r["time_to_first_token_p50_ms"]),
                _num(r["tokens_per_second"]),
                str(r["cases_run"]),
                str(r["errors"]),
            )
        console.print(table)
    finally:
        repo.close()


if __name__ == "__main__":
    setup_logging()
    app()