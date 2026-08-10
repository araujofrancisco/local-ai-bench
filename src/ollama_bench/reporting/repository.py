"""Report persistence — writes JSON, Markdown, and HTML artifacts per run."""

from __future__ import annotations

import html
import json
from pathlib import Path

from ollama_bench.domain.models import RunResult

DEFAULT_FORMATS = ["json", "markdown", "html"]


def _report_dir(base: str, run_id: str) -> Path:
    root = Path(base)
    root.mkdir(parents=True, exist_ok=True)
    d = root / run_id
    d.mkdir(exist_ok=True)
    return d


def write_report(
    config_output_dir: str,
    result: RunResult,
    formats: list[str] | None = None,
    include_raw_cases: bool = True,
) -> str:
    """Persist run results. Returns the report directory as a string."""
    d = _report_dir(config_output_dir, result.run_id)
    data = result.model_dump(mode="json", exclude_none=True)
    if not include_raw_cases:
        for model in data.get("models", []):
            model.pop("cases", None)
    enabled = [f.lower() for f in (formats or DEFAULT_FORMATS)]

    if "json" in enabled:
        (d / "report.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if "markdown" in enabled:
        (d / "report.md").write_text(_render_markdown(result), encoding="utf-8")
    if "html" in enabled:
        (d / "report.html").write_text(_render_html(result), encoding="utf-8")

    result.report_dir = str(d)
    return str(d)


def _fmt(x: float | None, digits: int = 1) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


_PID_LABELS = {
    "reasoning": "Reasoning",
    "translation": "Translation",
    "summarization": "Summarization",
    "structured_output": "Structured",
    "coding": "Coding",
    "vision": "Vision",
    "long_context": "Long Context",
    "smoke": "Smoke",
}


def _render_markdown(result: RunResult) -> str:
    lines: list[str] = []
    lines.append(f"# OllamaBench report · {result.run_id}")
    lines.append("")
    lines.append(f"- App version: {result.app_version}")
    lines.append(f"- Config hash: `{result.config_hash}`")
    lines.append(f"- Timestamp: {result.timestamp}")
    lines.append(f"- Hosts: {', '.join(h.name for h in result.hosts) if result.hosts else 'none'}")
    lines.append("")

    if result.errors:
        lines.append("## Errors")
        for err in result.errors:
            lines.append(f"- {err}")
        lines.append("")

    if not result.models:
        lines.append("## Models")
        lines.append("")
        lines.append("No models benchmarked.")
        lines.append("")
        return "\n".join(lines)

    lines.extend(_render_model_summary(result))
    lines.extend(_render_usecase_comparison(result))
    lines.extend(_render_context_recommendation(result))
    lines.extend(_render_per_case_detail(result))
    lines.extend(_render_context_window(result))
    lines.extend(_render_plugin_detail(result))
    lines.append("")
    lines.append("## Raw artifacts")
    lines.append("- JSON: `report.json`")
    lines.append("- Markdown: `report.md`")
    lines.append("- HTML: `report.html`")
    return "\n".join(lines)


def _render_model_summary(result: RunResult) -> list[str]:
    lines = ["## Model summary", ""]
    header = (
        "| Model | Context | Latency p50 (ms) | Latency p95 (ms) | TTFT p50 (ms) | "
        "Tokens/s | Cases | Errors | Score |"
    )
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for m in sorted(result.models, key=lambda x: (x.overall_score or 0), reverse=True):
        ctx = m.plugins[0].metrics.get("max_context_tokens") if m.plugins else None
        ctx_str = f"{ctx}" if ctx else "-"
        lines.append(
            f"| {m.model_name} | {ctx_str} | {_fmt(m.latency_p50_ms)} | "
            f"{_fmt(m.latency_p95_ms)} | {_fmt(m.time_to_first_token_p50_ms)} | "
            f"{_fmt(m.tokens_per_second)} | {m.cases_run} | {m.errors} | "
            f"{_fmt(m.overall_score, 3) if m.overall_score is not None else '-'} |"
        )
    lines.append("")
    return lines


def _collect_plugin_ids(result: RunResult) -> list[str]:
    seen: list[str] = []
    for m in result.models:
        for p in m.plugins:
            if p.plugin_id not in seen:
                seen.append(p.plugin_id)
    return seen


def _ctx_cell(per: dict[int, float | None], size: int) -> str:
    sc = per.get(size)
    if sc is None:
        return "-"
    return f"{sc:.0f}" if sc >= 1 else f"{sc:.1f}"


def _render_context_recommendation(result: RunResult) -> list[str]:
    recs = [(m.model_name, m.context_recommendation) for m in result.models if m.context_recommendation]
    if not recs:
        return []
    lines = ["## Context-window recommendations", ""]
    lines.append("| Model | Recommended ctx | Reason |")
    lines.append("|---|---|---|")
    for name, rec in recs:
        ctx = rec.get("recommended_context")
        ctx_str = f"{ctx}" if ctx else "—"
        lines.append(f"| {name} | {ctx_str} | {rec.get('reason') or ''} |")
    lines.append("")
    return lines


def _render_usecase_comparison(result: RunResult) -> list[str]:
    lines: list[str] = []
    pids = _collect_plugin_ids(result)
    if not pids:
        return lines
    lines.append("## Per-use-case comparison")
    lines.append("")
    lines.append("Which model is best for which task. Ranked by score (higher = better):")
    lines.append("")
    for pid in pids:
        rows = []
        for m in result.models:
            for p in m.plugins:
                if p.plugin_id == pid:
                    rows.append((m.model_name, p))
        rows.sort(key=lambda r: (r[1].score or -1), reverse=True)
        label = _PID_LABELS.get(pid, pid)
        lines.append(f"### `{pid}` ({label})")
        lines.append("")
        lines.append("| Model | Passed | Score |")
        lines.append("|---|---|---|")
        if not rows:
            lines.append("| _none_ | - | - |")
        for name, p in rows:
            lines.append(
                f"| {name} | {p.successful_cases}/{p.total_cases} | "
                f"{_fmt(p.score, 3)} |"
            )
        lines.append("")
    return lines


def _render_per_case_detail(result: RunResult) -> list[str]:
    lines: list[str] = ["## Per-case detail", ""]
    lines.append("| Plugin | Model | Case | Result | Score |")
    lines.append("|---|---|---|---|---|")
    for m in sorted(result.models, key=lambda x: x.model_name):
        for p in m.plugins:
            label = _PID_LABELS.get(p.plugin_id, p.plugin_id)
            per_ctx = p.metrics.get("per_context_score", {}) if p.metrics else {}
            if per_ctx:
                for ctx_size, sc in sorted(per_ctx.items()):
                    ok = sc is not None and sc > 0
                    lines.append(
                        f"| {label} | {m.model_name} | ctx={ctx_size} | "
                        f"{'✅' if ok else '❌'} | {_fmt(sc, 2)} |"
                    )
            else:
                passed = p.successful_cases == p.total_cases
                lines.append(
                    f"| {label} | {m.model_name} | (aggregate) | "
                    f"{'✅' if passed else '❌'} | {_fmt(p.score, 2)} |"
                )
    lines.append("")
    return lines


def _render_context_window(result: RunResult) -> list[str]:
    lines: list[str] = []
    lc = [
        (m.model_name, p)
        for m in result.models
        for p in m.plugins
        if p.plugin_id == "long_context"
    ]
    if not lc:
        return lines
    lines.append("## Context-window performance")
    lines.append("")
    lines.append("Accuracy (1.0 = correct, 0 = wrong) by context size:")
    lines.append("")
    lines.append("| Model | Max ctx (tokens) | Ctx 256 | Ctx 1024 | Ctx 4096 | Ctx 16k |")
    lines.append("|---|---|---|---|---|---|")
    for name, p in lc:
        per = p.metrics.get("per_context_score", {})
        max_ctx = p.metrics.get("max_context_tokens", "-")
        lines.append(
            f"| {name} | {max_ctx} | {_ctx_cell(per, 256)} | "
            f"{_ctx_cell(per, 1024)} | {_ctx_cell(per, 4096)} | "
            f"{_ctx_cell(per, 16384)} |"
        )
    lines.append("")
    return lines


def _render_plugin_detail(result: RunResult) -> list[str]:
    lines = ["## Plugin detail", ""]
    for m in result.models:
        lines.append(f"### {m.model_name}")
        lines.append("")
        for p in m.plugins:
            lines.append(
                f"- **{p.plugin_id}**: {p.total_cases} cases, "
                f"{p.successful_cases} passed, score {_fmt(p.score, 3)}"
            )
            if p.metrics:
                snippet = json.dumps(p.metrics, default=str)[:200]
                lines.append(f"  - metrics: `{snippet}`")
        lines.append("")
    return lines


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _render_html(result: RunResult) -> str:
    """Self-contained HTML report — no external assets, works offline."""
    h = _esc
    parts: list[str] = []
    parts.append(_html_header(result))
    parts.append(_html_errors(result))
    parts.append(_html_model_summary(result))
    parts.append(_html_usecase_comparison(result))
    parts.append(_html_context_recommendation(result))
    parts.append(_html_per_case_detail(result))
    parts.append(_html_context_window(result))
    parts.append(_html_plugin_detail(result))

    body = "\n".join(p for p in parts if p)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OllamaBench report · {h(result.run_id)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0 auto;
         max-width: 960px; padding: 1.5rem; line-height: 1.45; }}
  h1 {{ font-size: 1.5rem; }} h2 {{ font-size: 1.2rem; margin-top: 2rem;
       border-bottom: 1px solid #ccc; padding-bottom: .25rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ text-align: left; padding: .35rem .6rem;
           border-bottom: 1px solid #ddd; font-size: .9rem; }}
  th {{ border-bottom: 2px solid #999; }}
  code {{ background: rgba(127,127,127,.15); padding: .1em .35em; border-radius: 4px; }}
  .ok {{ color: #15803d; }} .bad {{ color: #b91c1c; }}
  .err {{ color: #b91c1c; }} .muted {{ color: #666; }}
  @media (prefers-color-scheme: dark) {{ .ok {{ color: #4ade80; }}
    .bad, .err {{ color: #f87171; }} .muted {{ color: #aaa; }} }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def _html_header(result: RunResult) -> str:
    lines = [
        "<h1>OllamaBench report</h1>",
        "<p class=\"muted\">Run <code>" + _esc(result.run_id) + "</code> · "
        + _esc(result.app_version) + " · "
        + _esc(result.timestamp) + "</p>",
        "<p>Config hash: <code>" + _esc(result.config_hash) + "</code></p>",
        "<p>Hosts: " + _esc(", ".join(h.name for h in result.hosts) if result.hosts else "none") + "</p>",
    ]
    return "\n".join(lines)


def _html_errors(result: RunResult) -> str:
    if not result.errors:
        return ""
    items = "\n".join(f"<li>{_esc(e)}</li>" for e in result.errors)
    return f'<h2>Errors</h2>\n<ul class="err">\n{items}\n</ul>'


def _html_model_summary(result: RunResult) -> str:
    if not result.models:
        return "<h2>Models</h2>\n<p class=\"muted\">No models benchmarked.</p>"
    rows = ["<tr><th>Model</th><th>Context</th><th>Latency p50 (ms)</th>"
            "<th>Latency p95 (ms)</th><th>TTFT p50 (ms)</th>"
            "<th>Tokens/s</th><th>Cases</th><th>Errors</th><th>Score</th></tr>"]
    for m in sorted(result.models, key=lambda x: (x.overall_score or 0), reverse=True):
        ctx = m.plugins[0].metrics.get("max_context_tokens") if m.plugins else None
        rows.append(
            f"<tr><td>{_esc(m.model_name)}</td><td>{_esc(ctx) if ctx else '-'}</td>"
            f"<td>{_fmt(m.latency_p50_ms)}</td><td>{_fmt(m.latency_p95_ms)}</td>"
            f"<td>{_fmt(m.time_to_first_token_p50_ms)}</td>"
            f"<td>{_fmt(m.tokens_per_second)}</td>"
            f"<td>{m.cases_run}</td><td>{m.errors}</td>"
            f"<td>{_fmt(m.overall_score, 3) if m.overall_score is not None else '-'}</td></tr>"
        )
    return "<h2>Model summary</h2>\n<table>\n" + "\n".join(rows) + "\n</table>"


def _html_usecase_comparison(result: RunResult) -> str:
    pids = _collect_plugin_ids(result)
    sections = []
    for pid in pids:
        rows = []
        for m in result.models:
            for p in m.plugins:
                if p.plugin_id == pid:
                    rows.append((m.model_name, p))
        rows.sort(key=lambda r: (r[1].score or -1), reverse=True)
        label = _PID_LABELS.get(pid, pid)
        cells = ["<tr><th>Model</th><th>Passed</th><th>Score</th></tr>"]
        if not rows:
            cells.append("<tr><td colspan=\"3\" class=\"muted\">none</td></tr>")
        for name, p in rows:
            cells.append(
                f"<tr><td>{_esc(name)}</td><td>{p.successful_cases}/{p.total_cases}</td>"
                f"<td>{_fmt(p.score, 3)}</td></tr>"
            )
        sections.append(
            f"<h2>{_esc(pid)} <span class=\"muted\">({_esc(label)})</span></h2>"
            f"\n<table>\n{'\n'.join(cells)}\n</table>"
        )
    return "\n".join(sections)


def _html_context_recommendation(result: RunResult) -> str:
    recs = [(m.model_name, m.context_recommendation) for m in result.models if m.context_recommendation]
    if not recs:
        return ""
    rows = ["<tr><th>Model</th><th>Recommended ctx</th><th>Reason</th></tr>"]
    for name, rec in recs:
        ctx = rec.get("recommended_context")
        rows.append(
            f"<tr><td>{_esc(name)}</td><td>{_esc(ctx) if ctx else '—'}</td>"
            f"<td>{_esc(rec.get('reason') or '')}</td></tr>"
        )
    return "<h2>Context-window recommendations</h2>\n<table>\n" + "\n".join(rows) + "\n</table>"


def _html_per_case_detail(result: RunResult) -> str:
    rows = ["<tr><th>Plugin</th><th>Model</th><th>Case</th><th>Result</th><th>Score</th></tr>"]
    for m in sorted(result.models, key=lambda x: x.model_name):
        for p in m.plugins:
            label = _PID_LABELS.get(p.plugin_id, p.plugin_id)
            per_ctx = p.metrics.get("per_context_score", {}) if p.metrics else {}
            if per_ctx:
                for ctx_size, sc in sorted(per_ctx.items()):
                    ok = sc is not None and sc > 0
                    mark = '<span class="ok">✓</span>' if ok else '<span class="bad">✗</span>'
                    rows.append(
                        f"<tr><td>{_esc(label)}</td><td>{_esc(m.model_name)}</td>"
                        f"<td>ctx={_esc(ctx_size)}</td><td>{mark}</td>"
                        f"<td>{_fmt(sc, 2)}</td></tr>"
                    )
            else:
                passed = p.successful_cases == p.total_cases
                mark = '<span class="ok">✓</span>' if passed else '<span class="bad">✗</span>'
                rows.append(
                    f"<tr><td>{_esc(label)}</td><td>{_esc(m.model_name)}</td>"
                    f"<td>(aggregate)</td><td>{mark}</td><td>{_fmt(p.score, 2)}</td></tr>"
                )
    return "<h2>Per-case detail</h2>\n<table>\n" + "\n".join(rows) + "\n</table>"


def _html_context_window(result: RunResult) -> str:
    lc = [
        (m.model_name, p)
        for m in result.models
        for p in m.plugins
        if p.plugin_id == "long_context"
    ]
    if not lc:
        return ""
    rows = ["<tr><th>Model</th><th>Max ctx (tokens)</th><th>Ctx 256</th><th>Ctx 1024</th>"
            "<th>Ctx 4096</th><th>Ctx 16k</th></tr>"]
    for name, p in lc:
        per = p.metrics.get("per_context_score", {})
        max_ctx = p.metrics.get("max_context_tokens", "-")
        rows.append(
            f"<tr><td>{_esc(name)}</td><td>{_esc(max_ctx)}</td>"
            f"<td>{_ctx_cell(per, 256)}</td><td>{_ctx_cell(per, 1024)}</td>"
            f"<td>{_ctx_cell(per, 4096)}</td><td>{_ctx_cell(per, 16384)}</td></tr>"
        )
    return "<h2>Context-window performance</h2>\n<table>\n" + "\n".join(rows) + "\n</table>"


def _html_plugin_detail(result: RunResult) -> str:
    if not result.models:
        return ""
    sections = []
    for m in result.models:
        items = []
        for p in m.plugins:
            extra = ""
            if p.metrics:
                snippet = json.dumps(p.metrics, default=str)[:200]
                extra = f"\n<div class=\"muted\"><code>{_esc(snippet)}</code></div>"
            items.append(
                f"<li><strong>{_esc(p.plugin_id)}</strong>: {p.total_cases} cases, "
                f"{p.successful_cases} passed, score {_fmt(p.score, 3)}{extra}</li>"
            )
        sections.append(
            f"<h3>{_esc(m.model_name)}</h3>\n<ul>\n" + "\n".join(items) + "\n</ul>"
        )
    return "<h2>Plugin detail</h2>\n" + "\n".join(sections)
