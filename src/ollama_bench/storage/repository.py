"""SQLite persistence for benchmark results.

Schema:
- runs: top-level run metadata
- models: per-model aggregated results within a run
- plugins: per-plugin aggregated results within a run
- cases: individual case results (responses, evaluations, timing)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, cast

from ollama_bench.domain.models import RunResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    timestamp TEXT,
    app_version TEXT,
    config_hash TEXT,
    hosts TEXT
);

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    host_name TEXT,
    model_name TEXT NOT NULL,
    model_digest TEXT,
    max_context_tokens INTEGER,
    completion_tokens_total INTEGER DEFAULT 0,
    cases_run INTEGER DEFAULT 0,
    errors INTEGER DEFAULT 0,
    latency_p50_ms REAL,
    latency_p95_ms REAL,
    time_to_first_token_p50_ms REAL,
    tokens_per_second REAL,
    overall_score REAL,
    context_recommendation TEXT,
    UNIQUE(run_id, model_name)
);

CREATE TABLE IF NOT EXISTS plugins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    plugin_id TEXT NOT NULL,
    total_cases INTEGER DEFAULT 0,
    successful_cases INTEGER DEFAULT 0,
    failed_cases INTEGER DEFAULT 0,
    skipped_cases INTEGER DEFAULT 0,
    score REAL,
    metrics TEXT,
    UNIQUE(run_id, model_name, plugin_id)
);

CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    plugin_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    passed INTEGER,
    score REAL,
    response_text TEXT,
    error TEXT,
    total_ms REAL,
    time_to_first_token_ms REAL,
    tokens_per_second REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    attempt INTEGER DEFAULT 1,
    raw_response TEXT,
    UNIQUE(run_id, model_name, plugin_id, case_id, attempt)
);

CREATE INDEX IF NOT EXISTS idx_models_run ON models(run_id);
CREATE INDEX IF NOT EXISTS idx_plugins_run ON plugins(run_id);
CREATE INDEX IF NOT EXISTS idx_cases_run ON cases(run_id);

CREATE TABLE IF NOT EXISTS plugin_options (
    plugin_id TEXT PRIMARY KEY,
    options TEXT
);
"""


def _like_escape(value: str) -> str:
    """Escape LIKE wildcards so user input is matched literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _normalize_run(row: dict[str, Any]) -> dict[str, Any]:
    """Shape a raw DB row into the API representation.

    ``hosts`` is stored as a JSON string; ``model_names`` is a comma-separated
    GROUP_CONCAT string. Both are converted to their structured forms here so
    callers never see the storage format.
    """
    names = row.pop("model_names", None)
    row["model_names"] = names.split(",") if names else []
    hosts = row.get("hosts")
    row["hosts"] = json.loads(hosts) if hosts else []
    return row


class BenchmarkRepository:
    """Persist RunResult into SQLite for later comparison and analysis."""

    def __init__(self, db_path: str | Path = "benchmark.db") -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Idempotently add columns introduced after the initial schema."""
        cols = [row[1] for row in self._conn.execute("PRAGMA table_info(models)").fetchall()]
        if "context_recommendation" not in cols:
            self._conn.execute("ALTER TABLE models ADD COLUMN context_recommendation TEXT")

    def close(self) -> None:
        self._conn.close()

    def save_run(self, result: RunResult) -> None:
        cur = self._conn.cursor()
        # Insert run
        cur.execute(
            "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?)",
            (
                result.run_id,
                result.timestamp,
                result.app_version,
                result.config_hash,
                json.dumps([h.model_dump() for h in result.hosts]),
            ),
        )
        for m in result.models:
            max_ctx = None
            if m.plugins and m.plugins[0].metrics:
                max_ctx = m.plugins[0].metrics.get("max_context_tokens")
            cur.execute(
                """INSERT OR REPLACE INTO models
                (run_id, host_name, model_name, model_digest, max_context_tokens,
                 completion_tokens_total, cases_run, errors,
                 latency_p50_ms, latency_p95_ms, time_to_first_token_p50_ms,
                 tokens_per_second, overall_score, context_recommendation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.run_id,
                    m.host_name,
                    m.model_name,
                    m.model_digest,
                    max_ctx,
                    m.completion_tokens_total,
                    m.cases_run,
                    m.errors,
                    m.latency_p50_ms,
                    m.latency_p95_ms,
                    m.time_to_first_token_p50_ms,
                    m.tokens_per_second,
                    m.overall_score,
                    json.dumps(m.context_recommendation, default=str) if m.context_recommendation else None,
                ),
            )
            for p in m.plugins:
                cur.execute(
                    "INSERT OR REPLACE INTO plugins VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        result.run_id,
                        m.model_name,
                        p.plugin_id,
                        p.total_cases,
                        p.successful_cases,
                        p.failed_cases,
                        p.skipped_cases,
                        p.score,
                        json.dumps(p.metrics, default=str),
                    ),
                )
            for c in m.cases:
                cur.execute(
                    """INSERT OR REPLACE INTO cases
                    (run_id, model_name, plugin_id, case_id, passed, score,
                     response_text, error, total_ms, time_to_first_token_ms,
                     tokens_per_second, prompt_tokens, completion_tokens,
                     attempt, raw_response)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        result.run_id,
                        m.model_name,
                        c.case.plugin_id,
                        c.case.id,
                        None if c.evaluation.passed is None else int(c.evaluation.passed),
                        c.evaluation.score,
                        c.response.text,
                        c.response.error,
                        c.response.timing.total_ms,
                        c.response.timing.time_to_first_token_ms,
                        c.response.tokens.tokens_per_second,
                        c.response.tokens.prompt_tokens,
                        c.response.tokens.completion_tokens,
                        c.attempt,
                        json.dumps(c.response.raw, default=str),
                    ),
                )
        self._conn.commit()

    def list_runs(
        self,
        *,
        search: str | None = None,
        model: str | None = None,
        host: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if search:
            where.append("r.run_id LIKE ?")
            params.append(f"%{_like_escape(search)}%")
        if model:
            where.append(
                "EXISTS (SELECT 1 FROM models m WHERE m.run_id = r.run_id AND m.model_name = ?)"
            )
            params.append(model)
        if host:
            where.append("r.hosts LIKE ? ESCAPE '\\'")
            params.append(f'%"name": "{_like_escape(host)}"%')
        if date_from:
            where.append("substr(r.timestamp, 1, 10) >= ?")
            params.append(date_from)
        if date_to:
            where.append("substr(r.timestamp, 1, 10) <= ?")
            params.append(date_to)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        rows = self._conn.execute(
            f"""
            SELECT r.run_id, r.timestamp, r.app_version, r.config_hash, r.hosts,
                   (SELECT GROUP_CONCAT(DISTINCT m.model_name)
                      FROM models m WHERE m.run_id = r.run_id) AS model_names
            FROM runs r
            {clause}
            ORDER BY r.timestamp DESC
            """,
            params,
        ).fetchall()
        return [_normalize_run(dict(row)) for row in rows]

    def distinct_filters(self) -> dict[str, list[str]]:
        """Distinct models and hosts across all runs, for filter dropdowns."""
        models = [
            row[0]
            for row in self._conn.execute(
                "SELECT DISTINCT model_name FROM models ORDER BY model_name"
            )
        ]
        host_set: set[str] = set()
        for row in self._conn.execute("SELECT hosts FROM runs"):
            try:
                hosts = json.loads(row["hosts"])
            except (TypeError, ValueError):
                continue
            for h in hosts:
                name = h.get("name") if isinstance(h, dict) else h
                if name:
                    host_set.add(name)
        return {"models": models, "hosts": sorted(host_set)}

    def delete_run(self, run_id: str) -> bool:
        if self._conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
            return False
        self._conn.execute("DELETE FROM cases WHERE run_id = ?", (run_id,))
        self._conn.execute("DELETE FROM plugins WHERE run_id = ?", (run_id,))
        self._conn.execute("DELETE FROM models WHERE run_id = ?", (run_id,))
        self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        self._conn.commit()
        return True

    def get_plugin_options(self, plugin_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT options FROM plugin_options WHERE plugin_id = ?", (plugin_id,)
        ).fetchone()
        if row is None:
            return None
        return cast("dict[str, Any]", json.loads(row["options"]))

    def set_plugin_options(self, plugin_id: str, options: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO plugin_options (plugin_id, options) VALUES (?, ?)",
            (plugin_id, json.dumps(options)),
        )
        self._conn.commit()

    def all_plugin_options(self) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT plugin_id, options FROM plugin_options"
        ).fetchall()
        return {row["plugin_id"]: json.loads(row["options"]) for row in rows}

    def get_model_history(self, model_name: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self._conn.execute(
                """
                SELECT r.timestamp, r.run_id, m.overall_score, m.latency_p50_ms,
                       m.latency_p95_ms, m.tokens_per_second, m.cases_run, m.errors
                FROM models m
                JOIN runs r ON r.run_id = m.run_id
                WHERE m.model_name = ?
                ORDER BY r.timestamp DESC
                """,
                (model_name,),
            )
        ]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT r.run_id, r.timestamp, r.app_version, r.config_hash, r.hosts,
                   (SELECT GROUP_CONCAT(DISTINCT m.model_name)
                      FROM models m WHERE m.run_id = r.run_id) AS model_names
            FROM runs r
            WHERE r.run_id = ?
            """,
            (run_id,),
        ).fetchone()
        return _normalize_run(dict(row)) if row else None

    def compare_models(self, run_id: str | None = None) -> list[dict[str, Any]]:
        q = """
        SELECT m.model_name, m.overall_score, m.latency_p50_ms, m.latency_p95_ms,
               m.time_to_first_token_p50_ms, m.tokens_per_second, m.cases_run, m.errors
        FROM models m
        %s
        ORDER BY m.overall_score DESC NULLS LAST, m.latency_p50_ms ASC NULLS LAST
        """
        params: list[Any] = []
        where = ""
        if run_id:
            where = "WHERE m.run_id = ?"
            params.append(run_id)
        cur = self._conn.execute(q % where, params)
        return [dict(row) for row in cur.fetchall()]
