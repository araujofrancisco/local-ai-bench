"""SQL generation benchmark — the model must write a correct SQLite query.

Each case supplies a small relational schema (DDL + seed rows) and a
natural-language question. The model must produce a ``SELECT`` statement; the
benchmark executes it against an in-memory SQLite database seeded identically to
the prompt and compares the resulting rows (order-insensitive, numeric-tolerant)
with the expected result set.

Execution is a safe, throwaway ``:memory:`` connection: a progress handler
aborts runaway queries, and a non-``SELECT`` first statement is rejected outright
so the model can never mutate or drop anything. This is the same spirit as the
``coding`` plugin's protected subprocess, but requires no host SQL server.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from collections.abc import Iterable
from typing import Any, ClassVar

from local_ai_bench.domain.models import (
    BenchmarkCase,
    BenchmarkCategory,
    Evaluation,
    Modality,
)
from local_ai_bench.plugins.builtin._base import BaseTextPlugin
from local_ai_bench.plugins.score import normalize_text

_SCHEMA = """\
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER NOT NULL,
    city TEXT NOT NULL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    product TEXT NOT NULL,
    amount REAL NOT NULL,
    status TEXT NOT NULL
);
"""

_SEED = """\
INSERT INTO users (id, name, age, city) VALUES
    (1, 'Alice', 29, 'Tokyo'),
    (2, 'Bob', 42, 'Osaka'),
    (3, 'Carol', 31, 'Tokyo'),
    (4, 'David', 25, 'Kyoto');

INSERT INTO orders (id, user_id, product, amount, status) VALUES
    (1, 1, 'Laptop', 899.0, 'shipped'),
    (2, 1, 'Mouse', 29.99, 'shipped'),
    (3, 2, 'Keyboard', 49.5, 'pending'),
    (4, 3, 'Monitor', 199.0, 'shipped'),
    (5, 4, 'USB cable', 12.0, 'pending');
"""

_CASES = [
    {
        "id": "sql_select_city_0001",
        "question": "Return the names of the users who live in Tokyo.",
        "expected": [["Alice"], ["Carol"]],
    },
    {
        "id": "sql_count_0002",
        "question": "How many users are older than 30?",
        "expected": [[2]],
    },
    {
        "id": "sql_group_0003",
        "question": "For each order status, how many orders are there? Show the status and the count.",
        "expected": [["pending", 2], ["shipped", 3]],
    },
    {
        "id": "sql_join_0004",
        "question": (
            "List the name of every user who has placed an order, together with "
            "the product they ordered."
        ),
        "expected": [
            ["Alice", "Laptop"],
            ["Alice", "Mouse"],
            ["Bob", "Keyboard"],
            ["Carol", "Monitor"],
            ["David", "USB cable"],
        ],
    },
    {
        "id": "sql_order_0005",
        "question": "Which single order had the highest amount? Show the product and its amount.",
        "expected": [["Laptop", 899.0]],
    },
    {
        "id": "sql_agg_0006",
        "question": "What is the total amount of all shipped orders?",
        "expected": [[1127.99]],
    },
]

_MAX_QUERY_OPS = 100_000


def _rows_as_prompt_rows() -> str:
    """Render the seed rows so the model sees concrete data values."""
    return _SEED.strip()


def _extract_sql(text: str) -> str:
    """Pull a ```sql fenced block, else the raw text, down to the first ';'."""
    m = re.search(r"```(?:sql)?[^\n]*\n(.*?)```", text, re.DOTALL)
    sql = m.group(1).strip() if m else text.strip()
    idx = sql.find(";")
    if idx != -1:
        sql = sql[:idx].strip()
    return sql


def _norm_cell(value: Any) -> Any:
    """Comparable cell: numbers as floats, NULL as '', strings normalized."""
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return normalize_text(str(value))


def _result_rows(cursor: sqlite3.Cursor) -> list[tuple[Any, ...]]:
    return [tuple(_norm_cell(c) for c in row) for row in cursor.fetchall()]


class SqlPlugin(BaseTextPlugin):
    id: ClassVar[str] = "sql"
    name: ClassVar[str] = "SQL Generation"
    description: ClassVar[str] = (
        "Model must write a correct SQLite SELECT statement for a described schema."
    )
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.SQL
    version: ClassVar[str] = "0.1.0"
    dataset_version: ClassVar[str] = "v1"
    modalities: ClassVar[set[Modality]] = {Modality.TEXT}

    def supports_model(self, model) -> bool:  # noqa: ANN001
        return True

    def cases(self, ctx) -> Iterable[BenchmarkCase]:  # noqa: ANN001
        for spec in _CASES:
            yield BenchmarkCase(
                id=spec["id"],
                plugin_id=self.id,
                dataset_version=self.dataset_version,
                input={
                    "schema": f"{_SCHEMA.strip()}\n\n{_rows_as_prompt_rows()}",
                    "prompt": spec["question"],
                },
                expected={"rows": spec["expected"]},
            )

    def build_request(self, case, model, ctx) -> dict[str, Any]:  # noqa: ANN001
        return {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You write SQLite SQL queries. Respond with a single "
                        "SELECT statement matching the request, using only the "
                        "schema given below. No prose.\n\nSchema and data:\n"
                        f"{case.input['schema']}"
                    ),
                },
                {"role": "user", "content": case.input["prompt"]},
            ],
            "options": {"temperature": 0.0, "num_predict": 128},
        }

    async def evaluate(self, case, response, ctx) -> Evaluation:  # noqa: ANN001
        sql = _extract_sql(response.text)
        expected: list[list[Any]] = (case.expected or {}).get("rows", [])

        if not sql.lower().startswith("select"):
            return Evaluation(
                score=0.0,
                passed=False,
                metrics={
                    "sql_extracted": False,
                    "error": "no SQL SELECT statement found",
                    "sql": sql[:120],
                },
            )

        da = sqlite3.connect(":memory:")
        try:
            da.executescript(f"{_SCHEMA}\n{_SEED}")
            ops = {"count": 0}

            def _guard() -> int:
                ops["count"] += 1
                return 1 if ops["count"] > _MAX_QUERY_OPS else 0

            da.set_progress_handler(_guard, 1000)
            cur = da.cursor()
            cur.execute(sql)
            actual = _result_rows(cur)
        except sqlite3.Error as exc:
            return Evaluation(
                score=0.0,
                passed=False,
                metrics={
                    "sql_extracted": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "sql": sql[:120],
                },
            )
        finally:
            da.close()

        wanted = [
            tuple(_norm_cell(c) for c in row)
            for row in expected
        ]
        wanted_counts = Counter(wanted)
        actual_counts = Counter(actual)
        matched = sum((actual_counts & wanted_counts).values())
        max_rows = max(len(wanted), len(actual))
        recall = round(matched / max_rows, 4) if max_rows else 1.0

        return Evaluation(
            score=recall,
            passed=recall == 1.0,
            metrics={
                "sql_extracted": True,
                "rows_expected": len(wanted),
                "rows_received": len(actual),
                "rows_matched": matched,
                "row_recall": recall,
            },
        )