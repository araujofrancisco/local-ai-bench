"""Coding benchmark — correctness against executable test cases.

Uses the Python language by default (deterministic, available everywhere) but
the design is language-agnostic: each case ships a small unit test harness.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from typing import Any, ClassVar

from ollama_bench.domain.models import (
    BenchmarkCase,
    BenchmarkCategory,
    Evaluation,
    Modality,
)
from ollama_bench.plugins.builtin._base import BaseTextPlugin
from ollama_bench.plugins.score import extract_code, function_defined, python_syntax_ok

_CASES = [
    {
        "id": "code_reverse_string_0001",
        "prompt": (
            "Write a Python function named `reverse_string` that takes a string `s` "
            "and returns the reversed string. Return only the function code, no explanation."
        ),
        "function_name": "reverse_string",
        "tests": [
            ("assert reverse_string('hello') == 'olleh'", True),
            ("assert reverse_string('') == ''", True),
            ("assert reverse_string('abc') == 'cba'", True),
        ],
    },
    {
        "id": "code_is_even_0002",
        "prompt": (
            "Write a Python function named `is_even` that takes an integer `n` and "
            "returns True if n is even, False otherwise. Return only the function code."
        ),
        "function_name": "is_even",
        "tests": [
            ("assert is_even(0) is True", True),
            ("assert is_even(1) is False", True),
            ("assert is_even(42) is True", True),
            ("assert is_even(-7) is False", True),
        ],
    },
]


class CodingPlugin(BaseTextPlugin):
    id: ClassVar[str] = "coding"
    name: ClassVar[str] = "Coding"
    description: ClassVar[str] = "Correctness of generated code against unit tests."
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.CODING
    version: ClassVar[str] = "0.1.0"
    dataset_version: ClassVar[str] = "v1"
    modalities: ClassVar[set[Modality]] = {Modality.TEXT}

    def cases(self, ctx) -> Iterable[BenchmarkCase]:
        for spec in _CASES:
            yield BenchmarkCase(
                id=spec["id"],
                plugin_id=self.id,
                dataset_version=self.dataset_version,
                input={"prompt": spec["prompt"]},
                expected={
                    "function_name": spec["function_name"],
                    "tests": spec["tests"],
                },
            )

    def build_request(self, case, model, ctx) -> dict[str, Any]:  # noqa: ANN001
        return {
            "messages": [{"role": "user", "content": case.input["prompt"]}],
            "options": {"temperature": 0.0, "num_predict": 256},
        }

    async def evaluate(self, case, response, ctx) -> Evaluation:  # noqa: ANN001
        expected = case.expected
        fn_name = expected["function_name"]
        tests: list[tuple[str, bool]] = expected["tests"]

        source = extract_code(response.text)
        metrics: dict[str, Any] = {
            "syntax_ok": False,
            "function_defined": False,
            "execute_code": bool(ctx.options.get("execute_code", False)),
        }

        if not python_syntax_ok(source):
            metrics["syntax_error"] = True
            return Evaluation(score=0.0, passed=False, metrics=metrics)
        metrics["syntax_ok"] = True

        if not function_defined(source, fn_name):
            metrics["function_not_found"] = True
            return Evaluation(score=0.0, passed=False, metrics=metrics)
        metrics["function_defined"] = True

        if not metrics["execute_code"]:
            return Evaluation(
                score=1.0,
                passed=True,
                metrics={**metrics, "tests_total": 0},
            )

        timeout = int(ctx.options.get("timeout_seconds", 30))
        passed_count = 0
        total = len(tests)
        for test_code, should_pass in tests:
            snippet = f"{source}\n{test_code}\n"
            ok = _run_test_snippet(snippet, timeout)
            if ok == should_pass:
                passed_count += 1

        score = passed_count / total if total else 0.0
        return Evaluation(
            score=round(score, 4),
            passed=passed_count == total,
            metrics={
                **metrics,
                "tests_passed": passed_count,
                "tests_total": total,
            },
        )


def _run_test_snippet(snippet: str, timeout: int = 30) -> bool:
    """Execute a tiny Python snippet in a subprocess and check for exceptions."""
    path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, delete_on_close=False
        ) as f:
            f.write(snippet)
            path = f.name
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
        return proc.returncode == 0
    except Exception:
        return False
    finally:
        try:
            import os
            os.unlink(path)
        except Exception:
            pass
