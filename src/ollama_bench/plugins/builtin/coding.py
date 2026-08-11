"""Coding benchmark — correctness against executable unit tests.

Uses the Python language by default (deterministic, available everywhere).
Each case ships a small unit-test harness with edge cases (empty inputs,
negatives, unicode, duplicates, large inputs) so that real-world models are
discriminated rather than collapsing to a perfect score.

Generated code is executed in an isolated subprocess per assertion
(``python -I``, per-case time budget) so a broken solution cannot hang the
benchmark or read the surrounding environment.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from contextlib import suppress
from typing import Any, ClassVar

from ollama_bench.domain.models import (
    BenchmarkCase,
    BenchmarkCategory,
    Evaluation,
    Modality,
)
from ollama_bench.plugins.builtin._base import BaseTextPlugin
from ollama_bench.plugins.score import extract_python, python_syntax_ok, symbol_defined

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
            ("assert reverse_string('a  b') == 'b  a'", True),
            ("assert reverse_string('héllo') == 'olléh'", True),
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
            ("assert is_even(10**12) is True", True),
        ],
    },
    {
        "id": "code_fizzbuzz_0003",
        "prompt": (
            "Write a Python function named `fizzbuzz` that takes an integer `n` and "
            "returns a list of strings for the numbers 1..n: 'Fizz' if divisible by 3, "
            "'Buzz' if divisible by 5, 'FizzBuzz' if divisible by both, otherwise the "
            "number itself as a string. Return only the function code."
        ),
        "function_name": "fizzbuzz",
        "tests": [
            ("assert fizzbuzz(1) == ['1']", True),
            ("assert fizzbuzz(5) == ['1','2','Fizz','4','Buzz']", True),
            ("assert fizzbuzz(15)[-1] == 'FizzBuzz'", True),
            ("assert fizzbuzz(15)[2] == 'Fizz'", True),
            ("assert fizzbuzz(0) == []", True),
        ],
    },
    {
        "id": "code_is_anagram_0004",
        "prompt": (
            "Write a Python function named `is_anagram` that takes two strings `a` and "
            "`b` and returns True if they are anagrams of each other (same characters, "
            "same counts), case-sensitive. Return only the function code."
        ),
        "function_name": "is_anagram",
        "tests": [
            ("assert is_anagram('anagram', 'nagaram') is True", True),
            ("assert is_anagram('rat', 'car') is False", True),
            ("assert is_anagram('', '') is True", True),
            ("assert is_anagram('aab', 'baa') is True", True),
            ("assert is_anagram('A', 'a') is False", True),
        ],
    },
    {
        "id": "code_is_palindrome_0005",
        "prompt": (
            "Write a Python function named `is_palindrome` that takes a string `s` and "
            "returns True if `s` is a palindrome ignoring case and non-alphanumeric "
            "characters. Return only the function code."
        ),
        "function_name": "is_palindrome",
        "tests": [
            ("assert is_palindrome('A man, a plan, a canal: Panama') is True", True),
            ("assert is_palindrome('race a car') is False", True),
            ("assert is_palindrome('') is True", True),
            ("assert is_palindrome('ab_a') is True", True),
            ("assert is_palindrome('abc') is False", True),
        ],
    },
    {
        "id": "code_two_sum_0006",
        "prompt": (
            "Write a Python function named `two_sum` that takes a list of integers "
            "`nums` and a target integer `target`, and returns a list of the two indices "
            "whose values add up to `target`, or None if no such pair exists. Return only "
            "the function code."
        ),
        "function_name": "two_sum",
        "tests": [
            ("assert sorted(two_sum([2, 7, 11, 15], 9)) == [0, 1]", True),
            ("assert sorted(two_sum([3, 2, 4], 6)) == [1, 2]", True),
            ("assert sorted(two_sum([3, 3], 6)) == [0, 1]", True),
            ("assert sorted(two_sum([5, -2, 3], 3)) == [1, 2]", True),
            ("assert two_sum([1], 2) is None", True),
        ],
    },
    {
        "id": "code_valid_parentheses_0007",
        "prompt": (
            "Write a Python function named `valid_parentheses` that takes a string `s` "
            "containing only '(', ')', '{', '}', '[' and ']' and returns True if the "
            "brackets are correctly closed in the right order. Return only the function code."
        ),
        "function_name": "valid_parentheses",
        "tests": [
            ("assert valid_parentheses('()') is True", True),
            ("assert valid_parentheses('()[]{}') is True", True),
            ("assert valid_parentheses('(]') is False", True),
            ("assert valid_parentheses('([)]') is False", True),
            ("assert valid_parentheses('{[]}') is True", True),
            ("assert valid_parentheses('(' * 100 + ')' * 100) is True", True),
            ("assert valid_parentheses('(') is False", True),
        ],
    },
    {
        "id": "code_merge_intervals_0008",
        "prompt": (
            "Write a Python function named `merge_intervals` that takes a list of "
            "intervals `intervals`, where each interval is a list [start, end], and returns "
            "a list of merged non-overlapping intervals sorted by start. The input may be "
            "unsorted. Return only the function code."
        ),
        "function_name": "merge_intervals",
        "tests": [
            ("assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]", True),
            ("assert merge_intervals([[1,4],[4,5]]) == [[1,5]]", True),
            ("assert merge_intervals([[2,3],[1,4]]) == [[1,4]]", True),
            ("assert merge_intervals([]) == []", True),
            ("assert merge_intervals([[1,4]]) == [[1,4]]", True),
        ],
    },
    {
        "id": "code_max_subarray_0009",
        "prompt": (
            "Write a Python function named `max_subarray` that takes a list of integers "
            "`nums` and returns the maximum sum of any contiguous subarray (Kadane's "
            "algorithm). For all-negative input, return the largest single element. "
            "Return only the function code."
        ),
        "function_name": "max_subarray",
        "tests": [
            ("assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6", True),
            ("assert max_subarray([1]) == 1", True),
            ("assert max_subarray([5,4,-1,7,8]) == 23", True),
            ("assert max_subarray([-2,-1]) == -1", True),
            ("assert max_subarray([-1,-2,-3]) == -1", True),
            ("assert max_subarray([]) == 0", True),
        ],
    },
    {
        "id": "code_longest_common_prefix_0010",
        "prompt": (
            "Write a Python function named `longest_common_prefix` that takes a list of "
            "strings `strs` and returns the longest common prefix string, or an empty "
            "string if there is none. Return only the function code."
        ),
        "function_name": "longest_common_prefix",
        "tests": [
            ("assert longest_common_prefix(['flower','flow','flight']) == 'fl'", True),
            ("assert longest_common_prefix(['dog','racecar','car']) == ''", True),
            ("assert longest_common_prefix(['a']) == 'a'", True),
            ("assert longest_common_prefix(['']) == ''", True),
            ("assert longest_common_prefix([]) == ''", True),
            ("assert longest_common_prefix(['ab','a']) == 'a'", True),
        ],
    },
    {
        "id": "code_lru_cache_0011",
        "prompt": (
            "Write a Python class named `LRUCache` with an __init__(self, capacity) "
            "method, plus get(self, key) returning the value or -1 if missing, and "
            "put(self, key, value). The least recently used key must be evicted when "
            "capacity is exceeded. get() and put() must be O(1) average. Return only the "
            "class code, no explanation."
        ),
        "function_name": "LRUCache",
        "tests": [
            ("cache = LRUCache(2)\ncache.put(1, 1)\ncache.put(2, 2)\nassert cache.get(1) == 1", True),
            (
                "cache = LRUCache(2)\ncache.put(1, 1)\ncache.put(2, 2)\n"
                "cache.get(1)\n"
                "cache.put(3, 3)\n"
                "assert cache.get(2) == -1",
                True,
            ),
            (
                "cache = LRUCache(2)\ncache.put(1,1)\ncache.put(2,2)\ncache.get(1)\n"
                "cache.put(3,3)\nassert cache.get(2) == -1\n"
                "cache.put(4,4)\nassert cache.get(1) == -1\n"
                "assert cache.get(3) == 3\nassert cache.get(4) == 4",
                True,
            ),
            (
                "cache = LRUCache(1)\ncache.put(1,1)\nassert cache.get(1)==1\n"
                "cache.put(2,2)\nassert cache.get(1)==-1\nassert cache.get(2)==2",
                True,
            ),
            ("cache = LRUCache(2)\nassert cache.get(9) == -1", True),
        ],
    },
    {
        "id": "code_trie_0012",
        "prompt": (
            "Write a Python class named `Trie` with __init__(self), insert(self, word), "
            "search(self, word) returning True only if word was inserted whole, and "
            "starts_with(self, prefix) returning True if any inserted word starts with "
            "prefix. Return only the class code, no explanation."
        ),
        "function_name": "Trie",
        "tests": [
            (
                "t = Trie()\nt.insert('apple')\n"
                "assert t.search('apple') is True\n"
                "assert t.search('app') is False\n"
                "assert t.starts_with('app') is True\n"
                "t.insert('app')\n"
                "assert t.search('app') is True",
                True,
            ),
            ("t = Trie()\nassert t.search('') is False", True),
            (
                "t = Trie()\nt.insert('hello')\nt.insert('hell')\n"
                "assert t.starts_with('he') is True\n"
                "assert t.starts_with('world') is False",
                True,
            ),
        ],
    },
    {
        "id": "code_edit_distance_0013",
        "prompt": (
            "Write a Python function named `edit_distance` that takes two strings `a` "
            "and `b` and returns the minimum number of single-character insertions, "
            "deletions or substitutions required to change `a` into `b` (Levenshtein). "
            "Return only the function code."
        ),
        "function_name": "edit_distance",
        "tests": [
            ("assert edit_distance('horse', 'ros') == 3", True),
            ("assert edit_distance('intention', 'execution') == 5", True),
            ("assert edit_distance('', '') == 0", True),
            ("assert edit_distance('a', '') == 1", True),
            ("assert edit_distance('', 'a') == 1", True),
            ("assert edit_distance('abc', 'abc') == 0", True),
        ],
    },
    {
        "id": "code_n_queens_0014",
        "prompt": (
            "Write a Python function named `n_queens` that takes an integer `n` and "
            "returns the number of distinct ways to place n queens on an n x n board so "
            "that no two queens attack each other. Return only the function code."
        ),
        "function_name": "n_queens",
        "tests": [
            ("assert n_queens(1) == 1", True),
            ("assert n_queens(4) == 2", True),
            ("assert n_queens(5) == 10", True),
            ("assert n_queens(6) == 4", True),
            ("assert n_queens(8) == 92", True),
        ],
    },
]


class CodingPlugin(BaseTextPlugin):
    id: ClassVar[str] = "coding"
    name: ClassVar[str] = "Coding"
    description: ClassVar[str] = "Correctness of generated code against unit tests."
    category: ClassVar[BenchmarkCategory] = BenchmarkCategory.CODING
    version: ClassVar[str] = "0.1.0"
    dataset_version: ClassVar[str] = "v2"
    modalities: ClassVar[set[Modality]] = {Modality.TEXT}

    def cases(self, ctx) -> Iterable[BenchmarkCase]:  # noqa: ANN001
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
            "options": {"temperature": 0.0, "num_predict": 768},
        }

    async def evaluate(self, case, response, ctx) -> Evaluation:  # noqa: ANN001
        expected = case.expected
        fn_name = expected["function_name"]
        tests: list[tuple[str, bool]] = expected["tests"]

        source = extract_python(response.text)
        metrics: dict[str, Any] = {
            "syntax_ok": False,
            "symbol_defined": False,
            "execute_code": bool(ctx.options.get("execute_code", True)),
        }

        if not python_syntax_ok(source):
            metrics["syntax_error"] = True
            return Evaluation(score=0.0, passed=False, metrics=metrics)
        metrics["syntax_ok"] = True

        if not symbol_defined(source, fn_name):
            metrics["symbol_not_found"] = True
            return Evaluation(score=0.0, passed=False, metrics=metrics)
        metrics["symbol_defined"] = True

        if not metrics["execute_code"]:
            return Evaluation(
                score=1.0,
                passed=True,
                metrics={**metrics, "tests_total": 0},
            )

        # Cumulative wall-clock budget across all assertions of the case so a
        # pathological solution cannot stall the whole benchmark.
        timeout = max(1, int(ctx.options.get("timeout_seconds", 30)))
        deadline = time.monotonic() + timeout
        passed_count = 0
        total = len(tests)
        stderr_tail = ""
        for test_code, should_pass in tests:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            snippet = f"{source}\n{test_code}\n"
            ok, err = _run_test_snippet(snippet, max(1, remaining))
            if err:
                stderr_tail = err[-400:]
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
                **({"stderr": stderr_tail} if stderr_tail else {}),
            },
        )


def _run_test_snippet(snippet: str, timeout: int = 30) -> tuple[bool, str]:
    """Execute a tiny Python snippet in an isolated subprocess.

    Returns ``(ok, stderr_tail)``; ``ok`` is True when the process exits cleanly.
    """
    path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, delete_on_close=False
        ) as f:
            f.write(snippet)
            path = f.name
        proc = subprocess.run(
            # -I = isolated: no user site-packages, no env overrides.
            [sys.executable, "-I", path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
        return proc.returncode == 0, (proc.stderr or "") + (proc.stdout or "")
    except (subprocess.TimeoutExpired, OSError, ValueError) as err:
        return False, type(err).__name__
    finally:
        with suppress(Exception):
            os.unlink(path)