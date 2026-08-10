"""Shared, deterministic scoring helpers used by built-in plugins."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from typing import Any


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for fuzzy comparison."""
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return text.strip()


def keyword_recall(text: str, keywords: list[str]) -> float:
    """Fraction of keywords found (normalized substring match) in text."""
    if not keywords:
        return 1.0
    norm = normalize_text(text)
    matched = [kw for kw in keywords if normalize_text(kw) in norm]
    return round(len(matched) / len(keywords), 4)


def normalize_answer(text: str) -> str:
    """Extract a bare numeric answer from noisy model output."""
    text = normalize_text(text)
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return matches[-1] if matches else text


def numeric_close(text: str, expected: str, tolerance: float = 0.5) -> bool:
    try:
        return abs(float(normalize_answer(text)) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return normalize_answer(text) == normalize_text(expected)


def valid_json_with_fields(raw: str, required: list[str]) -> tuple[float, dict[str, Any]]:
    """Score JSON/schema compliance.

    Returns (score in [0,1], metrics dict). Strips markdown fences and trailing
    prose, attempts light repair (extract first JSON object/array).
    """
    text = raw.strip()
    # Remove markdown fences.
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    parsed = None
    # Try direct parse first.
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Light repair: extract the first balanced { ... } / [ ... ].
        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            if start == -1:
                continue
            depth = 0
            for idx in range(start, len(text)):
                if text[idx] == opener:
                    depth += 1
                elif text[idx] == closer:
                    depth -= 1
                    if depth == 0:
                        candidates = text[start : idx + 1]
                        try:
                            parsed = json.loads(candidates)
                        except json.JSONDecodeError:
                            try:
                                parsed = json.loads(json.dumps(candidates))
                            except json.JSONDecodeError:
                                continue
                        break
            if parsed is not None:
                break

    metrics: dict[str, Any] = {"valid_json": False, "fields_present": 0, "fields_total": len(required)}
    if isinstance(parsed, dict):
        metrics["valid_json"] = True
        present = [f for f in required if f in parsed]
        metrics["fields_present"] = len(present)
        if required:
            return round(len(present) / len(required), 4), metrics
        return 1.0, metrics
    if isinstance(parsed, list):
        metrics["valid_json"] = True
    return 0.0, metrics


def is_json_valid(text: str) -> bool:
    try:
        json.loads(text.strip())
        return True
    except json.JSONDecodeError:
        # light repair attempt
        repaired = _extract_json(text)
        if repaired is None:
            return False
        try:
            json.loads(repaired)
            return True
        except json.JSONDecodeError:
            return False


def _extract_json(text: str) -> str | None:
    text = text.strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for idx in range(start, len(text)):
            if text[idx] == opener:
                depth += 1
            elif text[idx] == closer:
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
    return None


def python_syntax_ok(source: str) -> bool:
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def function_defined(source: str, function_name: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.FunctionDef) and node.name == function_name
        or (isinstance(node, ast.AsyncFunctionDef) and node.name == function_name)
        for node in ast.walk(tree)
    )


def extract_code(source: str) -> str:
    """Extract a ```python fenced block if present, else strip prose heuristically."""
    m = re.search(r"```python\n(.*?)```", source, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```(.*?)```", source, re.DOTALL)
    if m:
        return m.group(1)
    return source


def most_common(items: list[str]) -> str | None:
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]


def blend_scores(
    deterministic: float | None,
    judge: float | None,
    judge_weight: float = 0.4,
) -> float | None:
    """Blend a deterministic score with an optional judge score.

    When the judge score is missing, the deterministic score is used as-is.
    """
    if judge is None:
        return deterministic
    if deterministic is None:
        return judge
    return round(deterministic * (1 - judge_weight) + judge * judge_weight, 4)