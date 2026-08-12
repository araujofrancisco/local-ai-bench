"""LLM-as-judge evaluation helper (PLAN §20).

Plugins can ask a configured judge model to score subjective outputs
(translation fluency, summarization faithfulness, vision correctness). Judge
failures never break a benchmark: they return None and callers fall back to
deterministic scoring (PLAN §20.4).
"""

from __future__ import annotations

import json
import re
from typing import Any

from local_ai_bench.domain.models import BenchmarkCase, Evaluation, ModelResponse
from local_ai_bench.ollama.client import OllamaClient
from local_ai_bench.plugins.base import RunContext
from local_ai_bench.plugins.score import blend_scores

_JUDGE_PROMPT = """\
You are an evaluation judge.
Evaluate the candidate response using the rubric.
Return only valid JSON with fields: "score" (0.0 to 1.0), "passed" (true/false), "rationale" (short).

Rubric:
{rubric}

Input:
{input_text}

Expected:
{expected}

Candidate response:
{candidate}
"""


class Judge:
    """Scores candidate responses using a local judge model via Ollama."""

    def __init__(self, client: OllamaClient, model: str, temperature: float = 0.0) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature

    async def score(
        self,
        rubric: str,
        input_text: str,
        expected: str,
        candidate: str,
    ) -> Evaluation | None:
        """Score a candidate. Returns None if the judge is unavailable."""
        prompt = _JUDGE_PROMPT.format(
            rubric=rubric,
            input_text=input_text,
            expected=expected,
            candidate=candidate,
        )
        try:
            resp: ModelResponse = await self.client.chat(
                self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature, "num_predict": 256},
                stream=False,
            )
        except Exception:  # noqa: BLE001 - judge must never crash the run
            return None
        if resp.error is not None:
            return None
        return _parse_judge_output(resp.text, self.model)


async def judge_evaluation(
    ctx: RunContext,
    case: BenchmarkCase,
    response: ModelResponse,
    *,
    rubric: str,
    deterministic_score: float,
    passed: bool,
    pass_threshold: float,
    metrics: dict[str, Any],
    judge_weight: float = 0.4,
) -> Evaluation:
    """Deterministic score optionally blended with a judge score (PLAN §15).

    The composite score is ``(1 - judge_weight) * deterministic +
    judge_weight * judge`` when a judge is configured and responds; otherwise
    the deterministic score is returned unchanged.
    """
    base = Evaluation(score=deterministic_score, passed=passed, metrics=metrics)
    judge = getattr(ctx, "judge", None)
    if judge is None or not response.text.strip():
        return base

    expected = json.dumps(case.expected, default=str) if case.expected else "-"
    input_text = json.dumps(case.input, default=str)
    judged = await judge.score(rubric, input_text, expected, response.text)
    if judged is None:
        return base

    score = blend_scores(deterministic_score, judged.score, judge_weight)
    if score is None:
        return base
    return Evaluation(
        score=score,
        passed=score >= pass_threshold,
        metrics={
            **metrics,
            "judge_score": judged.score,
            "judge_rationale": judged.rationale,
        },
        judge_model=judged.judge_model,
    )


def _parse_judge_output(text: str, judge_model: str) -> Evaluation | None:
    """Extract {score, passed, rationale} from judge output, tolerating noise."""
    parsed = _extract_json_object(text)
    if parsed is None:
        return None
    try:
        score = float(parsed.get("score"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    score = max(0.0, min(1.0, score))
    passed = parsed.get("passed")
    if not isinstance(passed, bool):
        passed = score >= 0.5
    return Evaluation(
        score=score,
        passed=passed,
        rationale=str(parsed.get("rationale") or "")[:500],
        judge_model=judge_model,
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object in ``text``, tolerating fences and prose."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None
