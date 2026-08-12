"""Unit tests for the LLM-as-judge helper and the context optimizer."""

from __future__ import annotations

from local_ai_bench.context.optimizer import recommend
from local_ai_bench.judge import Judge, _parse_judge_output

# ---------- Judge output parsing ----------


def test_parse_judge_output_clean_json():
    ev = _parse_judge_output('{"score": 0.8, "passed": true, "rationale": "good"}', "j1")
    assert ev is not None
    assert ev.score == 0.8
    assert ev.passed is True
    assert ev.judge_model == "j1"


def test_parse_judge_output_tolerates_fences_and_prose():
    raw = 'Here is my evaluation:\n```json\n{"score": 0.5, "passed": false}\n```\nThanks'
    ev = _parse_judge_output(raw, "j1")
    assert ev is not None
    assert ev.score == 0.5
    assert ev.passed is False


def test_parse_judge_output_invalid_returns_none():
    assert _parse_judge_output("not json at all", "j1") is None


def test_parse_judge_output_clamps_score():
    ev = _parse_judge_output('{"score": 5.0, "passed": true}', "j1")
    assert ev is not None
    assert ev.score == 1.0


def test_judge_score_returns_none_when_chat_fails():
    class _FailingClient:
        async def chat(self, *args, **kwargs):
            raise RuntimeError("host unreachable")

    judge = Judge(_FailingClient(), "m")  # type: ignore[arg-type]

    async def _run() -> None:
        ev = await judge.score("rubric", "in", "exp", "cand")
        assert ev is None

    import asyncio

    asyncio.run(_run())


# ---------- Context optimizer (PLAN §16.6) ----------


def test_recommend_picks_smallest_acceptable_size():
    per = {512: 0.6, 1024: 0.95, 4096: 0.97, 16384: 0.3}
    rec = recommend(per, [512, 1024, 4096, 16384], quality_threshold=0.90)
    assert rec["recommended_context"] == 1024
    assert rec["reason"] == "smallest stable size meeting the quality threshold"
    assert len(rec["curve"]) == 4


def test_recommend_applies_latency_budget():
    per = {1024: 0.95, 4096: 0.97}
    lat = {1024: 500.0, 4096: 5000.0}
    rec = recommend(per, [1024, 4096], quality_threshold=0.90, latency_budget_ms=1000.0, per_context_latency=lat)
    assert rec["recommended_context"] == 1024


def test_recommend_none_meets_threshold_uses_largest():
    per = {512: 0.5, 1024: 0.6}
    lat = {512: 100.0, 1024: 5000.0}
    rec = recommend(per, [512, 1024], quality_threshold=0.90, latency_budget_ms=1000.0, per_context_latency=lat)
    assert rec["recommended_context"] == 1024
    assert "largest" in rec["reason"]


def test_recommend_no_results():
    rec = recommend({}, [512, 1024])
    assert rec["recommended_context"] is None


def test_recommend_ignores_sizes_outside_candidates():
    per = {512: 1.0, 9999: 1.0}
    rec = recommend(per, [512, 1024, 2048])
    assert rec["recommended_context"] == 512
