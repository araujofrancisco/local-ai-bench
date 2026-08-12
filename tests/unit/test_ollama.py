"""Unit tests for Ollama metric mapping and discovery."""

from local_ai_bench.ollama.discovery import model_info_from_entry
from local_ai_bench.ollama.metrics import timing_from_done, tokens_from_done

DONE = {
    "total_duration": 1_000_000_000,  # 1000 ms
    "load_duration": 900_000_000,
    "prompt_eval_count": 30,
    "prompt_eval_duration": 50_000_000,  # 50 ms
    "eval_count": 20,
    "eval_duration": 200_000_000,  # 200 ms -> 100 tok/s
}


def test_timing_from_done():
    t = timing_from_done(DONE, started_at=0.0, wall_ms=1000.0)
    assert t.total_ms == 1000.0
    assert t.load_ms == 900.0
    assert abs(t.time_to_first_token_ms - 50.0) < 1e-6
    assert t.generation_ms == 200.0


def test_tokens_from_done():
    tokens = tokens_from_done(DONE)
    assert tokens.prompt_tokens == 30
    assert tokens.completion_tokens == 20
    assert abs(tokens.tokens_per_second - 100.0) < 1e-6  # 20 / 0.2s


def test_model_info_from_entry_parses_capabilities():
    entry = {
        "name": "llava:7b",
        "digest": "abc123",
        "details": {"context_length": 4096, "quantization_level": "Q4_K_M"},
        "capabilities": ["completion", "vision", "tools"],
    }
    info = model_info_from_entry("host1", entry)
    assert info.model_name == "llava:7b"
    assert info.supports_vision is True
    assert info.supports_tools is True
    assert info.max_context_tokens == 4096


def test_model_info_capabilities_fallback_to_details():
    entry = {"name": "m", "details": {"capabilities": ["vision", "completion"]}}
    info = model_info_from_entry("h", entry)
    assert info.supports_vision is True


def test_model_info_name_fallback():
    info = model_info_from_entry("h", {"model": "other-name"})
    assert info.model_name == "other-name"