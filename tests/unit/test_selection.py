"""Unit tests for model selection (glob filtering + interactive parsing)."""

from local_ai_bench.domain.models import HostConfig, ModelInfo
from local_ai_bench.selection import (
    _resolve_indices,
    filter_models,
    matches_any,
    pick_hosts,
    split_patterns,
)


def _model(name: str) -> ModelInfo:
    return ModelInfo(host_name="h", model_name=name)


MODELS = [
    _model("llama3.2:latest"),
    _model("llama3.1:8b"),
    _model("qwen2.5-coder:14b"),
    _model("qwen3.5:0.8b"),
    _model("gemma4:12b"),
]


def test_split_patterns_comma_and_space():
    assert split_patterns("llama*,qwen* gemma*") == ["llama*", "qwen*", "gemma*"]
    assert split_patterns(None) == []


def test_matches_any_simple():
    assert matches_any("llama3.2:latest", ["llama*"])
    assert not matches_any("qwen:0.8b", ["llama*"])


def test_filter_models_no_patterns_keeps_all():
    assert filter_models(MODELS) == MODELS


def test_filter_models_include_glob():
    names = [m.model_name for m in filter_models(MODELS, include=["llama*"])]
    assert names == ["llama3.2:latest", "llama3.1:8b"]


def test_filter_models_include_exact():
    names = [m.model_name for m in filter_models(MODELS, include=["gemma4:12b"])]
    assert names == ["gemma4:12b"]


def test_filter_models_exclude_globs():
    names = [m.model_name for m in filter_models(MODELS, include=["*"], exclude=["*:0.8b", "gemma*"])]
    assert names == ["llama3.2:latest", "llama3.1:8b", "qwen2.5-coder:14b"]


def test_filter_models_combined():
    names = [m.model_name for m in filter_models(MODELS, include=["qwen*", "llama*"], exclude=["*:0.8b"])]
    assert names == ["llama3.2:latest", "llama3.1:8b", "qwen2.5-coder:14b"]


def test_resolve_indices_and_ranges():
    selected = _resolve_indices(MODELS, "0 2-3")
    assert [m.model_name for m in selected] == ["llama3.2:latest", "qwen2.5-coder:14b", "qwen3.5:0.8b"]


def test_resolve_indices_ignores_garbage():
    assert _resolve_indices(MODELS, "abc -2 99") == []


# --- pick_hosts (interactive host picker) ---


def _host(name: str, base_url: str = "http://example.invalid") -> HostConfig:
    return HostConfig(name=name, base_url=base_url)


HOSTS = [_host("local"), _host("lab-server"), _host("gpu-node")]


def test_pick_hosts_indices(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "0 2")
    assert [h.name for h in pick_hosts(HOSTS)] == ["local", "gpu-node"]


def test_pick_hosts_range(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "1-2")
    assert [h.name for h in pick_hosts(HOSTS)] == ["lab-server", "gpu-node"]


def test_pick_hosts_all(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "a")
    assert pick_hosts(HOSTS) == HOSTS


def test_pick_hosts_quit_returns_empty(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "q")
    assert pick_hosts(HOSTS) == []


def test_pick_hosts_empty_returns_empty(monkeypatch):
    assert pick_hosts([]) == []