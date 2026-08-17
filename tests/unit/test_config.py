"""Unit tests for the configuration system."""

import pytest
from pydantic import ValidationError

from local_ai_bench.config import (
    DEFAULT_CONFIG_TEXT,
    BenchmarkConfig,
    config_hash,
    load_config,
    write_default_config,
)

VALID_YAML = """\
app:
  name: LocalAIBench
  output_dir: ./reports

hosts:
  - name: a
    base_url: http://127.0.0.1:11434
  - name: b
    base_url: http://127.0.0.2:11434

models:
  - llama3.2:latest

plugins:
  enabled: [smoke]
"""


def test_config_hash_is_stable():
    a = BenchmarkConfig()
    b = BenchmarkConfig()
    assert config_hash(a) == config_hash(b)


def test_config_hash_changes_with_meaningful_options():
    a = BenchmarkConfig()
    b = BenchmarkConfig.model_validate({"runner": {"repetitions": 5}})
    assert config_hash(a) != config_hash(b)


def test_duplicate_host_names_rejected():
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(
            {
                "hosts": [
                    {"name": "x", "base_url": "http://a"},
                    {"name": "x", "base_url": "http://b"},
                ]
            }
        )


def test_load_valid_config(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(VALID_YAML)
    cfg = load_config(p)
    assert len(cfg.hosts) == 2
    assert cfg.models == ["llama3.2:latest"]


def test_write_default_config(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_SERVER_URL", "http://192.168.10.109:11434")
    target = tmp_path / "config.yaml"
    write_default_config(target)
    assert target.exists()
    cfg = load_config(target)
    assert cfg.app.name == "LocalAIBench"
    assert [h.name for h in cfg.hosts] == ["local", "lab-server"]


def test_default_config_falls_back_to_local_host(tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setenv("LAB_SERVER_URL", "http://192.168.10.109:11434")
    p = tmp_path / "config.yaml"
    write_default_config(p)
    cfg = load_config(p)
    assert [h.name for h in cfg.hosts] == ["local", "lab-server"]
    assert cfg.hosts[0].name == "local"
    assert cfg.hosts[0].base_url == "http://127.0.0.1:11434"
    assert cfg.hosts[1].base_url == "http://192.168.10.109:11434"


def test_default_local_host_honors_ollama_host(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.example:11435")
    cfg = BenchmarkConfig()
    assert cfg.hosts[0].base_url == "http://ollama.example:11435"


def test_default_host_uses_local_when_no_env(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    cfg = BenchmarkConfig()
    assert cfg.hosts[0].base_url == "http://127.0.0.1:11434"


# --- ${VAR} / ${VAR:-default} expansion in hosts.base_url ---


def test_env_expansion_uses_default_when_var_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    from local_ai_bench.domain.models import HostConfig

    h = HostConfig(name="local", base_url="${OLLAMA_HOST:-http://127.0.0.1:11434}")
    assert h.base_url == "http://127.0.0.1:11434"


def test_env_expansion_uses_env_value(monkeypatch):
    monkeypatch.setenv("LAB_SERVER_URL", "http://192.168.10.109:11434")
    from local_ai_bench.domain.models import HostConfig

    h = HostConfig(name="lab-server", base_url="${LAB_SERVER_URL}")
    assert h.base_url == "http://192.168.10.109:11434"


def test_env_expansion_required_missing_raises(monkeypatch):
    monkeypatch.delenv("LAB_SERVER_URL", raising=False)
    from local_ai_bench.domain.models import HostConfig

    with pytest.raises(ValidationError, match="LAB_SERVER_URL"):
        HostConfig(name="lab-server", base_url="${LAB_SERVER_URL}")


def test_env_expansion_in_config_load(monkeypatch, tmp_path):
    monkeypatch.setenv("LAB_SERVER_URL", "http://192.168.10.109:11434")
    p = tmp_path / "config.yaml"
    p.write_text(
        "hosts:\n"
        "  - name: local\n"
        "    base_url: http://127.0.0.1:11434\n"
        "  - name: lab-server\n"
        "    base_url: ${LAB_SERVER_URL}\n"
    )
    cfg = load_config(p)
    assert len(cfg.hosts) == 2
    assert cfg.hosts[1].base_url == "http://192.168.10.109:11434"


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_compare_default_defaults_empty():
    assert BenchmarkConfig().plugins.compare_default == []


def test_compare_default_from_yaml(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(VALID_YAML + "  compare_default: [translation, coding]\n")
    cfg = load_config(p)
    assert cfg.plugins.compare_default == ["translation", "coding"]


def test_plugin_options_are_plain_dicts():
    cfg = BenchmarkConfig()
    assert isinstance(cfg.plugins.options, dict)
    for inner in cfg.plugins.options.values():
        assert isinstance(inner, dict)


def test_default_config_matches_packaged(tmp_path):
    target = tmp_path / "config.yaml"
    write_default_config(target)
    assert target.read_text(encoding="utf-8").rstrip("\n") == DEFAULT_CONFIG_TEXT.rstrip("\n")


def test_packaged_default_config_falls_back(tmp_path, monkeypatch):
    import local_ai_bench.config as config_mod

    monkeypatch.setattr(config_mod, "_packaged_default_config", lambda: None)
    p = tmp_path / "config.yaml"
    write_default_config(p)
    assert p.read_text(encoding="utf-8").rstrip("\n") == DEFAULT_CONFIG_TEXT.rstrip("\n")