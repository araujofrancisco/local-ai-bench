"""Unit tests for the configuration system."""

import pytest
from pydantic import ValidationError

from ollama_bench.config import BenchmarkConfig, config_hash, load_config, write_default_config

VALID_YAML = """\
app:
  name: OllamaBench
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


def test_write_default_config(tmp_path):
    target = tmp_path / "config.yaml"
    write_default_config(target)
    assert target.exists()
    cfg = load_config(target)
    assert cfg.app.name == "OllamaBench"


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")