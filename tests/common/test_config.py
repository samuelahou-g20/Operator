import pytest

from src.common import config


def test_load_config_reads_yaml(monkeypatch, tmp_path):
    (tmp_path / "sample.yaml").write_text("a: 1\nb: [x, y]\n", encoding="utf-8")
    monkeypatch.setattr(config.paths, "CONFIG_DIR", tmp_path)
    assert config.load_config("sample.yaml") == {"a": 1, "b": ["x", "y"]}


def test_load_config_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(config.paths, "CONFIG_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        config.load_config("nope.yaml")


def test_load_config_not_a_mapping(monkeypatch, tmp_path):
    (tmp_path / "list.yaml").write_text("- 1\n- 2\n", encoding="utf-8")
    monkeypatch.setattr(config.paths, "CONFIG_DIR", tmp_path)
    with pytest.raises(ValueError):
        config.load_config("list.yaml")
