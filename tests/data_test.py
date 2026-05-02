from __future__ import annotations

import pytest

from src import data as data_module


def test_load_data_reads_valid_jsonl(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_path = data_dir / "dataset.jsonl"
    file_path.write_text('{"text":"مرحبا","label":"greeting"}\n', encoding="utf-8")

    monkeypatch.setattr(data_module, "DATA_DIR", data_dir)
    rows = data_module.load_data("dataset.jsonl")

    assert len(rows) == 1
    assert rows[0]["text"] == "مرحبا"
    assert rows[0]["label"] == "greeting"


def test_load_data_raises_on_invalid_json(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_path = data_dir / "bad.jsonl"
    file_path.write_text('not-json\n', encoding="utf-8")

    monkeypatch.setattr(data_module, "DATA_DIR", data_dir)

    with pytest.raises(ValueError):
        data_module.load_data("bad.jsonl")


def test_load_data_raises_when_file_missing(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(data_module, "DATA_DIR", data_dir)

    with pytest.raises(FileNotFoundError):
        data_module.load_data("missing.jsonl")


def test_load_data_rejects_missing_label(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_path = data_dir / "bad.jsonl"
    file_path.write_text('{"text":"مرحبا"}\n', encoding="utf-8")

    monkeypatch.setattr(data_module, "DATA_DIR", data_dir)

    with pytest.raises(ValueError, match="missing required field 'label'"):
        data_module.load_data("bad.jsonl")


def test_load_data_rejects_empty_text(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_path = data_dir / "bad.jsonl"
    file_path.write_text('{"text":"   ","label":"greeting"}\n', encoding="utf-8")

    monkeypatch.setattr(data_module, "DATA_DIR", data_dir)

    with pytest.raises(ValueError, match="field 'text' must not be empty"):
        data_module.load_data("bad.jsonl")
