from __future__ import annotations

import json
from pathlib import Path

from src import data as data_module
from src import train as train_module
from src.config import TrainingConfig
from src.inference import load_model, predict_texts


def test_full_pipeline_train_save_load_predict(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    log_dir = tmp_path / "logs"
    data_dir.mkdir()
    model_dir.mkdir()
    log_dir.mkdir()

    dataset = data_dir / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            [
                '{"text":"مرحبا","label":"greeting"}',
                '{"text":"اهلا","label":"greeting"}',
                '{"text":"ما اسمك","label":"question"}',
                '{"text":"اين انت","label":"question"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(data_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(train_module, "MODEL_DIR", model_dir)
    monkeypatch.setattr(train_module, "LOG_DIR", log_dir)

    config = TrainingConfig(test_size=0.5, random_state=42, min_samples=4)
    results = train_module.run_training_pipeline(
        data_file="dataset.jsonl",
        model_file="pipeline.joblib",
        config=config,
    )

    model_path = Path(results["model_path"])
    metrics_path = Path(results["metrics_path"])

    assert model_path.exists()
    assert metrics_path.exists()
    manifest_path = model_path.with_suffix(model_path.suffix + ".manifest.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "metrics" in manifest
    assert 0.0 <= results["metrics"]["accuracy"] <= 1.0
    assert 0.0 <= results["metrics"]["f1_macro"] <= 1.0

    loaded_model = load_model(model_path)
    predictions = predict_texts(loaded_model, ["مرحبا", "ما اسمك"])

    assert len(predictions) == 2
    assert set(predictions).issubset({"greeting", "question"})


def test_train_model_rejects_insufficient_samples():
    rows = [
        {"text": "مرحبا", "label": "greeting"},
        {"text": "اهلا", "label": "greeting"},
    ]

    try:
        train_module.train_model(rows, config=TrainingConfig(min_samples=4))
    except ValueError as exc:
        assert "Not enough samples" in str(exc)
    else:
        raise AssertionError("Expected ValueError for insufficient samples")
