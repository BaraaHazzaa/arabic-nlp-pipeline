from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import platform
import sklearn
from sklearn.model_selection import train_test_split

from src.config import (
    DEFAULT_DATA_FILE,
    DEFAULT_MODEL_FILE,
    LOG_DIR,
    MODEL_DIR,
    TrainingConfig,
    ensure_directories,
)
from src.data import load_data
from src.model import define_model
from src.preprocess import preprocess_data

LOGGER = logging.getLogger(__name__)


def _setup_logging() -> None:
    ensure_directories()
    if LOGGER.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def train_model(
    preprocessed_data: list[dict[str, Any]],
    label_key: str = "label",
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    """Train and evaluate a text classifier from preprocessed records."""
    config = config or TrainingConfig()
    if len(preprocessed_data) < config.min_samples:
        raise ValueError(
            f"Not enough samples for training. Need at least {config.min_samples}, got {len(preprocessed_data)}."
        )

    missing_label = [row for row in preprocessed_data if label_key not in row]
    if missing_label:
        raise ValueError(f"All rows must include '{label_key}' for supervised training.")

    texts = [row["text"] for row in preprocessed_data]
    labels = [row[label_key] for row in preprocessed_data]

    unique_labels = set(labels)
    stratify = labels if len(unique_labels) > 1 else None

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )

    model = define_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    label_order = sorted(unique_labels)
    report = classification_report(
        y_test,
        predictions,
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=label_order).tolist()

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_macro": f1_score(y_test, predictions, average="macro"),
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "labels": label_order,
        "classification_report": report,
        "confusion_matrix": matrix,
    }

    return {"model": model, "metrics": metrics}


def save_model(model: Any, file_name: str) -> Path:
    ensure_directories()
    model_path = MODEL_DIR / file_name
    joblib.dump(model, model_path)
    return model_path


def run_training_pipeline(
    data_file: str = DEFAULT_DATA_FILE,
    model_file: str = DEFAULT_MODEL_FILE,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    _setup_logging()
    records = load_data(data_file)
    preprocessed_records = preprocess_data(records)
    results = train_model(preprocessed_records, config=config)
    model_path = save_model(results["model"], model_file)

    metrics_path = LOG_DIR / "latest_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file_handle:
        json.dump(results["metrics"], file_handle, ensure_ascii=False, indent=2)

    # Write a manifest alongside the model file with metadata for reproducibility
    manifest = {
        "model_file": str(model_path.name),
        "metrics": results["metrics"],
        "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }
    manifest_path = MODEL_DIR / (Path(model_file).name + ".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)

    LOGGER.info("Training complete. Model saved to %s", model_path)
    LOGGER.info("Metrics saved to %s", metrics_path)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metrics": results["metrics"],
    }


if __name__ == "__main__":
    run_training_pipeline()
