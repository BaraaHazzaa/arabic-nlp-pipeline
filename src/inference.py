from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib

from src.config import MODEL_DIR
from src.preprocess import normalize_arabic_text


def load_model(model_file: str | Path):
    """Load a persisted sklearn pipeline from disk."""
    model_path = Path(model_file)
    if not model_path.is_absolute():
        model_path = MODEL_DIR / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def prepare_texts(texts: Iterable[str]) -> list[str]:
    return [normalize_arabic_text(text) for text in texts]


def predict_texts(model, texts: Iterable[str]) -> list[str]:
    """Normalize inputs and return predicted labels."""
    prepared_texts = prepare_texts(texts)
    return list(model.predict(prepared_texts))
