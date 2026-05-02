# Configuration for Arabic NLP pipeline.
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

DEFAULT_DATA_FILE = os.getenv("ARABIC_NLP_DATA_FILE", "dataset.jsonl")
DEFAULT_MODEL_FILE = os.getenv("ARABIC_NLP_MODEL_FILE", "arabic_text_classifier.joblib")


def _env_float(name: str, default: float) -> float:
	value = os.getenv(name)
	if value is None:
		return default
	try:
		return float(value)
	except ValueError as exc:
		raise ValueError(f"Environment variable {name} must be a float.") from exc


def _env_int(name: str, default: int) -> int:
	value = os.getenv(name)
	if value is None:
		return default
	try:
		return int(value)
	except ValueError as exc:
		raise ValueError(f"Environment variable {name} must be an integer.") from exc


@dataclass(frozen=True)
class TrainingConfig:
	test_size: float = _env_float("ARABIC_NLP_TEST_SIZE", 0.2)
	random_state: int = _env_int("ARABIC_NLP_RANDOM_STATE", 42)
	min_samples: int = _env_int("ARABIC_NLP_MIN_SAMPLES", 10)


def ensure_directories() -> None:
	"""Create required runtime directories if they do not exist."""
	DATA_DIR.mkdir(parents=True, exist_ok=True)
	MODEL_DIR.mkdir(parents=True, exist_ok=True)
	LOG_DIR.mkdir(parents=True, exist_ok=True)
