# Arabic NLP Pipeline

Production-oriented baseline for Arabic text classification with secure data loading, Arabic-aware normalization, model training, artifact persistence, and tests.

## What is implemented

- Safe JSONL data ingestion with schema checks
- Arabic text normalization (diacritics, Alef variants, tatweel, spacing)
- TF-IDF + Logistic Regression baseline classifier
- Train/test split with metrics (accuracy + macro F1)
- Saved model artifact and metrics report
- Unit tests for data loading and preprocessing

## Project layout

- `src/config.py`: paths and training configuration
- `src/data.py`: JSONL data loading and validation
- `src/preprocess.py`: Arabic normalization and preprocessing
- `src/model.py`: model definition and prediction helpers
- `src/train.py`: end-to-end training and persistence pipeline
- `tests/`: unit tests

## Expected dataset format

Store JSONL in `data/` (one object per line):

```json
{"text": "مرحبا كيف حالك", "label": "greeting"}
{"text": "اين موقع الفرع", "label": "question"}
```

## Setup

```bash
pip install -r requirements.txt
```

## Run training

```bash
python -m src.train
```

Outputs:

- Model: `models/arabic_text_classifier.joblib`
- Metrics: `logs/latest_metrics.json`
	- Includes accuracy, macro F1, per-class metrics, and confusion matrix

## Run inference

```python
from src.inference import load_model, predict_texts

model = load_model("arabic_text_classifier.joblib")
predictions = predict_texts(model, ["مرحبا", "ما اسمك"])
print(predictions)
```

## Run from the CLI

```bash
python -m src.cli train --data-file dataset.jsonl --model-file arabic_text_classifier.joblib
python -m src.cli predict --model-file arabic_text_classifier.joblib "مرحبا" "ما اسمك"
```

## Optional environment variables

- `ARABIC_NLP_DATA_FILE` (default: `dataset.jsonl`)
- `ARABIC_NLP_MODEL_FILE` (default: `arabic_text_classifier.joblib`)
- `ARABIC_NLP_TEST_SIZE` (default: `0.2`)
- `ARABIC_NLP_RANDOM_STATE` (default: `42`)
- `ARABIC_NLP_MIN_SAMPLES` (default: `10`)

## Tests

```bash
pytest -q
```

## Git safety defaults

This repository now includes defensive defaults to prevent accidental commits of secrets and local artifacts:

- `.gitignore` blocks `.env*`, keys/cert files, local data, logs, model binaries, virtualenv files, and caches.
- `.env.example` provides a safe template for local configuration.
- `.pre-commit-config.yaml` enables checks for private keys, merge conflicts, whitespace issues, large files, and secret scanning.
- `.secrets.baseline` is included for `detect-secrets` pre-commit scans.

Install and enable commit checks:

```bash
pip install -r requirements-dev.txt
pre-commit install
pre-commit run --all-files
```

## Recommended next production upgrades

- Add model/version metadata and reproducibility manifest
- Add CI pipeline with linting and test gates
- Add data drift and quality checks
- Add experiment tracking and model registry integration
- Add API serving layer with health checks and structured logging
