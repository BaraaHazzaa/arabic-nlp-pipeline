from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import DATA_DIR


def get_data_directory() -> Path:
    return DATA_DIR


def _validate_record(record: Any, line_number: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"Line {line_number}: each JSONL line must be a JSON object.")
    if "text" not in record:
        raise ValueError(f"Line {line_number}: missing required field 'text'.")
    if not isinstance(record["text"], str):
        raise ValueError(f"Line {line_number}: field 'text' must be a string.")
    if not record["text"].strip():
        raise ValueError(f"Line {line_number}: field 'text' must not be empty.")
    if "label" not in record:
        raise ValueError(f"Line {line_number}: missing required field 'label'.")
    if not isinstance(record["label"], str):
        raise ValueError(f"Line {line_number}: field 'label' must be a string.")
    if not record["label"].strip():
        raise ValueError(f"Line {line_number}: field 'label' must not be empty.")
    return record


def load_data(file_name: str) -> list[dict[str, Any]]:
    """Load and validate JSONL records from the data directory."""
    file_path = get_data_directory() / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
            records.append(_validate_record(record, line_number))

    return records
