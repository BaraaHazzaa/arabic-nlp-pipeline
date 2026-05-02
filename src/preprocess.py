from __future__ import annotations

import re
from typing import Any

ARABIC_DIACRITICS_PATTERN = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]")
ARABIC_ALEF_PATTERN = re.compile(r"[إأآٱ]")
NON_TEXT_PATTERN = re.compile(r"[^\u0600-\u06FFa-zA-Z0-9\s]")
EXTRA_SPACE_PATTERN = re.compile(r"\s+")


def normalize_arabic_text(text: str) -> str:
    """Apply common Arabic normalization rules for model stability."""
    text = text.strip().lower()
    text = ARABIC_DIACRITICS_PATTERN.sub("", text)
    text = text.replace("ـ", "")  # Tatweel removal
    text = ARABIC_ALEF_PATTERN.sub("ا", text)
    text = text.replace("ى", "ي")
    text = NON_TEXT_PATTERN.sub(" ", text)
    text = EXTRA_SPACE_PATTERN.sub(" ", text)
    return text.strip()


def preprocess_record(item: dict[str, Any]) -> dict[str, Any]:
    cleaned_item = dict(item)
    cleaned_item["text"] = normalize_arabic_text(cleaned_item["text"])
    return cleaned_item


def preprocess_data(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [preprocess_record(item) for item in data]


def return_preprocessed_data(preprocessed_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return preprocessed_data
