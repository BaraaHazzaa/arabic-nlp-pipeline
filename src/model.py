from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def define_model() -> Pipeline:
    """Create a strong baseline model for Arabic text classification."""
    return Pipeline(
        steps=[
            ("vectorizer", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def forward_pass(model: Pipeline, inputs: list[str]):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(inputs)
    return model.predict(inputs)


def backward_pass(loss: float) -> float:
    """Placeholder to preserve API compatibility with the original skeleton."""
    return max(0.0, float(loss))
