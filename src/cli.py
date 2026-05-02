from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Sequence

from src.config import TrainingConfig
from src.inference import load_model, predict_texts
from src.train import run_training_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Arabic NLP pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save a model")
    train_parser.add_argument("--data-file", default=None, help="JSONL file in the data directory")
    train_parser.add_argument("--model-file", default=None, help="Output model filename")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    train_parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    train_parser.add_argument("--min-samples", type=int, default=10, help="Minimum training samples")

    predict_parser = subparsers.add_parser("predict", help="Predict labels for one or more texts")
    predict_parser.add_argument("--model-file", required=True, help="Saved model filename or path")
    predict_parser.add_argument("text", nargs="+", help="One or more texts to classify")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        config = TrainingConfig(
            test_size=args.test_size,
            random_state=args.random_state,
            min_samples=args.min_samples,
        )
        result = run_training_pipeline(
            data_file=args.data_file or "dataset.jsonl",
            model_file=args.model_file or "arabic_text_classifier.joblib",
            config=config,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "predict":
        model = load_model(args.model_file)
        predictions = predict_texts(model, args.text)
        print(json.dumps({"predictions": predictions, "input_texts": list(args.text)}, ensure_ascii=False, indent=2))
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
