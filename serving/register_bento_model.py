from __future__ import annotations

import argparse
from pathlib import Path

import bentoml

from src.predictor import load_model_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register trained model bundle into BentoML model store.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/model.joblib",
        help="Path to trained joblib bundle produced by training step.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="who_health_model",
        help="BentoML model name.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")

    bundle = load_model_bundle(model_path)
    model = bentoml.picklable_model.save_model(
        args.name,
        bundle,
        metadata={
            "target_col": bundle.get("target_col"),
            "feature_count": len(bundle.get("features", [])),
        },
    )
    print(f"Saved BentoML model: {model.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

