"""Predict scores and constrained, model-estimated feature-group feedback for posters."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from score_feedback import ScoreOptimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/numeric")
    parser.add_argument("--features-csv", type=Path, default=ROOT / "data/posteriq/features.csv")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/posteriq")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image", type=Path, help="Extract features from one fresh image and return JSON")
    args = parser.parse_args()
    from autogluon.tabular import TabularPredictor

    report = json.loads((args.model_dir / "metrics.json").read_text())
    columns = report["feature_columns"]
    predictor = TabularPredictor.load(str(args.model_dir / "model"))
    recipe_path = args.model_dir / "feedback_recipe.json"
    if recipe_path.exists():
        optimizer = ScoreOptimizer.load_recipe(predictor, recipe_path)
        if optimizer.feature_names != columns:
            raise ValueError("Feedback recipe does not match model feature columns")
    else:
        features = pd.read_csv(args.features_csv, dtype={"id": str})
        train_ids = pd.read_csv(args.data_dir / "train.csv", dtype={"id": str})["id"]
        if features["id"].isna().any() or not features["id"].is_unique or not train_ids.is_unique:
            raise ValueError("Features and train IDs must be unique and non-missing")
        indexed = features.set_index("id")
        if not set(train_ids).issubset(indexed.index):
            raise ValueError("Feature table is missing training posters")
        optimizer = ScoreOptimizer(predictor, indexed.loc[train_ids, columns], columns, train_ids)
        optimizer.save_recipe(recipe_path)
    if args.image:
        from PIL import Image
        from features import extract_features
        with Image.open(args.image) as image:
            measurements, _ = extract_features(image)
        result = optimizer.analyze(measurements)
        result["image"] = str(args.image)
        text = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")
        print(text)
        return
    features = pd.read_csv(args.features_csv, dtype={"id": str})
    results = optimizer.analyze_many(features[columns])
    output = args.output or args.data_dir / "model_feedback.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [{"id": poster_id, **result} for poster_id, result in zip(features["id"], results)]
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    else:
        for record in records:
            for key in ("ranked_changes", "best_recommendation"):
                record[key] = json.dumps(record[key], allow_nan=False)
        pd.DataFrame(records).to_csv(output, index=False)
    print(f"Saved {len(records)} score/feedback rows to {output}")


if __name__ == "__main__":
    main()
