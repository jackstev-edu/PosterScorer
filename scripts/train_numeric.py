"""Train AutoGluon regression on numeric poster measurements and fixed splits."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/posteriq"
TARGET = "overall_design_score"
DEFAULT_FEATURES = ["character_count", "text_density", "aspect_ratio"]
RESERVED = {"id", "image_path", TARGET, "split"}


def read_table(path):
    frame = pd.read_csv(path, dtype={"id": "string"})
    if "id" not in frame or frame["id"].isna().any() or frame["id"].str.strip().eq("").any():
        raise ValueError(f"{path}: every row needs a poster id")
    if not frame["id"].is_unique:
        raise ValueError(f"{path}: duplicate poster IDs")
    return frame


def load_splits(data_dir, features_csv, feature_columns):
    """Join measurements by ID, preserving the existing partition membership."""
    data_dir, features_csv = Path(data_dir), Path(features_csv)
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Select at least one feature, without duplicate columns")
    if RESERVED.intersection(feature_columns):
        raise ValueError("IDs, image paths, split names, and the target cannot be input features")
    if not features_csv.exists():
        raise FileNotFoundError(
            f"Create {features_csv} from features_template.csv and fill in your measurements first."
        )
    features = read_table(features_csv)
    missing = set(feature_columns) - set(features.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    features = features[["id"] + feature_columns].copy()
    for column in feature_columns:
        features[column] = pd.to_numeric(features[column], errors="raise").astype(float)
        if np.isinf(features[column]).any():
            raise ValueError(f"{column}: infinite values are not allowed")
        measured = features[column].dropna()
        if column == "text_density" and not measured.between(0, 1).all():
            raise ValueError("text_density must be a fraction between 0 and 1")
        if column in {"character_count", "word_count"} and (measured < 0).any():
            raise ValueError(f"{column} cannot be negative")
        if column in {"aspect_ratio", "width_px", "height_px"} and (measured <= 0).any():
            raise ValueError(f"{column} must be positive")

    master = read_table(data_dir / "posters.csv").set_index("id")
    if set(features["id"]) != set(master.index):
        raise ValueError("Feature IDs must match posters.csv exactly, one row per poster")
    splits, seen_ids, seen_paths = {}, set(), set()
    for name in ("train", "validation", "test"):
        labels = read_table(data_dir / f"{name}.csv")
        if labels.empty or not {"image_path", TARGET}.issubset(labels.columns):
            raise ValueError(f"{name}.csv must contain IDs, image paths, and scores")
        ids, paths = set(labels["id"]), set(labels["image_path"])
        if seen_ids.intersection(ids) or seen_paths.intersection(paths):
            raise ValueError("Overlapping posters across the saved splits")
        if len(paths) != len(labels) or not ids.issubset(master.index):
            raise ValueError(f"{name}.csv has duplicate paths or unknown IDs")
        expected = master.loc[labels["id"]]
        if list(labels["image_path"]) != list(expected["image_path"]):
            raise ValueError(f"{name}.csv image paths differ from posters.csv")
        labels[TARGET] = pd.to_numeric(labels[TARGET], errors="raise")
        if not labels[TARGET].between(1, 10).all():
            raise ValueError(f"{name}.csv has missing or out-of-range design scores")
        if not np.array_equal(labels[TARGET].to_numpy(), expected[TARGET].to_numpy()):
            raise ValueError(f"{name}.csv scores differ from posters.csv")
        splits[name] = labels[["id", "image_path", TARGET]].merge(
            features, on="id", how="left", validate="one_to_one", sort=False
        )
        seen_ids.update(ids)
        seen_paths.update(paths)
    if seen_ids != set(master.index):
        raise ValueError("Saved splits must cover every poster exactly once")
    blank = [c for c in feature_columns if splits["train"][c].isna().all()]
    if blank:
        raise ValueError(f"Fill in training measurements for these entirely blank features: {blank}")
    return splits


def regression_metrics(truth, predictions):
    truth, predictions = np.asarray(truth, dtype=float), np.asarray(predictions, dtype=float)
    errors = predictions - truth
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "r2": float(1 - np.sum(errors ** 2) / denominator) if denominator else None,
    }


def evaluate_split(predictor, frame, feature_columns, mean_baseline):
    """Evaluate the already-selected model, with IDs retained only in the output."""
    predicted = predictor.predict(frame[feature_columns]).to_numpy()
    baseline = np.full(len(frame), mean_baseline)
    metrics = {
        "model": regression_metrics(frame[TARGET], predicted),
        "mean_baseline": regression_metrics(frame[TARGET], baseline),
    }
    predictions = frame[["id", "image_path", TARGET]].copy()
    predictions["predicted_score"] = predicted
    predictions["mean_baseline_score"] = baseline
    return metrics, predictions


def train_model(features_csv, feature_columns, data_dir, output_dir,
                time_limit=120, num_cpus=2, evaluate_test=False):
    """Fit with train and validation only; optionally score test after selection."""
    if time_limit <= 0 or num_cpus <= 0:
        raise ValueError("time_limit and num_cpus must be positive")
    data_dir, features_csv, output_dir = Path(data_dir), Path(features_csv), Path(output_dir)
    splits = load_splits(data_dir, features_csv, feature_columns)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose a new output directory; {output_dir} already contains a run")

    from autogluon.tabular import TabularPredictor

    output_dir.mkdir(parents=True, exist_ok=True)
    model_columns = feature_columns + [TARGET]
    predictor = TabularPredictor(
        label=TARGET, problem_type="regression", eval_metric="root_mean_squared_error",
        path=str(output_dir / "model"),
    ).fit(
        train_data=splits["train"][model_columns],
        tuning_data=splits["validation"][model_columns],
        presets="medium_quality", time_limit=time_limit,
        hyperparameters={"LR": {}, "RF": {}, "XT": {}, "GBM": {}},
        num_bag_folds=0, num_stack_levels=0, refit_full=False,
        num_cpus=num_cpus, num_gpus=0, fit_strategy="sequential",
    )
    baseline = float(splits["train"][TARGET].mean())
    report = {
        "target": TARGET, "feature_columns": feature_columns,
        "best_model": predictor.model_best, "training_mean_score": baseline,
        "split_counts": {name: len(frame) for name, frame in splits.items()},
        "time_limit_seconds": time_limit, "test_evaluated": evaluate_test,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [features_csv] + [data_dir / f"{s}.csv" for s in splits]
        },
    }
    for name in (["validation", "test"] if evaluate_test else ["validation"]):
        report[name], predictions = evaluate_split(predictor, splits[name], feature_columns, baseline)
        predictions.to_csv(output_dir / f"{name}_predictions.csv", index=False)
    # The leaderboard uses validation scores; never rank candidate models on test.
    predictor.leaderboard(silent=True).to_csv(output_dir / "validation_leaderboard.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return predictor, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", type=Path, default=DATA_DIR / "features.csv")
    parser.add_argument("--feature-columns", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/numeric")
    parser.add_argument("--time-limit", type=int, default=120)
    parser.add_argument("--num-cpus", type=int, default=2)
    parser.add_argument("--evaluate-test", action="store_true", help="Final test evaluation after model selection")
    args = parser.parse_args()
    train_model(**vars(args))


if __name__ == "__main__":
    main()
