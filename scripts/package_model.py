"""Package the trained score predictor and fitted feedback recipe for the GUI."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features import FEATURE_COLUMNS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/numeric")
    args = parser.parse_args()
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    for name in ("model/predictor.pkl", "metrics.json", "feedback_recipe.json"):
        if not (args.model_dir / name).is_file():
            raise FileNotFoundError(f"Required artifact missing: {name}")
    manifest = {
        "schema_version": 1, "target": "overall_design_score", "score_scale": [1, 10],
        "feature_columns": FEATURE_COLUMNS,
        "extractor_sha256": hashlib.sha256((ROOT / "features.py").read_bytes()).hexdigest(),
        "autogluon_version": version("autogluon.tabular"),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_packages": {p: version(p) for p in
                            ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "Pillow", "pytesseract")},
        "feedback_method": "Training-only quantile counterfactuals using the score model; not a second supervised target.",
    }
    (args.model_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = artifacts / "poster_scorer_model.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted((args.model_dir / "model").rglob("*")):
            if path.is_file() and "logs" not in path.parts:
                z.write(path, path.relative_to(args.model_dir))
        for name in ("metrics.json", "feedback_recipe.json", "model_manifest.json"):
            z.write(args.model_dir / name, name)
    for name in ("metrics.json", "model_manifest.json", "validation_leaderboard.csv",
                 "validation_predictions.csv", "test_predictions.csv", "feedback_recipe.json"):
        if (args.model_dir / name).exists():
            shutil.copyfile(args.model_dir / name, artifacts / name)
    (artifacts / "SHA256SUMS").write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n"
    )
    (ROOT / "requirements-inference.txt").write_text(
        "# Python " + manifest["python_version"] + "; trained bundle runtime. Install Tesseract separately.\n"
        + "autogluon.tabular[lightgbm]==" + manifest["autogluon_version"] + "\n"
        + "".join(f"{name}=={v}\n" for name, v in manifest["python_packages"].items())
        + "typing-extensions>=4.10,<5\n"
    )
    print(f"Packaged {archive} ({archive.stat().st_size / 1_000_000:.2f} MB)")


if __name__ == "__main__":
    main()
