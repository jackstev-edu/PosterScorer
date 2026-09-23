"""Stable GUI interface to the bundled poster score model and feedback recipe."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = ROOT / "artifacts/poster_scorer_model.zip"


def _validate_manifest(manifest, columns):
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported model bundle schema")
    if manifest.get("target") != "overall_design_score":
        raise ValueError("Bundle target is not overall_design_score")
    if manifest.get("feature_columns") != columns:
        raise ValueError("Model bundle feature contract does not match features.py")
    extractor_hash = hashlib.sha256((ROOT / "features.py").read_bytes()).hexdigest()
    if manifest.get("extractor_sha256") != extractor_hash:
        raise ValueError("Feature extractor differs from the version used to train this model")
    if manifest.get("autogluon_version") != version("autogluon.tabular"):
        raise ValueError("Install the AutoGluon version specified in the model manifest")
    expected_python = manifest.get("python_version")
    if expected_python and expected_python.split(".")[:2] != [str(sys.version_info.major), str(sys.version_info.minor)]:
        raise ValueError(f"This model bundle requires Python {expected_python}")


def _extract_bundle(bundle_path, cache_dir):
    """Unpack a content-addressed bundle, rejecting path traversal and symlinks."""
    bundle_path, cache_dir = Path(bundle_path).resolve(), Path(cache_dir).resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}")
    digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    destination = cache_dir / digest
    marker = destination / ".bundle_sha256"
    if marker.is_file() and marker.read_text().strip() == digest:
        return destination
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="unpack-", dir=cache_dir))
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            entries = archive.infolist()
            if len(entries) > 10000 or sum(item.file_size for item in entries) > 2_000_000_000:
                raise ValueError("Model bundle exceeds extraction limits")
            names = [item.filename for item in entries]
            if len(names) != len(set(names)):
                raise ValueError("Duplicate archive entries are not allowed")
            required = {"metrics.json", "feedback_recipe.json", "model_manifest.json", "model/predictor.pkl"}
            if not required.issubset(names):
                raise ValueError("Model bundle is missing required artifacts")
            for item in entries:
                name = item.filename
                path = PurePosixPath(name)
                mode = item.external_attr >> 16
                if (path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name
                        or stat.S_ISLNK(mode) or not (temporary / name).resolve().is_relative_to(temporary)):
                    raise ValueError(f"Unsafe model bundle entry: {name}")
            archive.extractall(temporary)
        (temporary / ".bundle_sha256").write_text(digest + "\n")
        try:
            temporary.rename(destination)
        except FileExistsError:
            if not marker.is_file() or marker.read_text().strip() != digest:
                raise ValueError("Incomplete model cache; remove the indicated cache directory")
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


class PosterScorer:
    """Load once per GUI process, then score any number of posters."""

    def __init__(self, bundle_path=None, cache_dir=None):
        from features import FEATURE_COLUMNS
        from autogluon.tabular import TabularPredictor
        from score_feedback import ScoreOptimizer

        self.bundle_dir = _extract_bundle(bundle_path or DEFAULT_BUNDLE, cache_dir or ROOT / "models/gui-cache")
        self.manifest = json.loads((self.bundle_dir / "model_manifest.json").read_text())
        _validate_manifest(self.manifest, FEATURE_COLUMNS)
        metrics = json.loads((self.bundle_dir / "metrics.json").read_text())
        if metrics.get("feature_columns") != FEATURE_COLUMNS or metrics.get("target") != self.manifest["target"]:
            raise ValueError("Metrics and manifest feature/target contracts differ")
        self.predictor = TabularPredictor.load(str(self.bundle_dir / "model"))
        self.optimizer = ScoreOptimizer.load_recipe(self.predictor, self.bundle_dir / "feedback_recipe.json")
        if self.optimizer.feature_names != FEATURE_COLUMNS:
            raise ValueError("Feedback recipe and extractor feature contracts differ")

    def score_features(self, features):
        """Return score, raw_score, one feedback sentence and ranked changes."""
        result = self.optimizer.analyze(features)
        result["raw_score"] = result["score"]
        result["score"] = min(10.0, max(1.0, result["raw_score"]))
        return result

    def score_image(self, image):
        """Accept a PIL image and return the score result, features and PIL overlay."""
        from features import extract_features
        features, overlay = extract_features(image)
        result = self.score_features(features)
        result.update(features=features, overlay=overlay)
        return result
