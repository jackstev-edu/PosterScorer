"""Stage a small, self-contained Hugging Face Space without training images."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
FILES = ["README.md", "CLAUDE.md", "api.py", "Dockerfile", "requirements-space.txt", "poster_inference.py", "score_feedback.py",
         "features.py", "requirements.txt", "requirements-inference.txt", "packages.txt",
         "artifacts/poster_scorer_model.zip", "artifacts/model_manifest.json", "artifacts/metrics.json",
         "docs/gui-handoff.md", "docs/huggingface-spaces.md", "data/posteriq/README.md"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "space-export")
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists() and any(destination.iterdir()):
        parser.error("Output must be empty so old files cannot enter the deployment.")
    for name in FILES:
        if not (ROOT / name).is_file():
            parser.error(f"Missing required file: {name}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    print(f"Ready: {destination} ({len(FILES)} files)")


if __name__ == "__main__":
    main()
