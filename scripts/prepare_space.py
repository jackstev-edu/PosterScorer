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
    parser.add_argument("--sdk", choices=["gradio", "docker"], default="gradio")
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
    if args.sdk == "gradio":
        shutil.copy2(ROOT / "deployment/huggingface/hf_space.py", destination / "hf_space.py")
        shutil.copy2(ROOT / "deployment/huggingface/README.md", destination / "README.md")
        requirements = (ROOT / "requirements-inference.txt").read_text() + "\n" + "\n".join(line for line in (ROOT / "requirements-space.txt").read_text().splitlines() if not line.startswith("-r ")) + "\ngradio==6.28.0\n"
        (destination / "requirements.txt").write_text(requirements)
        (destination / "packages.txt").write_text("tesseract-ocr\nlibgomp1\n")
        (destination / "Dockerfile").unlink()
    print(f"Ready: {destination} ({len(FILES)} files)")


if __name__ == "__main__":
    main()
