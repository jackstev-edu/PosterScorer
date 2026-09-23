"""Upload the Gradio Poster Scorer app and its trained model to a Hugging Face Space.

Usage: python scripts/deploy_space.py --repo-id user/PosterScorer
Log in first with `hf auth login` using a token with write access.
"""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parents[1]
# Space path -> repo path; the Space gets its own Gradio README
FILES = {
    "app.py": "app.py",
    "features.py": "features.py",
    "poster_inference.py": "poster_inference.py",
    "score_feedback.py": "score_feedback.py",
    "requirements.txt": "requirements.txt",
    "packages.txt": "packages.txt",
    "README.md": "space/README.md",
    "artifacts/poster_scorer_model.zip": "artifacts/poster_scorer_model.zip",
}
TEXT_SUFFIXES = {".py", ".txt", ".md"}


def read_bytes(path):
    """Read a file, converting Windows line endings so hashes match the committed bytes."""
    data = path.read_bytes()
    return data.replace(b"\r\n", b"\n") if path.suffix in TEXT_SUFFIXES else data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Space id, for example user/PosterScorer")
    parser.add_argument("--private", action="store_true", help="Create the Space as private")
    parser.add_argument("--commit-message", default="Deploy Poster Scorer")
    args = parser.parse_args()

    files = {dest: read_bytes(ROOT / src) for dest, src in FILES.items()}
    files.update({f"examples/{p.name}": p.read_bytes() for p in sorted((ROOT / "examples").glob("*.png"))})

    # The bundle refuses to load with a different extractor, so check before uploading
    manifest = json.loads((ROOT / "artifacts/model_manifest.json").read_text())
    if hashlib.sha256(files["features.py"]).hexdigest() != manifest["extractor_sha256"]:
        raise SystemExit("features.py does not match the extractor hash in artifacts/model_manifest.json")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
    operations = [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=data) for dest, data in files.items()]
    commit = api.create_commit(args.repo_id, operations, repo_type="space", commit_message=args.commit_message)
    print(f"Uploaded {len(operations)} files: {commit.commit_url}")


if __name__ == "__main__":
    main()
