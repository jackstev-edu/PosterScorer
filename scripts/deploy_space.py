"""Upload only the Poster Scorer app files to a Hugging Face Space.

Usage: python scripts/deploy_space.py --repo-id user/poster-scorer
The Space must already exist; log in first with `hf auth login`.
"""
import argparse
from pathlib import Path

from huggingface_hub import upload_folder

ROOT = Path(__file__).resolve().parents[1]
# Data, notebooks, and training scripts stay out of the Space
APP_FILES = ["app.py", "features.py", "requirements.txt", "packages.txt", "README.md", "examples/*"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Space id, for example user/poster-scorer")
    parser.add_argument("--commit-message", default="Deploy Poster Scorer app")
    args = parser.parse_args()
    commit = upload_folder(
        repo_id=args.repo_id, repo_type="space", folder_path=ROOT,
        allow_patterns=APP_FILES, commit_message=args.commit_message,
    )
    print(f"Uploaded to {commit.commit_url}")


if __name__ == "__main__":
    main()
