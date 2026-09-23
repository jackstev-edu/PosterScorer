"""Build the training table: run every poster in a folder through features.py.

Usage: python batch_extract.py --input posters/ --output data/features.csv
Add your score/label column to the CSV afterwards, keyed on the filename.
"""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from features import FEATURE_COLUMNS, MIN_SIDE_PX, extract_features

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="Extract poster features to CSV.")
    parser.add_argument("--input", required=True, type=Path, help="Folder of poster images")
    parser.add_argument("--output", required=True, type=Path, help="CSV path to write")
    parser.add_argument("--overlays", type=Path, help="Optional folder for overlay PNGs")
    args = parser.parse_args()

    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    rows, skipped = [], []
    for path in paths:
        with Image.open(path) as image:
            if min(image.size) < MIN_SIDE_PX:
                skipped.append(path.name)  # Same size rule as the app, keeps data consistent
                continue
            row, overlay = extract_features(image)
        rows.append({"filename": path.name, **row})  # Filename is the join key for labels
        if args.overlays:
            args.overlays.mkdir(parents=True, exist_ok=True)
            overlay.save(args.overlays / f"{path.stem}_overlay.png")  # Spot check OCR quality
        print(f"done  {path.name}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["filename", *FEATURE_COLUMNS]).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} rows to {args.output}; skipped {len(skipped)} too-small images {skipped}")


if __name__ == "__main__":
    main()
