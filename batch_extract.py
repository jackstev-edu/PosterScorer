"""Build the training table: run every poster in a folder through features.py.

Usage: python batch_extract.py --input posters/ --output data/features.csv
With --posters-csv, rows are keyed on the dataset's poster id so the CSV
can be passed to scripts/train_numeric.py as --features-csv.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from features import FEATURE_COLUMNS, MIN_SIDE_PX, extract_features

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def map_ids(posters_csv, paths):
    """Return {filename: id}; every image and every poster id must pair up."""
    posters = pd.read_csv(posters_csv, dtype={"id": "string", "image_path": "string"})
    missing = {"id", "image_path"} - set(posters.columns)
    if missing:
        sys.exit(f"{posters_csv} is missing columns {sorted(missing)}")
    base = posters_csv.resolve().parent  # image_path is relative to the CSV's folder
    id_by_path = {(base / p).resolve(): i for i, p in zip(posters["id"], posters["image_path"])}
    ids = {path.name: id_by_path.get(path.resolve()) for path in paths}
    no_id = sorted(name for name, i in ids.items() if i is None)
    no_image = sorted(set(posters["id"]) - set(ids.values()))
    if no_id or no_image:
        sys.exit(f"Images without an id in {posters_csv}: {no_id}\nIds without an image in --input: {no_image}")
    return ids


def main():
    parser = argparse.ArgumentParser(description="Extract poster features to CSV.")
    parser.add_argument("--input", required=True, type=Path, help="Folder of poster images")
    parser.add_argument("--output", required=True, type=Path, help="CSV path to write")
    parser.add_argument("--overlays", type=Path, help="Optional folder for overlay PNGs")
    parser.add_argument("--posters-csv", type=Path, help="Optional posters.csv; adds its id as the first column")
    args = parser.parse_args()

    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    ids = map_ids(args.posters_csv, paths) if args.posters_csv else None  # Fail before the slow OCR loop
    rows, skipped = [], []
    for path in paths:
        with Image.open(path) as image:
            if min(image.size) < MIN_SIDE_PX:
                skipped.append(path.name)  # Same size rule as the app, keeps data consistent
                continue
            row, overlay = extract_features(image)
        key = {"id": ids[path.name]} if ids else {}
        rows.append({**key, "filename": path.name, **row})  # Filename kept for spot checks
        if args.overlays:
            args.overlays.mkdir(parents=True, exist_ok=True)
            overlay.save(args.overlays / f"{path.stem}_overlay.png")  # Spot check OCR quality
        print(f"done  {path.name}")

    columns = (["id"] if ids else []) + ["filename", *FEATURE_COLUMNS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} rows to {args.output}; skipped {len(skipped)} too-small images {skipped}")
    if ids and skipped:
        sys.exit("Training needs a row for every poster id; the skipped images above have none.")


if __name__ == "__main__":
    main()
