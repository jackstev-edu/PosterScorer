"""Extract shared OCR/image features and append them to all saved poster CSVs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import importlib.metadata
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_THREAD_LIMIT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
from features import FEATURE_COLUMNS, MIN_SIDE_PX, extract_features

BASE_COLUMNS = ["id", "image_path", "overall_design_score"]


def extract_one(data_dir, row):
    image_bytes = (data_dir / row["image_path"]).read_bytes()
    key = hashlib.sha256(image_bytes + (ROOT / "features.py").read_bytes()).hexdigest()
    cache = ROOT / ".cache/feature_extraction" / f"{key}.json"
    if cache.exists():
        return {"id": row["id"], **json.loads(cache.read_text())}
    with Image.open(io.BytesIO(image_bytes)) as image:
        if min(image.size) < MIN_SIDE_PX:
            raise ValueError(f"Image smaller than app minimum {MIN_SIDE_PX}px")
        features, _ = extract_features(image)
    if list(features) != FEATURE_COLUMNS or not np.isfinite(list(features.values())).all():
        raise ValueError("Extractor returned missing or non-finite features")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(features))
    return {"id": row["id"], **features}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/posteriq")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    tables = {name: pd.read_csv(args.data_dir / f"{name}.csv") for name in
              ("posters", "train", "validation", "test")}
    master = tables["posters"]
    if not master["id"].is_unique or not master["image_path"].is_unique:
        raise ValueError("Duplicate poster IDs or image paths")
    split_rows = pd.concat([tables[s][BASE_COLUMNS] for s in ("train", "validation", "test")])
    if not split_rows["id"].is_unique or set(split_rows["id"]) != set(master["id"]):
        raise ValueError("Splits must cover every poster exactly once")
    pd.testing.assert_frame_equal(
        master[BASE_COLUMNS].sort_values("id").reset_index(drop=True),
        split_rows.sort_values("id").reset_index(drop=True),
    )
    extracted, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(extract_one, args.data_dir, row): row["id"]
                   for row in master[BASE_COLUMNS].to_dict("records")}
        for i, future in enumerate(as_completed(pending), 1):
            try:
                extracted.append(future.result())
            except Exception as error:
                failures.append({"id": pending[future], "error": str(error)})
            if i % 20 == 0 or i == len(master):
                print(f"Processed {i}/{len(master)}; failures: {len(failures)}", flush=True)
    if failures:
        raise RuntimeError("No CSVs changed. Extraction failures: " + json.dumps(failures))
    features = master[["id"]].merge(pd.DataFrame(extracted), on="id", validate="one_to_one")
    if len(features) != len(master) or features[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Missing feature rows")
    enriched = {}
    for name, table in tables.items():
        clean = table.drop(columns=[c for c in FEATURE_COLUMNS if c in table])
        enriched[name] = clean.merge(features, on="id", how="left", validate="one_to_one", sort=False)
        pd.testing.assert_frame_equal(table[BASE_COLUMNS], enriched[name][BASE_COLUMNS])

    # Finish every extraction and check before replacing any CSV.
    outputs = {"features": features, **enriched}
    for name, table in outputs.items():
        destination = args.data_dir / f"{name}.csv"
        temporary = destination.with_suffix(".csv.tmp")
        table.to_csv(temporary, index=False, lineterminator="\n")
        temporary.replace(destination)
    metadata = {
        "rows": len(features), "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "extractor_sha256": hashlib.sha256((ROOT / "features.py").read_bytes()).hexdigest(),
        "tesseract_version": str(pytesseract.get_tesseract_version()),
        "python_packages": {p: importlib.metadata.version(p) for p in ("pytesseract", "Pillow", "numpy", "pandas")},
        "omp_thread_limit": os.environ["OMP_THREAD_LIMIT"],
        "no_detected_words": int(features["word_count"].eq(0).sum()),
        "failed_images": failures,
        "note": "Measurements use features.py unchanged; OCR and background masks are estimates, not annotations.",
    }
    (args.data_dir / "feature_extraction.json").write_text(json.dumps(metadata, indent=2) + "\n")
    checksums_path = args.data_dir / "SHA256SUMS"
    existing = dict(line.split("  ", 1)[::-1] for line in checksums_path.read_text().splitlines())
    for name in [f"{n}.csv" for n in outputs] + ["feature_extraction.json"]:
        existing[name] = hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
    checksums_path.write_text("".join(f"{sha}  {name}\n" for name, sha in existing.items()))
    print(f"Wrote {len(FEATURE_COLUMNS)} features for all {len(features)} posters and updated every split.")


if __name__ == "__main__":
    main()
