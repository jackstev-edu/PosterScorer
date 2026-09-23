"""Download the pinned PosterIQ overall_rating subset and export images + CSV."""
import argparse
import csv
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

import pyarrow.parquet as pq
from PIL import Image

SOURCE = "creative-graphic-design/PosterIQ"
REVISION = "02bddb802658cb952d693d289af230644401bb26"
SHARDS = [
    "893092bb5c8830d4a516ee6bad85b9dbf26f46096fb248b6d956fe8264394e87",
    "66f0d99e47936c4eec55337abdfb0cfc2dc2ba2479923d83f8904d50ebbad115",
    "5e6031b1e869bf7bb44f42863628884ce7dd2b62a25cb195790310b6309ccc41",
    "810b7da1a269902ec9829092f5c5e2cdccd738f062639c7cc6e5d9b84c7f8d6c",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=root / ".cache/posteriq")
    parser.add_argument("--output-dir", type=Path, default=root / "data/posteriq")
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    images = args.output_dir / "images"
    images.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        i, expected_hash = item
        name = f"test-{i:05d}-of-00004.parquet"
        path = args.cache_dir / name
        if not path.exists() or digest(path.read_bytes()) != expected_hash:
            url = f"https://huggingface.co/datasets/{SOURCE}/resolve/{REVISION}/overall_rating/{name}"
            temporary = path.with_suffix(".part")
            subprocess.run(["curl", "-fLsS", "--retry", "3", url, "-o", str(temporary)], check=True)
            if digest(temporary.read_bytes()) != expected_hash:
                raise ValueError(f"Source checksum mismatch: {name}")
            temporary.replace(path)
        return path

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(fetch, enumerate(SHARDS)))
    source_rows = []
    for path in paths:
        source_rows.extend(pq.read_table(path).to_pylist())
    if len(source_rows) != 219 or len({r["id"] for r in source_rows}) != 219:
        raise ValueError("Expected 219 distinct poster records")
    if len({r["name"] for r in source_rows}) != 219:
        raise ValueError("Duplicate image names")

    records, checksums = [], []
    for row in source_rows:
        name = row["name"]
        if Path(name).name != name:
            raise ValueError(f"Unsafe image filename: {name}")
        score = json.loads(row["gt_json"])
        if type(score) not in (int, float) or not 1 <= score <= 10:
            raise ValueError(f"Invalid score for {row['id']}: {score}")
        image_bytes = row["image"]["bytes"]
        with Image.open(io.BytesIO(image_bytes)) as im:
            im.verify()
        with Image.open(io.BytesIO(image_bytes)) as im:
            im.load()
        relative_path = f"images/{name}"
        (images / name).write_bytes(image_bytes)
        records.append({"id": row["id"], "image_path": relative_path, "overall_design_score": score})
        checksums.append(f"{digest(image_bytes)}  {relative_path}")

    csv_path = args.output_dir / "posters.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "image_path", "overall_design_score"])
        writer.writeheader()
        writer.writerows(records)
    checksums.append(f"{digest(csv_path.read_bytes())}  posters.csv")
    (args.output_dir / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    info = {
        "source_url": f"https://huggingface.co/datasets/{SOURCE}",
        "source_revision": REVISION,
        "source_config": "overall_rating",
        "source_split": "test",
        "rows": len(records),
        "score_source_field": "gt_json",
        "score_scale": [1, 10],
        "observed_score_range": [min(r["overall_design_score"] for r in records), max(r["overall_design_score"] for r in records)],
        "images": "Original embedded image bytes; no resizing or re-encoding.",
        "declared_source_license": "non-commercial-research-license",
        "source_shard_sha256": {p.name: h for p, h in zip(paths, SHARDS)},
    }
    (args.output_dir / "source_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Validated and exported {len(records)} posters to {args.output_dir}")


if __name__ == "__main__":
    main()
