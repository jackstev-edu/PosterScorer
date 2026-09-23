"""Create reproducible 70/15/15 splits of the 219-poster dataset."""
import csv
import random
from pathlib import Path


SEED = 42
DATA_DIR = Path(__file__).resolve().parents[1] / "data/posteriq"
FIELDS = ["id", "image_path", "overall_design_score"]


def main():
    with (DATA_DIR / "posters.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Expected columns: {FIELDS}")
        rows = list(reader)
    if len(rows) != 219 or len({row["id"] for row in rows}) != 219:
        raise ValueError("Expected 219 unique poster IDs")
    if len({row["image_path"] for row in rows}) != 219:
        raise ValueError("Expected 219 unique image paths")
    for row in rows:
        if not (DATA_DIR / row["image_path"]).is_file():
            raise ValueError(f"Missing image: {row['image_path']}")
        if not 1 <= float(row["overall_design_score"]) <= 10:
            raise ValueError(f"Invalid score: {row['id']}")

    # Balance low, middle, and high scores without new dependencies.
    rows.sort(key=lambda row: (float(row["overall_design_score"]), row["id"]))
    rng = random.Random(SEED)
    splits = {"train": [], "validation": [], "test": []}
    for start in range(0, len(rows), 73):
        band = rows[start:start + 73]
        rng.shuffle(band)
        splits["train"].extend(band[:51])
        splits["validation"].extend(band[51:62])
        splits["test"].extend(band[62:])

    ids = [row["id"] for split in splits.values() for row in split]
    if len(ids) != len(set(ids)) or set(ids) != {row["id"] for row in rows}:
        raise ValueError("Splits must cover every poster exactly once")
    for name, split in splits.items():
        rng.shuffle(split)
        path = DATA_DIR / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(split)
        print(f"{name}: {len(split)} posters ({len(split) / len(rows):.2%})")


if __name__ == "__main__":
    main()
