# PosterScorer

219 poster images paired with their overall design scores, ready for text-density analysis and regression.

- **[Score table](data/posteriq/posters.csv)** — one row per image: `id`, `image_path`, `overall_design_score`.
- **[Images](data/posteriq/images)** — original image files, with no resizing or re-encoding.
- **[Dataset instructions and provenance](data/posteriq/README.md)** — loading example, score meaning, and source details.

Scores are copied directly from the [PosterIQ](https://huggingface.co/datasets/creative-graphic-design/PosterIQ) `overall_rating` subset, on its original 1–10 scale. Text quantity and “needs more text” labels are not supplied by this subset.

## Rebuild the dataset

Requires Python 3.10+ and `curl`:

```sh
python -m pip install -r requirements-data.txt
python scripts/prepare_posteriq.py
```

The script downloads a pinned source revision, checks source checksums, verifies all images and scores, and regenerates the CSV and image files. Source data is declared for non-commercial research; see the dataset instructions for attribution.
