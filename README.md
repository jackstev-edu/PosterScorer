# PosterScorer

219 poster images paired with their overall design scores, ready for text-density analysis and regression.

- **[Score table](data/posteriq/posters.csv)** — one row per image: `id`, `image_path`, `overall_design_score`.
- **Model splits:** [train](data/posteriq/train.csv) (153), [validation](data/posteriq/validation.csv) (33), and [test](data/posteriq/test.csv) (33), approximately 70/15/15, with no overlap.
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

Regenerate the model splits with `python scripts/split_posteriq.py` (Python standard library only). The reproducible split uses seed 42 and balances three score bands. Use train for fitting, validation for tuning, and reserve test for final evaluation.
