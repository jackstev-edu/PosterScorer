---
title: Poster Scorer
emoji: 🖼️
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# PosterScorer

219 poster images paired with their overall design scores, ready for text-density analysis and regression.

- **[Score table](data/posteriq/posters.csv)** — one row per image: ID, image path, overall score, and 14 extracted numeric features.
- **Model splits:** [train](data/posteriq/train.csv) (153), [validation](data/posteriq/validation.csv) (33), and [test](data/posteriq/test.csv) (33), approximately 70/15/15, with no overlap.
- **[Images](data/posteriq/images)** — original image files, with no resizing or re-encoding.
- **[Dataset instructions and provenance](data/posteriq/README.md)** — loading example, score meaning, and source details.

Scores are copied directly from the [PosterIQ](https://huggingface.co/datasets/creative-graphic-design/PosterIQ) `overall_rating` subset, on its original 1–10 scale. Text quantity and “needs more text” labels are not supplied by this subset.

## Train from numeric features

Use the [Jupyter notebook](notebooks/train_numeric.ipynb) or [training script](scripts/train_numeric.py) to predict `overall_design_score` with AutoGluon Tabular. Inputs are explicit numeric columns, such as character count, text density, and aspect ratio. Image pixels, filenames, and IDs are excluded from model inputs.

1. Install `requirements-training.txt` with Python 3.10–3.13 (3.12 recommended).
2. Use [features.csv](data/posteriq/features.csv): all 219 posters now have the 14 measurements from `features.py`. No manual text measurements are needed.
3. Train from the repository root:

```sh
python -m pip install -r requirements-training.txt
python scripts/train_numeric.py --features-csv data/posteriq/features.csv
```

The script fits on the existing 153 training posters, selects models using the 33 validation posters, and saves the predictor, validation predictions, leaderboard, and metrics. Test scoring is opt-in with `--evaluate-test`; reserve it for the final experiment. See [numeric training instructions](docs/numeric-training.md) for feature definitions, optional features, and saved outputs.

This predicts overall design quality. A score alone does not establish whether adding text would improve a poster.

## Trained model and one-sentence feedback

The GUI handoff is [documented here](docs/gui-handoff.md). Download the committed [model bundle](artifacts/poster_scorer_model.zip) or clone the repository, install the training requirements, and use:

```python
from poster_inference import PosterScorer
scorer = PosterScorer()
result = scorer.score_image(pil_image)
print(result["score"], result["feedback_sentence"])
```

The fitted regressor predicts the original **1–10 overall design score**. Its feedback engine tests coherent feature changes against that same model and returns one sentence. Feedback is model-estimated guidance, not separately labeled training data or a proven cause of design quality. The standalone Gradio baseline below uses a different 0–100 scale; use the wrapper for the trained model.

To regenerate OCR/features and update the master and all split CSVs:

```sh
python -m pip install -r requirements-extraction.txt
python scripts/extract_posteriq_features.py --workers 4
```

Tesseract is required (`brew install tesseract` on macOS, `apt-get install tesseract-ocr` on Debian/Ubuntu). The extractor processes all 219 posters, preserves IDs/scores/splits, and does not silently skip failed images. Its code, versions, and OCR counts are recorded in [feature_extraction.json](data/posteriq/feature_extraction.json).

## Rebuild the dataset

Requires Python 3.10+ and `curl`:

```sh
python -m pip install -r requirements-data.txt
python scripts/prepare_posteriq.py
```

The script downloads a pinned source revision, checks source checksums, verifies all images and scores, and regenerates the CSV and image files. Source data is declared for non-commercial research; see the dataset instructions for attribution.

Regenerate the model splits with `python scripts/split_posteriq.py` (Python standard library only). The reproducible split uses seed 42 and balances three score bands. Use train for fitting, validation for tuning, and reserve test for final evaluation.

## Poster Scorer app

A Gradio app ([app.py](app.py)) that takes an uploaded poster and gives a 0 to 100 score for its balance of text and graphics.

1. [features.py](features.py) runs Tesseract OCR and a background-color analysis to measure text area, graphic area, and text size. The result is one tabular row (`FEATURE_COLUMNS`).
2. A trained tabular model scores that row. Until a model is published, a labeled baseline rule scores it instead.
3. The app shows the score, a verdict, the feature table, and an overlay of what was measured.

The score reflects text versus graphic **area** only. It does not judge color, content, typography, or overall design quality. Gradient or photo backgrounds may be counted as graphics.

| File | Purpose |
|---|---|
| `app.py` | Gradio interface and scoring callback |
| `features.py` | Shared feature extraction for training and the app |
| `batch_extract.py` | Builds a feature CSV from a folder of posters |
| `tests/test_contract.py` | Output-shape and feature-contract checks |
| `examples/` | Synthetic example posters |
| `requirements.txt` | Python packages for the app |
| `packages.txt` | System packages for Hugging Face Spaces (Tesseract) |

### Run locally

```sh
python -m pip install -r requirements.txt
python app.py
```

Tesseract must be installed. On Windows, install it and set `TESSERACT_CMD` to the path of `tesseract.exe` if it is not on PATH.

### Extract features from a folder of posters

```sh
python batch_extract.py --input posters/ --output data/app_features.csv --overlays overlays/
```

To use a trained model in the app, publish the predictor, set `MODEL_REPO_ID` and `MODEL_REVISION` in `app.py`, and uncomment `autogluon.tabular` in `requirements.txt`. The app checks at startup that the model's features match `FEATURE_COLUMNS` in `features.py`.

### Test

```sh
python -m pip install pytest
pytest -q
```
