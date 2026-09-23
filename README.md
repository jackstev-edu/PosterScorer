---
title: Poster Scorer
emoji: 🖼️
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# PosterScorer

A trained poster design scorer: upload an image, receive a **continuous score from 1 to 10** and **one suggested improvement**. The committed model uses 14 OCR and layout measurements extracted from 219 PosterIQ posters.

## Start here

- **Claude / GUI agent:** read [CLAUDE.md](CLAUDE.md), then the [GUI integration contract](docs/gui-handoff.md).
- **Host the model:** follow [Hugging Face Spaces deployment](docs/huggingface-spaces.md). The trained model is already bundled; no retraining is needed.
- **Web app:** [app.py](app.py) is a Gradio interface that shows the score, the feedback sentence, and what was measured; see [Gradio app](#gradio-app-hugging-face-space).
- **Run locally:** use Python 3.12, install Tesseract and `requirements-space.txt`, then run `uvicorn api:app --host 0.0.0.0 --port 7860`.

The pipeline is: poster → shared OCR extractor → 14 numeric features → trained AutoGluon regressor → score + counterfactual feedback. Feedback is generated from the score model using a fitted training-data recipe; it is not a separately supervised feedback model.

## Dataset

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

The fitted regressor predicts the original **1–10 overall design score**. Its feedback engine tests coherent feature changes against that same model and returns one sentence. Feedback is model-estimated guidance, not separately labeled training data or a proven cause of design quality.

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

## Model backend; your GUI stays separate

[api.py](api.py) hosts the trained model. **POST `/score`** accepts a multipart image field named `image` and returns JSON with `score`, `feedback_sentence`, feature measurements, ranked suggested changes and a PNG overlay. **GET `/health`** confirms readiness; **GET `/docs`** provides interactive API documentation.

The Gradio GUI in [app.py](app.py) calls `PosterScorer` directly, so it does not need this backend; other frontends can send the uploaded file to it instead. Read [CLAUDE.md](CLAUDE.md) and the [copy-paste frontend integration](docs/gui-handoff.md).

```sh
docker build -t poster-scorer .
docker run --rm -p 7860:7860 poster-scorer
# Test with your own poster:
curl -X POST http://localhost:7860/score -F "image=@poster.png"
```

Deploy the same Dockerfile as a Hugging Face **Docker Space** using [these instructions](docs/huggingface-spaces.md). The held-out 33-poster test result is **RMSE 1.122, MAE 0.882, R² 0.408** on the original 1–10 scale. See [saved metrics](artifacts/metrics.json). Suggestions estimate changes the model favors; they are not causal explanations.

## Gradio app (Hugging Face Space)

[app.py](app.py) is the web interface. It loads `PosterScorer` once at startup, runs OCR once per upload, and shows:

- the predicted design score out of 10, with a verdict from the dataset quartiles: below 4.0 is "Below typical", 4.0 to below 5.9 is "Typical", 5.9 or above is "Above typical";
- the model's feedback sentence, shown exactly as `PosterScorer` returns it;
- the overlay and the 14 extracted features.

If the model bundle fails to load, the app fails at startup instead of falling back to a heuristic.

Run it locally with Python 3.12 and Tesseract installed:

```sh
python -m pip install -r requirements.txt gradio==6.28.0
python app.py
```

Deploy or update a Gradio Space (log in first with `hf auth login`). Hugging Face requires a PRO account or a paid organization to host Gradio Spaces on its CPU hardware:

```sh
python scripts/deploy_space.py --repo-id <user>/PosterScorer
```

The script uploads only the app, `features.py`, `poster_inference.py`, `score_feedback.py`, the model bundle, `requirements.txt`, `packages.txt`, the examples, and [space/README.md](space/README.md) as the Space README. It first checks that `features.py` matches the extractor hash in the model manifest.
