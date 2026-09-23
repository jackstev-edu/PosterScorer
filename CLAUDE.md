# Claude agent handoff: connect the trained poster scorer

## Hosted backend

- Space: https://huggingface.co/spaces/cmuchancel/poster-scorer
- API base URL: `https://cmuchancel-poster-scorer.hf.space`
- Upload endpoint: `POST https://cmuchancel-poster-scorer.hf.space/score` (multipart field `image`)
- Readiness: `GET https://cmuchancel-poster-scorer.hf.space/health`
- Interactive request tester: https://cmuchancel-poster-scorer.hf.space/docs

The Space was uploaded on 2026-09-23. Check `/health` before wiring a demo; startup/build completion is independent of the repository upload.


Your integration target is **`PosterScorer` in `poster_inference.py`**. A trained model and fitted feedback recipe are already committed; do not retrain to connect the GUI. The Hugging Face Docker Space serves the backend in `api.py`. Another teammate owns the GUI: preserve `app.py` and connect their interface to this API; do not replace or redesign it.

## Read these first

1. [`docs/gui-handoff.md`](docs/gui-handoff.md): exact Python and remote API contracts, UI behavior, and examples.
2. [`docs/huggingface-spaces.md`](docs/huggingface-spaces.md): deployment instructions and runtime files.
3. [`poster_inference.py`](poster_inference.py), [`api.py`](api.py), and [`artifacts/model_manifest.json`](artifacts/model_manifest.json): executable source of truth.

## What exists

- **219** PosterIQ images with continuous `overall_design_score` targets on a **1–10** scale.
- Fixed split: **153 train / 33 validation / 33 test** (70/15/15 rounded; seed 42).
- **14 numeric features**, extracted for every image and appended to the master and split CSVs. `features.FEATURE_COLUMNS` defines the order.
- One AutoGluon regression predictor (selected ensemble) and one fitted counterfactual feedback recipe. There is **no separately supervised feedback model**: this dataset has no feedback labels.
- `artifacts/poster_scorer_model.zip` contains the predictor, recipe, manifest, and metrics. The wrapper unpacks it into `models/gui-cache/` and validates its contract.
- Untouched test performance: RMSE **1.122**, MAE **0.882**, R² **0.408**; training-mean baseline RMSE **1.464**. These are estimates from only 33 test posters.

## Integration rules

- Instantiate `PosterScorer()` **once per server process**. Call `score_image(PIL_image)` per upload. Reuse `score_features(features)` if OCR already ran through the shared extractor.
- Display `result["score"]` as `/10` and `result["feedback_sentence"]` verbatim. Keep `raw_score` for diagnostics. Never apply the old app’s 0–100 bands or thresholds to these outputs.
- Treat `best_feature`, `best_gain`, `best_direction`, and `best_recommendation` as nullable. No positive tested gain is a valid result, not an exception.
- Feedback describes the most promising model-estimated change. Do not relabel it as a proven cause, causal attribution, guaranteed improvement, or human annotation.
- Keep `features.py` byte-for-byte unchanged for this bundle. Its SHA-256 is enforced at startup. Changing OCR preprocessing or feature formulas requires re-extraction, retraining, and repackaging.
- Use Python **3.12**, AutoGluon **1.5.0**, and `requirements-inference.txt` pins. Tesseract is a system dependency. Do not upgrade the serialized-model runtime casually.
- Surface loading/OCR/model errors; never silently substitute the heuristic score or fabricated features. A poster with zero detected words can still produce a valid prediction.
- Never use `id`, image paths, labels, generated feedback, or held-out scores as input features. Do not tune against the existing test split.
- Keep upload processing on the server; do not place a private Hugging Face token in browser code.

## Runtime pipeline

`uploaded image → features.extract_features → 14 numeric values → trained regression score → constrained candidate feature changes → highest positive estimated gain → one sentence`

The extractor normalizes the image, runs Tesseract, estimates text/graphics/background coverage, and draws an overlay. Orange boxes mark OCR text; blue shading marks estimated graphics. The feedback engine tests training-only quantiles, updates linked measurements together, and leaves image geometry fixed. It does not edit the uploaded image.

## Definition of done for the GUI agent

- A real uploaded poster yields a continuous score out of 10, one feedback sentence, and an optional overlay/details panel.
- Loading and error states are visible; empty upload is handled.
- The model is loaded once, and OCR is run once per request.
- The interface remains usable when recommendation fields are null.
- The UI obtains results from `PosterScorer` or the `POST /score` Space endpoint, not from the legacy heuristic.
- The committed model archive and manifest ship with the runtime. No training CSVs are required for inference.
- Run `python -m unittest discover -s tests -p 'test_feedback.py'` plus a real-image smoke test using the bundled predictor; run any additional Space tests present in the repo.

## Data and provenance

Dataset: [creative-graphic-design/PosterIQ](https://huggingface.co/datasets/creative-graphic-design/PosterIQ), revision `02bddb802658cb952d693d289af230644401bb26`. Its declared license is non-commercial research; repository inclusion does not grant additional image or model rights. Consult `data/posteriq/README.md` and the source terms before deployment outside that scope. Generated `data/posteriq/model_feedback.csv` is model output, not human feedback ground truth.
