# Connect the trained model to your GUI

## Hosted backend

- Space: https://huggingface.co/spaces/cmuchancel/poster-scorer
- API base URL: `https://cmuchancel-poster-scorer.hf.space`
- Upload endpoint: `POST https://cmuchancel-poster-scorer.hf.space/score` (multipart field `image`)
- Readiness: `GET https://cmuchancel-poster-scorer.hf.space/health`
- Interactive request tester: https://cmuchancel-poster-scorer.hf.space/docs

The Space was uploaded on 2026-09-23. Check `/health` before wiring a demo; startup/build completion is independent of the repository upload.


The complete inference interface is **`PosterScorer`**. The hostable backend is **`api.py`**, served by a Hugging Face Docker Space. Another teammate owns the GUI: connect their interface to this backend without replacing or redesigning their UI. Connecting requires no training job, dataset download, or external language model.

## Local Python integration

Use Python **3.12**. Install `requirements-inference.txt` and the Tesseract executable. For the hosted backend, follow the Docker configuration and [deployment instructions](huggingface-spaces.md). The Docker image installs the Python and system dependencies.

```python
from PIL import Image
from poster_inference import PosterScorer

scorer = PosterScorer()  # Once at server startup, not inside each callback.

with Image.open("poster.png") as image:
    result = scorer.score_image(image)

score = result["score"]                 # Display as e.g. 5.6 / 10.
feedback = result["feedback_sentence"]  # Ready-to-display one sentence.
overlay = result["overlay"]             # PIL.Image; orange text, blue graphics.
features = result["features"]           # Dict of 14 numeric measurements.
```

If your backend already used `features.extract_features(image)`, pass its measurements to `scorer.score_features(measurements)` to avoid duplicate OCR. That method returns the same prediction fields without `features` or `overlay`. Custom locations are supported through `PosterScorer(bundle_path=..., cache_dir=...)`; defaults resolve relative to this repository, not the current directory.

## Hosted API: `POST /score`

Use your Space’s app origin, for example `https://OWNER-SPACE.hf.space`; the `huggingface.co/spaces/OWNER/SPACE` address is the project page, not the API base URL. Copy the actual app URL from your Space rather than assuming its name conversion.

- `GET /health` returns `{"status": "ok"}` when the API responds.
- `POST /score` accepts multipart form data with an image file field named **`image`**.
- The successful response is a JSON object containing the prediction fields below, all 14 features, and **`overlay_png_base64`**, a base64-encoded PNG. It excludes the local Python `overlay` object.

### Browser / JavaScript

```javascript
// `file` is the File supplied by your existing upload control.
const SPACE_URL = "https://YOUR-SPACE.hf.space";
const form = new FormData();
form.append("image", file);
const response = await fetch(`${SPACE_URL}/score`, {
  method: "POST",
  body: form,
});
if (!response.ok) throw new Error(`Scoring failed (${response.status})`);
const result = await response.json();
const scoreLabel = `${result.score.toFixed(1)} / 10`;
const feedback = result.feedback_sentence;
const overlayUrl = `data:image/png;base64,${result.overlay_png_base64}`;
```

Do not set the multipart `Content-Type` yourself; the browser supplies its boundary. Keep the result fields intact and display the supplied sentence. Show progress while the Space starts or OCR runs, and catch errors for a useful upload/retry message. For a private Space, proxy authenticated requests through your backend rather than placing a Hugging Face token in browser code.

### Python client

```python
import requests

space_url = "https://YOUR-SPACE.hf.space"
with open("poster.png", "rb") as image:
    response = requests.post(
        f"{space_url}/score",
        files={"image": ("poster.png", image, "image/png")},
        timeout=180,
    )
response.raise_for_status()
result = response.json()
print(f"{result['score']:.1f} / 10", result["feedback_sentence"])
```

The remote client needs only an HTTP client, not AutoGluon, Tesseract, or the model archive. For local development, run `uvicorn api:app --host 0.0.0.0 --port 7860` in the configured Python 3.12 runtime.

## Prediction result schema

| Key | Type | Meaning |
|---|---|---|
| `score` | float | Display prediction clipped to 1–10 |
| `raw_score` | float | Unclipped regression prediction |
| `feedback_sentence` | string | One actionable sentence or neutral no-improvement message |
| `best_feature` | string or null | Actionable feature group with greatest positive tested gain |
| `best_gain` | float or null | Improvement in raw predicted score units, not a probability |
| `best_direction` | `increase`, `decrease`, or null | Suggested direction for that feature group |
| `best_recommendation` | object or null | Winning recommendation, with fields below |
| `ranked_changes` | array | Up to three best tested group changes; gains may be zero or negative |
| `supported_features` | array of strings | Groups supported by the fitted recipe |
| `message` | string | Explanation of the comparison result |
| `features` | object | All 14 extracted numeric inputs; image inference only |
| `overlay` | PIL image | Local `score_image` calls only; excluded from API JSON |
| `overlay_png_base64` | string | Hosted API only; PNG bytes encoded as base64 |

Each recommendation contains `feature`, `current_value`, `suggested_value`, `direction`, `coupled_feature_updates`, `predicted_score`, and signed `gain`. Candidate scores and gains use raw predictions, so they need not match differences between clipped display scores. When no candidate improves the prediction, the four `best_*` fields are null; display the supplied sentence and do not invent a recommendation.

## Feature contract

`features.FEATURE_COLUMNS` defines this exact ordered list:

```text
img_width_px, img_height_px, aspect_ratio,
word_count, char_count, text_block_count,
text_area_frac, graphic_area_frac, background_area_frac,
text_share, text_to_graphic_ratio,
mean_text_height_frac, max_text_height_frac, text_height_std_frac
```

There are **14 features and 219 images**, despite earlier planning references to 13 or 213. Counts come from confident OCR words. Area values are fractions, not percentages. `text_share` is text divided by text plus graphics; text-to-graphics ratio uses a 0.001 denominator floor. Word heights are normalized to processed image height. Zero OCR words means no confident detection, not proof of an empty poster.

The manifest enforces the extractor hash, feature order, label name, Python major/minor version, and AutoGluon version. Do not replace preprocessing or reorder columns to fix a startup error. Use the matching committed files and pinned dependencies. `features.py` is part of the trained artifact contract.

## What the feedback means

There is **one supervised continuous-score model plus a fitted feedback engine**. The recipe tests training-distribution quantiles for text coverage, graphic coverage, word count, and text size. Count and height changes update related measurements together; background/share/ratio are recomputed and geometry stays fixed. These assumptions produce plausible feature comparisons, not actual image edits or proven causal explanations.

The GUI should say “The model suggests …”, as the supplied sentence does. It should not claim the feature caused a bad score, that the estimated gain is guaranteed, or that this is a separately trained human-feedback model. The existing `app.py` uses a 0–100 heuristic baseline; its bands must not be applied to the trained 1–10 score. Leave GUI edits to the teammate responsible for that interface.

## Artifacts, metrics, and troubleshooting

`artifacts/poster_scorer_model.zip` contains `model/`, `feedback_recipe.json`, `model_manifest.json`, and `metrics.json`. Default extraction cache: `models/gui-cache/<archive-sha256>/`. Load only the trusted project bundle, because AutoGluon artifacts contain serialized Python objects.

| Symptom | Resolution |
|---|---|
| Python/AutoGluon mismatch | Use Python 3.12 and the exact inference requirements |
| Extractor hash mismatch | Restore the matching `features.py`; a changed extractor needs retraining/repackaging |
| Tesseract missing | Install the system package or set `TESSERACT_CMD` to its executable |
| LightGBM/OpenMP loading error | Install Linux `libgomp1` or macOS `libomp` |
| Model archive missing | Include `artifacts/poster_scorer_model.zip` in the deployed runtime |
| Startup fails | Surface the failure and repair it; do not silently substitute a heuristic |

Training uses 153 posters, validation 33, test 33. On the untouched test set: RMSE **1.122**, MAE **0.882**, R² **0.408**, versus mean-baseline RMSE **1.464**. See [metrics](../artifacts/metrics.json). The small holdout supports a hackathon prototype, not a universal design-quality claim.

To rebuild after an intentional training change, generate the fitted recipe with `python scripts/predict_feedback.py --model-dir models/numeric`, then run `python scripts/package_model.py`. Training details are in [numeric-training.md](numeric-training.md). Generated feedback CSV rows are predictions and must not be treated as human labels.
