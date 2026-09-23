# GUI scoring interface

Use Python 3.12 and run `pip install -r requirements-inference.txt`; install Tesseract separately. Use the Python version recorded in the bundled `model_manifest.json`. Keep `features.py` unchanged: the wrapper verifies its hash and the model's exact numeric feature contract before loading.

The committed `artifacts/poster_scorer_model.zip` contains the model fitted to actual overall design scores, validation metrics, and a feedback recipe learned from training features only. Its root contains `model/`, `metrics.json`, `feedback_recipe.json`, and `model_manifest.json`. The wrapper safely extracts it into the ignored `models/gui-cache/` directory and reuses that cache.

```python
from PIL import Image
from poster_inference import PosterScorer

# Initialize once when the GUI starts, not once per uploaded image.
scorer = PosterScorer()
with Image.open("my-poster.png") as image:
    result = scorer.score_image(image)

score = result["score"]                  # Continuous display score, 1–10.
feedback = result["feedback_sentence"]   # One sentence; ready for the GUI.
overlay = result["overlay"]              # PIL image with detected regions.
measurements = result["features"]        # Exact shared extraction output.
```

If the GUI already called `features.extract_features`, use `scorer.score_features(measurements)` to avoid running OCR twice. Custom bundle/cache locations can be passed as `PosterScorer(bundle_path=..., cache_dir=...)`; default paths resolve relative to the repository, regardless of the working directory.

The result also includes `raw_score` (unclipped regression output), `best_feature`, `best_gain`, `best_direction`, `best_recommendation`, `ranked_changes`, and `supported_features`. Display scores are clamped to 1–10; gains describe differences between raw model predictions. If none of the tested changes improves the prediction, best fields are `None` and the sentence says no tested improvement was found.

Feedback is available at every score. It identifies the most promising **model-estimated change**, not a proven cause of poor design. The feedback engine reuses the score predictor; there is no separately supervised feedback model because the dataset supplies no feedback labels. It tests training-distribution quantiles for text coverage, graphic coverage, word count, and text size. Counts and heights change in coherent groups, with proportional changes to text area and recomputed background/share/ratio features. Those proportional relationships are assumptions for the comparison, not edits to the image. Geometry is held fixed.

## Measured holdout performance

On the 33 untouched test posters, RMSE is 1.122 and MAE is 0.882 on the 1–10 score scale (R² 0.408). The training-mean baseline has RMSE 1.464. The model was selected using validation only. See `artifacts/metrics.json` and `artifacts/test_predictions.csv`.

Rebuild the bundle after training with `python scripts/predict_feedback.py --model-dir models/numeric` then `python scripts/package_model.py`. `data/posteriq/model_feedback.csv` holds generated feedback for all 219 images; these are model outputs, not human feedback labels.
