# Train a numeric poster-score model

The [notebook](../notebooks/train_numeric.ipynb) and [script](../scripts/train_numeric.py) use the same training function. They predict `overall_design_score` from selected numeric columns with AutoGluon Tabular. The notebook follows the example's install, load, check, and save pattern and adds model fitting and evaluation. No data augmentation is applied.

## 1. Install

Run from the repository root in a Python 3.10–3.13 environment; Python 3.12 is recommended:

```sh
python -m pip install -r requirements-training.txt
```

On macOS, LightGBM also needs the OpenMP runtime: `brew install libomp`. The requirements pin AutoGluon to 1.5.0. See the official [AutoGluon installation instructions](https://auto.gluon.ai/1.5/install.html) and [1.5.0 release notes](https://github.com/autogluon/autogluon/releases/tag/v1.5.0).

## 2. Use the extracted feature table

`data/posteriq/features.csv` is ready for training: one row per poster, joined to the existing splits by `id`. All 219 posters have extracted measurements. The same features are appended to `posters.csv`, `train.csv`, `validation.csv`, and `test.csv`. No manual measurements are required.

The default inputs are all **14** columns in `features.FEATURE_COLUMNS`:

| Columns | Meaning |
|---|---|
| `img_width_px`, `img_height_px`, `aspect_ratio` | Original dimensions and width divided by height |
| `word_count`, `char_count`, `text_block_count` | Confident OCR word, character, and text-block counts |
| `text_area_frac`, `graphic_area_frac`, `background_area_frac` | Fractions of image area assigned to text, graphics, and background |
| `text_share` | Text area divided by text plus graphic area; zero if both are absent |
| `text_to_graphic_ratio` | Text area divided by graphic area, with a 0.001 denominator floor |
| `mean_text_height_frac`, `max_text_height_frac`, `text_height_std_frac` | OCR word heights divided by image height: mean, maximum, and standard deviation |

See `features.py` for the shared extraction implementation. Text and graphics are deterministic estimates based on Tesseract OCR and a border-based background estimate. Training and inference must use this same function. Training on the saved CSV does not run OCR or require the Tesseract executable; extracting a new image does.

To regenerate all measurements after installing the extraction dependencies and Tesseract, run:

```sh
python scripts/extract_posteriq_features.py --workers 4
```

Selected inputs must be numeric; IDs, image paths, and the target are excluded. Some missing measurements can be imputed by AutoGluon, but a selected feature with no observed training values is rejected. The older `features_template.csv` is a historical manual template and is not used by this workflow.

## 3. Train and compare on validation

```sh
python scripts/train_numeric.py \
  --features-csv data/posteriq/features.csv \
  --time-limit 120 \
  --num-cpus 2 \
  --output-dir models/numeric
```

The default uses the complete shared feature list. For an experiment with a subset, pass `--feature-columns char_count text_area_frac aspect_ratio`. Unselected columns are ignored. For geometry alone, use `--feature-columns aspect_ratio img_width_px img_height_px` and a separate output directory. Models used by the app should retain the complete shared feature contract.

The script preserves the existing 153/33/33 train, validation, and test split. It checks feature joins and split separation, fits regression models on train, and supplies validation as AutoGluon's `tuning_data`. Candidate models are linear regression, random forest, extra trees, and LightGBM, with AutoGluon's weighted ensemble. The 120-second setting is a training budget, not a guarantee of total wall-clock time. See the official [`TabularPredictor.fit` API](https://auto.gluon.ai/1.5/api/autogluon.tabular.TabularPredictor.fit.html).

The report compares the selected model with a baseline that predicts the **training mean score** for every validation poster. RMSE and MAE are in score units and are better when smaller; R² is better when larger and can be negative. The AutoGluon leaderboard uses negative RMSE as its score, so values closer to zero are better. Validation performance helps choose features and settings; it is not an independent final estimate.

Use a different output directory for each experiment. The dataset splits stay fixed; model defaults and runtime conditions can still affect exact results.

## 4. Evaluate the final experiment

After choosing the feature set and training settings, run the final configuration with test scoring enabled:

```sh
python scripts/train_numeric.py \
  --features-csv data/posteriq/features.csv \
  --output-dir models/numeric-final \
  --evaluate-test
```

This still fits on train and selects using validation. Test rows are used only for reporting the final predictions and metrics. Avoid repeatedly changing features based on these 33 test results.

## 5. Load the saved model

Each output directory contains:

- `model/`: the saved AutoGluon predictor and its models.
- `metrics.json`: validation model and baseline metrics, plus test metrics when requested.
- `validation_predictions.csv`: predictions alongside IDs and actual scores.
- `validation_leaderboard.csv`: model comparison on validation.
- `test_predictions.csv`: written only when `--evaluate-test` is used.

For inference, pass the same numeric feature columns using the same measurement definitions:

```python
from pathlib import Path
from autogluon.tabular import TabularPredictor
from scripts.train_numeric import load_splits
from features import FEATURE_COLUMNS

feature_columns = list(FEATURE_COLUMNS)
splits = load_splits(
    Path("data/posteriq"), Path("data/posteriq/features.csv"), feature_columns
)
predictor = TabularPredictor.load("models/numeric/model")
# Demonstration using an existing measured validation row.
inputs = splits["validation"].loc[:, feature_columns].head(1)
print(predictor.predict(inputs))
```

The notebook also works in Colab: it clones this repository if needed, then installs the same requirements. The extracted `features.csv` is already included in the repository. Download the entire output directory to retain a Colab-trained model after the runtime ends.

These labels describe overall design quality, not a direct need for more text. A regression association between text measurements and score does not establish that adding or removing text will improve a particular poster. There are only 219 independent posters, so keep the number of features modest and compare against the baseline.
