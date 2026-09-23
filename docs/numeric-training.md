# Train a numeric poster-score model

The [notebook](../notebooks/train_numeric.ipynb) and [script](../scripts/train_numeric.py) use the same training function. They predict `overall_design_score` from selected numeric columns with AutoGluon Tabular. The notebook follows the example's install, load, check, and save pattern and adds model fitting and evaluation. No data augmentation is applied.

## 1. Install

Run from the repository root in a Python 3.10–3.13 environment; Python 3.12 is recommended:

```sh
python -m pip install -r requirements-training.txt
```

On macOS, LightGBM also needs the OpenMP runtime: `brew install libomp`. The requirements pin AutoGluon to 1.5.0. See the official [AutoGluon installation instructions](https://auto.gluon.ai/1.5/install.html) and [1.5.0 release notes](https://github.com/autogluon/autogluon/releases/tag/v1.5.0).

## 2. Fill the feature table

Copy `data/posteriq/features_template.csv` to `data/posteriq/features.csv`. Keep its 219 `id` values unchanged. The table joins to each existing split by `id`, so row order does not matter.

| Column | Definition | Supplied? |
|---|---|---|
| `id` | Poster identifier; used only for joining | Yes |
| `character_count` | Count of non-whitespace characters in the poster's text | Blank; fill from a consistent manual or OCR process |
| `word_count` | Word count using one documented tokenization rule | Blank; optional feature |
| `text_density` | Union area of text regions divided by total image area, from 0 to 1 | Blank; measure consistently |
| `aspect_ratio` | Image width divided by image height | Yes |
| `width_px` | Image width in pixels | Yes |
| `height_px` | Image height in pixels | Yes |

For text density, use the union of text-region boxes or polygons so overlaps are counted once. Use the same region convention for all posters and future predictions. A confirmed poster with no text has character count 0 and density 0; an unmeasured value stays blank. The template does not invent text measurements.

The default inputs are `character_count`, `text_density`, and `aspect_ratio`. You can add measured numeric columns and select them explicitly. Selected features must be numeric; IDs, image paths, and the target are not predictors. Some missing measurements are allowed and handled by AutoGluon, but a selected feature with no observed training values cannot be used. Filling the measurements before training is preferable to relying on imputation.

## 3. Train and compare on validation

```sh
python scripts/train_numeric.py \
  --features-csv data/posteriq/features.csv \
  --feature-columns character_count text_density aspect_ratio \
  --time-limit 120 \
  --num-cpus 2 \
  --output-dir models/numeric
```

To use more measurements, add their names after `--feature-columns`, for example `character_count word_count text_density aspect_ratio width_px height_px`. Unselected columns are ignored. To check the pipeline before text measurements are ready, select only `--feature-columns aspect_ratio width_px height_px` and use a separate output directory; that experiment measures only image geometry.

The script preserves the existing 153/33/33 train, validation, and test split. It checks feature joins and split separation, fits regression models on train, and supplies validation as AutoGluon's `tuning_data`. Candidate models are linear regression, random forest, extra trees, and LightGBM, with AutoGluon's weighted ensemble. The 120-second setting is a training budget, not a guarantee of total wall-clock time. See the official [`TabularPredictor.fit` API](https://auto.gluon.ai/1.5/api/autogluon.tabular.TabularPredictor.fit.html).

The report compares the selected model with a baseline that predicts the **training mean score** for every validation poster. RMSE and MAE are in score units and are better when smaller; R² is better when larger and can be negative. The AutoGluon leaderboard uses negative RMSE as its score, so values closer to zero are better. Validation performance helps choose features and settings; it is not an independent final estimate.

Use a different output directory for each experiment. The dataset splits stay fixed; model defaults and runtime conditions can still affect exact results.

## 4. Evaluate the final experiment

After choosing the feature set and training settings, run the final configuration with test scoring enabled:

```sh
python scripts/train_numeric.py \
  --features-csv data/posteriq/features.csv \
  --feature-columns character_count text_density aspect_ratio \
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

feature_columns = ["character_count", "text_density", "aspect_ratio"]
splits = load_splits(
    Path("data/posteriq"), Path("data/posteriq/features.csv"), feature_columns
)
predictor = TabularPredictor.load("models/numeric/model")
# Demonstration using an existing measured validation row.
inputs = splits["validation"].loc[:, feature_columns].head(1)
print(predictor.predict(inputs))
```

The notebook also works in Colab: it clones this repository if needed, then installs the same requirements. Supply your completed `features.csv` at the displayed path. Download the entire output directory to retain a Colab-trained model after the runtime ends.

These labels describe overall design quality, not a direct need for more text. A regression association between text measurements and score does not establish that adding or removing text will improve a particular poster. There are only 219 independent posters, so keep the number of features modest and compare against the baseline.
