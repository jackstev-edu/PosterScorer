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

# Poster Scorer

Upload a poster to predict its overall design score on the PosterIQ 1 to 10 scale and get one suggested layout change. The app measures text and layout with Tesseract OCR and background-color analysis, scores those 14 measurements with an AutoGluon regression model, and asks the same model which tested change would raise its score most.

**Model inputs:** the 14 numeric features in `features.py`, such as `char_count`, `text_area_frac`, `graphic_area_frac`, `text_share`, and text height.

**Results (219 posters, 153/33/33 split):**

| Split | Model RMSE | Mean baseline RMSE | Model R² |
|---|---|---|---|
| Validation | 1.36 | 1.48 | 0.14 |
| Test | 1.12 | 1.46 | 0.41 |

**Feedback:** the suggestion comes from testing training-data values of text area, graphic area, word count, and text size against the score model. It is a model estimate, not a proven cause of the score or a guaranteed improvement, and it is not trained on human feedback labels.

**Scope:** the score predicts overall design quality from text and layout measurements only. It does not judge color, content, or typography. OCR and graphic masks are estimates.

**Data:** [creative-graphic-design/PosterIQ](https://huggingface.co/datasets/creative-graphic-design/PosterIQ), `overall_rating` subset, listed for non-commercial research use. Code and data pipeline: [github.com/jackstev-edu/PosterScorer](https://github.com/jackstev-edu/PosterScorer).

**AI assistance:** code was developed with help from Claude (Anthropic).
