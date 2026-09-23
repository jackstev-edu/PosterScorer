---
title: PosterScorer
emoji: 🖼️
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: hf_space.py
pinned: false
---

# PosterScorer model backend

Trained regression model with OCR measurements and one-sentence feedback.

- Open `/docs` on the app's direct URL to upload a poster interactively.
- `POST /score`: multipart image field named `image`; returns `score` (1–10), `feedback_sentence`, 14 `features`, and `overlay_png_base64`.
- `GET /health`: model readiness.
- [Source and GUI integration](https://github.com/jackstev-edu/PosterScorer/blob/main/CLAUDE.md)

The application GUI is developed separately. This Space hosts its model API and a minimal landing page.

Model: AutoGluon 1.5.0, Python 3.12, 219 PosterIQ posters, 153/33/33 fixed split. Test RMSE 1.122; feedback estimates model-favored changes, not causal explanations. PosterIQ declares a non-commercial research license; no additional source or model rights are granted here.
