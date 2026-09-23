"""Poster Scorer: Gradio app that scores an uploaded poster.

Pipeline: poster image -> OCR + pixel analysis (features.py) -> one tabular
row -> trained model -> predicted PosterIQ design score on its 1-10 scale.
Until a trained model is published, a clearly labeled baseline rule scores
text to graphic balance from 0 to 100 instead.
"""

import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image

from features import FEATURE_COLUMNS, MIN_SIDE_PX, extract_features

# Model location. Leave MODEL_REPO_ID as None until the model is uploaded.
MODEL_REPO_ID = None  # e.g. "jackstev-edu/postscorer-autogluon"
MODEL_REVISION = None  # Pin a commit hash so the app never silently changes
ZIP_FILENAME = "autogluon_predictor_dir.zip"
LOCAL_MODEL_DIR = Path("model")  # Optional unzipped predictor committed to the repo
EXAMPLES_DIR = Path("examples")

# Trained model settings, on the PosterIQ overall_design_score scale
MODEL_SCORE_RANGE = (1.0, 10.0)
LOWER_QUARTILE = 4.0  # Dataset quartiles of overall_design_score
UPPER_QUARTILE = 5.9

# Baseline rule settings, used only when no trained model is available
BASELINE_TARGET_SHARE = 0.35  # Placeholder ideal text share, replace after EDA
BASELINE_WIDTH = 0.25  # How quickly the baseline score falls off
MIN_CONTENT_FRAC = 0.01  # Below this the poster is treated as blank


def _safe_extract(archive, dest):
    """Unzip while blocking path traversal, as in the class notebooks."""
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (dest / item.filename).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError("Unexpected path in model archive.")
        z.extractall(dest)


def check_predictor(predictor):
    """Startup contract check; returns the model's feature columns."""
    features = list(predictor.features())
    unknown = sorted(set(features) - set(FEATURE_COLUMNS))
    if unknown:
        # The model may use a subset, but every column must come from features.py
        raise ValueError(f"Model features not produced by features.py: {unknown}")
    if predictor.problem_type != "regression":
        raise ValueError("App expects a regression model for overall_design_score.")
    return features


def load_predictor():
    """Return (predictor or None, model feature columns or None, human-readable source label)."""
    model_dir = None
    if LOCAL_MODEL_DIR.exists() and any(LOCAL_MODEL_DIR.iterdir()):
        model_dir = LOCAL_MODEL_DIR  # Local folder wins so offline dev works
    elif MODEL_REPO_ID:
        from huggingface_hub import hf_hub_download  # Lazy import keeps baseline mode light

        archive = hf_hub_download(repo_id=MODEL_REPO_ID, filename=ZIP_FILENAME, revision=MODEL_REVISION)
        model_dir = Path(tempfile.mkdtemp(prefix="postscorer_model_"))
        _safe_extract(archive, model_dir)
    if model_dir is None:
        return None, None, "Baseline rule (no trained model loaded yet)"

    from autogluon.tabular import TabularPredictor  # Heavy import, only when a model exists

    predictor = TabularPredictor.load(str(model_dir))
    features = check_predictor(predictor)  # Fail at startup, not on a user's upload
    return predictor, features, f"Trained model: {predictor.model_best}"


PREDICTOR, MODEL_FEATURES, MODEL_SOURCE = load_predictor()

# Labels switch with the scoring mode so the UI never mixes the two scales
if PREDICTOR is None:
    TITLE = "Poster Scorer: text balance baseline"
    INTRO = "Upload a poster to score its balance of text and graphics from 0 to 100 with a placeholder rule. "
    SCORE_LABEL = "Score (0-100)"
    SCOPE_NOTE = "This score reflects text versus graphic area only, not color, content, or overall design quality."
else:
    TITLE = "Poster Scorer: predicted design score"
    INTRO = "Upload a poster to predict its PosterIQ overall design score on a 1 to 10 scale. "
    SCORE_LABEL = "Predicted design score (1-10)"
    SCOPE_NOTE = (
        "The model predicts PosterIQ overall design quality from text and layout measurements. "
        "It does not measure whether adding or removing text would improve the poster."
    )


def baseline_score(row):
    """Gaussian bump around a target text share; transparent placeholder."""
    z = (row["text_share"] - BASELINE_TARGET_SHARE) / BASELINE_WIDTH
    return 100.0 * float(np.exp(-0.5 * z * z))


def baseline_band(score):
    """Map a 0-100 baseline score to a short verdict."""
    if score >= 75:
        return "Well balanced"
    if score >= 50:
        return "Acceptable"
    if score >= 25:
        return "Needs rebalancing"
    return "Poorly balanced"


def design_band(score):
    """Map a 1-10 predicted design score to its dataset quartile band."""
    if score < LOWER_QUARTILE:
        return "Below typical"
    if score < UPPER_QUARTILE:
        return "Typical"
    return "Above typical"


def lean_note(row):
    """Describe which side dominates, straight from the measurements."""
    if row["word_count"] == 0:
        return "No readable text was detected."
    if row["text_share"] > 0.6:
        return "Leans text-heavy."
    if row["text_share"] < 0.2:
        return "Leans graphic-heavy."
    return "Text and graphics are roughly comparable in area."


def empty_outputs(message):
    """Every early exit returns the same five outputs as a success."""
    return None, "", pd.DataFrame(columns=["feature", "value"]), None, message


def score_poster(image):
    """Callback: poster image in, score + verdict + features + overlay + status out."""
    if image is None:
        return empty_outputs("Upload a poster image, then select Score poster.")
    if not isinstance(image, Image.Image):
        return empty_outputs("Use a PNG, JPG, or WEBP image file.")
    if min(image.size) < MIN_SIDE_PX:
        return empty_outputs(f"Image is too small; use at least {MIN_SIDE_PX}px on the shortest side.")

    try:
        row, overlay = extract_features(image)
    except pytesseract.TesseractNotFoundError:
        return empty_outputs("OCR engine not found. Install Tesseract or set TESSERACT_CMD.")

    # Long format table reads better than one very wide row
    table = pd.DataFrame({"feature": FEATURE_COLUMNS, "value": [round(float(row[c]), 4) for c in FEATURE_COLUMNS]})
    if row["text_area_frac"] + row["graphic_area_frac"] < MIN_CONTENT_FRAC:
        # Nearly blank input: a score here would be meaningless
        return None, "", table, overlay, "Almost no text or graphics were detected. Check that the image is a poster."

    # Round before banding so the shown score and its verdict always agree
    if PREDICTOR is None:
        score = round(float(np.clip(baseline_score(row), 0, 100)), 1)
        band = baseline_band(score)
    else:
        frame = pd.DataFrame([row], columns=MODEL_FEATURES)  # Only the columns the model was trained on
        score = round(float(np.clip(PREDICTOR.predict(frame).iloc[0], *MODEL_SCORE_RANGE)), 1)  # Regressors can overshoot
        band = design_band(score)

    status = f"{lean_note(row)} Source: {MODEL_SOURCE}. {SCOPE_NOTE}"
    return score, band, table, overlay, status


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(
        f"# {TITLE}\n"
        f"{INTRO}"
        "The overlay shows what was measured: **orange boxes** are detected text, "
        "**blue tint** is graphic area."
    )
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", sources=["upload", "clipboard"], label="Poster")
            score_btn = gr.Button("Score poster", variant="primary")
        with gr.Column():
            score_out = gr.Number(label=SCORE_LABEL, precision=1, interactive=False)
            band_out = gr.Textbox(label="Verdict", interactive=False)
            status_out = gr.Textbox(label="Interpretation or next step", interactive=False, lines=3)
    with gr.Row():
        overlay_out = gr.Image(label="What the model measured", interactive=False)
        table_out = gr.Dataframe(label="Extracted features", interactive=False)

    outputs = [score_out, band_out, table_out, overlay_out, status_out]
    score_btn.click(score_poster, image_in, outputs, api_name="score_poster", concurrency_limit=2)
    # Clearing the input also clears stale results
    image_in.clear(lambda: empty_outputs("Upload another poster."), outputs=outputs, api_name="clear_poster")

    example_paths = sorted(p for p in EXAMPLES_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if example_paths:
        gr.Examples(
            examples=[[str(p)] for p in example_paths], inputs=image_in, outputs=outputs,
            fn=score_poster, run_on_click=True, cache_examples=False,
            label="Click an example to score it",
        )

if __name__ == "__main__":
    demo.queue().launch()
