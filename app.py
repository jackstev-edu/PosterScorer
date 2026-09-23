"""Poster Scorer: Gradio app that scores an uploaded poster and suggests one change.

Pipeline: poster image -> OCR + pixel analysis (features.py) -> 14 numeric
features -> trained AutoGluon regressor -> PosterIQ design score on its 1-10
scale, plus one model-estimated feedback sentence (poster_inference.py).
"""

from pathlib import Path
from threading import Lock

import gradio as gr
import pandas as pd
import pytesseract
from PIL import Image

from features import FEATURE_COLUMNS, MIN_SIDE_PX, extract_features
from poster_inference import PosterScorer

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
LOWER_QUARTILE = 4.0  # Dataset quartiles of overall_design_score
UPPER_QUARTILE = 5.9
MIN_CONTENT_FRAC = 0.01  # Below this the poster is treated as blank
MAX_PIXELS = 40_000_000  # Same upload limit as api.py

# Load once per process; a broken bundle should fail startup, never fall back to a heuristic
SCORER = PosterScorer()
MODEL_SOURCE = f"Trained model: {SCORER.predictor.model_best}"
SCORING_LOCK = Lock()  # OCR runs in parallel, model calls one at a time as in api.py

SCOPE_NOTE = (
    "The model predicts PosterIQ overall design quality from text and layout measurements. "
    "The feedback is the change the model estimates would raise its score most, "
    "not a proven cause or a guaranteed improvement."
)


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
    """Every early exit returns the same six outputs as a success."""
    return None, "", "", pd.DataFrame(columns=["feature", "value"]), None, message


def score_poster(image):
    """Callback: poster image in, score + verdict + feedback + features + overlay + status out."""
    if image is None:
        return empty_outputs("Upload a poster image, then select Score poster.")
    if not isinstance(image, Image.Image):
        return empty_outputs("Use a PNG, JPG, or WEBP image file.")
    if min(image.size) < MIN_SIDE_PX:
        return empty_outputs(f"Image is too small; use at least {MIN_SIDE_PX}px on the shortest side.")
    if image.width * image.height > MAX_PIXELS:
        return empty_outputs("Image is too large; use at most 40 million pixels.")

    try:
        row, overlay = extract_features(image)  # OCR once; the scorer reuses these measurements
    except pytesseract.TesseractNotFoundError:
        return empty_outputs("OCR engine not found. Install Tesseract or set TESSERACT_CMD.")

    # Long format table reads better than one very wide row
    table = pd.DataFrame({"feature": FEATURE_COLUMNS, "value": [round(float(row[c]), 4) for c in FEATURE_COLUMNS]})
    if row["text_area_frac"] + row["graphic_area_frac"] < MIN_CONTENT_FRAC:
        # Nearly blank input: a score here would be meaningless
        return None, "", "", table, overlay, "Almost no text or graphics were detected. Check that the image is a poster."

    with SCORING_LOCK:
        result = SCORER.score_features(row)
    score = round(result["score"], 1)  # Already clipped to 1..10; round before banding so both agree
    status = f"{lean_note(row)} Source: {MODEL_SOURCE}. {SCOPE_NOTE}"
    return score, design_band(score), result["feedback_sentence"], table, overlay, status


with gr.Blocks(title="Poster Scorer") as demo:
    gr.Markdown(
        "# Poster Scorer\n"
        "Upload a poster to predict its PosterIQ overall design score on a 1 to 10 scale "
        "and get one suggested layout change from the model. "
        "The overlay shows what was measured: **orange boxes** are detected text, "
        "**blue tint** is graphic area."
    )
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", sources=["upload", "clipboard"], label="Poster")
            score_btn = gr.Button("Score poster", variant="primary")
        with gr.Column():
            score_out = gr.Number(label="Predicted design score (1-10)", precision=1, interactive=False)
            band_out = gr.Textbox(label="Verdict", interactive=False)
            feedback_out = gr.Textbox(label="Model feedback", interactive=False, lines=2)
            status_out = gr.Textbox(label="Interpretation", interactive=False, lines=3)
    with gr.Row():
        overlay_out = gr.Image(label="What the model measured", interactive=False)
        table_out = gr.Dataframe(label="Extracted features (model input)", interactive=False)

    outputs = [score_out, band_out, feedback_out, table_out, overlay_out, status_out]
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
