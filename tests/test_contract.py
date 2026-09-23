"""Contract checks in the style of the class notebooks: shape, not accuracy."""

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Import app from repo root

import app  # noqa: E402
from features import FEATURE_COLUMNS, extract_features  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_empty_input_keeps_output_shape():
    outputs = app.score_poster(None)
    assert len(outputs) == 6 and outputs[0] is None  # Same six outputs as success


def test_small_image_is_rejected():
    assert app.score_poster(Image.new("RGB", (50, 50)))[0] is None


def test_blank_poster_gets_no_score():
    assert app.score_poster(Image.new("RGB", (400, 400), "gray"))[0] is None


def test_design_bands_use_dataset_quartiles():
    assert [app.design_band(s) for s in (3.9, 4.0, 5.8, 5.9)] == [
        "Below typical", "Typical", "Typical", "Above typical"]


def test_feature_columns_match_contract():
    row, _ = extract_features(Image.open(EXAMPLES / "balanced.png"))
    assert list(row) == FEATURE_COLUMNS


def test_trained_model_scores_and_gives_feedback():
    score, band, feedback, *_, status = app.score_poster(Image.open(EXAMPLES / "balanced.png"))
    assert 1 <= score <= 10 and band and feedback
    assert "Trained model" in status


def test_text_share_orders_examples():
    # Sanity check on measurement, independent of any model
    share = {n: extract_features(Image.open(EXAMPLES / f"{n}.png"))[0]["text_share"]
             for n in ["text_heavy", "balanced", "graphic_heavy"]}
    assert share["text_heavy"] > share["balanced"] > share["graphic_heavy"]
