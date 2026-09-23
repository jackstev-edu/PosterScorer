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
    assert len(outputs) == 5 and outputs[0] is None  # Same five outputs as success


def test_small_image_is_rejected():
    assert app.score_poster(Image.new("RGB", (50, 50)))[0] is None


def test_blank_poster_gets_no_score():
    assert app.score_poster(Image.new("RGB", (400, 400), "gray"))[0] is None


def test_feature_columns_match_contract():
    row, _ = extract_features(Image.open(EXAMPLES / "balanced.png"))
    assert list(row) == FEATURE_COLUMNS


def test_score_is_in_range():
    score = app.score_poster(Image.open(EXAMPLES / "balanced.png"))[0]
    assert 0 <= score <= 100


def test_text_share_orders_examples():
    # Sanity check on measurement, independent of any model
    share = {n: extract_features(Image.open(EXAMPLES / f"{n}.png"))[0]["text_share"]
             for n in ["text_heavy", "balanced", "graphic_heavy"]}
    assert share["text_heavy"] > share["balanced"] > share["graphic_heavy"]
