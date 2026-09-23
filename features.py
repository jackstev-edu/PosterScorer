"""Poster feature extraction shared by training and the Gradio app.

Training and inference must call the same function, otherwise the model
sees differently computed columns at serving time (train/serve skew).
"""

import os

import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFilter, ImageOps

# Allow a custom Tesseract path on Windows without editing code
if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]

# Column order is the model contract; training must use this exact list
FEATURE_COLUMNS = [
    "img_width_px",
    "img_height_px",
    "aspect_ratio",
    "word_count",
    "char_count",
    "text_block_count",
    "text_area_frac",
    "graphic_area_frac",
    "background_area_frac",
    "text_share",
    "text_to_graphic_ratio",
    "mean_text_height_frac",
    "max_text_height_frac",
    "text_height_std_frac",
]

MAX_SIDE_PX = 1600  # Normalize scale so OCR speed and noise stay consistent
MIN_SIDE_PX = 200  # Below this OCR output is unreliable, so reject
MIN_WORD_CONF = 60  # Tesseract confidence floor for keeping a word
BOX_PAD_PX = 3  # Small pad so anti-aliased glyph edges count as text
BG_DIST_THRESHOLD = 40.0  # RGB distance separating content from background
BORDER_FRAC = 0.03  # Border strip width used to estimate background color
RATIO_EPS = 1e-3  # Caps text_to_graphic_ratio when no graphics are found


def prepare_image(image):
    """Apply EXIF rotation, force RGB, and downscale large posters."""
    image = ImageOps.exif_transpose(image).convert("RGB")  # Phone photos often carry rotation tags
    scale = MAX_SIDE_PX / max(image.size)  # Only shrink, never upscale small images
    if scale < 1:
        new_size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)  # Lanczos keeps small text edges sharp
    return image


def _ocr_words(image):
    """Run Tesseract and keep confident, non-empty word boxes."""
    # PSM 11 finds sparse text anywhere, which suits poster layouts
    gray = ImageOps.grayscale(image)  # Color shapes confuse Tesseract layout analysis
    data = pytesseract.image_to_data(
        gray, config="--psm 11", output_type=pytesseract.Output.DICT
    )
    words = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        conf = float(data["conf"][i])  # Tesseract returns -1 for non-word rows
        n_alnum = sum(c.isalnum() for c in text)
        if conf < MIN_WORD_CONF or (n_alnum < 2 and not text.isdigit()):
            continue  # Single glyphs on shapes are the most common false positive
        words.append({
            "text": text,
            "left": data["left"][i],
            "top": data["top"][i],
            "width": data["width"][i],
            "height": data["height"][i],
            "block": (data["page_num"][i], data["block_num"][i]),  # Unique block id
        })
    return words


def _background_color(rgb):
    """Estimate background as the median color of the border strip."""
    h, w, _ = rgb.shape
    b = max(1, int(min(h, w) * BORDER_FRAC))  # Strip width scales with poster size
    border = np.concatenate([
        rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
        rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3),
    ])
    return np.median(border, axis=0)  # Median resists logos touching the edge


def extract_features(image):
    """Return (feature dict, overlay image) for one poster.

    The overlay shades detected graphics blue and outlines text in orange
    so a user can check what the numbers were computed from.
    """
    original_w, original_h = image.size  # Report true size, not the resized one
    img = prepare_image(image)
    w, h = img.size

    words = _ocr_words(img)

    # Text mask: union of padded word boxes
    text_mask = np.zeros((h, w), dtype=bool)
    for wd in words:
        x0 = max(0, wd["left"] - BOX_PAD_PX)
        y0 = max(0, wd["top"] - BOX_PAD_PX)
        x1 = min(w, wd["left"] + wd["width"] + BOX_PAD_PX)
        y1 = min(h, wd["top"] + wd["height"] + BOX_PAD_PX)
        text_mask[y0:y1, x0:x1] = True

    # Content mask: pixels far from the background color after denoising
    smooth = np.asarray(img.filter(ImageFilter.MedianFilter(5)), dtype=np.float32)
    bg = _background_color(smooth)
    dist = np.linalg.norm(smooth - bg, axis=2)  # Per-pixel RGB distance to background
    content_mask = dist > BG_DIST_THRESHOLD

    # Graphics are content that is not inside a text box
    graphic_mask = content_mask & ~text_mask

    text_frac = float(text_mask.mean())
    graphic_frac = float(graphic_mask.mean())
    heights = np.array([wd["height"] / h for wd in words]) if words else np.zeros(1)

    row = {
        "img_width_px": original_w,
        "img_height_px": original_h,
        "aspect_ratio": original_w / original_h,
        "word_count": len(words),
        "char_count": sum(len(wd["text"]) for wd in words),
        "text_block_count": len({wd["block"] for wd in words}),
        "text_area_frac": text_frac,
        "graphic_area_frac": graphic_frac,
        "background_area_frac": max(0.0, 1.0 - text_frac - graphic_frac),
        # Bounded 0..1 share is easier for models than an unbounded ratio
        "text_share": text_frac / (text_frac + graphic_frac) if (text_frac + graphic_frac) > 0 else 0.0,
        "text_to_graphic_ratio": text_frac / max(graphic_frac, RATIO_EPS),
        "mean_text_height_frac": float(heights.mean()),
        "max_text_height_frac": float(heights.max()),
        "text_height_std_frac": float(heights.std()),
    }
    assert list(row) == FEATURE_COLUMNS, "Feature dict drifted from FEATURE_COLUMNS."
    return row, _draw_overlay(img, graphic_mask, words)


def _draw_overlay(img, graphic_mask, words):
    """Tint graphic pixels blue and outline each OCR word box."""
    base = np.asarray(img, dtype=np.float32)
    tint = np.array([40, 110, 255], dtype=np.float32)
    base[graphic_mask] = 0.55 * base[graphic_mask] + 0.45 * tint  # Semi-transparent blend
    overlay = Image.fromarray(base.astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    for wd in words:
        box = [wd["left"], wd["top"], wd["left"] + wd["width"], wd["top"] + wd["height"]]
        draw.rectangle(box, outline=(255, 140, 0), width=3)  # Orange reads on most palettes
    return overlay
