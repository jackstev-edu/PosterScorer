"""PosterScorer HTTP backend. The frontend is maintained separately."""
import base64
from contextlib import asynccontextmanager
import io
import logging
import os
from threading import Lock

os.environ.setdefault("OMP_THREAD_LIMIT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from poster_inference import PosterScorer

MAX_BYTES = 20 * 1024 * 1024
scorer = None
scoring_lock = Lock()


@asynccontextmanager
async def lifespan(app):
    global scorer
    scorer = PosterScorer()
    yield


app = FastAPI(title="PosterScorer API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                   allow_headers=["*"], allow_credentials=False)


@app.get("/health")
def health():
    if scorer is None:
        raise HTTPException(503, "Model is loading")
    return {"status": "ok"}


@app.post("/score")
def score(image: UploadFile = File(...)):
    """Upload a raster image; return score, feedback, measurements and PNG overlay."""
    content = image.file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(413, "Maximum image size is 20 MB")
    try:
        poster = Image.open(io.BytesIO(content))
        if poster.width * poster.height > 40_000_000:
            raise HTTPException(413, "Maximum image size is 40 million pixels")
        if min(poster.size) < 200:
            raise HTTPException(422, "Image must be at least 200 pixels wide and tall")
        poster.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "Upload a valid raster image, such as PNG or JPEG") from exc
    if scorer is None:
        raise HTTPException(503, "Model is loading")
    try:
        with scoring_lock:
            result = scorer.score_image(poster)
        overlay = result.pop("overlay")
        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG")
        result["overlay_png_base64"] = base64.b64encode(buffer.getvalue()).decode("ascii")
        return result
    except Exception as exc:
        logging.exception("Poster scoring failed")
        raise HTTPException(500, "Scoring failed; check server logs") from exc
