# Host the trained model on Hugging Face Spaces

**Current free-account route:** run `python scripts/prepare_space.py --output space-export-free` (defaults to Gradio), create a Gradio Space using **ZeroGPU Free** on this account, and upload that folder’s contents. It serves the same FastAPI `/score` and `/health` endpoints plus a minimal landing page; your teammate’s GUI is untouched. Some accounts restrict Docker to paid plans. The Docker alternative below is available using `--sdk docker`.

This is a **backend only**. Your teammate owns the GUI. The repository includes a trained model; hosting requires no training, GPU, or separate model download.

## Deploy in five steps

1. Run `python scripts/prepare_space.py --sdk docker` from this repository. This creates `space-export/` containing the runtime and trained bundle, without the dataset images.
2. Open [Create a Space](https://huggingface.co/new-space). Choose your namespace/name and select **Docker → Blank**, with CPU hardware.
3. Upload **the contents** of `space-export/` to the Space root, preserving directories. Replace the generated README with ours. `Dockerfile` must be at the root.
4. Wait for the build to become Running. Open the Space's direct `https://YOUR-SPACE-SUBDOMAIN.hf.space/health` URL. It must return `{"status":"ok"}`. Find the actual direct URL in the Space embed/open menu; do not guess it from the repository name.
5. Give that direct base URL to the GUI agent, together with [the integration handoff](gui-handoff.md). It will call **POST `/score`**.

The direct backend root `/` is intentionally not a GUI and returns 404. `/docs` is the interactive API explorer. Hugging Face uses the README's `sdk: docker` and `app_port: 7860` metadata. [Official Docker Spaces guide](https://huggingface.co/docs/hub/spaces-sdks-docker) · [Space configuration](https://huggingface.co/docs/hub/spaces-config-reference).

## Run the same container locally

```sh
docker build -t poster-scorer .
docker run --rm -p 7860:7860 poster-scorer
curl http://localhost:7860/health
curl -X POST http://localhost:7860/score -F "image=@poster.png"
```

Without Docker, use Python 3.12, install Tesseract and OpenMP (`brew install tesseract libomp` on macOS), install `requirements-space.txt`, and run `uvicorn api:app --host 0.0.0.0 --port 7860`. On Debian/Ubuntu the system packages are `tesseract-ocr libgomp1`.

## Runtime contract

`Dockerfile` installs Python 3.12 and the pinned inference packages, then starts one API worker on port 7860. It loads the committed ZIP once on startup, extracts each uploaded image with the original OCR code and serializes model scoring with a lock. It never retrains or falls back to heuristic scores. The model cache is disposable; persistent storage is not needed.

The export includes `api.py`, `Dockerfile`, `requirements-space.txt`, `requirements-inference.txt`, `poster_inference.py`, `features.py`, `score_feedback.py`, `artifacts/poster_scorer_model.zip`, documentation and model metadata. The existing GUI `app.py` is neither changed nor launched.

## Common fixes

| Symptom | Fix |
| --- | --- |
| Space shows 404 at `/` | Expected for this backend; use `/health` or `/docs`. |
| Model version mismatch | Keep Python 3.12 and the pinned inference versions. |
| Extractor hash mismatch | Restore `features.py`; changing it requires re-extraction and retraining. |
| Tesseract or libgomp missing | Use the included Dockerfile, which installs both. |
| Missing model ZIP | Preserve the `artifacts/` folder and upload the actual binary ZIP. |
| Browser request fails | Use the direct `.hf.space` URL, not the Hugging Face Space page URL. |
| Private Space needs authentication | Route through your backend and keep the HF token server-side. |
| First request is slow | Wait for `/health`; free Spaces may sleep and OCR takes time. |

CORS allows browser uploads from any origin without cookies; restrict origins if your deployment needs it. Accepted images must be at least 200×200, no more than 20 MB and no more than 40 million pixels. Uploads are processed in memory. PNG overlays are returned as base64 strings.

The bundle was trained and inference-tested on macOS/Python 3.12. A Linux Docker/Space build is still the deployment verification step; no live Space is claimed here. The original PosterIQ dataset declares a non-commercial research license; see [source provenance](../data/posteriq/README.md). This setup grants no additional source-data or model rights.

The live free host uses the ZeroGPU request wrapper required by Hugging Face. The underlying tabular model still computes on CPU. Requests through the custom `/score` route use the Space’s available quota; show a retry state if hosting quota or capacity is exhausted.
