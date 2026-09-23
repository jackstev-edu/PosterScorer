"""Hosted backend adapter; leaves the teammate-owned GUI untouched."""
import spaces
import gradio as gr
import api
from functools import lru_cache
from poster_inference import PosterScorer

@lru_cache(maxsize=1)
def worker_scorer():
    return PosterScorer()

@spaces.GPU(duration=30)
def hosted_inference(poster):
    return worker_scorer().score_image(poster)

api.run_inference = hosted_inference
with gr.Blocks(title="PosterScorer API") as landing:
    gr.Markdown("# PosterScorer\nThe trained model backend is ready.\n\n"
                "Upload a poster through **[the interactive API](/docs)**, or connect your GUI to **POST `/score`**.\n\n"
                "Each response includes a score out of 10, one feedback sentence, 14 measured features and a text overlay.")
if __name__ == "__main__":
    # Preload before Gradio's server startup timeout; launch handles Space routing.
    api.scorer = worker_scorer()
    landing.launch(
        server_name="0.0.0.0",
        app_kwargs={"routes": api.app.routes, "middleware": api.app.user_middleware},
        show_error=False,
    )
