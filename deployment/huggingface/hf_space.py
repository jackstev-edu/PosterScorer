"""Hosted backend landing page; leaves the teammate-owned GUI untouched."""
import gradio as gr
import uvicorn
from api import app

with gr.Blocks(title="PosterScorer API") as landing:
    gr.Markdown("# PosterScorer\nThe trained model backend is ready.\n\n"
                "Upload a poster through **[the interactive API](/docs)**, or connect your GUI to **POST `/score`**.\n\n"
                "Each response includes a score out of 10, one feedback sentence, 14 measured features and a text overlay.")
app = gr.mount_gradio_app(app, landing, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, workers=1)
