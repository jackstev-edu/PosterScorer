FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements-inference.txt requirements-space.txt ./
RUN pip install --no-cache-dir -r requirements-space.txt
COPY --chown=user:user api.py poster_inference.py score_feedback.py features.py ./
COPY --chown=user:user artifacts/poster_scorer_model.zip ./artifacts/poster_scorer_model.zip
RUN chown -R user:user /app
USER user
ENV HOME=/home/user PYTHONUNBUFFERED=1 OMP_THREAD_LIMIT=1 OMP_NUM_THREADS=1
EXPOSE 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
