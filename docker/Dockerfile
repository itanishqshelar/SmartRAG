# SmartRAG backend (FastAPI). Ollama runs as a separate service, see docker-compose.yml.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch keeps the image a few GB smaller; drop the index URL for CUDA builds
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

RUN useradd --create-home --uid 1000 smartrag \
    && mkdir -p /app/data /app/vector_db \
    && chown -R smartrag:smartrag /app
USER smartrag

COPY --chown=smartrag:smartrag config.yaml api.py ./
COPY --chown=smartrag:smartrag multimodal_rag ./multimodal_rag

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/home/smartrag/.cache/huggingface

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
