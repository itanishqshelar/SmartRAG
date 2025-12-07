# SmartRAG Docker Image - Production Ready
# Multi-stage build for optimized image size and security

# ============================================
# Stage 1: Builder - Install and cache models
# ============================================
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download Hugging Face models to cache directory
RUN mkdir -p /build/.cache/huggingface
ENV HF_HOME=/build/.cache/huggingface
RUN python3 -c "from transformers import BlipProcessor, BlipForConditionalGeneration; \
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base'); \
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base'); \
    print('✓ BLIP model cached successfully')"

# Pre-download Whisper model
RUN python3 -c "import whisper; \
    model = whisper.load_model('base'); \
    print('✓ Whisper model cached successfully')"

# Pre-download CLIP model for image embeddings
RUN python3 -c "from transformers import CLIPProcessor, CLIPModel; \
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
    print('✓ CLIP model cached successfully')"

# ============================================
# Stage 2: Runtime - Final production image
# ============================================
FROM python:3.10-slim

# Add metadata labels
LABEL maintainer="SmartRAG Team"
LABEL version="1.0"
LABEL description="Multimodal RAG System with Ollama"

# Install runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create non-root user for security
RUN groupadd -r smartrag --gid=1000 && \
    useradd -r -g smartrag --uid=1000 --create-home --shell /bin/bash smartrag

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder --chown=smartrag:smartrag /root/.local /home/smartrag/.local

# Copy cached models from builder
COPY --from=builder --chown=smartrag:smartrag /build/.cache /home/smartrag/.cache

# Update PATH to use user-installed packages
ENV PATH=/home/smartrag/.local/bin:$PATH
ENV HF_HOME=/home/smartrag/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/smartrag/.cache/huggingface

# Copy application code
COPY --chown=smartrag:smartrag . .

# Create necessary directories with proper permissions
RUN mkdir -p vector_db temp_uploads user_data logs data && \
    chown -R smartrag:smartrag vector_db temp_uploads user_data logs data

# Create directory for Ollama data
RUN mkdir -p /home/smartrag/.ollama && \
    chown -R smartrag:smartrag /home/smartrag/.ollama

# Expose ports
EXPOSE 8501 11434

# Create production-ready startup script with proper error handling
RUN echo '#!/bin/bash\n\
    set -euo pipefail\n\
    \n\
    # Trap signals for graceful shutdown\n\
    trap "echo Shutting down...; kill -TERM $OLLAMA_PID 2>/dev/null; exit 0" SIGTERM SIGINT\n\
    \n\
    echo "========================================"\n\
    echo "SmartRAG Production Startup"\n\
    echo "========================================"\n\
    \n\
    # Start Ollama service\n\
    echo "[1/5] Starting Ollama service..."\n\
    OLLAMA_HOST=0.0.0.0:11434 ollama serve &\n\
    OLLAMA_PID=$!\n\
    \n\
    # Wait for Ollama to be ready with timeout\n\
    echo "[2/5] Waiting for Ollama to be ready..."\n\
    MAX_RETRIES=30\n\
    RETRY_COUNT=0\n\
    while ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; do\n\
    RETRY_COUNT=$((RETRY_COUNT + 1))\n\
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then\n\
    echo "ERROR: Ollama failed to start after ${MAX_RETRIES} seconds"\n\
    exit 1\n\
    fi\n\
    echo "  Waiting for Ollama... (${RETRY_COUNT}/${MAX_RETRIES})"\n\
    sleep 1\n\
    done\n\
    echo "  ✓ Ollama is ready"\n\
    \n\
    # Pull required models with retry logic\n\
    echo "[3/5] Pulling required Ollama models..."\n\
    pull_model() {\n\
    local model=$1\n\
    local max_attempts=3\n\
    local attempt=1\n\
    \n\
    while [ $attempt -le $max_attempts ]; do\n\
    echo "  Pulling $model (attempt $attempt/$max_attempts)..."\n\
    if ollama pull $model; then\n\
    echo "  ✓ $model downloaded successfully"\n\
    return 0\n\
    fi\n\
    attempt=$((attempt + 1))\n\
    [ $attempt -le $max_attempts ] && sleep 5\n\
    done\n\
    \n\
    echo "  ⚠ Warning: Failed to pull $model after $max_attempts attempts"\n\
    return 1\n\
    }\n\
    \n\
    pull_model "llama3.1:8b"\n\
    pull_model "nomic-embed-text"\n\
    \n\
    # Verify Hugging Face models\n\
    echo "[4/5] Verifying cached models..."\n\
    python3 -c "\n\
    from transformers import BlipProcessor, CLIPProcessor\n\
    import whisper\n\
    try:\n\
    BlipProcessor.from_pretrained('\''Salesforce/blip-image-captioning-base'\'')\n\
    print('\''  ✓ BLIP model verified'\'')\n\
    CLIPProcessor.from_pretrained('\''openai/clip-vit-base-patch32'\'')\n\
    print('\''  ✓ CLIP model verified'\'')\n\
    whisper.load_model('\''base'\'')\n\
    print('\''  ✓ Whisper model verified'\'')\n\
    except Exception as e:\n\
    print(f'\''  ⚠ Warning: Model verification failed: {e}'\'')\n\
    " || echo "  ⚠ Warning: Model verification failed"\n\
    \n\
    # Start SmartRAG application\n\
    echo "[5/5] Starting SmartRAG application..."\n\
    echo "========================================"\n\
    echo "Application ready at http://localhost:8501"\n\
    echo "Ollama API available at http://localhost:11434"\n\
    echo "========================================"\n\
    \n\
    exec streamlit run chatbot_app.py \\\n\
    --server.port=8501 \\\n\
    --server.address=0.0.0.0 \\\n\
    --server.headless=true \\\n\
    --server.fileWatcherType=none \\\n\
    --browser.gatherUsageStats=false\n\
    ' > /app/start.sh && chmod +x /app/start.sh

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    OLLAMA_HOST=http://localhost:11434 \
    OLLAMA_MODELS=/home/smartrag/.ollama/models \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Security: Set file size limits via environment
ENV MAX_FILE_SIZE_MB=50 \
    MAX_UPLOAD_SIZE=52428800

# Switch to non-root user
USER smartrag

# Comprehensive health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health && \
    curl -f http://localhost:11434/api/tags > /dev/null || exit 1

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Run startup script
CMD ["/bin/bash", "/app/start.sh"]
