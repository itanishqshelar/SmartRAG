# SmartRAG Docker Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p vector_db temp_uploads user_data logs data

# Expose Streamlit port
EXPOSE 8501

# Expose Ollama port
EXPOSE 11434

# Create startup script
RUN echo '#!/bin/bash\n\
    set -e\n\
    \n\
    # Start Ollama service in background\n\
    ollama serve &\n\
    OLLAMA_PID=$!\n\
    \n\
    # Wait for Ollama to be ready\n\
    echo "Waiting for Ollama to start..."\n\
    sleep 5\n\
    \n\
    # Pull required models\n\
    echo "Pulling Llama 3.1 8B model..."\n\
    ollama pull llama3.1:8b || echo "Failed to pull llama3.1:8b"\n\
    \n\
    echo "Pulling Nomic Embed Text model..."\n\
    ollama pull nomic-embed-text || echo "Failed to pull nomic-embed-text"\n\
    \n\
    # Start Streamlit\n\
    echo "Starting SmartRAG application..."\n\
    streamlit run chatbot_app.py --server.port=8501 --server.address=0.0.0.0\n\
    ' > /app/start.sh && chmod +x /app/start.sh

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV OLLAMA_HOST=http://localhost:11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run startup script
CMD ["/app/start.sh"]
