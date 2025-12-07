# SmartRAG - Intelligent Multimodal RAG System

A production-ready RAG system enabling intelligent conversations with documents, images, and audio files. Built with local-first AI models for complete privacy and offline operation.

![SmartRAG Interface](https://github.com/user-attachments/assets/7b413c33-3208-405b-a4f9-b18381807216)

## Quick Start

```bash
# Standard deployment
docker-compose up -d

# Access at http://localhost:8501
```

## Core Features

**Multimodal Processing**

- Documents: PDF, DOCX, TXT, MD with intelligent chunking
- Images: OCR + visual understanding via BLIP
- Audio: Automatic transcription with Whisper

**Local AI Stack**

- Ollama (Llama 3.1 8B) for generation
- Nomic Embed Text (768-dim) for embeddings
- ChromaDB for vector storage
- Complete offline operation

**Production Ready**

- Docker deployment with multi-stage builds
- Non-root user execution
- Health checks and auto-healing
- Resource management and monitoring
- Security hardening included

## Technology Stack

| Component      | Technology                  |
| -------------- | --------------------------- |
| **LLM**        | Llama 3.1 8B via Ollama     |
| **Embeddings** | Nomic Embed Text (768-dim)  |
| **Vector DB**  | ChromaDB / FAISS            |
| **Vision**     | BLIP + CLIP + Tesseract OCR |
| **Audio**      | OpenAI Whisper (base)       |
| **UI**         | Streamlit                   |
| **Storage**    | SQLite3                     |

## Architecture

<img width="1600" height="676" alt="image" src="https://github.com/user-attachments/assets/b4e96e9f-d797-409e-be6b-ceb295e91615" />

## Installation

### Docker (Recommended)

```bash
git clone https://github.com/itanishqshelar/SmartRAG.git
cd SmartRAG/docker

# Development
docker-compose up -d

# Production with full stack (PostgreSQL, Redis, Nginx)
docker-compose -f docker-compose.prod.yml up -d
```

### Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install Ollama and models
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# Install system dependencies
# macOS: brew install tesseract ffmpeg
# Ubuntu: apt-get install tesseract-ocr ffmpeg
# Windows: Download from GitHub releases

# Run application
streamlit run chatbot_app.py
```

## Configuration

SmartRAG uses a single `config.yaml` with Pydantic validation:

```yaml
models:
  llm_model: "llama3.1:8b"
  embedding_model: "nomic-embed-text"
  vision_model: "Salesforce/blip-image-captioning-base"
  whisper_model: "base"

vector_store:
  type: "chromadb"
  embedding_dimension: 768

processing:
  chunk_size: 1000
  chunk_overlap: 200
  ocr_enabled: true

generation:
  temperature: 0.7
  max_tokens: 2000
  context_window: 4096
```

Override via environment variables:

```bash
export SMARTRAG_LLM_MODEL=llama2:7b
export SMARTRAG_TEMPERATURE=0.5
```

## Usage

**Web Interface**

1. Upload files via drag-and-drop
2. Ask questions about your content
3. View source documents inline
4. Manage chat history and files

**Python API**

```python
from multimodal_rag.system import MultimodalRAGSystem

system = MultimodalRAGSystem()

# Ingest content
system.ingest_file("document.pdf")
system.ingest_file("screenshot.png")
system.ingest_file("recording.mp3")

# Query with context
response = system.query("Summarize the key points")
print(response.answer)
```

**Batch Processing**

```python
# Process directories
results = system.ingest_directory("./docs/", recursive=True)
print(f"Processed {len(results)} files")
```

## Project Structure

```
smartrag/
├── chatbot_app.py              # Streamlit application
├── config.yaml                 # Configuration
├── requirements.txt            # Dependencies
├── multimodal_rag/
│   ├── system.py              # RAG orchestrator
│   ├── processors/            # File type handlers
│   │   ├── document_processor.py
│   │   ├── image_processor.py
│   │   └── audio_processor.py
│   └── vector_stores/         # DB implementations
│       ├── chroma_store.py
│       └── faiss_store.py
├── docker/                    # Production deployment
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── docker-compose.prod.yml
└── tests/                     # Test suite
```

## Deployment Options

**Standard** - All-in-one container with Ollama

```bash
docker-compose up -d
```

**Lightweight** - External Ollama on host

```bash
docker-compose -f docker-compose.lite.yml up -d
```

**Production** - Full stack with PostgreSQL, Redis, Nginx

```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Development

```bash
# Run tests
pytest tests/

# Code formatting
black multimodal_rag/ tests/

# Linting
flake8 multimodal_rag/ tests/
```

## Performance

- **Image size**: 4.2GB 
- **Memory**: 4-8GB recommended
- **CPU**: 2-4 cores recommended
- **Startup time**: ~90s (includes model downloads)
- **Query latency**: <3s typical

## Security

- Local inference - no external API calls
- Non-root container execution
- File size limits enforced (50MB default)
- No privilege escalation
- Security headers in production setup

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with ChromaDB, Ollama, Hugging Face Transformers, OpenAI Whisper, and Tesseract OCR.

---

**SmartRAG** - Local-first multimodal AI for document intelligence.
