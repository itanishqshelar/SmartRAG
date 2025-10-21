# SmartRAG - Multimodal Document Chat System

A sophisticated RAG (Retrieval-Augmented Generation) system that enables intelligent conversations with documents, images, and audio files through a clean ChatGPT-style interface.

<img width="1919" height="1078" alt="Screenshot 2025-10-18 125810" src="https://github.com/user-attachments/assets/7b413c33-3208-405b-a4f9-b18381807216" />

## 🚀 Quick Start

```bash
# Start the application
streamlit run chatbot_app.py
```

## 🏗️ Tech Stack

### **Core Framework**

- **Python 3.8+** - Primary language
- **Streamlit** - Web interface and UI framework
- **SQLite3** - File metadata storage and management

### **AI/ML Models**

- **Ollama** - Local LLM hosting (Llama 3.1 8B model)
- **PyTorch** - Deep learning framework
- **Transformers** - Hugging Face model library
- **OpenAI Whisper** - Speech-to-text conversion (base model)
- **BLIP** - Image captioning (Salesforce/blip-image-captioning-base)

### **Vector Database & Embeddings**

- **ChromaDB** - Vector storage and similarity search
- **Nomic Embed Text** - Text embeddings via Ollama (768-dim vectors)
- **CLIP** - Visual embeddings for images (openai/clip-vit-base-patch32)
- **FAISS** - Alternative vector search (Facebook AI)

### **Document Processing**

- **PyPDF2** - PDF text extraction
- **python-docx** - Word document processing
- **pdfplumber** - Advanced PDF parsing
- **python-pptx** - PowerPoint file support

### **Image Processing**

- **Pillow (PIL)** - Image manipulation
- **OpenCV** - Computer vision operations
- **Tesseract OCR** - Text extraction from images
- **pytesseract** - Python wrapper for Tesseract

### **Audio Processing**

- **PyDub** - Audio file manipulation
- **librosa** - Audio analysis and processing
- **Whisper** - Audio transcription

### **Utilities**

- **NumPy** - Numerical computations
- **PyYAML** - Configuration management
- **tqdm** - Progress bars
- **requests** - HTTP client

## 📁 Project Structure

```
smartrag/
├── chatbot_app.py              # Main Streamlit application
├── config.yaml                 # System configuration
├── requirements.txt            # Python dependencies
├── multimodal_rag/            # Core RAG system
│   ├── system.py              # Main RAG orchestrator
│   ├── base.py                # Base classes and interfaces
│   ├── processors/            # File processors
│   │   ├── document_processor.py  # PDF, DOCX, TXT
│   │   ├── image_processor.py     # Images with OCR
│   │   └── audio_processor.py     # Audio transcription
│   └── vector_stores/         # Vector database implementations
│       ├── chroma_store.py    # ChromaDB integration
│       └── faiss_store.py     # FAISS integration
├── file_storage.db            # SQLite database
├── vector_db/                 # ChromaDB persistence
└── user_data/                 # User session data
```

## 🎯 Features

### **Multimodal Support**

- **Documents**: PDF, DOCX, DOC, TXT, MD, RTF
- **Images**: JPG, PNG, BMP, TIFF, WEBP (with OCR)
- **Audio**: MP3, WAV, M4A, OGG, FLAC, AAC

### **AI Capabilities**

- Local LLM inference with Ollama
- Semantic search with vector embeddings
- Image understanding and captioning
- Speech-to-text transcription
- Context-aware document retrieval

### **User Interface**

- ChatGPT-style conversation interface
- File upload and management
- Real-time processing feedback
- Document viewer for stored files
- Recent uploads tracking

## ⚙️ Configuration

SmartRAG uses a **single source of truth** configuration system with Pydantic validation.

### Quick Configuration

The system uses `config.yaml` with priority chain:

```
CLI Overrides > Environment Variables > config.yaml > Defaults
```

**Example `config.yaml`:**

```yaml
system:
  name: "SmartRAG System"
  debug: false
  log_level: "INFO"

models:
  llm_model: "llama3.1:8b" # Ollama Llama 3.1 8B
  embedding_model: "nomic-embed-text" # 768-dim embeddings
  vision_model: "Salesforce/blip-image-captioning-base"
  whisper_model: "base"

vector_store:
  type: "chromadb"
  embedding_dimension: 768 # Must match embedding model

processing:
  chunk_size: 1000
  chunk_overlap: 200
  ocr_enabled: true # Tesseract OCR
```

### Environment Variable Overrides

```bash
export SMARTRAG_LLM_MODEL=llama2:7b
export SMARTRAG_TEMPERATURE=0.5
export SMARTRAG_DEBUG=true
```

### Programmatic Overrides

```python
from config_schema import load_config

config = load_config(
    "config.yaml",
    models__llm_model="llama2:7b",
    generation__temperature=0.5
)
```

📖 **See [CONFIG.md](CONFIG.md) for comprehensive configuration documentation**

## 🔧 Requirements

- **Python**: 3.8 or higher
- **Ollama**: For local LLM inference
- **Tesseract OCR**: For image text extraction
- **FFmpeg**: For audio processing (optional)

## 🏃‍♂️ Installation

### Option 1: Docker (Recommended) 🐳

```bash
# Clone the repository
git clone https://github.com/itanishqshelar/SmartRAG.git
cd SmartRAG

# Start with Docker Compose
cd docker
docker-compose up -d

# Access at http://localhost:8501
```

See [docker/README.md](docker/README.md) for detailed Docker deployment instructions.

### Option 2: Local Installation

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Install Ollama and pull models:**

   ```bash
   # Install Ollama (see ollama.ai)
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```

3. **Install Tesseract OCR:**

   - Windows: Download from GitHub releases
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

4. **Run the application:**
   ```bash
   streamlit run chatbot_app.py
   ```

## 📊 Architecture

```
[User Input] → [Streamlit UI] → [RAG System] → [File Processors]
                    ↓                              ↓
                                        [Document: PyPDF2/python-docx]
                                        [Image: Tesseract OCR + BLIP]
                                        [Audio: Whisper Transcription]
                    ↓                              ↓
[SQLite DB] ← [Text Chunks] → [Nomic Embed Text (Ollama)] → [ChromaDB]
                                                                ↓
[Vector Search] → [Context Retrieval] → [Llama 3.1 8B (Ollama)] → [Response]
```

### Processing Pipeline

• **Text Documents**: Extracted with PyPDF2/python-docx → Chunked → Embedded with Nomic Embed Text
• **Images**: OCR with Tesseract + Captioning with BLIP → Combined text → Embedded with Nomic Embed Text  
• **Audio**: Transcribed with Whisper → Chunked → Embedded with Nomic Embed Text
• **Storage**: All embeddings stored in ChromaDB (768-dim vectors) for semantic search
• **Generation**: Retrieved context fed to Llama 3.1 8B via Ollama for response generation

## 🎮 Usage

1. **Upload Files**: Drag & drop or browse files in the sidebar
2. **Chat**: Ask questions about your uploaded content
3. **View Files**: Use the eye icon to preview stored documents
4. **Manage Data**: Clear chat history or uploaded files as needed

## 🔒 Privacy

- **Fully Offline**: All processing happens locally
- **No Data Sent**: No external API calls for LLM inference
- **Local Storage**: Files and embeddings stored on your machine

# Ingest mixed content

system.ingest_file("presentation.pdf") # Slides
system.ingest_file("screenshot.png") # Image with text
system.ingest_file("meeting_recording.mp3") # Audio transcript

# Query across all modalities

response = system.query("What was discussed about the Q4 budget?")

````

### Batch Processing

```python
# Process entire directories
results = system.ingest_directory("./company_docs/", recursive=True)

# Get processing summary
successful = sum(1 for r in results.values() if r.success)
total_chunks = sum(len(r.chunks) for r in results.values() if r.success)
print(f"Processed {successful} files, created {total_chunks} chunks")
````

## 🧪 Testing

Run the test suite to verify installation:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python tests/test_system.py

# Run with coverage
pip install coverage
coverage run tests/test_system.py
coverage report
```

## 🔧 Advanced Configuration

### Vector Store Options

**ChromaDB** (Default - Recommended)

```yaml
vector_store:
  type: "chromadb"
  persist_directory: "./vector_db"
  collection_name: "documents"
  embedding_dimension: 768 # For nomic-embed-text
```

**FAISS** (Alternative - High performance)

```yaml
vector_store:
  type: "faiss"
  persist_directory: "./faiss_db"
  embedding_dimension: 768 # Must match nomic-embed-text
```

## 🚀 Deployment

### Local Development

```bash
python cli.py interactive
```

### Docker Container

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
CMD ["python", "cli.py", "interactive"]
```

### API Server

```python
from fastapi import FastAPI
from multimodal_rag.system import MultimodalRAGSystem

app = FastAPI()
system = MultimodalRAGSystem()

@app.post("/query")
async def query_endpoint(query: str):
    response = system.query(query)
    return {"answer": response.answer, "sources": len(response.sources)}
```

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python -m pytest

# Format code
black multimodal_rag/ tests/ examples/

# Lint code
flake8 multimodal_rag/ tests/ examples/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Hugging Face Transformers](https://huggingface.co/transformers/) for language models
- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text extraction

---

**SmartRAG** - Intelligent multimodal document understanding for the modern age.
