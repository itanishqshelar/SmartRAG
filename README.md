# SmartRAG - Multimodal Document Chat System

A sophisticated RAG (Retrieval-Augmented Generation) system that enables intelligent conversations with documents, images, and audio files through a clean ChatGPT-style interface.

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

- **Ollama** - Local LLM hosting (Llama 3.1 8B)
- **PyTorch** - Deep learning framework
- **Transformers** - Hugging Face model library
- **OpenAI Whisper** - Speech-to-text conversion
- **BLIP** - Image captioning (Salesforce/blip-image-captioning-base)

### **Vector Database & Embeddings**

- **ChromaDB** - Vector storage and similarity search
- **Sentence Transformers** - Text embeddings
- **Nomic Embed Text** - Embedding model via Ollama
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

The system uses `config.yaml` for configuration:

```yaml
models:
  llm_model: "llama3.1:8b" # Ollama model
  embedding_model: "nomic-embed-text" # Embedding model
  vision_model: "Salesforce/blip-image-captioning-base"
  whisper_model: "base" # Whisper size

vector_store:
  type: "chromadb"
  persist_directory: "./vector_db"
  collection_name: "traditional_multimodal_documents"

processing:
  chunk_size: 1000
  chunk_overlap: 200
  max_image_size: [1024, 1024]
  ocr_enabled: true
```

## 🔧 Requirements

- **Python**: 3.8 or higher
- **Ollama**: For local LLM inference
- **Tesseract OCR**: For image text extraction
- **FFmpeg**: For audio processing (optional)

## 🏃‍♂️ Installation

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

## � Architecture

```
[User Input] → [Streamlit UI] → [RAG System] → [File Processors]
                    ↓                              ↓
[SQLite DB] ← [ChromaDB] ← [Embeddings] ← [Text Chunks]
                    ↓
[Vector Search] → [Context Retrieval] → [Ollama LLM] → [Response]
```

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

**ChromaDB** (Default - Good for development)

```yaml
vector_store:
  type: "chromadb"
  persist_directory: "./chroma_db"
  collection_name: "documents"
```

**FAISS** (High performance - Good for production)

```yaml
vector_store:
  type: "faiss"
  persist_directory: "./faiss_db"
  embedding_dimension: 384
```

### Model Configuration

**Lightweight Setup** (Lower resource usage)

```yaml
models:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  llm_model: "microsoft/DialoGPT-small"
  whisper_model: "tiny"
```

**High Performance Setup** (Better quality)

```yaml
models:
  embedding_model: "sentence-transformers/all-mpnet-base-v2"
  llm_model: "microsoft/DialoGPT-large"
  whisper_model: "large"
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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

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

## 🙏 Acknowledgments

- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Hugging Face Transformers](https://huggingface.co/transformers/) for language models
- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text extraction

## 📞 Support

- 📧 Email: support@smartrag.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/smartrag/issues)
- 📖 Documentation: [Wiki](https://github.com/your-repo/smartrag/wiki)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-repo/smartrag/discussions)

---

**SmartRAG** - Intelligent multimodal document understanding for the modern age. 🚀
