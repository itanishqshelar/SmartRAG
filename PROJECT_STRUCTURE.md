# SmartRAG Project Structure

## 📁 Clean Project Structure

```
smartrag/
├── 🚀 start.py                    # Simple launcher script (FastAPI)
├── 🤖 start_chatbot.py           # Streamlit chatbot launcher
├── 🌐 enhanced_app.py             # FastAPI web application
├── 💬 chatbot_app.py             # Streamlit ChatGPT-style interface
├── ⚙️  config.yaml                # Configuration file
├── 📦 requirements.txt            # Python dependencies (includes streamlit)
├── 📚 README.md                   # Documentation
├── 📋 LICENSE                     # License file
│
├── 🔧 Setup Scripts
│   ├── setup_complete_ollama.py   # Complete Ollama setup
│   ├── setup_ollama.py           # Basic Ollama setup
│   ├── install_ollama_models.py  # Install required models
│   └── setup.py                  # Package setup
│
├── 🧹 Utilities
│   ├── clean_db.py               # Clean vector database
│   ├── INSTALLATION.md           # Installation guide
│   └── OFFLINE_GUIDE.md          # Offline usage guide
│
├── 🧠 Core System
│   └── multimodal_rag/           # Main RAG system package
│       ├── system.py             # Core system logic
│       ├── base.py              # Base classes and data models
│       └── processors/          # Document processors
│           ├── document_processor.py  # PDF, DOCX, TXT
│           ├── image_processor.py     # Images with OCR
│           └── audio_processor.py     # Audio transcription
│       └── vector_stores/       # Vector database integration
│           └── chroma_store.py   # ChromaDB with Ollama embeddings
│
├── 💾 Data Storage
│   ├── vector_db/               # ChromaDB vector database
│   ├── user_data/               # User upload history & streamlit data
│   └── data/                    # Sample/test data
```

## 🚀 Quick Start

1. **Setup Ollama and models:**

   ```bash
   python setup_complete_ollama.py
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Start the application:**

   #### Option 1: Streamlit ChatBot (Recommended)

   ```bash
   python start_chatbot.py
   # Opens at: http://localhost:8501
   ```

   #### Option 2: FastAPI Web Interface

   ```bash
   python start.py
   # OR: python enhanced_app.py
   # Opens at: http://localhost:8000
   ```

## ✨ Key Features

- ✅ **Complete Ollama Integration**: llama3.1:8b + nomic-embed-text
- ✅ **Dual Web Interfaces**:
  - 💬 **Streamlit ChatBot**: ChatGPT-style minimalistic interface
  - 🌐 **FastAPI UI**: Full-featured web application
- ✅ **File Management**: Upload, view, and delete documents
- ✅ **Multimodal Support**: PDF, DOCX, TXT, Images, Audio
- ✅ **Offline Operation**: No internet required after setup
- ✅ **Clean Architecture**: Removed all test scripts and duplicates
- ✅ **Production Ready**: Optimized for performance and reliability

## 🎯 Usage (Streamlit ChatBot)

1. **Launch**: `python start_chatbot.py`
2. **Upload**: Use sidebar to upload documents (PDF, DOCX, TXT, Images, Audio)
3. **Manage**: View uploaded files, delete unwanted ones
4. **Chat**: Ask questions in the main chat interface
5. **Get Answers**: Receive detailed, comprehensive responses
6. **All Local**: Everything processes locally with Ollama

The project is now clean, optimized, and ready for production use!
