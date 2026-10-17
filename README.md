# SmartRAG

Chat with your documents, images and audio, privately. SmartRAG indexes your files locally and answers questions with inline citations that link back to the exact source, using local models through [Ollama](https://ollama.com). No data leaves your machine.

## Features

- **Multimodal ingestion**: PDF, DOCX, TXT and Markdown; images via Tesseract OCR and BLIP captioning; audio via Whisper transcription
- **Grounded answers**: streamed responses with numbered citations; click a citation to open the source at the cited page
- **Library management**: drag-and-drop uploads, previews, deletion, and scoping a chat to selected documents
- **Voice input**: dictate questions; they're transcribed locally with Whisper
- **Conversations**: multiple chats with follow-up context, saved in your browser

## Architecture

```
frontend/  (Next.js, port 3000)  ──HTTP──▶  api.py  (FastAPI, port 8000)
                                               │
                                     multimodal_rag/
                                       ├── processors/   PDF · DOCX · text · image (OCR + BLIP) · audio (Whisper)
                                       ├── vector_store  ChromaDB, embeddings from Ollama (nomic-embed-text)
                                       ├── llm           Ollama chat (llama3.1:8b)
                                       └── documents     uploaded files + SQLite registry (./data)
```

## Local setup

**Prerequisites**

- Python 3.10–3.12, Node.js 20+
- [Ollama](https://ollama.com/download), then pull the models:
  ```bash
  ollama pull llama3.1:8b
  ollama pull nomic-embed-text
  ```
- [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) (image OCR) and [ffmpeg](https://ffmpeg.org/download.html) (audio)
  - Windows: `winget install UB-Mannheim.TesseractOCR Gyan.FFmpeg`
  - macOS: `brew install tesseract ffmpeg`
  - Ubuntu: `sudo apt install tesseract-ocr ffmpeg`

**Install**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cd frontend && npm install
```

**Run**

```bash
python start.py
```

This starts the API on http://localhost:8000 and the web app on http://localhost:3000. To run them separately:

```bash
uvicorn api:app --reload --port 8000   # backend
cd frontend && npm run dev             # frontend
```

The image captioning model (~1 GB) and Whisper model (~140 MB) are downloaded from Hugging Face the first time you upload an image or audio file.

## Docker

```bash
cd docker
docker compose up -d --build
```

Open http://localhost:3000. The first start pulls the Ollama models (~5 GB). To use an Ollama instance already running on the host instead:

```bash
SMARTRAG_OLLAMA_HOST=http://host.docker.internal:11434 docker compose up -d --build backend frontend
```

## Configuration

Settings live in [`config.yaml`](config.yaml). Any value can be overridden with an environment variable named `SMARTRAG_<SECTION>__<KEY>`:

```bash
SMARTRAG_MODELS__LLM_MODEL=llama3.2:3b
SMARTRAG_RETRIEVAL__TOP_K=8
SMARTRAG_GENERATION__TEMPERATURE=0.3
```

The frontend reads `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).

## API

Interactive docs: http://localhost:8000/docs

| Method   | Endpoint                    | Description                                              |
| -------- | --------------------------- | -------------------------------------------------------- |
| `GET`    | `/api/health`               | Ollama/model status, counts, supported file types        |
| `GET`    | `/api/documents`            | List documents                                           |
| `POST`   | `/api/documents`            | Upload and index a file (multipart `file`)               |
| `GET`    | `/api/documents/{id}/file`  | Original file                                            |
| `DELETE` | `/api/documents/{id}`       | Delete a document and its vectors                        |
| `DELETE` | `/api/documents`            | Delete everything                                        |
| `POST`   | `/api/chat`                 | Ask a question; streams NDJSON (`sources`, `token`, `done`/`error`) |
| `POST`   | `/api/transcribe`           | Speech-to-text for an audio file                         |

## Python usage

```python
from multimodal_rag import RAGSystem

rag = RAGSystem()
rag.ingest_file("report.pdf")
rag.ingest_file("whiteboard.png")

response = rag.query("What are the action items?")
print(response.answer)
for source in response.sources:
    print(source.index, source.filename, source.page)
```

## License

MIT, see [LICENSE](LICENSE).
