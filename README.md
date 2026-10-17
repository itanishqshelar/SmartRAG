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

### 1. Prerequisites

| Tool | Version | Used for |
| ---- | ------- | -------- |
| [Python](https://www.python.org/downloads/) | 3.10–3.12 | Backend API and RAG pipeline |
| [Node.js](https://nodejs.org) | 20+ | Next.js frontend |
| [Ollama](https://ollama.com/download) | latest | Local LLM and embeddings |
| [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) | 5.x | Text extraction from images (OCR) |
| [ffmpeg](https://ffmpeg.org/download.html) | any | Audio decoding for Whisper |

Install Tesseract and ffmpeg:

```bash
# Windows
winget install UB-Mannheim.TesseractOCR Gyan.FFmpeg
# macOS
brew install tesseract ffmpeg
# Ubuntu / Debian
sudo apt install tesseract-ocr ffmpeg
```

On Windows, Tesseract is picked up automatically from `C:\Program Files\Tesseract-OCR` even if it isn't on your `PATH`. Without Tesseract, images are still indexed using captions only.

### 2. Pull the Ollama models

Make sure Ollama is running (the desktop app starts it automatically; otherwise run `ollama serve`), then:

```bash
ollama pull llama3.1:8b        # chat model (~4.9 GB)
ollama pull nomic-embed-text   # embedding model (~270 MB)
```

### 3. Clone and install

```bash
git clone https://github.com/itanishqshelar/SmartRAG.git
cd SmartRAG

# Python backend
python -m venv .venv
.venv\Scripts\activate             # Windows (PowerShell / cmd)
source .venv/bin/activate          # macOS / Linux
pip install -r requirements.txt

# Next.js frontend
cd frontend
npm install
cd ..
```

## Running

### One command

From the project root:

```bash
python start.py
```

This:

- starts the FastAPI backend on http://localhost:8000 (with auto-reload)
- starts the Next.js dev server on http://localhost:3000
- uses the `.venv` interpreter automatically, even if the venv isn't activated
- runs `npm install` in `frontend/` if `node_modules` is missing
- warns if Ollama isn't reachable on http://localhost:11434

Open **http://localhost:3000** and upload a document to get started. Press `Ctrl+C` to stop both servers.

### Separately

In two terminals, both from the project root with the venv activated:

```bash
# Terminal 1: backend
uvicorn api:app --reload --port 8000

# Terminal 2: frontend
cd frontend
npm run dev
```

### Verify

- http://localhost:8000/api/health shows whether Ollama and the models are available
- http://localhost:8000/docs has the interactive API docs

The image captioning model (~1 GB) and Whisper model (~140 MB) are downloaded from Hugging Face the first time you upload an image or audio file, so the first upload of each type takes longer.

### Troubleshooting

- **"Ollama is not reachable"**: start Ollama (`ollama serve` or the desktop app) and check `ollama list` shows both models.
- **Model not found**: pull the model named in `config.yaml` (`models.llm_model` / `models.embedding_model`), or change it to one you have.
- **Frontend can't reach the API**: make sure the backend is on port 8000. If you run it elsewhere, set `NEXT_PUBLIC_API_URL` in `frontend/.env.local` and add the frontend origin to `api.cors_origins` in `config.yaml`.
- **Image text isn't extracted**: check `tesseract --version` works (Windows: confirm it's installed to `C:\Program Files\Tesseract-OCR`).
- **Audio upload fails**: check `ffmpeg -version` works in the same terminal.
- **Port already in use**: stop whatever is using port 3000/8000, or run the servers separately on other ports (see above).

## Docker

Runs Ollama, the backend and the frontend in containers; only Docker is required.

```bash
cd docker
docker compose up -d --build
```

Open http://localhost:3000. On first start the `ollama-pull` job downloads the models (~5 GB), so chat won't work until it finishes. Follow progress with:

```bash
docker compose logs -f ollama-pull
```

Uploaded files, the vector index, models and the Hugging Face cache are kept in Docker volumes across restarts.

```bash
docker compose down        # stop (keeps data)
docker compose down -v     # stop and delete all data and models
```

To use an Ollama instance already running on the host instead of the bundled one:

```bash
SMARTRAG_OLLAMA_HOST=http://host.docker.internal:11434 docker compose up -d --build backend frontend
```

For NVIDIA GPU acceleration, uncomment the `deploy` block under the `ollama` service in [`docker/docker-compose.yml`](docker/docker-compose.yml).

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
