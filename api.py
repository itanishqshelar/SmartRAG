"""
SmartRAG HTTP API.

Run with:  uvicorn api:app --reload --port 8000
"""

import json
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from multimodal_rag import IngestionError, RAGSystem, load_config

config = load_config()
logging.basicConfig(level=config.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smartrag.api")

rag: RAGSystem


@asynccontextmanager
async def lifespan(_: FastAPI):
    global rag
    rag = RAGSystem(config)
    status = rag.status()
    if not status["ready"]:
        logger.warning(f"Ollama is not ready yet: {status}. Start Ollama and pull the configured models.")
    yield


app = FastAPI(title="SmartRAG API", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: List[ChatMessage] = []
    document_ids: Optional[List[str]] = None


def _check_upload(file: UploadFile) -> None:
    limit = config.api.max_upload_size_mb
    if file.size is not None and file.size > limit * 1024 * 1024:
        raise HTTPException(413, f"{file.filename} is larger than {limit} MB")


@app.get("/api/health")
def health() -> Dict:
    return {
        **rag.status(),
        "supported_extensions": rag.supported_extensions,
        "max_upload_size_mb": config.api.max_upload_size_mb,
    }


@app.get("/api/documents")
def list_documents() -> List[Dict]:
    return [doc.to_dict() for doc in rag.documents.list()]


@app.post("/api/documents", status_code=201)
def upload_document(file: UploadFile) -> Dict:
    _check_upload(file)
    try:
        doc = rag.ingest(file.file, file.filename or "upload", file.content_type)
    except IngestionError as e:
        raise HTTPException(422, str(e))
    return doc.to_dict()


@app.get("/api/documents/{document_id}/file")
def get_document_file(document_id: str) -> FileResponse:
    doc = rag.documents.get(document_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    return FileResponse(
        rag.documents.file_path(doc.id, doc.filename),
        media_type=doc.content_type,
        filename=doc.filename,
        content_disposition_type="inline",
    )


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: str) -> None:
    if not rag.delete_document(document_id):
        raise HTTPException(404, "Document not found")


@app.delete("/api/documents", status_code=204)
def delete_all_documents() -> None:
    rag.clear()


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    """Stream the answer as newline-delimited JSON events: sources, token..., done | error."""

    def events():
        try:
            sources, tokens = rag.stream_answer(
                request.message,
                [m.model_dump() for m in request.history],
                request.document_ids or None,
            )
            yield _event("sources", sources=[s.to_dict() for s in sources])
            for token in tokens:
                yield _event("token", content=token)
            yield _event("done")
        except Exception as e:
            logger.exception("Chat failed")
            message = str(e)
            if "connect" in message.lower():
                message = "Can't reach Ollama. Make sure it is running (`ollama serve`)."
            yield _event("error", message=message)

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/api/transcribe")
def transcribe(file: UploadFile) -> Dict:
    _check_upload(file)
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        with tmp:
            tmp.write(file.file.read())
        return {"text": rag.transcribe(Path(tmp.name))}
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _event(type_: str, **data) -> str:
    return json.dumps({"type": type_, **data}) + "\n"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
