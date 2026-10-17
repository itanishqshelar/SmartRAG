"""
RAG orchestrator: ingests files, retrieves relevant chunks and generates grounded answers.
"""

import logging
import mimetypes
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, List, Optional

from .base import BaseProcessor, RetrievedChunk
from .config import SmartRAGConfig, load_config
from .documents import Document, DocumentStore
from .llm import OllamaLLM
from .processors import AudioProcessor, DOCXProcessor, ImageProcessor, PDFProcessor, TextProcessor
from .vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are SmartRAG, a helpful assistant that answers questions using the user's own documents, images and audio transcripts.

- Context excerpts are retrieved automatically and may be irrelevant. Use only the ones that help answer the question, and cite them inline with their bracketed numbers, e.g. [1] or [2][3]. Never cite excerpts you didn't use.
- For casual messages (greetings, thanks, small talk), reply naturally and briefly; ignore the excerpts and don't mention them.
- If the excerpts don't contain the answer, say so plainly, then help from general knowledge if appropriate.
- Be concise and use Markdown (lists, bold, tables, code blocks) where it improves readability."""

EXCERPT_PREVIEW_CHARS = 600


class IngestionError(Exception):
    pass


@dataclass
class Source:
    index: int
    document_id: Optional[str]
    filename: str
    document_type: str
    page: Optional[int]
    score: float
    excerpt: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class QueryResponse:
    answer: str
    sources: List[Source] = field(default_factory=list)


class RAGSystem:
    def __init__(self, config: Optional[SmartRAGConfig] = None):
        self.config = config or load_config()
        self.llm = OllamaLLM(self.config)
        self.vector_store = ChromaVectorStore(self.config, self.llm)
        self.documents = DocumentStore(self.config.storage.data_directory)
        self.audio = AudioProcessor(self.config)
        self.processors: List[BaseProcessor] = [
            TextProcessor(self.config),
            PDFProcessor(self.config),
            DOCXProcessor(self.config),
            ImageProcessor(self.config),
            self.audio,
        ]

    # ------------------------------------------------------------------ status

    @property
    def supported_extensions(self) -> List[str]:
        return sorted(ext for p in self.processors for ext in p.supported_extensions)

    def status(self) -> Dict:
        models = self.llm.status()
        return {
            "ready": all(models.values()),
            **models,
            "llm_model": self.config.models.llm_model,
            "embedding_model": self.config.models.embedding_model,
            "documents": len(self.documents.list()),
            "chunks": self.vector_store.count(),
        }

    # --------------------------------------------------------------- ingestion

    def _processor_for(self, path: Path) -> Optional[BaseProcessor]:
        return next((p for p in self.processors if p.can_process(path)), None)

    def ingest(self, file: BinaryIO, filename: str, content_type: Optional[str] = None) -> Document:
        """Store a file, extract and embed its content. Raises IngestionError on failure."""
        filename = Path(filename).name
        if not content_type or content_type == "application/octet-stream":
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        processor = self._processor_for(Path(filename))
        if processor is None:
            raise IngestionError(f"Unsupported file type: {Path(filename).suffix or filename}")

        document_id = self.documents.new_id()
        path = self.documents.file_path(document_id, filename)
        with open(path, "wb") as out:
            shutil.copyfileobj(file, out)

        try:
            result = processor.extract_content(path)
            if not result.success:
                raise IngestionError(result.error_message or "Could not extract any content")
            for chunk in result.chunks:
                chunk.metadata.update(document_id=document_id, filename=filename)
            self.vector_store.add(result.chunks)
        except IngestionError:
            path.unlink(missing_ok=True)
            raise
        except Exception as e:
            path.unlink(missing_ok=True)
            self.vector_store.delete_document(document_id)
            logger.exception(f"Failed to ingest {filename}")
            raise IngestionError(str(e)) from e

        logger.info(f"Ingested {filename}: {len(result.chunks)} chunks in {result.processing_time:.1f}s")
        return self.documents.add(
            document_id=document_id,
            filename=filename,
            content_type=content_type,
            document_type=processor.document_type,
            size=path.stat().st_size,
            chunks=len(result.chunks),
        )

    def ingest_file(self, path: str | Path) -> Document:
        path = Path(path)
        with open(path, "rb") as f:
            return self.ingest(f, path.name)

    def delete_document(self, document_id: str) -> bool:
        self.vector_store.delete_document(document_id)
        return self.documents.delete(document_id) is not None

    def clear(self) -> None:
        for doc in self.documents.list():
            self.documents.delete(doc.id)
        self.vector_store.clear()

    def transcribe(self, path: Path) -> str:
        return self.audio.transcribe(path).get("text", "").strip()

    # --------------------------------------------------------------- answering

    def retrieve(self, query: str, document_ids: Optional[List[str]] = None) -> List[RetrievedChunk]:
        hits = self.vector_store.search(query, self.config.retrieval.top_k, document_ids)
        threshold = self.config.retrieval.similarity_threshold
        return [h for h in hits if h.score >= threshold]

    def stream_answer(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        document_ids: Optional[List[str]] = None,
    ) -> tuple[List[Source], Iterator[str]]:
        """Retrieve context, then return (sources, token stream)."""
        hits = self.retrieve(message, document_ids)
        sources = [_to_source(i, hit) for i, hit in enumerate(hits, 1)]

        system_prompt = SYSTEM_PROMPT
        if hits:
            context = "\n\n".join(
                f"[{s.index}] {s.filename}{f' (page {s.page})' if s.page else ''}:\n{hit.chunk.content}"
                for s, hit in zip(sources, hits)
            )
            prompt = f"Possibly relevant excerpts from my documents:\n\n{context}\n\n---\n\n{message}"
        else:
            prompt = message
            if self.vector_store.count() > 0:
                system_prompt += "\n\nNo excerpts from the user's documents matched this message."

        max_history = self.config.generation.history_messages
        prior = [m for m in (history or []) if m.get("role") in ("user", "assistant") and m.get("content")]
        messages = [
            {"role": "system", "content": system_prompt},
            *(prior[-max_history:] if max_history else []),
            {"role": "user", "content": prompt},
        ]
        return sources, self.llm.stream_chat(messages)

    def query(self, message: str, history: Optional[List[Dict[str, str]]] = None,
              document_ids: Optional[List[str]] = None) -> QueryResponse:
        sources, tokens = self.stream_answer(message, history, document_ids)
        return QueryResponse(answer="".join(tokens), sources=sources)


def _to_source(index: int, hit: RetrievedChunk) -> Source:
    meta = hit.chunk.metadata
    excerpt = hit.chunk.content
    if len(excerpt) > EXCERPT_PREVIEW_CHARS:
        excerpt = excerpt[:EXCERPT_PREVIEW_CHARS].rsplit(" ", 1)[0] + "…"
    return Source(
        index=index,
        document_id=meta.get("document_id"),
        filename=meta.get("filename", "Unknown"),
        document_type=hit.chunk.document_type,
        page=hit.chunk.page_number,
        score=round(hit.score, 3),
        excerpt=excerpt,
    )
