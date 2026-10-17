"""
ChromaDB vector store using Ollama embeddings.
"""

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from .base import DocumentChunk, RetrievedChunk
from .config import SmartRAGConfig
from .llm import OllamaLLM

logger = logging.getLogger(__name__)

# nomic-embed-text is trained with task prefixes; using them noticeably improves retrieval
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
EMBED_BATCH = 32


class ChromaVectorStore:
    def __init__(self, config: SmartRAGConfig, llm: OllamaLLM):
        self.llm = llm
        self.use_prefixes = "nomic" in config.models.embedding_model
        persist_dir = config.vector_store.persist_directory
        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(persist_dir), settings=Settings(anonymized_telemetry=False)
        )
        # Embeddings are always computed here and passed explicitly, so Chroma needs no embedding function
        self.collection = client.get_or_create_collection(
            name=config.vector_store.collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Vector store ready: collection '{self.collection.name}' in {persist_dir}")

    def _embed(self, texts: List[str], prefix: str) -> List[List[float]]:
        if self.use_prefixes:
            texts = [prefix + t for t in texts]
        embeddings = []
        for i in range(0, len(texts), EMBED_BATCH):
            embeddings.extend(self.llm.embed(texts[i : i + EMBED_BATCH]))
        return embeddings

    def add(self, chunks: List[DocumentChunk]) -> None:
        if not chunks:
            return
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.content for c in chunks],
            embeddings=self._embed([c.content for c in chunks], DOC_PREFIX),
            metadatas=[_clean_metadata({**c.metadata, "document_type": c.document_type}) for c in chunks],
        )

    def search(self, query: str, k: int, document_ids: Optional[List[str]] = None) -> List[RetrievedChunk]:
        if self.collection.count() == 0:
            return []
        where = None
        if document_ids:
            where = {"document_id": {"$in": document_ids}}
        results = self.collection.query(
            query_embeddings=self._embed([query], QUERY_PREFIX),
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        retrieved = []
        for chunk_id, doc, meta, distance in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            chunk = DocumentChunk(
                content=doc,
                metadata=meta,
                document_type=meta.get("document_type", "text"),
                chunk_id=chunk_id,
                page_number=meta.get("page_number"),
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=1 - distance))
        return retrieved

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    def clear(self) -> None:
        ids = self.collection.get(include=[])["ids"]
        if ids:
            self.collection.delete(ids=ids)

    def count(self) -> int:
        return self.collection.count()


def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata values must be str, int, float or bool."""
    return {
        k: v if isinstance(v, (str, int, float, bool)) else str(v)
        for k, v in metadata.items()
        if v is not None
    }
