"""
Core data types and the base class for content processors.
"""

import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import SmartRAGConfig

PAGE_MARKER = re.compile(r"---\s*Page\s+(\d+)\s*---")


@dataclass
class DocumentChunk:
    """A chunk of extracted content from any modality."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_type: str = "text"
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    page_number: Optional[int] = None


@dataclass
class ProcessingResult:
    """Outcome of processing a single file."""

    chunks: List[DocumentChunk]
    success: bool = True
    error_message: Optional[str] = None
    processing_time: Optional[float] = None


@dataclass
class RetrievedChunk:
    """A chunk returned from similarity search, with its cosine similarity."""

    chunk: DocumentChunk
    score: float


class BaseProcessor(ABC):
    """Extracts text from a file and splits it into chunks."""

    supported_extensions: List[str] = []
    document_type = "text"

    def __init__(self, config: SmartRAGConfig):
        self.config = config
        self.chunk_size = config.processing.chunk_size
        self.chunk_overlap = config.processing.chunk_overlap

    def can_process(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.supported_extensions

    @abstractmethod
    def extract_content(self, file_path: Path) -> ProcessingResult:
        """Extract and chunk content from the file."""

    def _create_chunks(self, content: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split content into overlapping chunks, preferring paragraph/sentence/word boundaries."""
        content = content.strip()
        if not content:
            return []

        markers = [(m.start(), int(m.group(1))) for m in PAGE_MARKER.finditer(content)]
        pieces = []
        start = 0
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            if end < len(content):
                window = content[start:end]
                for sep in ("\n\n", "\n", ". ", " "):
                    cut = window.rfind(sep)
                    if cut > self.chunk_size // 2:
                        end = start + cut + len(sep)
                        break
            text = content[start:end].strip()
            if text:
                pieces.append((start, text))
            if end >= len(content):
                break
            start = max(end - self.chunk_overlap, start + 1)

        chunks = []
        for index, (offset, text) in enumerate(pieces):
            # Page of a chunk = last page marker at or before where the chunk starts (or inside it)
            page = next((p for pos, p in reversed(markers) if pos <= offset), None)
            if page is None and markers and markers[0][0] < offset + len(text):
                page = markers[0][1]
            chunk_meta = {**metadata, "chunk_index": index, "total_chunks": len(pieces)}
            if page is not None:
                chunk_meta["page_number"] = page
            chunks.append(
                DocumentChunk(
                    content=PAGE_MARKER.sub("", text).strip() or text,
                    metadata=chunk_meta,
                    document_type=self.document_type,
                    page_number=page,
                )
            )
        return chunks
