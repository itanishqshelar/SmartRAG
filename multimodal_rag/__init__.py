"""
SmartRAG: local-first multimodal retrieval-augmented generation.
"""

from .config import SmartRAGConfig, load_config
from .documents import Document
from .system import IngestionError, QueryResponse, RAGSystem, Source

__all__ = ["Document", "IngestionError", "QueryResponse", "RAGSystem", "SmartRAGConfig", "Source", "load_config"]
