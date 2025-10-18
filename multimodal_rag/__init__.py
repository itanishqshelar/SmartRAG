"""
Initialization file for the multimodal_rag package.
"""

from .base import (
    DocumentChunk,
    ProcessingResult,
    RetrievalResult,
    BaseProcessor,
    BaseVectorStore,
    BaseEmbedding,
    BaseLLM,
    QueryRequest,
    QueryResponse
)

__version__ = "1.0.0"
__author__ = "SmartRAG Team"

__all__ = [
    "DocumentChunk",
    "ProcessingResult", 
    "RetrievalResult",
    "BaseProcessor",
    "BaseVectorStore",
    "BaseEmbedding",
    "BaseLLM",
    "QueryRequest",
    "QueryResponse"
]