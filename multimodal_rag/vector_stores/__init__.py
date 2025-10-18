"""
Vector stores package initialization.
"""

from .chroma_store import ChromaVectorStore

# Optional imports with fallback
try:
    from .faiss_store import FAISSVectorStore
    FAISS_AVAILABLE = True
except ImportError:
    FAISSVectorStore = None
    FAISS_AVAILABLE = False

__all__ = ['ChromaVectorStore']
if FAISS_AVAILABLE:
    __all__.append('FAISSVectorStore')