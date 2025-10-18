"""
LLaVA system stub - not implemented in this version.
"""

import logging
from typing import Dict, Any, Union
from .base import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

# Availability flag
LLAVA_AVAILABLE = False


class LLaVAQueryRequest(QueryRequest):
    """LLaVA-specific query request."""
    
    def __init__(self, query: str = "", **kwargs):
        self.image_path = kwargs.pop('image_path', None)
        super().__init__(query, **kwargs)


class LLaVAQueryResponse(QueryResponse):
    """LLaVA-specific query response."""
    pass


class LLaVARAGSystem:
    """LLaVA RAG system stub - not implemented."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        logger.warning("LLaVA system is not implemented in this version")
        
    def is_available(self) -> bool:
        return False
    
    def query(self, query: Union[str, QueryRequest]) -> QueryResponse:
        return QueryResponse(
            answer="LLaVA system not implemented in this version",
            sources=[],
            query=str(query),
            confidence=0.0
        )