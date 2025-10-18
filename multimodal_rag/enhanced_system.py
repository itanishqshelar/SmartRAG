"""
Enhanced system - simplified version that uses SimpleRAGSystem.
"""

import logging
from typing import Dict, Any, Union
from .base import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)


class MultimodalQueryRequest(QueryRequest):
    """Extended query request with multimodal capabilities."""
    
    def __init__(self, query: str = "", *args, **kwargs):
        self.search_type: str = kwargs.pop('search_type', 'hybrid')
        self.visual_query_image: str = kwargs.pop('visual_query_image', None)
        
        if not query and args:
            query = args[0]
            args = args[1:]
        
        super().__init__(query, *args, **kwargs)


class EnhancedMultimodalRAGSystem:
    """Enhanced multimodal RAG system - simplified version."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize the enhanced system."""
        from .system import SimpleRAGSystem
        
        self.config = config_dict
        self._simple_system = SimpleRAGSystem(config_dict)
        logger.info("Enhanced Multimodal RAG System (simplified) initialized")
    
    def is_available(self) -> bool:
        """Check if the system is available."""
        return self._simple_system.is_available()
    
    def ingest_file(self, file_path):
        """Ingest a file into the system."""
        return self._simple_system.ingest_file(file_path)
    
    def query(self, query: Union[str, QueryRequest]) -> QueryResponse:
        """Process a query and return response."""
        return self._simple_system.query(query)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        status = self._simple_system.get_system_status()
        status['system_type'] = 'enhanced_traditional_simplified'
        return status
