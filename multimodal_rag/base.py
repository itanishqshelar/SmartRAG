"""
Base classes and interfaces for the multimodal RAG system.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of processed content from any modality."""
    
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    document_type: str = "text"
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Ensure metadata has required fields."""
        if 'chunk_index' not in self.metadata:
            self.metadata['chunk_index'] = 0
        if 'total_chunks' not in self.metadata:
            self.metadata['total_chunks'] = 1


@dataclass
class ProcessingResult:
    """Result of processing a file through any processor."""
    
    chunks: List[DocumentChunk]
    success: bool = True
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class RetrievalResult:
    """Result of a semantic search query."""
    
    chunks: List[DocumentChunk]
    scores: List[float]
    query: str
    total_results: int
    retrieval_time: Optional[float] = None


class BaseProcessor(ABC):
    """Abstract base class for all content processors."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_extensions = []
        self.processor_type = "base"
    
    @abstractmethod
    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if this processor can handle the given file."""
        pass
    
    @abstractmethod
    def extract_content(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Extract and chunk content from the file."""
        pass
    
    def _get_file_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract common file metadata."""
        path = Path(file_path)
        return {
            'filename': path.name,
            'file_extension': path.suffix.lower(),
            'file_size': path.stat().st_size,
            'created_date': datetime.fromtimestamp(path.stat().st_ctime),
            'modified_date': datetime.fromtimestamp(path.stat().st_mtime),
            'absolute_path': str(path.absolute())
        }
    
    def _create_chunks(self, content: str, metadata: Dict[str, Any], 
                      chunk_size: int = 1000, overlap: int = 200) -> List[DocumentChunk]:
        """Create overlapping chunks from content."""
        if not content.strip():
            return []
        
        chunks = []
        content_length = len(content)
        
        if content_length <= chunk_size:
            # Content fits in one chunk - extract page number if available
            page_number = self._extract_page_number_from_content(content)
            chunk = DocumentChunk(
                content=content,
                metadata={**metadata, 'chunk_index': 0, 'total_chunks': 1},
                document_type=self.processor_type,
                source_file=metadata.get('absolute_path'),
                page_number=page_number
            )
            chunks.append(chunk)
        else:
            # Split into multiple chunks
            total_chunks = (content_length - overlap) // (chunk_size - overlap) + 1
            
            for i in range(0, content_length, chunk_size - overlap):
                chunk_content = content[i:i + chunk_size]
                if not chunk_content.strip():
                    continue
                
                # Extract page number from chunk content
                page_number = self._extract_page_number_from_content(chunk_content)
                
                chunk_metadata = {
                    **metadata,
                    'chunk_index': len(chunks),
                    'total_chunks': total_chunks,
                    'start_index': i,
                    'end_index': min(i + chunk_size, content_length)
                }
                
                # Add page number to metadata if found
                if page_number:
                    chunk_metadata['page_number'] = page_number
                
                chunk = DocumentChunk(
                    content=chunk_content,
                    metadata=chunk_metadata,
                    document_type=self.processor_type,
                    source_file=metadata.get('absolute_path'),
                    page_number=page_number
                )
                chunks.append(chunk)
        
        return chunks
    
    def _extract_page_number_from_content(self, content: str) -> Optional[int]:
        """Extract page number from content that contains page markers."""
        import re
        # Look for page markers like "--- Page 2 ---"
        page_match = re.search(r'---\s*Page\s+(\d+)\s*---', content)
        if page_match:
            return int(page_match.group(1))
        return None


class BaseVectorStore(ABC):
    """Abstract base class for vector storage backends."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collection_name = config.get('collection_name', 'default')
    
    @abstractmethod
    def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to the vector store."""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 5, 
                         filter_dict: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        """Perform similarity search and return results."""
        pass
    
    @abstractmethod
    def delete_documents(self, chunk_ids: List[str]) -> bool:
        """Delete documents by chunk IDs."""
        pass
    
    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        pass


class BaseEmbedding(ABC):
    """Abstract base class for embedding models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('embedding_model', 'nomic-embed-text')
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        pass


class BaseLLM(ABC):
    """Abstract base class for language models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('llm_model', 'microsoft/DialoGPT-medium')
    
    @abstractmethod
    def generate_response(self, prompt: str, context: str = "", 
                         max_tokens: int = 512) -> str:
        """Generate a response given a prompt and context."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available and loaded."""
        pass


@dataclass
class QueryRequest:
    """Represents a user query request."""
    
    query: str
    query_type: str = "general"  # general, document, image, audio
    filters: Dict[str, Any] = field(default_factory=dict)
    top_k: int = 5
    include_metadata: bool = True
    rerank: bool = False
    generation_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResponse:
    """Represents the response to a user query."""
    
    answer: str
    sources: List[DocumentChunk]
    query: str
    confidence_score: Optional[float] = None
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class OllamaLLM(BaseLLM):
    """Ollama LLM implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            import ollama
            self.ollama = ollama
        except ImportError:
            raise ImportError("Ollama not available. Install with: pip install ollama")
        
        self.model_name = config.get('models', {}).get('llm_model', 'llama3.1:8b')
        self.host = config.get('models', {}).get('ollama_host', 'http://localhost:11434')
    
    def is_available(self) -> bool:
        """Check if the LLM is available."""
        try:
            # Try to connect to Ollama server
            logger.info(f"Checking LLM availability for model: {self.model_name}")
            response = self.ollama.list()
            # Check if our model is available
            available_models = [model.model for model in response.models]
            logger.info(f"Available models: {available_models}")
            
            # Check for exact match or partial match (handles version suffixes)
            model_found = False
            for available_model in available_models:
                if self.model_name == available_model or self.model_name in available_model or available_model in self.model_name:
                    model_found = True
                    logger.info(f"Model match found: {available_model} matches {self.model_name}")
                    break
            
            logger.info(f"Model {self.model_name} found: {model_found}")
            return model_found
        except Exception as e:
            logger.error(f"Error checking LLM availability: {str(e)}")
            return False
        
    def generate_response(self, prompt: str, context: str = "", **kwargs) -> str:
        """Generate response using Ollama."""
        try:
            # Use the prompt as-is if it already includes system instructions
            # Otherwise, add context if provided
            full_prompt = prompt
            if context and context.strip() and "Context from documents:" not in prompt:
                full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
            
            response = self.ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': full_prompt}],
                options={
                    'temperature': kwargs.get('temperature', 0.7),
                    'top_p': kwargs.get('top_p', 0.9),
                    'max_tokens': kwargs.get('max_tokens', 2048)
                }
            )
            return response['message']['content']
        except Exception as e:
            raise Exception(f"Error generating response with Ollama: {str(e)}")
    
    def generate_streaming_response(self, prompt: str, **kwargs):
        """Generate streaming response using Ollama."""
        try:
            stream = self.ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,
                options={
                    'temperature': kwargs.get('temperature', 0.7),
                    'top_p': kwargs.get('top_p', 0.9),
                    'max_tokens': kwargs.get('max_tokens', 2048)
                }
            )
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
        except Exception as e:
            raise Exception(f"Error generating streaming response with Ollama: {str(e)}")