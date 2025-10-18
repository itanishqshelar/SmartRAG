"""
Main multimodal RAG system implementation.
Simple traditional s            logger.info(f"Ingesting file: {file_path}")
            
            # Determine processor based on file type
            if self.document_processor and self.document_processor.can_process(file_path):
                result = self.document_processor.process_file(file_path)
            elif self.image_processor and self.image_processor.can_process(file_path):
                result = self.image_processor.process_file(file_path)
            elif self.audio_processor and self.audio_processor.can_process(file_path):
                result = self.audio_processor.process_file(file_path)
            else:
                return ProcessingResult(
                    chunks=[],
                    success=False,
                    error_message=f"Unsupported file type: {file_path.suffix}"
                )ma + basic processors.
"""

import logging
import time
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from .base import (
    QueryRequest, QueryResponse, DocumentChunk, ProcessingResult, OllamaLLM
)

logger = logging.getLogger(__name__)


class SimpleRAGSystem:
    """Simple RAG system using available components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("Initializing Simple RAG System...")
        
        try:
            # Initialize LLM
            self.llm = OllamaLLM(config)
            logger.info("LLM initialized")
            
            # Initialize processors
            from .processors import DocumentProcessorManager, ImageProcessorManager, AudioProcessorManager
            self.document_processor = DocumentProcessorManager(config)
            self.image_processor = ImageProcessorManager(config)
            self.audio_processor = AudioProcessorManager(config)
            logger.info("Processors initialized")
            
            # Initialize vector store
            from .vector_stores.chroma_store import ChromaVectorStore
            self.vector_store = ChromaVectorStore(config)
            logger.info("Vector store initialized")
            
            logger.info("Simple RAG System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Simple RAG System: {e}")
            self.llm = None
            self.document_processor = None
            self.image_processor = None
            self.audio_processor = None
            self.vector_store = None
    
    def is_available(self) -> bool:
        """Check if the system is available."""
        return (self.llm is not None and 
                hasattr(self.llm, 'is_available') and 
                self.llm.is_available())
    
    def ingest_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Ingest a file into the system."""
        if not self.is_available():
            return ProcessingResult(
                chunks=[],
                success=False,
                error_message="System not available"
            )
        
        try:
            file_path = Path(file_path)
            logger.info(f"Ingesting file: {file_path}")
            
            # Determine processor based on file type
            if self.document_processor and self.document_processor.can_process(file_path):
                result = self.document_processor.process_file(file_path)
            elif self.image_processor and self.image_processor.can_process(file_path):
                result = self.image_processor.extract_content(file_path)
            elif self.audio_processor and self.audio_processor.can_process(file_path):
                result = self.audio_processor.extract_content(file_path)
            else:
                return ProcessingResult(
                    chunks=[],
                    success=False,
                    error_message=f"Unsupported file type: {file_path.suffix}"
                )
            
            # Store chunks in vector database
            if result.success and result.chunks and self.vector_store:
                self.vector_store.add_documents(result.chunks)
                logger.info(f"Added {len(result.chunks)} chunks to vector store")
            
            return result
            
        except Exception as e:
            logger.error(f"Error ingesting file {file_path}: {e}")
            return ProcessingResult(
                chunks=[],
                success=False,
                error_message=str(e)
            )
    
    def query(self, query: Union[str, QueryRequest]) -> QueryResponse:
        """Process a query and return response."""
        if not self.is_available():
            return QueryResponse(
                answer="System not available",
                sources=[],
                query=query if isinstance(query, str) else query.query,
                confidence=0.0
            )
        
        try:
            if isinstance(query, str):
                query_text = query
                query_obj = QueryRequest(query)
            else:
                query_text = query.query
                query_obj = query
            
            logger.info(f"Processing query: {query_text}")
            
            # Check if this is a conversational query
            is_conversational = self._is_conversational_query(query_text)
            
            context = ""
            sources = []
            
            # Only retrieve documents for non-conversational queries
            if not is_conversational and self.vector_store:
                retrieval_result = self.vector_store.similarity_search(
                    query_text,
                    k=self.config.get('retrieval', {}).get('top_k', 5)
                )
                
                # Filter to only truly relevant chunks
                relevant_chunks = self._filter_relevant_context(retrieval_result.chunks, query_text)
                
                # Generate context from relevant chunks
                if relevant_chunks:
                    context = self._build_context(relevant_chunks)
                    sources = relevant_chunks
                
                logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks for query")
            else:
                logger.info(f"Treating as conversational query: {query_text}")
            
            # Generate response using LLM with appropriate prompting
            response = self._generate_contextual_response(query_text, context, is_conversational, **query_obj.generation_params)
            
            return QueryResponse(
                answer=response,
                sources=sources,
                query=query_text,
                confidence_score=0.8 if sources else 0.9  # Higher confidence for conversational
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return QueryResponse(
                answer=f"Error processing query: {str(e)}",
                sources=[],
                query=query_text if 'query_text' in locals() else str(query),
                confidence_score=0.0
            )
    
    def _is_conversational_query(self, query_text: str) -> bool:
        """Determine if a query is conversational vs document-based."""
        conversational_patterns = [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'what is your name', 'who are you', 'thanks', 'thank you',
            'bye', 'goodbye', 'see you', 'nice to meet you', 'how do you do'
        ]
        
        query_lower = query_text.lower().strip()
        
        # Check for exact matches or if query starts with conversational patterns
        for pattern in conversational_patterns:
            if query_lower == pattern or query_lower.startswith(pattern):
                return True
        
        # Check if query is very short and likely conversational
        if len(query_lower) <= 10 and not any(c in query_lower for c in ['?', 'what', 'how', 'when', 'where', 'why']):
            return True
            
        return False
    
    def _filter_relevant_context(self, chunks: List[DocumentChunk], query_text: str, min_score: float = 0.1) -> List[DocumentChunk]:
        """Filter chunks to only include truly relevant ones."""
        if not chunks:
            return []
        
        # For now, use a simple filtering approach
        # In a real implementation, you'd use semantic similarity scores
        query_lower = query_text.lower()
        relevant_chunks = []
        
        for chunk in chunks[:3]:  # Limit to top 3 most relevant chunks
            # Simple relevance check - in production, use actual similarity scores
            chunk_lower = chunk.content.lower()
            if len(chunk_lower) > 50:  # Skip very short chunks
                relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def _build_context(self, chunks: List[DocumentChunk]) -> str:
        """Build context string from retrieved chunks with source attribution."""
        context_parts = []
        for i, chunk in enumerate(chunks):
            # Extract source information
            source_file = chunk.source_file or chunk.metadata.get('filename', 'Unknown Document')
            if source_file and source_file != 'Unknown Document':
                # Get just the filename without path
                source_name = source_file.split('/')[-1].split('\\')[-1]
            else:
                source_name = chunk.metadata.get('filename', 'Unknown Document')
            
            # Include page number if available
            page_info = ""
            if chunk.page_number:
                page_info = f" (Page {chunk.page_number})"
            elif 'page_number' in chunk.metadata:
                page_info = f" (Page {chunk.metadata['page_number']})"
            
            # Build context entry with source attribution
            context_parts.append(f"[{i+1}] From '{source_name}'{page_info}:\n{chunk.content}")
        return "\n\n".join(context_parts)
    
    def _generate_contextual_response(self, query_text: str, context: str, is_conversational: bool, **kwargs) -> str:
        """Generate response with appropriate prompting based on query type."""
        if is_conversational:
            # For conversational queries, use simple, friendly prompting
            system_prompt = "You are a helpful AI assistant. Respond naturally and conversationally to greetings and casual interactions. Keep responses brief and friendly."
            full_prompt = f"{system_prompt}\n\nUser: {query_text}\nAssistant:"
        else:
            # For document queries, use RAG-style prompting
            if context and context.strip():
                full_prompt = f"""You are a helpful AI assistant. Use the provided context to answer the user's question. Always mention the specific document name(s) where you found the information. If the context doesn't contain relevant information, say so and provide what general help you can.

Context from documents:
{context}

Question: {query_text}

Answer (include specific document names in your response):"""
            else:
                full_prompt = f"""You are a helpful AI assistant. The user asked a question but no relevant documents were found. Provide a helpful general response.

Question: {query_text}

Answer:"""
        
        # Generate response using LLM
        response = self.llm.generate_response(
            prompt=full_prompt,
            context="",  # Context already included in prompt
            **kwargs
        )
        
        return response
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            return {
                'system_type': 'simple_traditional',
                'llm_available': self.is_available(),
                'processors_available': {
                    'documents': self.document_processor is not None,
                    'images': self.image_processor is not None,
                    'audio': self.audio_processor is not None
                },
                'vector_store_available': self.vector_store is not None,
                'model_name': self.config.get('models', {}).get('llm_model', 'unknown')
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'system_type': 'simple_traditional',
                'error': str(e),
                'llm_available': False
            }


class MultimodalRAGSystem:
    """Main multimodal RAG system - simplified version."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize the multimodal RAG system."""
        
        # Load configuration
        if config_dict:
            config = config_dict
        elif config_path:
            config = self._load_config(config_path)
        else:
            config = self._get_default_config()
        
        # Try enhanced system first, fallback to simple system
        logger.info("Initializing multimodal RAG system...")
        try:
            from .enhanced_system import EnhancedMultimodalRAGSystem
            self._system = EnhancedMultimodalRAGSystem(config)
            if self._system.is_available():
                self.system_type = "traditional"
                logger.info("Enhanced Multimodal RAG System initialized successfully")
            else:
                logger.warning("Enhanced system initialized but LLM not available")
                self._system = None
                self.system_type = "none"
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced system: {e}")
            try:
                # Fallback to simple system
                self._system = SimpleRAGSystem(config)
                if self._system.is_available():
                    self.system_type = "traditional"
                    logger.info("Simple RAG System initialized as fallback")
                else:
                    logger.error("Simple RAG System not available - LLM connection failed")
                    self._system = None
                    self.system_type = "none"
            except Exception as e2:
                logger.error(f"Failed to initialize fallback system: {e2}")
                self._system = None
                self.system_type = "none"
    
    def _load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration (LLaVA-first)."""
        return {
            'system': {
                'name': 'SmartRAG LLaVA System',
                'offline_mode': True,
                'debug': False
            },
            'models': {
                'llm_type': 'llava',
                'llava_model': 'llava-hf/llava-1.5-7b-hf',
                'embedding_model': 'nomic-embed-text'
            },
            'vector_store': {
                'type': 'chromadb',
                'persist_directory': './vector_db',
                'collection_name': 'llava_documents',
                'embedding_dimension': 384
            },
            'processing': {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'max_image_size': [512, 512],
                'ocr_enabled': False,  # LLaVA processes images natively
                'batch_size': 16
            },
            'retrieval': {
                'top_k': 5,
                'similarity_threshold': 0.7
            },
            'generation': {
                'max_tokens': 2048,
                'temperature': 0.7
            }
        }
    
    def _initialize_traditional_system(self, config: Dict[str, Any]):
        """Initialize traditional system as fallback."""
        # Implementation for traditional system fallback
        self.config = config
        self._system = None  # Placeholder for traditional system
        logger.warning("Traditional system fallback not fully implemented")
    
    # Delegate methods to the underlying system
    def ingest_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Ingest file using the active system."""
        if self._system:
            return self._system.ingest_file(file_path)
        else:
            logger.error("No active system available")
            return ProcessingResult(chunks=[], success=False, error_message="No active system")
    
    def is_available(self) -> bool:
        """Check if the system is available and ready."""
        return self._system is not None and self._system.is_available()
    
    def query(self, query: Union[str, QueryRequest]) -> QueryResponse:
        """Query using the active system."""
        if not self._system:
            logger.error("No active system available")
            return QueryResponse(
                answer="No active system available. Please check Ollama is running and llama3.1:8b model is available.",
                sources=[],
                query=str(query),
                confidence=0.0
            )
        
        # Use the simple system query method
        return self._system.query(query)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        if self._system and hasattr(self._system, 'get_system_status'):
            stats = self._system.get_system_status()
            stats['wrapper_system_type'] = self.system_type
            return stats
        else:
            return {
                'wrapper_system_type': self.system_type,
                'active_system': None,
                'error': 'No active system'
            }


# Backward compatibility aliases
MultimodalRAG = MultimodalRAGSystem