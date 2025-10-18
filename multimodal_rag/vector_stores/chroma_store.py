"""
ChromaDB vector store implementation.
"""

import logging
import time
import uuid
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
    import ollama
    import requests
except ImportError:
    chromadb = None
    ollama = None

from ..base import BaseVectorStore, DocumentChunk, RetrievalResult

logger = logging.getLogger(__name__)


class OllamaEmbeddingFunction:
    """Custom embedding function for Ollama embeddings."""
    
    def __init__(self, model_name: str = "nomic-embed-text", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        if ollama is None:
            raise ImportError("Ollama not available. Install with: pip install ollama")
            
        # Check if Ollama server is running
        self._check_ollama_connection()
        
        # Ensure the embedding model is pulled
        self._ensure_model_available()
    
    def name(self) -> str:
        """Return the name of this embedding function (required by ChromaDB)."""
        return f"ollama-{self.model_name}"
    
    def _check_ollama_connection(self):
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"Ollama server is running at {self.host}")
            else:
                raise ConnectionError(f"Ollama server returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama server at {self.host}. Make sure Ollama is running. Error: {e}")
    
    def _ensure_model_available(self):
        """Ensure the embedding model is available in Ollama."""
        try:
            # Check if model exists
            models_response = ollama.list()
            # Extract model names from the response objects
            model_names = [model.model for model in models_response.models]
            
            # Check for exact match or with :latest tag
            model_found = False
            actual_model_name = self.model_name
            
            if self.model_name in model_names:
                model_found = True
                actual_model_name = self.model_name
            elif f"{self.model_name}:latest" in model_names:
                model_found = True
                actual_model_name = f"{self.model_name}:latest"
            elif any(self.model_name in name for name in model_names):
                # Find the actual model name
                matching_models = [name for name in model_names if self.model_name in name]
                actual_model_name = matching_models[0]
                model_found = True
            
            if not model_found:
                logger.info(f"Model {self.model_name} not found. Pulling model...")
                ollama.pull(self.model_name)
                logger.info(f"Successfully pulled model {self.model_name}")
                actual_model_name = self.model_name
            else:
                logger.info(f"Model {actual_model_name} is available")
            
            # Update the model name to use the actual available name
            self.model_name = actual_model_name
                
        except Exception as e:
            logger.error(f"Error checking/pulling model {self.model_name}: {e}")
            raise
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for input texts (ChromaDB interface)."""
        try:
            embeddings = []
            
            for text in input:
                # Generate embedding for each text
                response = ollama.embeddings(
                    model=self.model_name,
                    prompt=text
                )
                
                embeddings.append(response['embedding'])
            
            logger.debug(f"Generated embeddings for {len(input)} texts using Ollama")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating Ollama embeddings: {str(e)}")
            raise
    
    def embed_query(self, *args, **kwargs) -> List[float]:
        """Generate embedding for a single query (ChromaDB interface)."""
        try:
            # Debug what ChromaDB is calling us with
            logger.debug(f"embed_query called with args: {args}, kwargs: {kwargs}")
            
            # Extract query from arguments
            if args:
                query = args[0]
            elif 'query' in kwargs:
                query = kwargs['query']
            elif 'input' in kwargs:
                query = kwargs['input']
            else:
                raise ValueError("No query provided to embed_query")
            
            # Handle both single string and list inputs for compatibility
            if isinstance(query, list):
                query = query[0] if query else ""
            
            response = ollama.embeddings(
                model=self.model_name,
                prompt=str(query)
            )
            embedding = response['embedding']
            logger.debug(f"Generated embedding type: {type(embedding)}, length: {len(embedding) if hasattr(embedding, '__len__') else 'N/A'}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating Ollama query embedding: {str(e)}")
            raise


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB implementation of vector store."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if chromadb is None:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        if ollama is None:
            raise ImportError("Ollama not available. Install with: pip install ollama")
        
        self.persist_directory = config.get('persist_directory', './vector_db')
        self.embedding_function_name = config.get('embedding_model', 'nomic-embed-text')
        
        # Initialize ChromaDB client
        self.client = None
        self.collection = None
        self.embedding_function = None
        
        self._initialize_client()
        self._initialize_collection()
    
    def _initialize_client(self):
        """Initialize ChromaDB client."""
        try:
            settings = Settings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False
            )
            
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=settings
            )
            
            logger.info(f"Initialized ChromaDB client with persist directory: {self.persist_directory}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise
    
    def _initialize_collection(self):
        """Initialize or get collection."""
        try:
            # Always use Ollama embedding function to avoid confusion
            # Extract Ollama host from config if available
            ollama_host = self.config.get('ollama_host', 'http://localhost:11434')
            self.embedding_function = OllamaEmbeddingFunction(
                model_name=self.embedding_function_name,
                host=ollama_host
            )
            logger.info(f"Using Ollama embedding function: {self.embedding_function_name}")
            logger.info("Note: System configured to use ONLY Ollama embeddings to prevent embedding confusion")
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Retrieved existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize collection: {str(e)}")
            raise
    
    def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to the vector store."""
        try:
            if not chunks:
                return True
            
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for chunk in chunks:
                # Use existing chunk_id or generate new one
                chunk_id = chunk.chunk_id or str(uuid.uuid4())
                ids.append(chunk_id)
                documents.append(chunk.content)
                
                # Prepare metadata (ChromaDB requires dict with string values)
                metadata = self._prepare_metadata(chunk.metadata)
                metadata.update({
                    'document_type': chunk.document_type,
                    'source_file': chunk.source_file or 'unknown',
                    'timestamp': chunk.timestamp.isoformat() if chunk.timestamp else None
                })
                
                # Remove None values
                metadata = {k: v for k, v in metadata.items() if v is not None}
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(chunks)} chunks to ChromaDB collection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 5, 
                         filter_dict: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        """Perform similarity search and return results."""
        start_time = time.time()
        
        try:
            # Prepare where clause for filtering
            where_clause = None
            if filter_dict:
                where_clause = self._prepare_where_clause(filter_dict)
            
            # Generate query embedding manually to avoid ChromaDB interface issues
            try:
                query_embedding = self.embedding_function.embed_query(query)
                logger.debug(f"Generated query embedding length: {len(query_embedding)}")
                
                # Perform query with precomputed embedding
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    where=where_clause,
                    include=['documents', 'metadatas', 'distances']
                )
            except Exception as e:
                logger.error(f"Error with manual embedding, trying query_texts: {str(e)}")
                # Fallback to query_texts if manual embedding fails
                results = self.collection.query(
                    query_texts=[query],
                    n_results=k,
                    where=where_clause,
                    include=['documents', 'metadatas', 'distances']
                )
            
            # Convert results to DocumentChunks
            chunks = []
            scores = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance
                    scores.append(similarity_score)
                    
                    # Create DocumentChunk
                    chunk = DocumentChunk(
                        content=doc,
                        metadata=metadata,
                        document_type=metadata.get('document_type', 'unknown'),
                        chunk_id=results['ids'][0][i] if results['ids'] else str(uuid.uuid4()),
                        source_file=metadata.get('source_file')
                    )
                    chunks.append(chunk)
            
            retrieval_time = time.time() - start_time
            
            return RetrievalResult(
                chunks=chunks,
                scores=scores,
                query=query,
                total_results=len(chunks),
                retrieval_time=retrieval_time
            )
            
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            return RetrievalResult(
                chunks=[],
                scores=[],
                query=query,
                total_results=0,
                retrieval_time=time.time() - start_time
            )
    
    def delete_documents(self, chunk_ids: List[str]) -> bool:
        """Delete documents by chunk IDs."""
        try:
            if not chunk_ids:
                return True
            
            self.collection.delete(ids=chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} documents from ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'embedding_function': self.embedding_function_name,
                'persist_directory': self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {}
    
    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB (convert to string values)."""
        prepared = {}
        
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                prepared[key] = str(value)
            elif isinstance(value, (list, dict)):
                prepared[key] = str(value)
            else:
                prepared[key] = str(value)
        
        return prepared
    
    def _prepare_where_clause(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare where clause for ChromaDB filtering."""
        where_clause = {}
        
        for key, value in filter_dict.items():
            if isinstance(value, str):
                where_clause[key] = value
            elif isinstance(value, list):
                where_clause[key] = {"$in": value}
            else:
                where_clause[key] = str(value)
        
        return where_clause
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            # Get all document IDs
            results = self.collection.get()
            if results['ids']:
                self.collection.delete(ids=results['ids'])
            
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            return False