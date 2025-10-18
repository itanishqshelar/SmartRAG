"""
FAISS vector store implementation.
"""

import logging
import pickle
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from ..base import BaseVectorStore, DocumentChunk, RetrievalResult

logger = logging.getLogger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """FAISS implementation of vector store."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if faiss is None or np is None:
            raise ImportError("FAISS or NumPy not available. Install with: pip install faiss-cpu numpy")
        
        if SentenceTransformer is None:
            raise ImportError("SentenceTransformers not available. Install with: pip install sentence-transformers")
        
        self.persist_directory = Path(config.get('persist_directory', './vector_db'))
        # Configuration
        self.embedding_model_name = config.get('embedding_model', 'nomic-embed-text')
        self.embedding_dimension = config.get('embedding_dimension', 384)
        
        # Initialize components
        self.embedding_model = None
        self.index = None
        self.document_store = {}  # Store actual documents with metadata
        self.id_to_index_map = {}  # Map chunk IDs to FAISS indices
        self.index_to_id_map = {}  # Map FAISS indices to chunk IDs
        
        # Create persist directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self._initialize_embedding_model()
        self._initialize_or_load_index()
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model."""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Initialized embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise
    
    def _initialize_or_load_index(self):
        """Initialize or load existing FAISS index."""
        index_path = self.persist_directory / f"{self.collection_name}.index"
        metadata_path = self.persist_directory / f"{self.collection_name}.metadata"
        
        if index_path.exists() and metadata_path.exists():
            self._load_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        try:
            # Create FAISS index (using Inner Product for cosine similarity)
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            
            # Initialize empty stores
            self.document_store = {}
            self.id_to_index_map = {}
            self.index_to_id_map = {}
            
            logger.info(f"Created new FAISS index with dimension {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {str(e)}")
            raise
    
    def _load_index(self):
        """Load existing FAISS index and metadata."""
        try:
            index_path = self.persist_directory / f"{self.collection_name}.index"
            metadata_path = self.persist_directory / f"{self.collection_name}.metadata"
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.document_store = metadata['document_store']
                self.id_to_index_map = metadata['id_to_index_map']
                self.index_to_id_map = metadata['index_to_id_map']
            
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {str(e)}")
            # Fallback to creating new index
            self._create_new_index()
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            index_path = self.persist_directory / f"{self.collection_name}.index"
            metadata_path = self.persist_directory / f"{self.collection_name}.metadata"
            
            # Save FAISS index
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata = {
                'document_store': self.document_store,
                'id_to_index_map': self.id_to_index_map,
                'index_to_id_map': self.index_to_id_map
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.debug("Saved FAISS index and metadata to disk")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {str(e)}")
    
    def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to the vector store."""
        try:
            if not chunks:
                return True
            
            # Generate embeddings
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            start_index = self.index.ntotal
            self.index.add(embeddings)
            
            # Store documents and update mappings
            for i, chunk in enumerate(chunks):
                chunk_id = chunk.chunk_id or str(uuid.uuid4())
                faiss_index = start_index + i
                
                # Store document
                self.document_store[chunk_id] = {
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'document_type': chunk.document_type,
                    'source_file': chunk.source_file,
                    'timestamp': chunk.timestamp.isoformat() if chunk.timestamp else None
                }
                
                # Update mappings
                self.id_to_index_map[chunk_id] = faiss_index
                self.index_to_id_map[faiss_index] = chunk_id
            
            # Save to disk
            self._save_index()
            
            logger.info(f"Added {len(chunks)} chunks to FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to FAISS: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 5, 
                         filter_dict: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        """Perform similarity search and return results."""
        start_time = time.time()
        
        try:
            if self.index.ntotal == 0:
                return RetrievalResult(
                    chunks=[],
                    scores=[],
                    query=query,
                    total_results=0,
                    retrieval_time=time.time() - start_time
                )
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Perform search
            search_k = min(k * 2, self.index.ntotal)  # Get more results for filtering
            scores, indices = self.index.search(query_embedding, search_k)
            
            # Convert results to DocumentChunks
            chunks = []
            final_scores = []
            
            for score, index in zip(scores[0], indices[0]):
                if index == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                chunk_id = self.index_to_id_map.get(index)
                if not chunk_id or chunk_id not in self.document_store:
                    continue
                
                doc_data = self.document_store[chunk_id]
                
                # Apply filters if specified
                if filter_dict and not self._matches_filter(doc_data, filter_dict):
                    continue
                
                # Create DocumentChunk
                chunk = DocumentChunk(
                    content=doc_data['content'],
                    metadata=doc_data['metadata'],
                    document_type=doc_data['document_type'],
                    chunk_id=chunk_id,
                    source_file=doc_data['source_file']
                )
                
                chunks.append(chunk)
                final_scores.append(float(score))
                
                if len(chunks) >= k:
                    break
            
            retrieval_time = time.time() - start_time
            
            return RetrievalResult(
                chunks=chunks,
                scores=final_scores,
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
            # Note: FAISS doesn't support deletion, so we mark as deleted
            # and rebuild index if too many deletions accumulate
            deleted_count = 0
            
            for chunk_id in chunk_ids:
                if chunk_id in self.document_store:
                    del self.document_store[chunk_id]
                    
                    if chunk_id in self.id_to_index_map:
                        faiss_index = self.id_to_index_map[chunk_id]
                        del self.id_to_index_map[chunk_id]
                        del self.index_to_id_map[faiss_index]
                    
                    deleted_count += 1
            
            if deleted_count > 0:
                self._save_index()
                logger.info(f"Marked {deleted_count} documents as deleted")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            return {
                'collection_name': self.collection_name,
                'document_count': len(self.document_store),
                'faiss_index_size': self.index.ntotal,
                'embedding_dimension': self.embedding_dimension,
                'embedding_model': self.embedding_model_name,
                'persist_directory': str(self.persist_directory)
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {}
    
    def _matches_filter(self, doc_data: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if document matches filter criteria."""
        for key, value in filter_dict.items():
            if key in doc_data:
                doc_value = doc_data[key]
                if isinstance(value, list):
                    if doc_value not in value:
                        return False
                elif doc_value != value:
                    return False
            elif key in doc_data.get('metadata', {}):
                doc_value = doc_data['metadata'][key]
                if isinstance(value, list):
                    if doc_value not in value:
                        return False
                elif str(doc_value) != str(value):
                    return False
            else:
                return False
        
        return True
    
    def rebuild_index(self) -> bool:
        """Rebuild FAISS index to remove deleted documents."""
        try:
            if not self.document_store:
                self._create_new_index()
                return True
            
            # Collect all valid documents
            chunks = []
            for chunk_id, doc_data in self.document_store.items():
                chunk = DocumentChunk(
                    content=doc_data['content'],
                    metadata=doc_data['metadata'],
                    document_type=doc_data['document_type'],
                    chunk_id=chunk_id,
                    source_file=doc_data['source_file']
                )
                chunks.append(chunk)
            
            # Create new index
            self._create_new_index()
            
            # Re-add all documents
            return self.add_documents(chunks)
            
        except Exception as e:
            logger.error(f"Failed to rebuild index: {str(e)}")
            return False