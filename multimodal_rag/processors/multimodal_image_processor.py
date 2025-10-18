"""
Multimodal image processor for enhanced visual processing.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False

from ..base import BaseProcessor, DocumentChunk, ProcessingResult

logger = logging.getLogger(__name__)


class MultimodalDocumentChunk(DocumentChunk):
    """Extended document chunk with multimodal capabilities."""
    
    def __init__(self, content: str, **kwargs):
        # Extract multimodal-specific kwargs
        self.image_path: Optional[str] = kwargs.pop('image_path', None)
        self.image_features: Optional[List[float]] = kwargs.pop('image_features', None)
        self.visual_description: Optional[str] = kwargs.pop('visual_description', None)
        
        # Pass remaining kwargs to parent
        super().__init__(content, **kwargs)


class MultimodalImageProcessorManager:
    """Manager for multimodal image processing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Try to import image processor
        try:
            from .image_processor import ImageProcessorManager
            self.image_processor = ImageProcessorManager(config)
            self.available = True
            logger.info("Multimodal image processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize image processor: {e}")
            self.image_processor = None
            self.available = False
    
    def is_available(self) -> bool:
        """Check if the processor is available."""
        return self.available and self.image_processor is not None
    
    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if this processor can handle the given file."""
        if not self.is_available():
            return False
        return self.image_processor.can_process(file_path)
    
    def extract_content(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Extract content from image file."""
        if not self.is_available():
            return ProcessingResult(
                chunks=[],
                success=False,
                error_message="Multimodal image processor not available"
            )
        
        # Use the regular image processor as fallback
        result = self.image_processor.extract_content(file_path)
        
        # Convert regular chunks to multimodal chunks
        if result.success:
            multimodal_chunks = []
            for chunk in result.chunks:
                multimodal_chunk = MultimodalDocumentChunk(
                    content=chunk.content,
                    metadata=chunk.metadata,
                    document_type=chunk.document_type,
                    source_file=chunk.source_file,
                    image_path=str(file_path)
                )
                multimodal_chunks.append(multimodal_chunk)
            
            result.chunks = multimodal_chunks
        
        return result
    
    def process_visual_query(self, image_path: Union[str, Path]) -> Optional[List[float]]:
        """Process an image for visual query (placeholder)."""
        if not self.is_available():
            return None
        
        try:
            # This is a placeholder - in a real implementation, 
            # this would use CLIP or similar model to generate image features
            logger.info(f"Processing visual query for: {image_path}")
            return [0.0] * 512  # Placeholder feature vector
        except Exception as e:
            logger.error(f"Error processing visual query: {e}")
            return None
