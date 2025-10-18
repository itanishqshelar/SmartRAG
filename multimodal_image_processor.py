"""""""""

Multimodal image processor for enhanced visual processing.

"""Multimodal image processor for enhanced visual processing.Enhanced multimodal image processor with visual embeddings support.



import logging"""Supports both text-based and visual similarity search.

from pathlib import Path

from typing import Dict, Any, List, Optional, Union"""



try:import logging

    from PIL import Image

    PIL_AVAILABLE = Truefrom pathlib import Pathimport logging

except ImportError:

    Image = Nonefrom typing import Dict, Any, List, Optional, Unionimport time

    PIL_AVAILABLE = False

from PIL import Imageimport base64

from ..base import BaseProcessor, DocumentChunk, ProcessingResult

import numpy as npimport io

logger = logging.getLogger(__name__)

from pathlib import Path



class MultimodalDocumentChunk(DocumentChunk):from ..base import BaseProcessor, DocumentChunk, ProcessingResultfrom typing import Union, List, Dict, Any, Optional, Tuple

    """Extended document chunk with multimodal capabilities."""

    

    def __init__(self, content: str, **kwargs):

        # Extract multimodal-specific kwargslogger = logging.getLogger(__name__)try:

        self.image_path: Optional[str] = kwargs.pop('image_path', None)

        self.image_features: Optional[List[float]] = kwargs.pop('image_features', None)    from PIL import Image

        self.visual_description: Optional[str] = kwargs.pop('visual_description', None)

            import pytesseract

        # Pass remaining kwargs to parent

        super().__init__(content, **kwargs)class MultimodalDocumentChunk(DocumentChunk):except ImportError:



    """Extended document chunk with multimodal capabilities."""    Image = None

class MultimodalImageProcessorManager:

    """Manager for multimodal image processing."""        pytesseract = None

    

    def __init__(self, config: Dict[str, Any]):    def __init__(self, content: str, **kwargs):

        self.config = config

                # Extract multimodal-specific kwargstry:

        # Try to import image processor

        try:        self.image_path: Optional[str] = kwargs.pop('image_path', None)    import cv2

            from .image_processor import ImageProcessorManager

            self.image_processor = ImageProcessorManager(config)        self.image_features: Optional[List[float]] = kwargs.pop('image_features', None)    import numpy as np

            self.available = True

            logger.info("Multimodal image processor initialized successfully")        self.visual_description: Optional[str] = kwargs.pop('visual_description', None)except ImportError:

        except Exception as e:

            logger.error(f"Failed to initialize image processor: {e}")            cv2 = None

            self.image_processor = None

            self.available = False        # Pass remaining kwargs to parent    np = None

    

    def is_available(self) -> bool:        super().__init__(content, **kwargs)

        """Check if the processor is available."""

        return self.available and self.image_processor is not Nonetry:

    

    def can_process(self, file_path: Union[str, Path]) -> bool:    from transformers import CLIPProcessor, CLIPModel

        """Check if this processor can handle the given file."""

        if not self.is_available():class MultimodalImageProcessorManager:    import torch

            return False

        return self.image_processor.can_process(file_path)    """Manager for multimodal image processing."""except ImportError:

    

    def extract_content(self, file_path: Union[str, Path]) -> ProcessingResult:        CLIPProcessor = None

        """Extract content from image file."""

        if not self.is_available():    def __init__(self, config: Dict[str, Any]):    CLIPModel = None

            return ProcessingResult(

                chunks=[],        self.config = config    torch = None

                success=False,

                error_message="Multimodal image processor not available"        

            )

                # Try to import image processortry:

        # Use the regular image processor as fallback

        result = self.image_processor.extract_content(file_path)        try:    from sentence_transformers import SentenceTransformer

        

        # Convert regular chunks to multimodal chunks            from .image_processor import ImageProcessorManagerexcept ImportError:

        if result.success:

            multimodal_chunks = []            self.image_processor = ImageProcessorManager(config)    SentenceTransformer = None

            for chunk in result.chunks:

                multimodal_chunk = MultimodalDocumentChunk(            self.available = True

                    content=chunk.content,

                    metadata=chunk.metadata,            logger.info("Multimodal image processor initialized successfully")from ..base import BaseProcessor, ProcessingResult, DocumentChunk

                    document_type=chunk.document_type,

                    source_file=chunk.source_file,        except Exception as e:

                    image_path=str(file_path)

                )            logger.error(f"Failed to initialize image processor: {e}")logger = logging.getLogger(__name__)

                multimodal_chunks.append(multimodal_chunk)

                        self.image_processor = None

            result.chunks = multimodal_chunks

                    self.available = False

        return result

        class MultimodalDocumentChunk(DocumentChunk):

    def process_visual_query(self, image_path: Union[str, Path]) -> Optional[List[float]]:

        """Process an image for visual query (placeholder)."""    def is_available(self) -> bool:    """Extended document chunk with multimodal capabilities."""

        if not self.is_available():

            return None        """Check if the processor is available."""    

        

        try:        return self.available and self.image_processor is not None    def __init__(self, visual_embedding: Optional[List[float]] = None, 

            # This is a placeholder - in a real implementation, 

            # this would use CLIP or similar model to generate image features                     image_base64: Optional[str] = None, 

            logger.info(f"Processing visual query for: {image_path}")

            return [0.0] * 512  # Placeholder feature vector    def can_process(self, file_path: Union[str, Path]) -> bool:                 has_visual_content: bool = False, 

        except Exception as e:

            logger.error(f"Error processing visual query: {e}")        """Check if this processor can handle the given file."""                 *args, **kwargs):

            return None
        if not self.is_available():        # Remove visual-specific kwargs before calling parent

            return False        kwargs.pop('visual_embedding', None)

        return self.image_processor.can_process(file_path)        kwargs.pop('image_base64', None)

            kwargs.pop('has_visual_content', None)

    def extract_content(self, file_path: Union[str, Path]) -> ProcessingResult:        

        """Extract content from image file."""        super().__init__(*args, **kwargs)

        if not self.is_available():        self.visual_embedding: Optional[List[float]] = visual_embedding

            return ProcessingResult(        self.image_base64: Optional[str] = image_base64

                chunks=[],        self.has_visual_content: bool = has_visual_content

                success=False,

                error_message="Multimodal image processor not available"

            )class MultimodalImageProcessor(BaseProcessor):

            """Enhanced image processor with visual embeddings and multimodal search."""

        # Use the regular image processor as fallback    

        result = self.image_processor.extract_content(file_path)    def __init__(self, config: Dict[str, Any]):

                super().__init__(config)

        # Convert regular chunks to multimodal chunks        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

        if result.success:        self.processor_type = "multimodal_image"

            multimodal_chunks = []        

            for chunk in result.chunks:        # Configuration

                multimodal_chunk = MultimodalDocumentChunk(        self.max_image_size = tuple(config.get('processing', {}).get('max_image_size', [1024, 1024]))

                    content=chunk.content,        self.ocr_enabled = config.get('processing', {}).get('ocr_enabled', True)

                    metadata=chunk.metadata,        self.store_image_data = config.get('processing', {}).get('store_image_data', True)

                    document_type=chunk.document_type,        self.visual_embedding_model = config.get('models', {}).get('visual_embedding_model', 'openai/clip-vit-base-patch32')

                    source_file=chunk.source_file,        

                    image_path=str(file_path)        # Check availability

                )        self.pil_available = Image is not None

                multimodal_chunks.append(multimodal_chunk)        self.tesseract_available = pytesseract is not None

                    self.cv2_available = cv2 is not None

            result.chunks = multimodal_chunks        self.clip_available = all([CLIPProcessor, CLIPModel, torch])

                

        return result        # Initialize CLIP model for visual embeddings

            self.clip_processor = None

    def process_visual_query(self, image_path: Union[str, Path]) -> Optional[List[float]]:        self.clip_model = None

        """Process an image for visual query (placeholder)."""        if self.clip_available:

        if not self.is_available():            try:

            return None                self._load_clip_model()

                    except Exception as e:

        try:                logger.warning(f"Failed to load CLIP model: {str(e)}")

            # This is a placeholder - in a real implementation,                 self.clip_available = False

            # this would use CLIP or similar model to generate image features        

            logger.info(f"Processing visual query for: {image_path}")        # Configure Tesseract if available

            return [0.0] * 512  # Placeholder feature vector        if self.tesseract_available:

        except Exception as e:            import os

            logger.error(f"Error processing visual query: {e}")            tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

            return None            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def _load_clip_model(self):
        """Load CLIP model for visual embeddings."""
        try:
            self.clip_processor = CLIPProcessor.from_pretrained(self.visual_embedding_model)
            self.clip_model = CLIPModel.from_pretrained(self.visual_embedding_model)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
            
            logger.info(f"Loaded CLIP model: {self.visual_embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            raise
    
    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if file is a supported image format."""
        path = Path(file_path)
        return (path.suffix.lower() in self.supported_extensions and 
                self.pil_available)
    
    def extract_content(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Extract multimodal content from image file."""
        start_time = time.time()
        
        try:
            path = Path(file_path)
            
            # Load and preprocess image
            image = Image.open(path)
            original_size = image.size
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large (but keep original for embeddings)
            processed_image = image.copy()
            if (image.width > self.max_image_size[0] or 
                image.height > self.max_image_size[1]):
                processed_image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
            
            # Get file metadata
            metadata = self._get_file_metadata(path)
            metadata['original_size'] = original_size
            metadata['processed_size'] = processed_image.size
            metadata['image_mode'] = image.mode
            
            # Generate visual embedding
            visual_embedding = None
            if self.clip_available:
                try:
                    visual_embedding = self._generate_visual_embedding(processed_image)
                    metadata['has_visual_embedding'] = True
                except Exception as e:
                    logger.warning(f"Visual embedding generation failed for {path}: {str(e)}")
            
            # Extract text using OCR
            ocr_text = ""
            if self.ocr_enabled and self.tesseract_available:
                try:
                    ocr_text = pytesseract.image_to_string(processed_image, config='--psm 3')
                    if ocr_text.strip():
                        metadata['ocr_confidence'] = self._get_ocr_confidence(processed_image)
                        metadata['has_ocr_text'] = True
                except Exception as e:
                    logger.warning(f"OCR failed for {path}: {str(e)}")
            
            # Generate detailed image description using CLIP
            image_description = ""
            if self.clip_available:
                try:
                    image_description = self._generate_clip_description(processed_image)
                    metadata['has_clip_description'] = True
                except Exception as e:
                    logger.warning(f"CLIP description failed for {path}: {str(e)}")
            
            # Combine all text content
            content_parts = []
            if ocr_text.strip():
                content_parts.append(f"OCR Text: {ocr_text.strip()}")
            if image_description:
                content_parts.append(f"Visual Description: {image_description}")
            
            # Add basic image metadata as searchable content
            content_parts.append(f"Image Metadata: {path.name}, {original_size[0]}x{original_size[1]} pixels")
            
            content = '\n'.join(content_parts) if content_parts else f"Image file: {path.name}"
            
            # Convert image to base64 for storage (optional)
            image_base64 = None
            if self.store_image_data:
                image_base64 = self._image_to_base64(processed_image)
            
            # Create multimodal chunk
            chunk = MultimodalDocumentChunk(
                content=content,
                metadata=metadata,
                document_type=self.processor_type,
                source_file=str(path),
                visual_embedding=visual_embedding,
                image_base64=image_base64,
                has_visual_content=True
            )
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                chunks=[chunk],
                success=True,
                processing_time=processing_time,
                metadata={
                    'chunks_created': 1,
                    'has_visual_embedding': visual_embedding is not None,
                    'has_ocr_text': bool(ocr_text.strip()),
                    'has_image_data': image_base64 is not None
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing image file {file_path}: {str(e)}")
            return ProcessingResult(
                chunks=[],
                success=False,
                error_message=f"Failed to process image: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def _generate_visual_embedding(self, image: Any) -> List[float]:
        """Generate visual embedding using CLIP."""
        try:
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            # Move inputs to same device as model
            if torch.cuda.is_available() and next(self.clip_model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # Normalize the features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            return image_features.cpu().numpy().flatten().tolist()
            
        except Exception as e:
            logger.error(f"Visual embedding generation error: {str(e)}")
            return None
    
    def _generate_clip_description(self, image: Any) -> str:
        """Generate descriptive text using CLIP with various prompts."""
        try:
            # Various descriptive prompts to get comprehensive descriptions
            prompts = [
                "a photo of",
                "an image showing",
                "a picture containing",
                "a document with",
                "a chart showing",
                "a diagram of",
                "a screenshot of",
                "text that says"
            ]
            
            inputs = self.clip_processor(
                text=prompts, 
                images=[image] * len(prompts), 
                return_tensors="pt", 
                padding=True
            )
            
            # Move inputs to same device as model
            if torch.cuda.is_available() and next(self.clip_model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get the most likely description
            best_idx = probs.argmax().item()
            confidence = probs[0][best_idx].item()
            
            if confidence > 0.3:  # Threshold for confidence
                return f"{prompts[best_idx]} content (confidence: {confidence:.2f})"
            else:
                return "visual content with mixed elements"
                
        except Exception as e:
            logger.error(f"CLIP description error: {str(e)}")
            return "image content"
    
    def _get_ocr_confidence(self, image: Any) -> Optional[float]:
        """Get OCR confidence score."""
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [float(conf) for conf in data['conf'] if int(conf) > 0]
            return sum(confidences) / len(confidences) if confidences else None
        except Exception:
            return None
    
    def _image_to_base64(self, image: Any) -> str:
        """Convert PIL Image to base64 string."""
        try:
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Image to base64 conversion error: {str(e)}")
            return None
    
    def search_similar_images(self, query_image_path: Union[str, Path], 
                            stored_chunks: List[MultimodalDocumentChunk], 
                            top_k: int = 5) -> List[Tuple[MultimodalDocumentChunk, float]]:
        """Search for visually similar images."""
        if not self.clip_available:
            logger.warning("CLIP not available for visual similarity search")
            return []
        
        try:
            # Generate embedding for query image
            query_image = Image.open(query_image_path)
            if query_image.mode != 'RGB':
                query_image = query_image.convert('RGB')
            
            query_embedding = self._generate_visual_embedding(query_image)
            if query_embedding is None:
                return []
            
            # Calculate similarities
            similarities = []
            for chunk in stored_chunks:
                if chunk.visual_embedding is not None:
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_embedding, chunk.visual_embedding)
                    similarities.append((chunk, similarity))
            
            # Sort by similarity and return top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Visual similarity search error: {str(e)}")
            return []
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            a = np.array(a)
            b = np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        except Exception:
            return 0.0


class MultimodalImageProcessorManager:
    """Manager for multimodal image processing operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processor = MultimodalImageProcessor(config)
    
    def process_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Process an image file with multimodal capabilities."""
        if not self.processor.can_process(file_path):
            return ProcessingResult(
                chunks=[],
                success=False,
                error_message=f"Cannot process file: {file_path}"
            )
        
        return self.processor.extract_content(file_path)
    
    def search_similar_images(self, query_image_path: Union[str, Path], 
                            stored_chunks: List[MultimodalDocumentChunk], 
                            top_k: int = 5) -> List[Tuple[MultimodalDocumentChunk, float]]:
        """Search for visually similar images."""
        return self.processor.search_similar_images(query_image_path, stored_chunks, top_k)
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported image extensions."""
        return self.processor.supported_extensions
    
    def is_available(self) -> bool:
        """Check if multimodal image processing is available."""
        return self.processor.pil_available and self.processor.clip_available
    
    def has_visual_embeddings(self) -> bool:
        """Check if visual embeddings are available."""
        return self.processor.clip_available