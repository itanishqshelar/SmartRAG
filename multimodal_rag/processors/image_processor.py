"""
Image processor for visual content with OCR and vision model support.
"""

import logging
import time
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

try:
    from PIL import Image
    import pytesseract
except ImportError:
    Image = None
    pytesseract = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
except ImportError:
    BlipProcessor = None
    BlipForConditionalGeneration = None
    torch = None

from ..base import BaseProcessor, ProcessingResult, DocumentChunk

logger = logging.getLogger(__name__)


class ImageProcessor(BaseProcessor):
    """Processor for image files with OCR and vision capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        self.processor_type = "image"
        self.chunk_size = config.get('processing', {}).get('chunk_size', 1000)
        self.chunk_overlap = config.get('processing', {}).get('chunk_overlap', 200)
        
        # Configuration
        self.max_image_size = tuple(config.get('processing', {}).get('max_image_size', [1024, 1024]))
        self.ocr_enabled = config.get('processing', {}).get('ocr_enabled', True)
        self.vision_model_name = config.get('models', {}).get('vision_model', 'Salesforce/blip-image-captioning-base')
        
        # Check availability
        self.pil_available = Image is not None
        self.tesseract_available = pytesseract is not None
        self.cv2_available = cv2 is not None
        self.vision_model_available = all([BlipProcessor, BlipForConditionalGeneration, torch])
        
        if not self.pil_available:
            logger.warning("PIL not available. Install with: pip install Pillow")
        
        # Configure Tesseract path if available
        if self.tesseract_available:
            import os
            tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                logger.info(f"Configured Tesseract path: {tesseract_path}")
            else:
                logger.warning("Tesseract executable not found at expected path")
        
        if self.ocr_enabled and not self.tesseract_available:
            logger.warning("Tesseract not available. Install with: pip install pytesseract")
        
        # Initialize vision model
        self.vision_processor = None
        self.vision_model = None
        if self.vision_model_available:
            try:
                self._load_vision_model()
            except Exception as e:
                logger.warning(f"Failed to load vision model: {str(e)}")
                self.vision_model_available = False
    
    def _load_vision_model(self):
        """Load the vision model for image captioning."""
        try:
            self.vision_processor = BlipProcessor.from_pretrained(self.vision_model_name)
            self.vision_model = BlipForConditionalGeneration.from_pretrained(self.vision_model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.vision_model = self.vision_model.cuda()
            
            logger.info(f"Loaded vision model: {self.vision_model_name}")
        except Exception as e:
            logger.error(f"Failed to load vision model: {str(e)}")
            raise
    
    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if file is a supported image format."""
        path = Path(file_path)
        return (path.suffix.lower() in self.supported_extensions and 
                self.pil_available)
    
    def extract_content(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Extract content from image file."""
        start_time = time.time()
        
        try:
            path = Path(file_path)
            
            # Load and preprocess image
            image = Image.open(path)
            original_size = image.size
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large
            if (image.width > self.max_image_size[0] or 
                image.height > self.max_image_size[1]):
                image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
            
            # Get file metadata
            metadata = self._get_file_metadata(path)
            metadata['original_size'] = original_size
            metadata['processed_size'] = image.size
            metadata['image_mode'] = image.mode
            
            content_parts = []
            
            # Extract text using OCR
            if self.ocr_enabled and self.tesseract_available:
                try:
                    ocr_text = pytesseract.image_to_string(image, config='--psm 3')
                    if ocr_text.strip():
                        content_parts.append(f"OCR Text: {ocr_text.strip()}")
                        metadata['ocr_confidence'] = self._get_ocr_confidence(image)
                except Exception as e:
                    logger.warning(f"OCR failed for {path}: {str(e)}")
            
            # Generate image caption using vision model
            if self.vision_model_available:
                try:
                    caption = self._generate_caption(image)
                    if caption:
                        content_parts.append(f"Image Description: {caption}")
                        metadata['has_caption'] = True
                except Exception as e:
                    logger.warning(f"Caption generation failed for {path}: {str(e)}")
            
            # Extract EXIF data if available
            try:
                exif_data = self._extract_exif_data(Image.open(path))
                if exif_data:
                    metadata['exif_data'] = exif_data
            except Exception as e:
                logger.debug(f"EXIF extraction failed: {str(e)}")
            
            if not content_parts:
                return ProcessingResult(
                    chunks=[],
                    success=False,
                    error_message="No text or description could be extracted from image"
                )
            
            content = '\n'.join(content_parts)
            metadata['extraction_methods'] = len(content_parts)
            
            # Create chunks
            chunks = self._create_chunks(content, metadata, self.chunk_size, self.chunk_overlap)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                chunks=chunks,
                success=True,
                processing_time=processing_time,
                metadata={'chunks_created': len(chunks)}
            )
            
        except Exception as e:
            logger.error(f"Error processing image file {file_path}: {str(e)}")
            return ProcessingResult(
                chunks=[],
                success=False,
                error_message=f"Failed to process image: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def _generate_caption(self, image) -> Optional[str]:
        """Generate a caption for the image using the vision model."""
        try:
            inputs = self.vision_processor(image, return_tensors="pt")
            
            # Move inputs to same device as model
            if torch.cuda.is_available() and next(self.vision_model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                # Try different generation parameters for better captions
                outputs = self.vision_model.generate(
                    **inputs, 
                    max_length=100,  # Increased for more detailed captions
                    num_beams=8,     # More beams for better quality
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            caption = self.vision_processor.decode(outputs[0], skip_special_tokens=True)
            
            # Filter out poor quality captions
            if caption and len(caption.strip()) > 10:
                # Check for repetitive patterns
                words = caption.split()
                if len(set(words)) < len(words) * 0.5:  # Too many repeated words
                    logger.warning(f"Repetitive caption detected, using fallback: {caption}")
                    return f"Image content: Screenshot or document image (automatic caption failed)"
                return f"Visual content: {caption.strip()}"
            else:
                return f"Image content: Screenshot or document image"
            
        except Exception as e:
            logger.error(f"Caption generation error: {str(e)}")
            return f"Image content: Screenshot or document image (caption generation failed)"
    
    def _get_ocr_confidence(self, image) -> Optional[float]:
        """Get OCR confidence score."""
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [float(conf) for conf in data['conf'] if int(conf) > 0]
            return sum(confidences) / len(confidences) if confidences else None
        except Exception:
            return None
    
    def _extract_exif_data(self, image) -> Optional[Dict[str, Any]]:
        """Extract EXIF metadata from image."""
        try:
            exif_dict = {}
            if hasattr(image, '_getexif') and image._getexif():
                exif = image._getexif()
                for tag_id, value in exif.items():
                    tag = Image.ExifTags.TAGS.get(tag_id, tag_id)
                    exif_dict[tag] = value
            return exif_dict if exif_dict else None
        except Exception:
            return None


class ImageProcessorManager:
    """Manager for image processing operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processor = ImageProcessor(config)
    
    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if this processor can handle the given file."""
        return self.processor.can_process(file_path)
    
    def process_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Process an image file."""
        return self.extract_content(file_path)
    
    def extract_content(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Extract content from an image file."""
        if not self.processor.can_process(file_path):
            return ProcessingResult(
                chunks=[],
                success=False,
                error_message=f"Cannot process file: {file_path}"
            )
        
        return self.processor.extract_content(file_path)
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported image extensions."""
        return self.processor.supported_extensions
    
    def is_available(self) -> bool:
        """Check if image processing is available."""
        return self.processor.pil_available