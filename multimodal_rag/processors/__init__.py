"""
Processor package initialization.
"""

from .document_processor import (
    TextProcessor,
    PDFProcessor, 
    DOCXProcessor,
    DocumentProcessorManager
)

from .image_processor import (
    ImageProcessor,
    ImageProcessorManager
)

from .audio_processor import (
    AudioProcessor,
    AudioProcessorManager
)

__all__ = [
    'TextProcessor',
    'PDFProcessor',
    'DOCXProcessor', 
    'DocumentProcessorManager',
    'ImageProcessor',
    'ImageProcessorManager',
    'AudioProcessor',
    'AudioProcessorManager'
]