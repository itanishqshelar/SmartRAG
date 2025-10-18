"""
Document processors for text-based formats (PDF, DOCX, TXT, etc.).
"""

import logging
import time
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

try:
    import PyPDF2
    from PyPDF2 import PdfReader
except ImportError:
    PyPDF2 = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from ..base import BaseProcessor, ProcessingResult, DocumentChunk

logger = logging.getLogger(__name__)


class TextProcessor(BaseProcessor):
    """Processor for plain text files."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.supported_extensions = ['.txt', '.md', '.rtf']
        self.processor_type = "text"
        self.chunk_size = config.get('processing', {}).get('chunk_size', 1000)
        self.chunk_overlap = config.get('processing', {}).get('chunk_overlap', 200)
    
    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if file is a supported text format."""
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions
    
    def extract_content(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Extract content from text file."""
        start_time = time.time()
        
        try:
            path = Path(file_path)
            
            # Read text content
            with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            
            if not content.strip():
                return ProcessingResult(
                    chunks=[],
                    success=False,
                    error_message="File is empty or contains no readable text"
                )
            
            # Get file metadata
            metadata = self._get_file_metadata(path)
            metadata['word_count'] = len(content.split())
            metadata['char_count'] = len(content)
            
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
            logger.error(f"Error processing text file {file_path}: {str(e)}")
            return ProcessingResult(
                chunks=[],
                success=False,
                error_message=f"Failed to process text file: {str(e)}",
                processing_time=time.time() - start_time
            )


class PDFProcessor(BaseProcessor):
    """Processor for PDF files."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.supported_extensions = ['.pdf']
        self.processor_type = "pdf"
        self.chunk_size = config.get('processing', {}).get('chunk_size', 1000)
        self.chunk_overlap = config.get('processing', {}).get('chunk_overlap', 200)
        
        # Check for available PDF libraries
        self.use_pdfplumber = pdfplumber is not None
        self.use_pypdf2 = PyPDF2 is not None
        
        if not (self.use_pdfplumber or self.use_pypdf2):
            logger.warning("No PDF processing libraries available. Install pdfplumber or PyPDF2.")
    
    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if file is a PDF."""
        path = Path(file_path)
        return (path.suffix.lower() == '.pdf' and 
                (self.use_pdfplumber or self.use_pypdf2))
    
    def extract_content(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Extract content from PDF file."""
        start_time = time.time()
        
        try:
            path = Path(file_path)
            
            if self.use_pdfplumber:
                content, page_info = self._extract_with_pdfplumber(path)
            elif self.use_pypdf2:
                content, page_info = self._extract_with_pypdf2(path)
            else:
                return ProcessingResult(
                    chunks=[],
                    success=False,
                    error_message="No PDF processing library available"
                )
            
            if not content.strip():
                return ProcessingResult(
                    chunks=[],
                    success=False,
                    error_message="PDF contains no extractable text"
                )
            
            # Get file metadata
            metadata = self._get_file_metadata(path)
            metadata.update(page_info)
            metadata['word_count'] = len(content.split())
            metadata['char_count'] = len(content)
            
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
            logger.error(f"Error processing PDF file {file_path}: {str(e)}")
            return ProcessingResult(
                chunks=[],
                success=False,
                error_message=f"Failed to process PDF: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def _extract_with_pdfplumber(self, path: Path) -> tuple[str, Dict[str, Any]]:
        """Extract text using pdfplumber."""
        content_parts = []
        page_info = {'total_pages': 0, 'extracted_pages': 0}
        
        with pdfplumber.open(path) as pdf:
            page_info['total_pages'] = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        content_parts.append(f"\n--- Page {page_num} ---\n{page_text}")
                        page_info['extracted_pages'] += 1
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {str(e)}")
        
        return '\n'.join(content_parts), page_info
    
    def _extract_with_pypdf2(self, path: Path) -> tuple[str, Dict[str, Any]]:
        """Extract text using PyPDF2."""
        content_parts = []
        page_info = {'total_pages': 0, 'extracted_pages': 0}
        
        with open(path, 'rb') as file:
            pdf_reader = PdfReader(file)
            page_info['total_pages'] = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        content_parts.append(f"\n--- Page {page_num} ---\n{page_text}")
                        page_info['extracted_pages'] += 1
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {str(e)}")
        
        return '\n'.join(content_parts), page_info


class DOCXProcessor(BaseProcessor):
    """Processor for DOCX files."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.supported_extensions = ['.docx', '.doc']
        self.processor_type = "docx"
        self.chunk_size = config.get('processing', {}).get('chunk_size', 1000)
        self.chunk_overlap = config.get('processing', {}).get('chunk_overlap', 200)
        
        if DocxDocument is None:
            logger.warning("python-docx not available. Install with: pip install python-docx")
    
    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if file is a DOCX."""
        path = Path(file_path)
        return (path.suffix.lower() in self.supported_extensions and 
                DocxDocument is not None)
    
    def extract_content(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Extract content from DOCX file."""
        start_time = time.time()
        
        try:
            path = Path(file_path)
            
            # Only process .docx files (not .doc)
            if path.suffix.lower() == '.doc':
                return ProcessingResult(
                    chunks=[],
                    success=False,
                    error_message="Legacy .doc format not supported. Convert to .docx first."
                )
            
            doc = DocxDocument(path)
            
            # Extract paragraphs
            content_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    content_parts.append(paragraph.text)
            
            content = '\n'.join(content_parts)
            
            if not content.strip():
                return ProcessingResult(
                    chunks=[],
                    success=False,
                    error_message="DOCX contains no extractable text"
                )
            
            # Get file metadata
            metadata = self._get_file_metadata(path)
            metadata['paragraph_count'] = len(content_parts)
            metadata['word_count'] = len(content.split())
            metadata['char_count'] = len(content)
            
            # Add document properties if available
            try:
                core_props = doc.core_properties
                metadata['doc_title'] = core_props.title or 'Unknown'
                metadata['doc_author'] = core_props.author or 'Unknown'
                metadata['doc_subject'] = core_props.subject or 'Unknown'
            except Exception:
                pass
            
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
            logger.error(f"Error processing DOCX file {file_path}: {str(e)}")
            return ProcessingResult(
                chunks=[],
                success=False,
                error_message=f"Failed to process DOCX: {str(e)}",
                processing_time=time.time() - start_time
            )


class DocumentProcessorManager:
    """Manager class to handle different document types."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processors = [
            TextProcessor(config),
            PDFProcessor(config),
            DOCXProcessor(config)
        ]
    
    def get_processor(self, file_path: Union[str, Path]) -> Optional[BaseProcessor]:
        """Get the appropriate processor for a file."""
        for processor in self.processors:
            if processor.can_process(file_path):
                return processor
        return None
    
    def process_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Process a file using the appropriate processor."""
        processor = self.get_processor(file_path)
        if processor is None:
            return ProcessingResult(
                chunks=[],
                success=False,
                error_message=f"No processor available for file: {file_path}"
            )
        
        return processor.extract_content(file_path)
    
    def get_supported_extensions(self) -> List[str]:
        """Get all supported file extensions."""
        extensions = []
        for processor in self.processors:
            extensions.extend(processor.supported_extensions)
        return sorted(list(set(extensions)))
    
    def is_available(self) -> bool:
        """Check if document processors are available."""
        return len(self.processors) > 0
    
    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if any processor can handle this file."""
        return self.get_processor(file_path) is not None