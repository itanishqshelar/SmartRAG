"""
Image processor: Tesseract OCR for text plus BLIP captioning for visual content.
"""

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import pytesseract
from PIL import Image

from ..base import BaseProcessor, ProcessingResult

logger = logging.getLogger(__name__)

WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


def _configure_tesseract() -> bool:
    """Point pytesseract at a Tesseract binary. Returns whether one was found."""
    if shutil.which("tesseract"):
        return True
    if WINDOWS_TESSERACT.exists():
        pytesseract.pytesseract.tesseract_cmd = str(WINDOWS_TESSERACT)
        return True
    return False


class ImageProcessor(BaseProcessor):
    supported_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    document_type = "image"

    def __init__(self, config):
        super().__init__(config)
        self.max_image_size = tuple(config.processing.max_image_size)
        self.ocr_enabled = config.processing.ocr_enabled and _configure_tesseract()
        if config.processing.ocr_enabled and not self.ocr_enabled:
            logger.warning("Tesseract not found on PATH; image OCR is disabled")
        self.vision_model_name = config.models.vision_model
        self._vision = None  # (processor, model), loaded on first use
        self._lock = threading.Lock()

    def extract_content(self, file_path: Path) -> ProcessingResult:
        start = time.time()
        image = Image.open(file_path).convert("RGB")
        original_size = image.size
        image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)

        parts = []
        if self.ocr_enabled:
            try:
                text = pytesseract.image_to_string(image).strip()
                if text:
                    parts.append(f"Text in image:\n{text}")
            except Exception as e:
                logger.warning(f"OCR failed for {file_path.name}: {e}")

        caption = self._caption(image)
        if caption:
            parts.append(f"Image description: {caption}")

        if not parts:
            return ProcessingResult(
                chunks=[], success=False, error_message="No text or description could be extracted"
            )
        meta = {"width": original_size[0], "height": original_size[1]}
        chunks = self._create_chunks("\n\n".join(parts), meta)
        return ProcessingResult(chunks=chunks, processing_time=time.time() - start)

    def _load_vision_model(self):
        with self._lock:
            if self._vision is None:
                import torch
                from transformers import BlipForConditionalGeneration, BlipProcessor

                logger.info(f"Loading vision model {self.vision_model_name}...")
                processor = BlipProcessor.from_pretrained(self.vision_model_name)
                model = BlipForConditionalGeneration.from_pretrained(self.vision_model_name)
                model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
                self._vision = (processor, model)
        return self._vision

    def _caption(self, image: Image.Image) -> Optional[str]:
        try:
            import torch

            processor, model = self._load_vision_model()
            inputs = processor(image, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=60, num_beams=5)
            caption = processor.decode(output[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.warning(f"Caption generation failed: {e}")
            return None

        words = caption.split()
        # BLIP occasionally degenerates into repeated words; such captions add noise
        if len(words) < 3 or len(set(words)) < len(words) * 0.5:
            return None
        return caption
