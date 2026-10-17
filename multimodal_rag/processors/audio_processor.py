"""
Audio processor: speech-to-text with OpenAI Whisper (runs locally, needs ffmpeg).
"""

import logging
import threading
import time
from pathlib import Path

import whisper

from ..base import BaseProcessor, ProcessingResult

logger = logging.getLogger(__name__)


class AudioProcessor(BaseProcessor):
    supported_extensions = [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".webm"]
    document_type = "audio"

    def __init__(self, config):
        super().__init__(config)
        self.model_name = config.models.whisper_model
        self.max_duration = config.processing.max_audio_duration
        self._model = None  # loaded on first use
        self._lock = threading.Lock()

    def _load_model(self):
        with self._lock:
            if self._model is None:
                logger.info(f"Loading Whisper model '{self.model_name}'...")
                self._model = whisper.load_model(self.model_name)
        return self._model

    def transcribe(self, file_path: Path) -> dict:
        """Return Whisper's raw result: {'text': ..., 'segments': [...], 'language': ...}."""
        audio = whisper.load_audio(str(file_path))
        duration = len(audio) / whisper.audio.SAMPLE_RATE
        if duration > self.max_duration:
            raise ValueError(f"Audio is {duration:.0f}s long; the limit is {self.max_duration}s")
        result = self._load_model().transcribe(audio, fp16=False)
        result["duration"] = duration
        return result

    def extract_content(self, file_path: Path) -> ProcessingResult:
        start = time.time()
        result = self.transcribe(file_path)
        transcript = result.get("text", "").strip()
        if not transcript:
            return ProcessingResult(chunks=[], success=False, error_message="No speech detected")
        meta = {"duration": round(result["duration"], 1), "language": result.get("language", "")}
        chunks = self._create_chunks(transcript, meta)
        return ProcessingResult(chunks=chunks, processing_time=time.time() - start)
