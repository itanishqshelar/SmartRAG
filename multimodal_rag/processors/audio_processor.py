"""
Audio processor for speech-to-text conversion and audio content extraction.
"""

import logging
import time
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

try:
    import whisper
except ImportError:
    whisper = None

try:
    from pydub import AudioSegment
    from pydub.utils import which
except ImportError:
    AudioSegment = None
    which = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import librosa
    import numpy as np
except ImportError:
    librosa = None
    np = None

from ..base import BaseProcessor, ProcessingResult, DocumentChunk

logger = logging.getLogger(__name__)


class AudioProcessor(BaseProcessor):
    """Processor for audio files with speech-to-text capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.supported_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac']
        self.processor_type = "audio"
        self.chunk_size = config.get('processing', {}).get('chunk_size', 1000)
        self.chunk_overlap = config.get('processing', {}).get('chunk_overlap', 200)
        
        # Configuration
        self.whisper_model_name = config.get('models', {}).get('whisper_model', 'base')
        self.max_audio_duration = config.get('processing', {}).get('max_audio_duration', 300)  # seconds
        self.sample_rate = config.get('processing', {}).get('audio_sample_rate', 16000)
        
        # Check availability
        self.whisper_available = whisper is not None
        self.pydub_available = AudioSegment is not None
        self.speech_recognition_available = sr is not None
        self.librosa_available = librosa is not None
        
        if not self.whisper_available:
            logger.warning("Whisper not available. Install with: pip install openai-whisper")
        
        if not self.pydub_available:
            logger.warning("pydub not available. Install with: pip install pydub")
        
        # Initialize Whisper model
        self.whisper_model = None
        if self.whisper_available:
            try:
                self._load_whisper_model()
            except Exception as e:
                logger.warning(f"Failed to load Whisper model: {str(e)}")
                self.whisper_available = False
        
        # Initialize speech recognition
        self.speech_recognizer = None
        if self.speech_recognition_available:
            self.speech_recognizer = sr.Recognizer()
    
    def _load_whisper_model(self):
        """Load the Whisper model for speech-to-text."""
        try:
            self.whisper_model = whisper.load_model(self.whisper_model_name)
            logger.info(f"Loaded Whisper model: {self.whisper_model_name}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise
    
    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if file is a supported audio format."""
        path = Path(file_path)
        return (path.suffix.lower() in self.supported_extensions and 
                (self.whisper_available or self.speech_recognition_available))
    
    def extract_content(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Extract content from audio file."""
        start_time = time.time()
        
        try:
            path = Path(file_path)
            
            # Get audio metadata
            metadata = self._get_file_metadata(path)
            audio_info = self._get_audio_info(path)
            metadata.update(audio_info)
            
            # Check duration limit
            if audio_info.get('duration', 0) > self.max_audio_duration:
                return ProcessingResult(
                    chunks=[],
                    success=False,
                    error_message=f"Audio too long: {audio_info.get('duration', 0)}s > {self.max_audio_duration}s"
                )
            
            # Convert audio to text
            transcript = None
            confidence = None
            
            if self.whisper_available:
                transcript, confidence = self._transcribe_with_whisper(path)
            elif self.speech_recognition_available:
                transcript, confidence = self._transcribe_with_speech_recognition(path)
            
            if not transcript or not transcript.strip():
                return ProcessingResult(
                    chunks=[],
                    success=False,
                    error_message="No speech could be detected or transcribed from audio"
                )
            
            # Add transcription metadata
            metadata['transcript_length'] = len(transcript)
            metadata['word_count'] = len(transcript.split())
            if confidence is not None:
                metadata['transcription_confidence'] = confidence
            
            # Create chunks from transcript
            chunks = self._create_chunks(transcript, metadata, self.chunk_size, self.chunk_overlap)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                chunks=chunks,
                success=True,
                processing_time=processing_time,
                metadata={'chunks_created': len(chunks)}
            )
            
        except Exception as e:
            logger.error(f"Error processing audio file {file_path}: {str(e)}")
            return ProcessingResult(
                chunks=[],
                success=False,
                error_message=f"Failed to process audio: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def _get_audio_info(self, path: Path) -> Dict[str, Any]:
        """Extract audio metadata."""
        audio_info = {}
        
        try:
            if self.pydub_available:
                audio = AudioSegment.from_file(path)
                audio_info.update({
                    'duration': len(audio) / 1000.0,  # Convert to seconds
                    'sample_rate': audio.frame_rate,
                    'channels': audio.channels,
                    'sample_width': audio.sample_width,
                    'frame_count': audio.frame_count()
                })
            elif self.librosa_available:
                # Use librosa as fallback
                duration = librosa.get_duration(filename=str(path))
                audio_info['duration'] = duration
            
        except Exception as e:
            logger.warning(f"Failed to extract audio info: {str(e)}")
        
        return audio_info
    
    def _transcribe_with_whisper(self, path: Path) -> tuple[Optional[str], Optional[float]]:
        """Transcribe audio using Whisper."""
        try:
            result = self.whisper_model.transcribe(str(path))
            
            transcript = result.get('text', '').strip()
            
            # Calculate average confidence from segments
            segments = result.get('segments', [])
            if segments:
                confidences = [seg.get('avg_logprob', 0) for seg in segments if 'avg_logprob' in seg]
                avg_confidence = sum(confidences) / len(confidences) if confidences else None
            else:
                avg_confidence = None
            
            return transcript, avg_confidence
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {str(e)}")
            return None, None
    
    def _transcribe_with_speech_recognition(self, path: Path) -> tuple[Optional[str], Optional[float]]:
        """Transcribe audio using SpeechRecognition library."""
        try:
            # Convert to WAV if necessary
            if self.pydub_available and path.suffix.lower() != '.wav':
                audio = AudioSegment.from_file(path)
                # Convert to temporary WAV
                temp_wav_path = path.with_suffix('.temp.wav')
                audio.export(temp_wav_path, format="wav")
                wav_path = temp_wav_path
            else:
                wav_path = path
            
            # Transcribe using speech_recognition
            with sr.AudioFile(str(wav_path)) as source:
                audio_data = self.speech_recognizer.record(source)
            
            try:
                # Try Google Speech Recognition (requires internet)
                transcript = self.speech_recognizer.recognize_google(audio_data)
                confidence = None  # Google API doesn't return confidence
            except sr.RequestError:
                # Fallback to offline recognition if available
                try:
                    transcript = self.speech_recognizer.recognize_sphinx(audio_data)
                    confidence = None
                except (sr.RequestError, sr.UnknownValueError):
                    transcript = None
                    confidence = None
            
            # Clean up temporary file
            if 'temp_wav_path' in locals() and temp_wav_path.exists():
                temp_wav_path.unlink()
            
            return transcript, confidence
            
        except Exception as e:
            logger.error(f"Speech recognition failed: {str(e)}")
            return None, None
    
    def _segment_long_audio(self, path: Path, segment_duration: int = 30) -> List[Path]:
        """Split long audio into smaller segments."""
        if not self.pydub_available:
            return [path]
        
        try:
            audio = AudioSegment.from_file(path)
            segment_length_ms = segment_duration * 1000
            
            segments = []
            for i, start_ms in enumerate(range(0, len(audio), segment_length_ms)):
                end_ms = min(start_ms + segment_length_ms, len(audio))
                segment = audio[start_ms:end_ms]
                
                segment_path = path.with_name(f"{path.stem}_segment_{i}{path.suffix}")
                segment.export(segment_path, format=path.suffix[1:])
                segments.append(segment_path)
            
            return segments
            
        except Exception as e:
            logger.warning(f"Audio segmentation failed: {str(e)}")
            return [path]


class AudioProcessorManager:
    """Manager for audio processing operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processor = AudioProcessor(config)
    
    def process_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Process an audio file."""
        return self.extract_content(file_path)
    
    def extract_content(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Extract content from an audio file."""
        if not self.processor.can_process(file_path):
            return ProcessingResult(
                chunks=[],
                success=False,
                error_message=f"Cannot process file: {file_path}"
            )
        
        return self.processor.extract_content(file_path)
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported audio extensions."""
        return self.processor.supported_extensions
    
    def is_available(self) -> bool:
        """Check if audio processing is available."""
        return self.processor.whisper_available or self.processor.speech_recognition_available
    
    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if the audio processor can handle this file."""
        return self.processor.can_process(file_path)