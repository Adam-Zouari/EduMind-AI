"""
Audio extraction using Whisper
"""
import os
import whisper
from pathlib import Path
from typing import Dict
import time
from core.base_extractor import BaseExtractor, ExtractionResult
from config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_LANGUAGE, FFMPEG_PATH

# Configure FFmpeg path for Whisper
if os.path.exists(FFMPEG_PATH):
    # Set the directory containing ffmpeg.exe in PATH
    ffmpeg_dir = str(Path(FFMPEG_PATH).parent)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

class AudioExtractor(BaseExtractor):
    """Extract text from audio files using Whisper"""
    
    def __init__(self, model_name: str = WHISPER_MODEL):
        super().__init__()
        self.logger.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name, device=WHISPER_DEVICE)
        self.logger.info("Whisper model loaded successfully")
    
    def extract(self, file_path: Path, **kwargs) -> ExtractionResult:
        """Extract text from audio file"""
        start_time = time.time()
        self.logger.info(f"Transcribing audio: {file_path}")
        
        try:
            # Transcribe
            result = self.model.transcribe(
                str(file_path),
                language=WHISPER_LANGUAGE,
                verbose=False
            )
            
            text = result['text']
            
            # Extract segments with timestamps
            segments = []
            for segment in result.get('segments', []):
                segments.append({
                    "start": segment['start'],
                    "end": segment['end'],
                    "text": segment['text']
                })
            
            metadata = {
                "language": result.get('language', 'unknown'),
                "duration": result.get('duration', 0),
                "num_segments": len(segments),
                "segments": segments[:10],  # Store first 10 segments
                "model": WHISPER_MODEL,
                "extractor": "whisper"
            }
            
            extraction_time = time.time() - start_time
            
            return ExtractionResult(
                text=text,
                metadata=metadata,
                format_type="audio",
                file_path=str(file_path),
                extraction_time=extraction_time,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Audio extraction failed: {e}")
            return self._create_error_result(file_path, str(e))