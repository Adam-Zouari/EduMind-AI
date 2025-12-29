"""
Video extraction using FFmpeg and Whisper
"""
import os

# CRITICAL: Set these BEFORE importing torch/whisper
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Try to import and configure torch
import sys
WHISPER_AVAILABLE = False
try:
    import torch
    torch.set_num_threads(1)
    import whisper
    WHISPER_AVAILABLE = True
except Exception as e:
    print(f"ERROR: Could not load Whisper/PyTorch: {e}")
    print("Video extraction will not be available.")
    whisper = None

import ffmpeg
from pathlib import Path
from typing import Dict, Optional
import time
import tempfile
from core.base_extractor import BaseExtractor, ExtractionResult
from config import WHISPER_MODEL, WHISPER_DEVICE, VIDEO_FRAME_RATE

class VideoExtractor(BaseExtractor):
    """Extract text from video files"""

    # Class-level cache for Whisper model (shared across instances)
    _whisper_model = None

    def __init__(self, model_name: str = WHISPER_MODEL):
        super().__init__()
        self.model = None

        # Check if Whisper is available
        if not WHISPER_AVAILABLE or whisper is None:
            self.logger.error("Whisper is not available. Video extraction will fail.")
            self.logger.error("Install PyTorch CPU version to fix:")
            self.logger.error("  pip uninstall torch torchvision torchaudio")
            self.logger.error("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
            return

        # Use cached model if available
        if VideoExtractor._whisper_model is None:
            self.logger.info(f"Loading Whisper model for video: {model_name} (CPU mode)")
            try:
                # Force CPU mode to avoid CUDA/DLL issues
                VideoExtractor._whisper_model = whisper.load_model(model_name, device="cpu")
                self.logger.info("Whisper model loaded successfully on CPU")
            except Exception as e:
                self.logger.error(f"Failed to load Whisper model: {e}")
                self.logger.info("Attempting to load with default settings...")
                try:
                    VideoExtractor._whisper_model = whisper.load_model(model_name)
                    self.logger.info("Whisper model loaded with default settings")
                except Exception as e2:
                    self.logger.error(f"Failed to load Whisper model with default settings: {e2}")
                    return
        else:
            self.logger.info(f"Using cached Whisper model for video")

        self.model = VideoExtractor._whisper_model
    
    def extract(self, file_path: Path, **kwargs) -> ExtractionResult:
        """Extract audio and transcribe from video"""
        start_time = time.time()
        self.logger.info(f"Extracting from video: {file_path}")

        # Check if model is available
        if self.model is None:
            error_msg = "Whisper model not available. Install PyTorch CPU version to enable video extraction."
            self.logger.error(error_msg)
            return self._create_error_result(file_path, error_msg)

        try:
            # Extract audio from video
            audio_path = self._extract_audio(file_path)
            
            # Transcribe audio
            result = self.model.transcribe(
                str(audio_path),
                verbose=False
            )
            
            text = result['text']
            
            # Get video metadata
            video_info = self._get_video_info(file_path)
            
            # Extract segments
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
                "segments": segments[:10],
                "video_info": video_info,
                "model": WHISPER_MODEL,
                "extractor": "video"
            }
            
            # Clean up temporary audio file
            audio_path.unlink(missing_ok=True)
            
            extraction_time = time.time() - start_time
            
            return ExtractionResult(
                text=text,
                metadata=metadata,
                format_type="video",
                file_path=str(file_path),
                extraction_time=extraction_time,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Video extraction failed: {e}")
            return self._create_error_result(file_path, str(e))
    
    def _extract_audio(self, video_path: Path) -> Path:
        """Extract audio track from video"""
        audio_path = Path(tempfile.mktemp(suffix=".wav"))
        
        try:
            (
                ffmpeg
                .input(str(video_path))
                .output(str(audio_path), acodec='pcm_s16le', ac=1, ar='16k')
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            self.logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise
        
        return audio_path
    
    def _get_video_info(self, video_path: Path) -> Dict:
        """Get video metadata"""
        try:
            probe = ffmpeg.probe(str(video_path))
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            
            if video_stream:
                return {
                    "width": video_stream.get('width'),
                    "height": video_stream.get('height'),
                    "codec": video_stream.get('codec_name'),
                    "fps": eval(video_stream.get('r_frame_rate', '0/1')),
                    "duration": float(probe['format'].get('duration', 0))
                }
        except Exception as e:
            self.logger.warning(f"Could not extract video info: {e}")
        
        return {}