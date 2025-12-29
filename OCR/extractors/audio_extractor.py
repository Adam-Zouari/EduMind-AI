"""
Audio extraction using Whisper
"""
import os

# CRITICAL: Set these BEFORE importing torch/whisper
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Try to import and configure torch
import sys
import os
from pathlib import Path

# FIX for [WinError 127] on Windows: Add torch/lib to PATH before importing torch
try:
    # User-provided hardcoded path to torch/lib
    torch_lib_path = r"C:\Users\ademz\Desktop\9raya\MLOps\Project\venv_ocr\Lib\site-packages\torch\lib"
    
    print(f"DEBUG: Checking for torch DLLs in: {torch_lib_path}")
    if os.path.exists(torch_lib_path):
        if os.path.isdir(torch_lib_path):
            files = os.listdir(torch_lib_path)
            print(f"DEBUG: Found {len(files)} files in torch lib dir. First 5: {files[:5]}")
            
            # 1. Add to PATH (Legacy/General fix)
            os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ.get('PATH', '')
            
            # 2. Add via os.add_dll_directory (Python 3.8+ specific fix)
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(torch_lib_path)
                    print(f"DEBUG: Successfully called os.add_dll_directory('{torch_lib_path}')")
                except Exception as e:
                    print(f"DEBUG: Failed to call os.add_dll_directory: {e}")
                
            print(f"INFO: Added torch lib path to DLL search: {torch_lib_path}")
        else:
            print(f"DEBUG: Path exists but is not a directory: {torch_lib_path}")
    else:
        print(f"Warning: Hardcoded torch lib path does not exist: {torch_lib_path}")
        # Fallback: Try assuming we are in a standard site-packages layout if path is wrong
        import site
        # This might return the same as above or list of paths
        try:
            packages = site.getsitepackages()
            for pkg_path in packages:
                 path_candidate = os.path.join(pkg_path, 'torch', 'lib')
                 if os.path.exists(path_candidate) and path_candidate != torch_lib_path:
                     os.environ['PATH'] = path_candidate + os.pathsep + os.environ.get('PATH', '')
                     print(f"INFO: Added torch lib path to PATH (via site): {path_candidate}")
                     break
        except Exception as e:
            print(f"DEBUG: Error in fallback site packages lookup: {e}")
            
except Exception as e:
    print(f"Warning: Could not automatically add torch/lib to PATH: {e}")

WHISPER_AVAILABLE = False
try:
    import torch
    # Force CPU mode
    torch.set_num_threads(1)
    import whisper
    WHISPER_AVAILABLE = True
    print("SUCCESS: Torch and Whisper imported successfully!")
except Exception as e:
    print(f"ERROR: Could not load Whisper/PyTorch: {e}")
    # Print more debug info if it's an ImportError or OSError
    import traceback
    traceback.print_exc()
    
    print("Audio extraction will not be available.")
    print("To fix: Install PyTorch CPU version:")
    print("  pip uninstall torch torchvision torchaudio")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    whisper = None

from pathlib import Path
from typing import Dict, Optional
import time
from core.base_extractor import BaseExtractor, ExtractionResult
from config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_LANGUAGE, FFMPEG_PATH

# Configure FFmpeg path for Whisper
if os.path.exists(FFMPEG_PATH):
    # Set the directory containing ffmpeg.exe in PATH
    ffmpeg_dir = str(Path(FFMPEG_PATH).parent)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

class AudioExtractor(BaseExtractor):
    """Extract text from audio files using Whisper with model caching"""

    # Class-level cache for Whisper model (shared across instances)
    _whisper_models = {}

    def __init__(self, model_name: str = WHISPER_MODEL):
        super().__init__()
        self.model_name = model_name
        self.model = None

        # Check if Whisper is available
        if not WHISPER_AVAILABLE or whisper is None:
            self.logger.error("Whisper is not available. Audio extraction will fail.")
            self.logger.error("Install PyTorch CPU version to fix:")
            self.logger.error("  pip uninstall torch torchvision torchaudio")
            self.logger.error("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
            return

        # Use cached model if available
        if model_name not in AudioExtractor._whisper_models:
            self.logger.info(f"Loading Whisper model: {model_name} (CPU mode)")
            try:
                # Force CPU mode to avoid CUDA/DLL issues
                AudioExtractor._whisper_models[model_name] = whisper.load_model(
                    model_name, device="cpu"
                )
                self.logger.info("Whisper model loaded successfully on CPU")
            except Exception as e:
                self.logger.error(f"Failed to load Whisper model: {e}")
                self.logger.info("Attempting to load with default settings...")
                # Try without specifying device
                try:
                    AudioExtractor._whisper_models[model_name] = whisper.load_model(model_name)
                    self.logger.info("Whisper model loaded with default settings")
                except Exception as e2:
                    self.logger.error(f"Failed to load Whisper model with default settings: {e2}")
                    return
        else:
            self.logger.info(f"Using cached Whisper model: {model_name}")

        self.model = AudioExtractor._whisper_models.get(model_name)
    
    def extract(self, file_path: Path, **kwargs) -> ExtractionResult:
        """Extract text from audio file"""
        start_time = time.time()
        self.logger.info(f"Transcribing audio: {file_path}")

        # Check if model is available
        if self.model is None:
            error_msg = "Whisper model not available. Install PyTorch CPU version to enable audio extraction."
            self.logger.error(error_msg)
            return self._create_error_result(file_path, error_msg)

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