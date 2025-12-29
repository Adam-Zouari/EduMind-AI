"""
Configuration file for data ingestion system
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"

# Create directories if they don't exist
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Tika Configuration
TIKA_SERVER_JAR = os.getenv("TIKA_SERVER_JAR", None)

# OCR Configuration
# PaddleOCR is preferred for better accuracy (95%+ vs 85-90% for Tesseract)
OCR_USE_PADDLE = os.getenv("OCR_USE_PADDLE", "true").lower() == "true"  # Use PaddleOCR (recommended)
OCR_USE_GPU = os.getenv("OCR_USE_GPU", "false").lower() == "true"  # Enable GPU for PaddleOCR (requires cuDNN)

# Tesseract fallback configuration
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
OCR_LANGUAGES = ["eng", "fra", "spa", "deu"]  # Add more as needed
OCR_CONFIDENCE_THRESHOLD = 60
OCR_ENABLE_CACHING = True  # Cache OCR results to avoid reprocessing
OCR_CACHE_DIR = CACHE_DIR / "ocr"
OCR_ADAPTIVE_PREPROCESSING = True  # Adjust preprocessing based on image quality
OCR_ROTATION_CORRECTION = True  # Auto-detect and correct rotation
OCR_PERSPECTIVE_CORRECTION = True  # Auto-detect and correct perspective distortion
OCR_QUALITY_THRESHOLD = 50  # Minimum image quality score (0-100)
OCR_USE_ANGLE_CLS = os.getenv("OCR_USE_ANGLE_CLS", "false").lower() == "true"  # Disable by default to prevent IndexErrors
OCR_ENABLE_HANDWRITING = False  # Enable handwriting recognition (requires additional models)
OCR_PARALLEL_PROCESSING = True  # Enable parallel batch processing

# Create OCR cache directory
OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Whisper Configuration
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large, large-v3
WHISPER_DEVICE = "cpu"  # Options: cuda, cpu
WHISPER_LANGUAGE = None  # Auto-detect if None

# FFmpeg Configuration
FFMPEG_PATH = os.getenv("FFMPEG_PATH", r"C:\Users\ademz\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe")
VIDEO_FRAME_RATE = 1  # Extract 1 frame per second

# Web Scraping Configuration
WEB_TIMEOUT = 30  # seconds
USER_AGENT = "Mozilla/5.0 (compatible; DataIngestionBot/1.0)"

# Text Cleaning Configuration
REMOVE_HEADERS_FOOTERS = True
NORMALIZE_WHITESPACE = True
PRESERVE_LATEX = True
MIN_TEXT_LENGTH = 10  # Minimum characters for valid text

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"

# Supported file extensions
SUPPORTED_FORMATS = {
    "pdf": [".pdf"],
    "docx": [".docx", ".doc"],
    "image": [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"],
    "video": [".mp4", ".avi", ".mov", ".mkv", ".flv"],
    "audio": [".mp3", ".wav", ".m4a", ".flac", ".ogg"],
    "web": [".html", ".htm", ".xml"]
}