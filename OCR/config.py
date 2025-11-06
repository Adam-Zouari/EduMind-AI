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

# Create directories if they don't exist
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Tika Configuration
TIKA_SERVER_JAR = os.getenv("TIKA_SERVER_JAR", None)

# OCR Configuration
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
OCR_LANGUAGES = ["eng", "fra", "spa", "deu"]  # Add more as needed
OCR_CONFIDENCE_THRESHOLD = 60

# Whisper Configuration
WHISPER_MODEL = "large-v3"  # Options: tiny, base, small, medium, large, large-v3
WHISPER_DEVICE = "cuda"  # Options: cuda, cpu
WHISPER_LANGUAGE = None  # Auto-detect if None

# FFmpeg Configuration
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
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