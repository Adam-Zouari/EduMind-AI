"""
Format detection using Apache Tika and python-magic
"""
import magic
from pathlib import Path
from typing import Optional, Dict
from tika import detector
from utils.logger import get_logger
from config import SUPPORTED_FORMATS

logger = get_logger(__name__)

class FormatDetector:
    """Detects file format using multiple methods"""
    
    def __init__(self):
        self.magic = magic.Magic(mime=True)
    
    def detect(self, file_path: Path) -> Dict[str, str]:
        """
        Detect file format using multiple methods
        
        Returns:
            Dict with keys: format_type, mime_type, extension
        """
        logger.info(f"Detecting format for: {file_path}")
        
        # Get extension
        extension = file_path.suffix.lower()
        
        # Detect using python-magic
        mime_type = self._detect_with_magic(file_path)
        
        # Detect using Tika
        tika_mime = self._detect_with_tika(file_path)
        
        # Use Tika result if available, otherwise use magic
        final_mime = tika_mime or mime_type
        
        # Determine format type
        format_type = self._map_to_format_type(extension, final_mime)
        
        result = {
            "format_type": format_type,
            "mime_type": final_mime,
            "extension": extension
        }
        
        logger.info(f"Detected format: {result}")
        return result
    
    def _detect_with_magic(self, file_path: Path) -> Optional[str]:
        """Detect MIME type using python-magic"""
        try:
            mime_type = self.magic.from_file(str(file_path))
            logger.debug(f"Magic detected: {mime_type}")
            return mime_type
        except Exception as e:
            logger.warning(f"Magic detection failed: {e}")
            return None
    
    def _detect_with_tika(self, file_path: Path) -> Optional[str]:
        """Detect MIME type using Apache Tika"""
        try:
            mime_type = detector.from_file(str(file_path))
            logger.debug(f"Tika detected: {mime_type}")
            return mime_type
        except Exception as e:
            logger.warning(f"Tika detection failed: {e}")
            return None
    
    def _map_to_format_type(self, extension: str, mime_type: str) -> str:
        """Map extension and MIME type to format category"""
        
        # Check extension first
        for format_type, extensions in SUPPORTED_FORMATS.items():
            if extension in extensions:
                return format_type
        
        # Check MIME type
        if mime_type:
            if "pdf" in mime_type:
                return "pdf"
            elif "word" in mime_type or "officedocument" in mime_type:
                return "docx"
            elif "image" in mime_type:
                return "image"
            elif "video" in mime_type:
                return "video"
            elif "audio" in mime_type:
                return "audio"
            elif "html" in mime_type or "xml" in mime_type:
                return "web"
        
        logger.warning(f"Unknown format: {extension}, {mime_type}")
        return "unknown"