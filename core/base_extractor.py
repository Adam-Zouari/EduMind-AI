"""
Base class for all extractors
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ExtractionResult:
    """Data class for extraction results"""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    format_type: str = ""
    file_path: str = ""
    extraction_time: float = 0.0
    success: bool = True
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "metadata": self.metadata,
            "format_type": self.format_type,
            "file_path": self.file_path,
            "extraction_time": self.extraction_time,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp.isoformat()
        }

class BaseExtractor(ABC):
    """Abstract base class for all extractors"""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def extract(self, file_path: Path, **kwargs) -> ExtractionResult:
        """
        Extract content from file
        
        Args:
            file_path: Path to the file
            **kwargs: Additional arguments
            
        Returns:
            ExtractionResult object
        """
        pass
    
    def _create_error_result(self, file_path: Path, error: str) -> ExtractionResult:
        """Create an error result"""
        return ExtractionResult(
            text="",
            file_path=str(file_path),
            success=False,
            error=error
        )