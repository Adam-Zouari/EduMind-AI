"""
OCR extraction using Tesseract and PaddleOCR
"""
import pytesseract
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List
import time
from core.base_extractor import BaseExtractor, ExtractionResult
from config import TESSERACT_CMD, OCR_LANGUAGES, OCR_CONFIDENCE_THRESHOLD

# Try to import PaddleOCR (optional)
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("Warning: PaddleOCR not available. Only Tesseract will be used.")

class OCRExtractor(BaseExtractor):
    """Extract text from images using OCR"""
    
    def __init__(self, use_paddle: bool = False):
        super().__init__()
        self.use_paddle = use_paddle and PADDLE_AVAILABLE
        self.paddle_ocr = None

        if use_paddle and not PADDLE_AVAILABLE:
            self.logger.warning("PaddleOCR requested but not available. Using Tesseract instead.")
            self.use_paddle = False

        if self.use_paddle:
            try:
                self.logger.info("Initializing PaddleOCR...")
                # Note: show_log parameter may not be available in all PaddleOCR versions
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
                self.logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize PaddleOCR: {e}")
                self.logger.warning("Falling back to Tesseract")
                self.use_paddle = False

        if not self.use_paddle:
            # Only set Tesseract path if we're using Tesseract
            self.logger.info(f"Using Tesseract OCR: {TESSERACT_CMD}")
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    
    def extract(self, file_path: Path, **kwargs) -> ExtractionResult:
        """Extract text from image using OCR"""
        start_time = time.time()
        self.logger.info(f"Extracting text via OCR: {file_path}")
        
        try:
            # Preprocess image
            preprocessed = self._preprocess_image(file_path)
            
            # Extract text
            if self.use_paddle and self.paddle_ocr:
                text, confidence = self._extract_with_paddle(preprocessed)
            else:
                text, confidence = self._extract_with_tesseract(preprocessed)
            
            metadata = {
                "ocr_engine": "paddleocr" if self.use_paddle else "tesseract",
                "confidence": confidence,
                "languages": OCR_LANGUAGES,
                "extractor": "ocr"
            }
            
            extraction_time = time.time() - start_time
            
            return ExtractionResult(
                text=text,
                metadata=metadata,
                format_type="image",
                file_path=str(file_path),
                extraction_time=extraction_time,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return self._create_error_result(file_path, str(e))
    
    def _preprocess_image(self, file_path: Path) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Read image
        img = cv2.imread(str(file_path))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Denoise
        denoised = cv2.medianBlur(threshold, 3)
        
        return denoised
    
    def _extract_with_tesseract(self, image: np.ndarray) -> tuple[str, float]:
        """Extract text using Tesseract"""
        # Get text with confidence
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Filter by confidence
        text_parts = []
        confidences = []
        
        for i, conf in enumerate(data['conf']):
            if conf > OCR_CONFIDENCE_THRESHOLD:
                text = data['text'][i].strip()
                if text:
                    text_parts.append(text)
                    confidences.append(conf)
        
        text = " ".join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return text, avg_confidence
    
    def _extract_with_paddle(self, image: np.ndarray) -> tuple[str, float]:
        """Extract text using PaddleOCR"""
        result = self.paddle_ocr.ocr(image, cls=True)
        
        text_parts = []
        confidences = []
        
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                conf = line[1][1]
                text_parts.append(text)
                confidences.append(conf * 100)  # Convert to percentage
        
        text = " ".join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return text, avg_confidence