"""
Text cleaning and normalization
"""
import re
import ftfy
from unidecode import unidecode
from typing import List
from utils.logger import get_logger
from config import REMOVE_HEADERS_FOOTERS, NORMALIZE_WHITESPACE, MIN_TEXT_LENGTH

logger = get_logger(__name__)

class TextCleaner:
    """Clean and normalize extracted text"""
    
    @staticmethod
    def clean(text: str, preserve_latex: bool = True) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw extracted text
            preserve_latex: Keep LaTeX notation
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Fix encoding issues
        text = ftfy.fix_text(text)
        
        # Remove headers/footers (common patterns)
        if REMOVE_HEADERS_FOOTERS:
            text = TextCleaner._remove_headers_footers(text)
        
        # Normalize whitespace
        if NORMALIZE_WHITESPACE:
            text = TextCleaner._normalize_whitespace(text)
        
        # Remove page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove multiple dashes/underscores (often used as dividers)
        text = re.sub(r'[-_]{4,}', '', text)
        
        # Fix common OCR errors
        text = TextCleaner._fix_ocr_errors(text)
        
        # Preserve or remove LaTeX
        if not preserve_latex:
            text = re.sub(r'\$.*?\$', '', text)
            text = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', text)
        
        return text.strip()
    
    @staticmethod
    def _remove_headers_footers(text: str) -> str:
        """Remove common header/footer patterns"""
        lines = text.split('\n')
        
        # Common header/footer patterns
        patterns = [
            r'^\s*page\s+\d+',
            r'^\s*\d+\s*$',
            r'confidential',
            r'proprietary',
            r'copyright\s+©',
        ]
        
        cleaned_lines = []
        for line in lines:
            is_header_footer = False
            for pattern in patterns:
                if re.search(pattern, line.lower()):
                    is_header_footer = True
                    break
            
            if not is_header_footer:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    @staticmethod
    def _fix_ocr_errors(text: str) -> str:
        """Fix common OCR errors"""
        # Common OCR substitutions
        ocr_fixes = {
            r'\b0\b': 'O',  # Zero to O
            r'\b1\b': 'I',  # One to I (context-dependent)
            r'rn': 'm',     # rn to m
            r'vv': 'w',     # vv to w
        }
        
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """Extract sentences from text"""
        # Simple sentence splitter
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > MIN_TEXT_LENGTH]