"""
OCR-to-RAG Pipeline
A complete pipeline for converting OCR-extracted text into embeddings 
and storing them in a vector database for RAG applications.
"""

from .ocr_processor import OCRProcessor
from .text_chunker import TextChunker
from .embedder import Embedder
from .vector_store import VectorStore
from .rag_pipeline import RAGPipeline

__version__ = "1.0.0"

__all__ = [
    "OCRProcessor",
    "TextChunker",
    "Embedder",
    "VectorStore",
    "RAGPipeline",
]

