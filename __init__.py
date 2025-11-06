"""
Data Ingestion and Preprocessing System

A unified content extraction system for heterogeneous data sources.
"""
from core.pipeline import DataIngestionPipeline
from core.base_extractor import ExtractionResult
from extractors import *

__version__ = "1.0.0"
__all__ = ["DataIngestionPipeline", "ExtractionResult"]