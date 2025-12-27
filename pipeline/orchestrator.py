"""
OCR-RAG Orchestrator
Integrates OCR extraction and RAG pipeline for end-to-end document processing
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import logging
import importlib.util

# Setup paths
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent

# Add project root to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add OCR directory to path (for its relative imports)
ocr_dir = project_root / "OCR"
if str(ocr_dir) not in sys.path:
    sys.path.insert(0, str(ocr_dir))

# Import OCR modules (now available via relative imports from OCR dir)
from core.pipeline import DataIngestionPipeline
from core.base_extractor import ExtractionResult

# Import RAG modules
from RAG.src.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRRAGOrchestrator:
    """
    Orchestrates the complete pipeline:
    1. OCR extraction from various file formats
    2. RAG ingestion and querying
    """
    
    def __init__(self, use_llm: bool = True, rag_config_path: str = None):
        """
        Initialize the orchestrator with OCR and RAG pipelines.
        
        Args:
            use_llm: Whether to initialize LLM for answer generation
            rag_config_path: Path to RAG config file (default: RAG/config/config.yaml)
        """
        logger.info("Initializing OCR-RAG Orchestrator...")
        
        # Initialize OCR pipeline
        self.ocr_pipeline = DataIngestionPipeline()
        logger.info("✓ OCR Pipeline initialized")
        
        # Initialize RAG pipeline
        if rag_config_path is None:
            rag_config_path = str(project_root / "RAG" / "config" / "config.yaml")
        
        self.rag_pipeline = RAGPipeline(config_path=rag_config_path, use_llm=use_llm)
        logger.info("✓ RAG Pipeline initialized")
        
        logger.info("OCR-RAG Orchestrator ready!")
    
    def process_file(self, 
                    file_path: str | Path,
                    ingest_to_rag: bool = True,
                    clean_text: bool = True,
                    **kwargs) -> Dict[str, Any]:
        """
        Process a file through OCR and optionally ingest to RAG.
        
        Args:
            file_path: Path to the file to process
            ingest_to_rag: Whether to automatically ingest to RAG
            clean_text: Whether to clean extracted text
            **kwargs: Additional arguments for OCR extractor
            
        Returns:
            Dictionary with OCR result and RAG ingestion info
        """
        logger.info(f"Processing file: {file_path}")
        
        # Step 1: OCR Extraction
        ocr_result = self.ocr_pipeline.process_file(
            file_path=file_path,
            clean_text=clean_text,
            **kwargs
        )
        
        result = {
            'ocr_success': ocr_result.success,
            'ocr_error': ocr_result.error,
            'text': ocr_result.text,
            'metadata': ocr_result.metadata,
            'file_path': ocr_result.file_path,
            'format_type': ocr_result.format_type,
            'extraction_time': ocr_result.extraction_time,
            'rag_ingested': False,
            'rag_chunks': 0
        }
        
        # Step 2: RAG Ingestion (if requested and OCR succeeded)
        if ingest_to_rag and ocr_result.success and ocr_result.text:
            try:
                # Convert OCR result to RAG format
                document = ocr_result.to_dict()
                
                # Ingest to RAG
                num_chunks = self.rag_pipeline.ingest_document(document)
                
                result['rag_ingested'] = True
                result['rag_chunks'] = num_chunks
                logger.info(f"✓ Ingested to RAG: {num_chunks} chunks created")
                
            except Exception as e:
                logger.error(f"RAG ingestion failed: {e}")
                result['rag_error'] = str(e)
        
        return result
    
    def process_batch(self,
                     file_paths: List[str | Path],
                     ingest_to_rag: bool = True,
                     **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple files through OCR and RAG.
        
        Args:
            file_paths: List of file paths to process
            ingest_to_rag: Whether to automatically ingest to RAG
            **kwargs: Additional arguments for OCR extractor
            
        Returns:
            List of result dictionaries
        """
        logger.info(f"Processing batch of {len(file_paths)} files")
        
        results = []
        for file_path in file_paths:
            result = self.process_file(file_path, ingest_to_rag, **kwargs)
            results.append(result)
        
        total_chunks = sum(r.get('rag_chunks', 0) for r in results)
        logger.info(f"Batch processing complete: {total_chunks} total chunks ingested")
        
        return results
    
    def query(self, 
             query_text: str,
             top_k: int = 5,
             generate_answer: bool = True,
             **kwargs) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            query_text: The query text
            top_k: Number of results to return
            generate_answer: Whether to generate LLM answer (requires use_llm=True)
            **kwargs: Additional arguments for RAG query
            
        Returns:
            Dictionary with query results and optional answer
        """
        if generate_answer and self.rag_pipeline.llm_generator:
            # Generate answer with LLM
            return self.rag_pipeline.generate_answer(
                query=query_text,
                top_k=top_k,
                **kwargs
            )
        else:
            # Just retrieve documents
            results = self.rag_pipeline.query(query_text, top_k, **kwargs)
            return {
                'results': results,
                'query': query_text
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from both pipelines."""
        rag_stats = self.rag_pipeline.get_stats()
        
        return {
            'rag': rag_stats,
            'ocr_extractors': list(self.ocr_pipeline.extractors.keys())
        }
    
    def reset_rag(self):
        """Reset the RAG vector store."""
        self.rag_pipeline.reset()
        logger.info("RAG database reset")

