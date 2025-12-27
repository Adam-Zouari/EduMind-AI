"""
API-based Orchestrator
Coordinates OCR and RAG services via HTTP APIs
No dependency conflicts - just makes HTTP calls!
"""

import requests
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIOrchestrator:
    """
    Orchestrates OCR and RAG via microservices
    No direct imports - uses HTTP APIs
    """
    
    def __init__(self, ocr_url: str = "http://localhost:8000", rag_url: str = "http://localhost:8001"):
        """
        Initialize orchestrator with service URLs
        
        Args:
            ocr_url: URL of OCR service
            rag_url: URL of RAG service
        """
        self.ocr_url = ocr_url
        self.rag_url = rag_url
        
        # Check services are running
        self._check_services()
    
    def _check_services(self):
        """Check if both services are running"""
        try:
            ocr_health = requests.get(f"{self.ocr_url}/health", timeout=5)
            logger.info(f"✓ OCR Service: {ocr_health.json()}")
        except Exception as e:
            logger.error(f"❌ OCR Service not available at {self.ocr_url}")
            logger.error(f"   Start it with: uvicorn pipeline.ocr_service:app --port 8000")
            raise ConnectionError(f"OCR service not available: {e}")
        
        try:
            rag_health = requests.get(f"{self.rag_url}/health", timeout=5)
            logger.info(f"✓ RAG Service: {rag_health.json()}")
        except Exception as e:
            logger.error(f"❌ RAG Service not available at {self.rag_url}")
            logger.error(f"   Start it with: uvicorn pipeline.rag_service:app --port 8001")
            raise ConnectionError(f"RAG service not available: {e}")
    
    def process_file(self, file_path: str | Path, ingest_to_rag: bool = True) -> Dict[str, Any]:
        """
        Process a file through OCR and optionally ingest to RAG
        
        Args:
            file_path: Path to file
            ingest_to_rag: Whether to ingest to RAG
            
        Returns:
            Dictionary with OCR results and RAG status
        """
        file_path = Path(file_path)
        
        # Step 1: Extract text with OCR service
        logger.info(f"Extracting text from {file_path.name}...")
        
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'application/octet-stream')}
            response = requests.post(f"{self.ocr_url}/extract", files=files)
        
        if response.status_code != 200:
            raise Exception(f"OCR extraction failed: {response.text}")
        
        ocr_result = response.json()
        logger.info(f"✓ Extracted {len(ocr_result['text'])} characters")
        
        # Step 2: Ingest to RAG if requested
        rag_chunks = 0
        if ingest_to_rag and ocr_result['success']:
            logger.info("Ingesting to RAG...")
            
            ingest_data = {
                "text": ocr_result['text'],
                "metadata": ocr_result['metadata']
            }
            
            response = requests.post(f"{self.rag_url}/ingest", json=ingest_data)
            
            if response.status_code == 200:
                rag_result = response.json()
                rag_chunks = rag_result.get('chunks', 0)
                logger.info(f"✓ Created {rag_chunks} RAG chunks")
            else:
                logger.warning(f"RAG ingestion failed: {response.text}")
        
        return {
            "success": ocr_result['success'],
            "text": ocr_result['text'],
            "metadata": ocr_result['metadata'],
            "format_type": ocr_result['format_type'],
            "extraction_time": ocr_result['extraction_time'],
            "rag_ingested": ingest_to_rag and rag_chunks > 0,
            "rag_chunks": rag_chunks
        }
    
    def query(self, query_text: str, top_k: int = 5, generate_answer: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            query_text: Question to ask
            top_k: Number of results
            generate_answer: Whether to generate LLM answer
            
        Returns:
            Query results with answer and sources
        """
        logger.info(f"Querying: {query_text}")
        
        query_data = {
            "query": query_text,
            "top_k": top_k,
            "generate_answer": generate_answer
        }
        
        response = requests.post(f"{self.rag_url}/query", json=query_data)
        
        if response.status_code != 200:
            raise Exception(f"Query failed: {response.text}")
        
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from both services"""
        rag_stats = requests.get(f"{self.rag_url}/stats").json()
        ocr_formats = requests.get(f"{self.ocr_url}/formats").json()
        
        return {
            "rag": rag_stats,
            "ocr_formats": ocr_formats['formats']
        }
    
    def reset_database(self):
        """Reset the RAG database"""
        response = requests.delete(f"{self.rag_url}/reset")
        return response.json()

