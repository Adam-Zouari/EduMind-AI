"""
Text Chunker Module
Handles intelligent text splitting for OCR-extracted content.
"""

from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import yaml
import logging
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextChunker:
    """
    Handles text chunking with configurable parameters.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the TextChunker with configuration.

        Args:
            config_path: Path to the configuration file
        """
        # Handle relative path from RAG directory
        if not os.path.isabs(config_path) and not os.path.exists(config_path):
            # Try to find config relative to this file's location
            current_dir = Path(__file__).parent.parent
            config_path = str(current_dir / config_path)

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        chunk_config = self.config['chunking']
        self.chunk_size = chunk_config['chunk_size']
        self.chunk_overlap = chunk_config['chunk_overlap']
        self.separators = chunk_config['separators']
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
        
        logger.info(f"TextChunker initialized with chunk_size={self.chunk_size}, "
                   f"overlap={self.chunk_overlap}")
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        # Split the text
        chunks = self.text_splitter.split_text(text)
        
        # Create chunk objects with metadata
        chunk_objects = []
        for idx, chunk in enumerate(chunks):
            chunk_obj = {
                'text': chunk,
                'chunk_index': idx,
                'total_chunks': len(chunks),
            }
            
            # Add provided metadata
            if metadata:
                chunk_obj.update(metadata)
            
            chunk_objects.append(chunk_obj)
        
        logger.info(f"Created {len(chunk_objects)} chunks from text of length {len(text)}")
        return chunk_objects
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of documents with 'text' and optional metadata
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            text = doc.get('text', '')
            metadata = {k: v for k, v in doc.items() if k != 'text'}
            metadata['document_index'] = doc_idx
            
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} total chunks")
        return all_chunks


if __name__ == "__main__":
    # Example usage
    chunker = TextChunker()
    
    sample_text = """
    This is a sample document that demonstrates the text chunking functionality.
    The chunker will split this text into smaller, manageable pieces while maintaining
    context through overlapping chunks.
    
    This is especially useful for OCR-extracted text which can be quite long.
    The chunks will be used to create embeddings for the RAG system.
    """
    
    chunks = chunker.chunk_text(sample_text, metadata={'source': 'example.pdf', 'page': 1})
    
    print(f"\nCreated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"Text: {chunk['text'][:100]}...")
        print(f"Metadata: {chunk}")

