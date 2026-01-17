"""
Text Chunker Module
Handles intelligent text splitting for OCR-extracted content.
"""

from typing import List, Dict, Any

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TextChunker:
    """
    Handles intelligent text splitting for OCR-extracted content using Semantic Chunking.
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
        self.chunk_size = chunk_config.get('chunk_size', 1000)
        self.min_chunk_size = 500 # Minimum size to be useful
        
        # Initialize embedding model for semantic splitting
        # We use a small, fast model specifically for chunking decisions
        embedding_config = self.config.get('embedding', {})
        model_name = embedding_config.get('model_name', 'all-MiniLM-L6-v2')
        device = embedding_config.get('device', 'cpu')
        
        logger.info(f"Loading chunking model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        
        logger.info(f"TextChunker initialized with semantic splitting")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter."""
        # Split by common sentence terminators but keep them
        # This is a basic regex, could be improved with nltk/spacy
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks using semantic similarity.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
            
        # If text is short, return as single chunk
        if len(text) < self.chunk_size:
            chunk_obj = {'text': text, 'chunk_index': 0, 'total_chunks': 1}
            if metadata:
                chunk_obj.update(metadata)
            return [chunk_obj]

        # 1. Compute embeddings for all sentences
        embeddings = self.model.encode(sentences)
        
        # 2. Calculate cosine similarity between adjacent sentences
        # distances[i] = similarity between sentence i and i+1
        distances = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            distances.append(sim)
        
        # 3. Determine threshold (e.g., 10th percentile of similarity = 90th percentile distance)
        # Lower similarity means higher "distance" in topic
        # We want to split where similarity is LOW
        if distances:
            breakpoint_percentile_threshold = 10 # Split at the 10% lowest similarity points
            threshold = np.percentile(distances, breakpoint_percentile_threshold)
            
            # Find indices where similarity < threshold
            indices_above_thresh = [i for i, x in enumerate(distances) if x < threshold]
        else:
            indices_above_thresh = []

        # 4. Construct chunks
        chunks = []
        start_index = 0
        
        # Add end index
        break_indices = indices_above_thresh + [len(sentences)]
        
        current_chunk_sentences = []
        current_req_len = 0
        
        for i, sentence in enumerate(sentences):
            current_chunk_sentences.append(sentence)
            current_req_len += len(sentence)
            
            # Check if we should split here
            # Split if:
            # a) We are at a semantic breakpoint AND chunk is big enough
            # b) Chunk is getting too big (hard limit)
            
            is_semantic_break = i in indices_above_thresh
            is_too_big = current_req_len >= self.chunk_size
            is_big_enough = current_req_len >= self.min_chunk_size
            is_last = i == len(sentences) - 1
            
            if (is_semantic_break and is_big_enough) or (is_too_big) or (is_last):
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(chunk_text)
                
                # Reset
                current_chunk_sentences = []
                current_req_len = 0
        
        # Create chunk objects with metadata
        chunk_objects = []
        for idx, chunk_text in enumerate(chunks):
            chunk_obj = {
                'text': chunk_text,
                'chunk_index': idx,
                'total_chunks': len(chunks),
            }
            
            # Add provided metadata
            if metadata:
                chunk_obj.update(metadata)
            
            chunk_objects.append(chunk_obj)
        
        logger.info(f"Created {len(chunk_objects)} semantic chunks from text of length {len(text)}")
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
    Machine Learning is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence.
    
    Deep learning is a subset of machine learning using neural networks with three or more layers. These neural networks attempt to simulate the behavior of the human brain—albeit far from matching its ability—allowing it to “learn” from large amounts of data.
    
    The history of machine learning traces back to the mid-20th century. Arthur Samuel coined the term "machine learning" in 1959.
    """
    
    chunks = chunker.chunk_text(sample_text, metadata={'source': 'example.pdf', 'page': 1})
    
    print(f"\nCreated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"Text: {chunk['text'][:100]}...")
        print(f"Length: {len(chunk['text'])}")


