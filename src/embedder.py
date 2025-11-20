"""
Embedder Module
Generates embeddings using sentence-transformers.
"""

from typing import List, Union
from sentence_transformers import SentenceTransformer
import numpy as np
import yaml
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Embedder:
    """
    Handles embedding generation using sentence-transformers.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the Embedder with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        embed_config = self.config['embedding']
        self.model_name = embed_config['model_name']
        self.device = embed_config['device']
        self.embedding_dim = embed_config['embedding_dim']
        
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(f"Model loaded successfully on device: {self.device}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            Numpy array of embeddings
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.embedding_dim)
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            logger.warning("Empty text list provided for embedding")
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=32
        )
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def embed_chunks(self, chunks: List[dict], text_key: str = 'text') -> List[dict]:
        """
        Generate embeddings for chunks and add them to the chunk dictionaries.
        
        Args:
            chunks: List of chunk dictionaries
            text_key: Key in the dictionary containing the text
            
        Returns:
            List of chunks with embeddings added
        """
        if not chunks:
            logger.warning("Empty chunks list provided")
            return []
        
        # Extract texts
        texts = [chunk.get(text_key, '') for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding.tolist()  # Convert to list for JSON serialization
        
        logger.info(f"Added embeddings to {len(chunks)} chunks")
        return chunks
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dim


if __name__ == "__main__":
    # Example usage
    embedder = Embedder()
    
    # Single text embedding
    sample_text = "This is a sample text for embedding generation."
    embedding = embedder.embed_text(sample_text)
    print(f"\nSingle text embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Multiple texts embedding
    sample_texts = [
        "First document about machine learning.",
        "Second document about natural language processing.",
        "Third document about computer vision."
    ]
    embeddings = embedder.embed_texts(sample_texts)
    print(f"\nMultiple texts embeddings shape: {embeddings.shape}")
    
    # Chunk embedding
    sample_chunks = [
        {'text': 'Chunk 1 text', 'source': 'doc1.pdf'},
        {'text': 'Chunk 2 text', 'source': 'doc2.pdf'},
    ]
    chunks_with_embeddings = embedder.embed_chunks(sample_chunks)
    print(f"\nChunks with embeddings: {len(chunks_with_embeddings)}")
    print(f"First chunk keys: {chunks_with_embeddings[0].keys()}")

