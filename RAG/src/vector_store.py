"""
Vector Store Module
Handles ChromaDB operations for storing and retrieving embeddings.
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import yaml
import logging
import uuid
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



import pickle
from rank_bm25 import BM25Okapi
import numpy as np

class VectorStore:
    """
    Handles vector database operations using ChromaDB and BM25 for hybrid search.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the VectorStore with configuration.

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

        vectordb_config = self.config['vectordb']
        self.collection_name = vectordb_config['collection_name']
        self.persist_directory = vectordb_config['persist_directory']
        self.distance_metric = vectordb_config['distance_metric']

        # Handle relative persist_directory path
        if not os.path.isabs(self.persist_directory):
            # Make it relative to the RAG directory (parent of src)
            current_dir = Path(__file__).parent.parent
            self.persist_directory = str(current_dir / self.persist_directory)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric}
        )
        
        # Initialize BM25
        self.bm25_path = Path(self.persist_directory) / "bm25_index.pkl"
        self.bm25 = None
        self.bm25_corpus = [] # List of terms
        self.doc_map = [] # List of unique IDs mapping to BM25 indices
        self._load_bm25()

        logger.info(f"VectorStore initialized with collection: {self.collection_name}")
        logger.info(f"Persist directory: {self.persist_directory}")
    
    def _load_bm25(self):
        """Load BM25 index from disk."""
        if self.bm25_path.exists():
            try:
                with open(self.bm25_path, 'rb') as f:
                    data = pickle.load(f)
                    self.bm25 = data['index']
                    self.bm25_corpus = data['corpus']
                    self.doc_map = data['doc_map']
                logger.info(f"Loaded BM25 index with {len(self.doc_map)} documents")
            except Exception as e:
                logger.error(f"Failed to load BM25 index: {e}")
                
    def _save_bm25(self):
        """Save BM25 index to disk."""
        if self.bm25:
            try:
                with open(self.bm25_path, 'wb') as f:
                    pickle.dump({
                        'index': self.bm25,
                        'corpus': self.bm25_corpus,
                        'doc_map': self.doc_map
                    }, f)
                logger.info("Saved BM25 index to disk")
            except Exception as e:
                logger.error(f"Failed to save BM25 index: {e}")

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store and BM25 index.
        Automatically batches large additions to respect ChromaDB's batch size limit.
        
        Args:
            chunks: List of chunk dictionaries with 'text', 'embedding', and metadata
        """
        if not chunks:
            logger.warning("No chunks provided to add")
            return
        
        # ChromaDB has a maximum batch size (~5461), so we batch if necessary
        BATCH_SIZE = 5000
        total_chunks = len(chunks)
        
        if total_chunks > BATCH_SIZE:
            logger.info(f"Batching {total_chunks} documents into chunks of {BATCH_SIZE}")
            for i in range(0, total_chunks, BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE]
                self._add_batch(batch)
                logger.info(f"Added batch {i//BATCH_SIZE + 1}/{(total_chunks + BATCH_SIZE - 1)//BATCH_SIZE}")
        else:
            self._add_batch(chunks)
        
        logger.info(f"Added {len(chunks)} documents to vector store and BM25 index")
    
    def _add_batch(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Internal method to add a single batch of documents.
        
        Args:
            chunks: List of chunk dictionaries with 'text', 'embedding', and metadata
        """
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        # Prepare data for BM25
        new_corpus = []
        
        for chunk in chunks:
            # Generate unique ID
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            self.doc_map.append(chunk_id)
            
            # Extract embedding
            embedding = chunk.get('embedding', [])
            if isinstance(embedding, list):
                embeddings.append(embedding)
            else:
                embeddings.append(embedding.tolist())
            
            # Extract text
            text = chunk.get('text', '')
            documents.append(text)
            
            # Tokenize for BM25 (simple whitespace tokenization for now)
            # In production, use the same tokenizer as the cleaner
            new_corpus.append(text.lower().split())
            
            # Extract metadata (exclude text and embedding)
            metadata = {k: v for k, v in chunk.items() 
                       if k not in ['text', 'embedding']}
            # Convert all metadata values to strings for ChromaDB
            metadata = {k: str(v) for k, v in metadata.items()}
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        # Update BM25
        self.bm25_corpus.extend(new_corpus)
        self.bm25 = BM25Okapi(self.bm25_corpus)
        self._save_bm25()
    
    def query(self, query_embedding: List[float], top_k: int = 5, 
              filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the vector store (Dense Retrieval only).
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )
        
        logger.info(f"Dense query returned {len(results['documents'][0])} results")
        return results
    
    def query_hybrid(self, query_text: str, query_embedding: List[float], 
                    top_k: int = 5, alpha: float = 0.3) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (BM25 + Vector) using Weighted Reciprocal Rank Fusion.
        
        Args:
            query_text: Text for BM25
            query_embedding: Embedding for ChromaDB
            top_k: Number of results
            alpha: Weight for BM25 (0.0=Pure Vector, 1.0=Pure BM25)
            
        Returns:
            List of combined results
        """
        # 1. Dense Search (Vector)
        dense_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2, # Fetch more for re-ranking
            include=['documents', 'metadatas', 'distances']
        )
        
        # Normalize Dense Scores (Cosine Distance: lower is better -> Convert to Similarity)
        dense_hits = {}
        if dense_results['ids'] and dense_results['ids'][0]:
            ids = dense_results['ids'][0]
            dists = dense_results['distances'][0]
            docs = dense_results['documents'][0]
            metas = dense_results['metadatas'][0]
            
            for i, doc_id in enumerate(ids):
                # Similarity = 1 - distance (approx)
                score = 1 - dists[i] 
                dense_hits[doc_id] = {
                    'score': score, 
                    'doc': docs[i], 
                    'meta': metas[i]
                }
        
        # 2. Sparse Search (BM25)
        bm25_hits = {}
        if self.bm25:
            tokenized_query = query_text.lower().split()
            # Get top N scores
            doc_scores = self.bm25.get_scores(tokenized_query)
            top_n_indices = np.argsort(doc_scores)[::-1][:top_k * 2]
            
            # Normalize BM25 scores (MinMax)
            # Avoid division by zero
            max_score = np.max(doc_scores) if len(doc_scores) > 0 else 1
            if max_score > 0:
                normalized_scores = doc_scores / max_score
            else:
                normalized_scores = doc_scores

            for idx in top_n_indices:
                if idx < len(self.doc_map):
                    doc_id = self.doc_map[idx]
                    bm25_hits[doc_id] = normalized_scores[idx]
        
        # 3. Fusion using Weighted Sum
        # Combined Score = (1-alpha) * Vector_Score + alpha * BM25_Score
        all_ids = set(dense_hits.keys()) | set(bm25_hits.keys())
        combined_results = []
        
        for doc_id in all_ids:
            # Get scores (default to 0 if not found)
            dense_score = dense_hits.get(doc_id, {}).get('score', 0.0)
            bm25_score = bm25_hits.get(doc_id, 0.0)
            
            final_score = ((1 - alpha) * dense_score) + (alpha * bm25_score)
            
            # Get content (prefer dense hit data as we need metadata)
            # If item only in BM25, we need to fetch it (Optimization: In real DB we'd fetch by ID)
            # Here for simplicity, we skip if we don't have the doc content from dense hits
            # Ideally: ChromaDB get([doc_id])
            
            if doc_id in dense_hits:
                combined_results.append({
                    'id': doc_id,
                    'document': dense_hits[doc_id]['doc'],
                    'metadata': dense_hits[doc_id]['meta'],
                    'score': final_score,
                    'dense_score': dense_score,
                    'bm25_score': bm25_score
                })
            else:
                 # Fetch missing content from ChromaDB
                try:
                    fetched = self.collection.get(ids=[doc_id])
                    if fetched['documents']:
                        combined_results.append({
                            'id': doc_id,
                            'document': fetched['documents'][0],
                            'metadata': fetched['metadatas'][0],
                            'score': final_score,
                            'dense_score': dense_score,
                            'bm25_score': bm25_score
                        })
                except Exception as e:
                    logger.warning(f"Could not fetch doc {doc_id} for hybrid search: {e}")

        # Sort by final score
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        return combined_results[:top_k]

    def query_by_text(self, query_text: str, embedder, top_k: int = 5,
                     filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query using Hybrid Search.
        """
        # Generate query embedding
        query_embedding = embedder.embed_text(query_text)
        
        # Use Hybrid Search
        hybrid_results = self.query_hybrid(query_text, query_embedding.tolist(), top_k=top_k)
        
        # Format results
        formatted_results = []
        for res in hybrid_results:
            result = {
                'id': res['id'],
                'document': res['document'],
                'metadata': res['metadata'],
                'distance': 1 - res['score'] # Convert back to distance-like for compatibility
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def delete_collection(self) -> None:
        """Delete collection and BM25 index."""
        self.client.delete_collection(name=self.collection_name)
        # Delete BM25 index file
        if self.bm25_path.exists():
            os.remove(self.bm25_path)
            self.bm25 = None
            self.bm25_corpus = []
            self.doc_map = []
        logger.info(f"Deleted collection: {self.collection_name}")
    
    def get_collection_count(self) -> int:
        """
        Get the number of documents in the collection.
        
        Returns:
            Number of documents
        """
        count = self.collection.count()
        logger.info(f"Collection contains {count} documents")
        return count
    
    def reset_collection(self) -> None:
        """
        Reset the collection (delete and recreate).
        """
        self.delete_collection()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric}
        )
        logger.info(f"Reset collection: {self.collection_name}")


if __name__ == "__main__":
    # Example usage
    from embedder import Embedder
    
    vector_store = VectorStore()
    embedder = Embedder()
    
    # Sample chunks with embeddings
    sample_chunks = [
        {
            'text': 'Machine learning is a subset of artificial intelligence.',
            'source': 'ml_doc.pdf',
            'page': 1
        },
        {
            'text': 'Natural language processing deals with text and speech.',
            'source': 'nlp_doc.pdf',
            'page': 1
        }
    ]
    
    # Add embeddings to chunks
    chunks_with_embeddings = embedder.embed_chunks(sample_chunks)
    
    # Add to vector store
    vector_store.add_documents(chunks_with_embeddings)
    
    # Query
    query_text = "What is machine learning?"
    results = vector_store.query_by_text(query_text, embedder, top_k=2)
    
    print(f"\nQuery: {query_text}")
    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. Document: {result['document'][:100]}...")
        print(f"   Metadata: {result['metadata']}")
        print(f"   Similarity Score: {1 - result['distance']:.4f}")
    
    # Get collection count
    count = vector_store.get_collection_count()
    print(f"\nTotal documents in collection: {count}")


