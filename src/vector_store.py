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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Handles vector database operations using ChromaDB.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the VectorStore with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        vectordb_config = self.config['vectordb']
        self.collection_name = vectordb_config['collection_name']
        self.persist_directory = vectordb_config['persist_directory']
        self.distance_metric = vectordb_config['distance_metric']
        
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
        
        logger.info(f"VectorStore initialized with collection: {self.collection_name}")
        logger.info(f"Persist directory: {self.persist_directory}")
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'text', 'embedding', and metadata
        """
        if not chunks:
            logger.warning("No chunks provided to add")
            return
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            # Generate unique ID
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            
            # Extract embedding
            embedding = chunk.get('embedding', [])
            if isinstance(embedding, list):
                embeddings.append(embedding)
            else:
                embeddings.append(embedding.tolist())
            
            # Extract text
            documents.append(chunk.get('text', ''))
            
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
        
        logger.info(f"Added {len(chunks)} documents to vector store")
    
    def query(self, query_embedding: List[float], top_k: int = 5, 
              filter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_embedding: The query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            Dictionary containing query results
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )
        
        logger.info(f"Query returned {len(results['documents'][0])} results")
        return results
    
    def query_by_text(self, query_text: str, embedder, top_k: int = 5,
                     filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query the vector store using text (will generate embedding).
        
        Args:
            query_text: The query text
            embedder: Embedder instance to generate query embedding
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of result dictionaries
        """
        # Generate query embedding
        query_embedding = embedder.embed_text(query_text)
        
        # Query the vector store
        results = self.query(query_embedding.tolist(), top_k, filter_metadata)
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            result = {
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def delete_collection(self) -> None:
        """
        Delete the current collection.
        """
        self.client.delete_collection(name=self.collection_name)
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
        print(f"   Distance: {result['distance']}")
    
    # Get collection count
    count = vector_store.get_collection_count()
    print(f"\nTotal documents in collection: {count}")

