"""
RAG Pipeline Module
End-to-end pipeline for OCR-to-RAG processing.
"""

from typing import List, Dict, Any, Optional
import yaml
import logging
from pathlib import Path

from .ocr_processor import OCRProcessor
from .text_chunker import TextChunker
from .embedder import Embedder
from .vector_store import VectorStore
from .llm_generator import OllamaGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete pipeline for processing OCR text and enabling RAG.
    """
    
    def __init__(self, config_path: str = "config/config.yaml", use_llm: bool = False):
        """
        Initialize the RAG Pipeline with all components.

        Args:
            config_path: Path to the configuration file
            use_llm: Whether to initialize LLM generator (default: False)
        """
        self.config_path = config_path

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize all components
        logger.info("Initializing RAG Pipeline components...")
        self.ocr_processor = OCRProcessor()
        self.text_chunker = TextChunker(config_path)
        self.embedder = Embedder(config_path)
        self.vector_store = VectorStore(config_path)

        # Initialize LLM generator if requested
        self.llm_generator = None
        if use_llm:
            llm_config = self.config.get('llm', {})
            model_name = llm_config.get('model_name', 'qwen2.5:1.5b')
            self.llm_generator = OllamaGenerator(
                model_name=model_name,
                temperature=llm_config.get('temperature', 0.7),
                max_tokens=llm_config.get('max_tokens', 2048)
            )
            logger.info(f"LLM Generator initialized with model: {model_name}")

        self.rag_config = self.config['rag']
        self.top_k = self.rag_config['top_k']
        self.score_threshold = self.rag_config['score_threshold']

        logger.info("RAG Pipeline initialized successfully")
    
    def ingest_document(self, document: Dict[str, Any]) -> int:
        """
        Ingest a single document into the RAG system.

        Args:
            document: Document dictionary with 'text' and metadata from OCR JSON

        Returns:
            Number of chunks created
        """
        text = document.get('text', '')
        metadata = {k: v for k, v in document.items() if k != 'text'}

        # Chunk the text
        chunks = self.text_chunker.chunk_text(text, metadata)

        # Generate embeddings
        chunks_with_embeddings = self.embedder.embed_chunks(chunks)

        # Store in vector database
        self.vector_store.add_documents(chunks_with_embeddings)

        logger.info(f"Ingested document: {len(chunks)} chunks created")
        return len(chunks)
    
    def ingest_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Ingest multiple documents into the RAG system.

        Args:
            documents: List of documents with 'text' and metadata from OCR JSON

        Returns:
            Total number of chunks created
        """
        total_chunks = 0

        for doc in documents:
            num_chunks = self.ingest_document(doc)
            total_chunks += num_chunks

        logger.info(f"Ingested {len(documents)} documents: {total_chunks} total chunks")
        return total_chunks

    def ingest_from_json(self, json_path: str) -> int:
        """
        Ingest documents from OCR JSON file.

        Args:
            json_path: Path to the JSON file from OCR system

        Returns:
            Total number of chunks created
        """
        documents = self.ocr_processor.load_from_json(json_path)
        return self.ingest_documents(documents)
    
    def query(self, query_text: str, top_k: Optional[int] = None, 
             filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query the RAG system.
        
        Args:
            query_text: The query text
            top_k: Number of results to return (uses config default if None)
            filter_metadata: Optional metadata filter
            
        Returns:
            List of relevant document chunks
        """
        if top_k is None:
            top_k = self.top_k
        
        results = self.vector_store.query_by_text(
            query_text, 
            self.embedder, 
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        # Filter by score threshold if using distance metric
        if results and 'distance' in results[0] and results[0]['distance'] is not None:
            results = [r for r in results if r['distance'] <= (1 - self.score_threshold)]
        
        logger.info(f"Query returned {len(results)} results")
        return results
    
    def generate_context(self, query_text: str, top_k: Optional[int] = None) -> str:
        """
        Generate context for RAG by retrieving relevant chunks.
        
        Args:
            query_text: The query text
            top_k: Number of results to return
            
        Returns:
            Concatenated context from retrieved chunks
        """
        results = self.query(query_text, top_k)
        
        if not results:
            return ""
        
        # Concatenate retrieved documents
        context_parts = []
        for i, result in enumerate(results):
            context_parts.append(f"[Document {i+1}]\n{result['document']}\n")
        
        context = "\n".join(context_parts)
        logger.info(f"Generated context with {len(results)} documents")
        return context
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_documents': self.vector_store.get_collection_count(),
            'embedding_model': self.embedder.model_name,
            'embedding_dimension': self.embedder.embedding_dim,
            'chunk_size': self.text_chunker.chunk_size,
            'chunk_overlap': self.text_chunker.chunk_overlap,
            'collection_name': self.vector_store.collection_name
        }
        return stats
    
    def reset(self) -> None:
        """
        Reset the vector store (delete all documents).
        """
        self.vector_store.reset_collection()
        logger.info("RAG Pipeline reset complete")

    def generate_answer(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an answer using RAG + LLM.

        Args:
            query: User's question
            top_k: Number of documents to retrieve (default: from config)
            filter_metadata: Optional metadata filter
            stream: Whether to stream the LLM response
            system_prompt: Optional custom system prompt

        Returns:
            Dictionary with 'answer', 'sources', and 'context'
        """
        if self.llm_generator is None:
            raise ValueError(
                "LLM generator not initialized. "
                "Create pipeline with use_llm=True: RAGPipeline(use_llm=True)"
            )

        # Retrieve relevant documents
        if top_k is None:
            top_k = self.top_k

        results = self.query(query, top_k, filter_metadata)

        if not results:
            return {
                'answer': "I couldn't find any relevant information to answer your question.",
                'sources': [],
                'context': ''
            }

        # Generate answer using LLM
        answer = self.llm_generator.generate_with_results(
            query=query,
            results=results,
            system_prompt=system_prompt,
            stream=stream
        )

        # Extract sources
        sources = []
        for result in results:
            metadata = result.get('metadata', {})
            sources.append({
                'source': metadata.get('source', 'Unknown'),
                'page': metadata.get('page', 'N/A'),
                'similarity': f"{(1 - result.get('distance', 1)) * 100:.1f}%"
            })

        # Get context
        context = self.generate_context(query, top_k)

        return {
            'answer': answer,
            'sources': sources,
            'context': context
        }


if __name__ == "__main__":
    # Example usage
    pipeline = RAGPipeline()

    # Example: Load from OCR JSON and ingest
    # Assuming your collaborator provides a JSON file like:
    # [
    #   {"text": "Document text here...", "source": "doc1.pdf", "page": 1},
    #   {"text": "More text...", "source": "doc2.pdf", "page": 1}
    # ]

    # total_chunks = pipeline.ingest_from_json('path/to/ocr_output.json')
    # print(f"Ingested documents: {total_chunks} total chunks")

    # Query the system
    # results = pipeline.query("Your question here", top_k=5)

    print("RAG Pipeline ready to ingest OCR JSON files")

