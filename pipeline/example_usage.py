"""
Example usage of the OCR-RAG Orchestrator
Demonstrates how to use the pipeline programmatically
"""

from orchestrator import OCRRAGOrchestrator
from pathlib import Path

def main():
    print("=" * 60)
    print("OCR-RAG Pipeline Example")
    print("=" * 60)
    
    # Initialize the orchestrator
    print("\n1. Initializing OCR-RAG Orchestrator...")
    orchestrator = OCRRAGOrchestrator(use_llm=True)
    print("✓ Orchestrator initialized!")
    
    # Example 1: Process a single file
    print("\n2. Processing a PDF file...")
    example_pdf = Path("../OCR/examples/Rebhi_Mohamed_Amine_Grades.pdf")
    
    if example_pdf.exists():
        result = orchestrator.process_file(
            file_path=example_pdf,
            ingest_to_rag=True,
            clean_text=True
        )
        
        print(f"✓ File processed successfully!")
        print(f"  - Format: {result['format_type']}")
        print(f"  - Extraction time: {result['extraction_time']:.2f}s")
        print(f"  - Text length: {len(result['text'])} characters")
        print(f"  - RAG chunks created: {result['rag_chunks']}")
        print(f"  - Text preview: {result['text'][:200]}...")
    else:
        print(f"⚠ Example file not found: {example_pdf}")
        print("  Please provide your own file path")
    
    # Example 2: Query the knowledge base
    print("\n3. Querying the knowledge base...")
    
    if orchestrator.rag_pipeline.get_stats()['total_documents'] > 0:
        query = "What is the student's name and average grade?"
        
        print(f"  Query: {query}")
        
        answer = orchestrator.query(
            query_text=query,
            top_k=3,
            generate_answer=True
        )
        
        print(f"\n  Answer: {answer['answer']}")
        print(f"\n  Sources:")
        for i, source in enumerate(answer['sources'], 1):
            print(f"    {i}. {source['source']} (Page {source['page']}) - {source['similarity']}")
    else:
        print("  ⚠ No documents in knowledge base. Process files first.")
    
    # Example 3: Get statistics
    print("\n4. Pipeline Statistics:")
    stats = orchestrator.get_stats()
    print(f"  - Total documents: {stats['rag']['total_documents']}")
    print(f"  - Embedding model: {stats['rag']['embedding_model']}")
    print(f"  - Chunk size: {stats['rag']['chunk_size']}")
    print(f"  - Available OCR extractors: {', '.join(stats['ocr_extractors'])}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Run the Streamlit app: streamlit run app.py")
    print("  - Process your own files")
    print("  - Ask questions about your documents")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. Ollama is running: ollama serve")
        print("  2. Qwen model is available: ollama pull qwen3:1.7b")
        print("  3. All dependencies are installed: pip install -r requirements.txt")

