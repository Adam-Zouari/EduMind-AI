"""
RAG Microservice
Runs RAG pipeline as a separate FastAPI service
Install: pip install fastapi uvicorn
Run from project root: uvicorn pipeline.rag_service:app --port 8001
"""

import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Setup paths - must be run from project root
project_root = Path(__file__).parent.parent.resolve()
rag_dir = project_root / "RAG"

# Change to RAG directory for imports to work
original_cwd = os.getcwd()
os.chdir(str(rag_dir))
sys.path.insert(0, str(rag_dir))

from src.rag_pipeline import RAGPipeline

# Change back to original directory
os.chdir(original_cwd)

app = FastAPI(title="RAG Service", version="1.0.0")

# Initialize RAG pipeline
config_path = str(rag_dir / "config" / "config.yaml")
rag_pipeline = RAGPipeline(config_path=config_path, use_llm=True)

class IngestRequest(BaseModel):
    text: str
    metadata: Dict[str, Any]

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    generate_answer: bool = True

@app.get("/")
def root():
    return {"service": "RAG Service", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ingest")
def ingest_document(request: IngestRequest):
    """Ingest a document into the RAG system"""
    try:
        # RAG pipeline expects a dictionary with 'text' and metadata fields
        document = {
            "text": request.text,
            **request.metadata
        }
        result = rag_pipeline.ingest_document(document)
        return {"success": True, "chunks": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query_documents(request: QueryRequest):
    """Query the RAG system"""
    try:
        if request.generate_answer:
            result = rag_pipeline.generate_answer(
                query=request.query,
                top_k=request.top_k
            )
        else:
            result = rag_pipeline.query(
                query_text=request.query,
                top_k=request.top_k
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_stats():
    """Get RAG system statistics"""
    try:
        return rag_pipeline.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/reset")
def reset_database():
    """Reset the vector database"""
    try:
        rag_pipeline.reset_database()
        return {"success": True, "message": "Database reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

