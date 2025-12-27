"""
OCR Microservice
Runs OCR extraction as a separate FastAPI service
Install: pip install fastapi uvicorn
Run from project root: uvicorn pipeline.ocr_service:app --port 8000
"""

import sys
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import json

# Setup paths - must be run from project root
project_root = Path(__file__).parent.parent.resolve()
ocr_dir = project_root / "OCR"

# Change to OCR directory for imports to work
original_cwd = os.getcwd()
os.chdir(str(ocr_dir))
sys.path.insert(0, str(ocr_dir))

from core.pipeline import DataIngestionPipeline

# Change back to original directory
os.chdir(original_cwd)

app = FastAPI(title="OCR Service", version="1.0.0")
ocr_pipeline = DataIngestionPipeline()

@app.get("/")
def root():
    return {"service": "OCR Extraction Service", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from uploaded file
    Returns: JSON with extracted text and metadata
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Process with OCR
        result = ocr_pipeline.process_file(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        # Return result
        return JSONResponse(content={
            "success": result.success,
            "text": result.text,
            "metadata": result.metadata,
            "format_type": result.format_type,
            "extraction_time": result.extraction_time
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/formats")
def supported_formats():
    """List supported file formats"""
    return {
        "formats": ["pdf", "docx", "png", "jpg", "jpeg", "html", "mp3", "wav", "mp4", "avi"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

