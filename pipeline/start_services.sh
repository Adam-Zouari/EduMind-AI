#!/bin/bash
# Start OCR and RAG microservices

echo "========================================"
echo "Starting OCR-RAG Microservices"
echo "========================================"
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found!"
    exit 1
fi

# Check if uvicorn is installed
if ! python3 -c "import uvicorn" &> /dev/null; then
    echo "ERROR: uvicorn not installed!"
    echo "Install with: pip install fastapi uvicorn"
    exit 1
fi

echo "Starting services..."
echo ""

# Start OCR Service
echo "[1/2] Starting OCR Service on port 8000..."
cd "$PROJECT_ROOT"
python3 -m uvicorn pipeline.ocr_service:app --port 8000 &
OCR_PID=$!
sleep 2

# Start RAG Service
echo "[2/2] Starting RAG Service on port 8001..."
python3 -m uvicorn pipeline.rag_service:app --port 8001 &
RAG_PID=$!
sleep 2

echo ""
echo "========================================"
echo "Services started!"
echo "========================================"
echo ""
echo "OCR Service: http://localhost:8000 (PID: $OCR_PID)"
echo "RAG Service: http://localhost:8001 (PID: $RAG_PID)"
echo ""
echo "Starting Streamlit App..."
echo ""

# Start Streamlit App
cd "$SCRIPT_DIR"
streamlit run app_microservices.py

# Cleanup on exit
echo ""
echo "Stopping services..."
kill $OCR_PID $RAG_PID 2>/dev/null
echo "Done!"

