# OCR-RAG Pipeline - Microservices Architecture

## 🎯 Problem: Dependency Conflicts

OCR and RAG have conflicting Python dependencies:
- **OCR** needs: paddleocr, pyannote.audio, opencv-python, etc.
- **RAG** needs: langchain==0.1.0, chromadb==0.4.22, etc.
- These dependencies conflict and cannot coexist in the same environment

## ✅ Solution: Microservices Architecture

Run OCR and RAG as **separate services** in **separate environments**:

```
┌─────────────────┐
│  Streamlit App  │  (Lightweight - just HTTP calls)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│  OCR  │ │  RAG  │  (Separate Python environments)
│Service│ │Service│
└───────┘ └───────┘
```

### Benefits:
✅ **No dependency conflicts** - each service has its own environment  
✅ **Independent scaling** - scale OCR and RAG separately  
✅ **Easy deployment** - deploy services independently  
✅ **Better isolation** - failures don't cascade  
✅ **Technology flexibility** - use different Python versions if needed

## 🚀 Quick Start

### Option 1: Automated Startup (Recommended)

**Windows:**
```bash
cd pipeline
start_services.bat
```

**Linux/Mac:**
```bash
cd pipeline
chmod +x start_services.sh
./start_services.sh
```

This will:
1. Start OCR service on port 8000
2. Start RAG service on port 8001
3. Launch Streamlit app

### Option 2: Manual Setup

#### Step 1: Setup OCR Environment

```bash
# Create OCR environment
python -m venv venv_ocr
source venv_ocr/bin/activate  # Windows: venv_ocr\Scripts\activate

# Install OCR dependencies
cd OCR
pip install -r requirements.txt
pip install fastapi uvicorn

# Start OCR service
cd ..
uvicorn pipeline.ocr_service:app --port 8000
```

#### Step 2: Setup RAG Environment (New Terminal)

```bash
# Create RAG environment
python -m venv venv_rag
source venv_rag/bin/activate  # Windows: venv_rag\Scripts\activate

# Install RAG dependencies
cd RAG
pip install -r requirements.txt
pip install fastapi uvicorn

# Start Ollama
ollama serve  # In another terminal
ollama pull qwen3:1.7b

# Start RAG service
cd ..
uvicorn pipeline.rag_service:app --port 8001
```

#### Step 3: Start Streamlit App (New Terminal)

```bash
# Lightweight environment - just needs streamlit and requests
python -m venv venv_app
source venv_app/bin/activate  # Windows: venv_app\Scripts\activate

pip install streamlit requests

cd pipeline
streamlit run app_microservices.py
```

## 📁 File Structure

```
pipeline/
├── ocr_service.py          # OCR microservice (FastAPI)
├── rag_service.py          # RAG microservice (FastAPI)
├── orchestrator_api.py     # API-based orchestrator
├── app_microservices.py    # Streamlit app (uses APIs)
├── start_services.bat      # Windows startup script
├── start_services.sh       # Linux/Mac startup script
└── MICROSERVICES_GUIDE.md  # This file
```

## 🔌 API Endpoints

### OCR Service (Port 8000)

- `GET /` - Service info
- `GET /health` - Health check
- `POST /extract` - Extract text from file
- `GET /formats` - List supported formats

### RAG Service (Port 8001)

- `GET /` - Service info
- `GET /health` - Health check
- `POST /ingest` - Ingest document
- `POST /query` - Query documents
- `GET /stats` - Get statistics
- `DELETE /reset` - Reset database

## 🧪 Testing the Services

### Test OCR Service:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/formats
```

### Test RAG Service:
```bash
curl http://localhost:8001/health
curl http://localhost:8001/stats
```

### Test via Python:
```python
from pipeline.orchestrator_api import APIOrchestrator

# Initialize
orch = APIOrchestrator()

# Process a file
result = orch.process_file("document.pdf", ingest_to_rag=True)
print(f"Extracted {len(result['text'])} characters")

# Query
answer = orch.query("What is this about?")
print(answer['answer'])
```

## 🐳 Docker Deployment (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  ocr:
    build: ./OCR
    ports:
      - "8000:8000"
    command: uvicorn pipeline.ocr_service:app --host 0.0.0.0 --port 8000
  
  rag:
    build: ./RAG
    ports:
      - "8001:8001"
    command: uvicorn pipeline.rag_service:app --host 0.0.0.0 --port 8001
  
  app:
    build: ./pipeline
    ports:
      - "8501:8501"
    depends_on:
      - ocr
      - rag
    command: streamlit run app_microservices.py
```

Run: `docker-compose up`

## 🔧 Troubleshooting

**Services won't start:**
- Check if ports 8000/8001 are already in use
- Verify dependencies are installed in each environment

**Connection errors:**
- Ensure all services are running
- Check firewall settings
- Verify URLs in Streamlit sidebar

**OCR extraction fails:**
- Check OCR service logs
- Ensure file format is supported
- Verify Tesseract/FFmpeg are installed

**RAG queries fail:**
- Ensure Ollama is running: `ollama serve`
- Check if model is available: `ollama list`
- Verify documents are ingested first

## 📊 Monitoring

View service logs:
- OCR Service: Check terminal running port 8000
- RAG Service: Check terminal running port 8001
- Streamlit App: Check terminal running streamlit

Access API docs:
- OCR: http://localhost:8000/docs
- RAG: http://localhost:8001/docs

