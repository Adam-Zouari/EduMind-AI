# 🚀 OCR-RAG Pipeline - Complete Setup Guide

## ✅ What's Been Done

I've created **3 separate virtual environments** to avoid dependency conflicts:

1. **`venv_ocr`** - For OCR service (✅ INSTALLED)
2. **`venv_rag`** - For RAG service (⏳ PENDING)
3. **`venv_main`** - For Streamlit app (⏳ PENDING)

## 📋 Next Steps - Complete Installation

### Step 1: Install RAG Dependencies

Open a **NEW terminal** and run:

```powershell
cd C:\Users\ademz\Desktop\9raya\MLOps\Project
venv_rag\Scripts\activate
cd RAG
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
```

### Step 2: Install Main App Dependencies

Open **ANOTHER NEW terminal** and run:

```powershell
cd C:\Users\ademz\Desktop\9raya\MLOps\Project
venv_main\Scripts\activate
pip install streamlit requests
```

### Step 3: Start Ollama

Open **ANOTHER terminal** and run:

```powershell
ollama serve
```

Then in a separate terminal:

```powershell
ollama pull qwen3:1.7b
```

## 🎯 How to Run the Pipeline

Once all dependencies are installed, you need **4 terminals**:

### Terminal 1: OCR Service
```powershell
cd C:\Users\ademz\Desktop\9raya\MLOps\Project
venv_ocr\Scripts\activate
uvicorn pipeline.ocr_service:app --port 8000
```

### Terminal 2: RAG Service
```powershell
cd C:\Users\ademz\Desktop\9raya\MLOps\Project
venv_rag\Scripts\activate
uvicorn pipeline.rag_service:app --port 8001
```

### Terminal 3: Ollama (if not already running)
```powershell
ollama serve
```

### Terminal 4: Streamlit App
```powershell
cd C:\Users\ademz\Desktop\9raya\MLOps\Project
venv_main\Scripts\activate
cd pipeline
streamlit run app_microservices.py
```

## 🌐 Access the Services

Once all services are running:

- **Streamlit App**: http://localhost:8501
- **OCR Service API**: http://localhost:8000/docs
- **RAG Service API**: http://localhost:8001/docs

## 🔍 Verify Services

Check if services are running:

```powershell
# Check OCR
curl http://localhost:8000/health

# Check RAG
curl http://localhost:8001/health
```

## 📁 Project Structure

```
Project/
├── venv_ocr/          # OCR virtual environment
├── venv_rag/          # RAG virtual environment  
├── venv_main/         # Main app virtual environment
├── OCR/               # OCR code
├── RAG/               # RAG code
└── pipeline/          # Integration layer
    ├── ocr_service.py         # OCR microservice
    ├── rag_service.py         # RAG microservice
    ├── orchestrator_api.py    # API orchestrator
    └── app_microservices.py   # Streamlit UI
```

## ⚡ Quick Start (Automated)

Alternatively, use the startup script:

```powershell
cd pipeline
.\start_services.bat
```

This will start all services automatically.

## 🐛 Troubleshooting

**Port already in use:**
```powershell
# Kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Module not found:**
- Make sure you activated the correct virtual environment
- Check that dependencies are installed

**Ollama not running:**
```powershell
ollama serve
```

## 📝 Notes

- Each service runs in its own virtual environment to avoid dependency conflicts
- OCR and RAG have conflicting dependencies (numpy, protobuf, torch versions)
- The microservices architecture solves this by isolating each component
- The Streamlit app only needs `streamlit` and `requests` - very lightweight!

## 🎉 Usage

1. Open http://localhost:8501 in your browser
2. Click "🚀 Connect to Services" in the sidebar
3. Upload files in the "Upload & Process" tab
4. Ask questions in the "Ask Questions" tab
5. View history and processed files in other tabs

## 📚 Documentation

- **Microservices Guide**: `pipeline/MICROSERVICES_GUIDE.md`
- **Pipeline README**: `pipeline/README.md`
- **RAG README**: `RAG/README.md`

