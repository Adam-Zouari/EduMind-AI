# 🚀 How to Run the OCR-RAG Pipeline

## ✅ Status: All Dependencies Installed

All three virtual environments are ready:
- ✅ **venv_ocr** - OCR service dependencies
- ✅ **venv_rag** - RAG service dependencies  
- ✅ **venv_main** - Streamlit app dependencies

---

## 🔧 Important Fixes Applied

I've fixed the following issues:

### 1. **Import Error Fixed**
- Updated `pipeline/__init__.py` to not auto-import the old orchestrator
- Services now run independently without import conflicts

### 2. **libmagic Error Fixed**
- Made `python-magic` optional in `OCR/core/format_detector.py`
- OCR will fall back to extension-based detection if libmagic is not available
- This is acceptable for most use cases

### 3. **Service Startup Fixed**
- Services now properly change to their respective directories for imports
- Updated startup scripts with correct commands

---

## 🚀 How to Run (Choose One Method)

### **Method 1: Automated Startup (Easiest)**

Simply double-click or run:

```powershell
start_all_services.bat
```

This will open 4 windows:
1. OCR Service (port 8000)
2. RAG Service (port 8001)
3. Ollama Server (if not running)
4. Streamlit App (port 8501)

**Wait 10-15 seconds** for all services to start, then open your browser to:
- **http://localhost:8501** (Streamlit App)

---

### **Method 2: Manual Startup (More Control)**

Open **4 separate PowerShell/CMD terminals**:

#### **Terminal 1 - OCR Service:**
```powershell
cd C:\Users\ademz\Desktop\9raya\MLOps\Project
call venv_ocr\Scripts\activate.bat
python -m uvicorn pipeline.ocr_service:app --port 8000
```

Wait for: `Uvicorn running on http://0.0.0.0:8000`

#### **Terminal 2 - RAG Service:**
```powershell
cd C:\Users\ademz\Desktop\9raya\MLOps\Project
call venv_rag\Scripts\activate.bat
python -m uvicorn pipeline.rag_service:app --port 8001
```

Wait for: `Uvicorn running on http://0.0.0.0:8001`

#### **Terminal 3 - Ollama:**
```powershell
ollama serve
```

Wait for: `Listening on 127.0.0.1:11434`

#### **Terminal 4 - Streamlit:**
```powershell
cd C:\Users\ademz\Desktop\9raya\MLOps\Project
call venv_main\Scripts\activate.bat
cd pipeline
streamlit run app_microservices.py
```

Wait for: `You can now view your Streamlit app in your browser.`

---

## 🌐 Access the Application

Once all services are running:

| Service | URL | Purpose |
|---------|-----|---------|
| **Streamlit App** | http://localhost:8501 | Main web interface |
| **OCR API Docs** | http://localhost:8000/docs | OCR service API |
| **RAG API Docs** | http://localhost:8001/docs | RAG service API |
| **Ollama** | http://localhost:11434 | LLM server |

---

## 📋 How to Use the App

1. **Open** http://localhost:8501 in your browser

2. **Connect to Services** (in sidebar):
   - Click "🚀 Connect to Services"
   - Default URLs should work automatically

3. **Upload Files** (Upload & Process tab):
   - Click "Browse files" or drag & drop
   - Supported: PDF, DOCX, PNG, JPG, MP3, WAV, MP4, HTML
   - Files are automatically processed through OCR and stored in RAG

4. **Ask Questions** (Ask Questions tab):
   - Type your question about the uploaded documents
   - Click "Get Answer"
   - View AI-generated answer with source citations

5. **View History**:
   - **Chat History** tab: See all previous Q&A
   - **Processed Files** tab: See all uploaded documents

---

## 🔍 Verify Services Are Running

### Check Service Health:

```powershell
# Check OCR
curl http://localhost:8000/health

# Check RAG  
curl http://localhost:8001/health

# Check Ollama
curl http://localhost:11434/api/tags
```

Expected responses:
- OCR/RAG: `{"status":"healthy"}`
- Ollama: JSON with model list

---

## 🐛 Troubleshooting

### **Service won't start**

**Error**: `ModuleNotFoundError` or import errors
- **Solution**: Make sure you're running from the project root directory
- **Check**: `cd C:\Users\ademz\Desktop\9raya\MLOps\Project`

**Error**: `Address already in use`
- **Solution**: Kill the process using the port
```powershell
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### **Ollama not running**

```powershell
# Start Ollama
ollama serve

# In another terminal, pull the model
ollama pull qwen3:1.7b
```

### **Streamlit shows "Service Unavailable"**

- Make sure OCR and RAG services are running first
- Check the service URLs in the sidebar
- Click "🚀 Connect to Services" again

---

## 📊 Architecture

```
User Browser (localhost:8501)
    ↓
Streamlit App (venv_main)
    ↓ HTTP API calls
    ├─→ OCR Service (venv_ocr, port 8000)
    │   └─→ Extracts text from files
    │
    └─→ RAG Service (venv_rag, port 8001)
        ├─→ Stores text in ChromaDB
        └─→ Queries Ollama LLM (qwen3:1.7b)
```

**Key Benefits:**
- ✅ No dependency conflicts (each service has its own venv)
- ✅ Services communicate via HTTP (no import issues)
- ✅ Can scale each service independently
- ✅ Easy to debug (each service runs in its own window)

---

## 🎯 Next Steps

1. **Run the pipeline**: `start_all_services.bat`
2. **Upload a test document**: Try a PDF or DOCX file
3. **Ask questions**: Test the Q&A functionality
4. **Explore the API**: Visit http://localhost:8000/docs and http://localhost:8001/docs

---

## 📚 Additional Documentation

- **QUICK_START.txt** - Quick reference card
- **START_HERE.md** - Complete setup guide
- **INSTALLATION_COMPLETE.md** - Installation summary
- **pipeline/MICROSERVICES_GUIDE.md** - Architecture details

---

## 🎉 You're Ready!

Everything is set up and ready to go. Just run `start_all_services.bat` and start processing documents!

If you encounter any issues, check the terminal windows for error messages.

