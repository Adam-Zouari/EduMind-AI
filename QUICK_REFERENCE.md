# 🚀 Quick Technical Reference

## 📊 Technologies at a Glance

### **AI/ML Models**
| Component | Technology | Size | Purpose |
|-----------|-----------|------|---------|
| **LLM** | Qwen 3 | 1.7B params | Answer generation |
| **Embeddings** | all-MiniLM-L6-v2 | 384 dims | Semantic search |
| **OCR** | Tesseract 5.x | - | Image text extraction |
| **Speech-to-Text** | Whisper (base) | 74M params | Audio transcription |

### **Databases**
| Type | Technology | Purpose |
|------|-----------|---------|
| **Vector DB** | ChromaDB 0.4.22 | Store embeddings |
| **Persistence** | SQLite | ChromaDB backend |

### **Key Algorithms**
| Algorithm | Implementation | Purpose |
|-----------|---------------|---------|
| **Text Chunking** | RecursiveCharacterTextSplitter | Split documents |
| **Similarity Search** | Cosine Similarity + HNSW | Find relevant docs |
| **RAG** | Retrieval + LLM Generation | Answer questions |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                         │
│                    (venv_main, Port 8501)               │
└────────────────┬────────────────────────────────────────┘
                 │ HTTP REST API
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐  ┌──────────────┐
│ OCR Service  │  │ RAG Service  │
│ (venv_ocr)   │  │ (venv_rag)   │
│ Port: 8000   │  │ Port: 8001   │
└──────┬───────┘  └──────┬───────┘
       │                 │
       ▼                 ▼
  Text Extract     Vector Store
  (Tesseract,      (ChromaDB +
   Whisper,         Embeddings +
   PyMuPDF)         Qwen LLM)
```

---

## 🔧 Configuration

**File**: `RAG/config/config.yaml`

```yaml
# Embedding
embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  embedding_dim: 384

# Chunking
chunking:
  chunk_size: 1000
  chunk_overlap: 200

# Vector DB
vectordb:
  collection_name: "ocr_documents"
  distance_metric: "cosine"

# LLM
llm:
  model_name: "qwen3:1.7b"
  temperature: 0.7
  max_tokens: 2048

# RAG
rag:
  top_k: 5
  score_threshold: 0.5
```

---

## 📈 Performance Metrics

| Operation | Speed | Memory |
|-----------|-------|--------|
| Embedding | ~14K sentences/sec | ~500MB |
| Vector Search | <100ms (10K docs) | - |
| LLM Generation | ~20-50 tokens/sec | ~2GB |
| PDF OCR | ~1-2 pages/sec | - |
| Image OCR | ~2-5 sec/image | - |
| Audio Transcription | ~0.1x realtime | ~1GB |

---

## 🎯 Key Features

### **OCR Capabilities**
- ✅ PDF (PyMuPDF + pdfplumber)
- ✅ DOCX (python-docx)
- ✅ Images (Tesseract)
- ✅ Audio (Whisper)
- ✅ Video (Whisper + FFmpeg)
- ✅ Web (Trafilatura + Newspaper)

### **RAG Capabilities**
- ✅ Semantic search (not keyword)
- ✅ Source attribution
- ✅ Metadata filtering
- ✅ Persistent storage
- ✅ Batch processing

---

## 🔍 How It Works

### **Document Processing**
```
Upload → OCR Extract → Clean → Chunk → Embed → Store
```

### **Question Answering**
```
Query → Embed → Search → Retrieve → LLM → Answer + Sources
```

---

## 📚 Main Libraries

**Backend**: FastAPI, Uvicorn, Streamlit  
**ML**: sentence-transformers, openai-whisper, torch  
**OCR**: pytesseract, PyMuPDF, pdfplumber, python-docx  
**Vector DB**: ChromaDB  
**LLM**: Ollama (Qwen)  
**Text Processing**: LangChain, NLTK  

---

## 🚀 Quick Start

```powershell
# Start all services
.\start_all_services.bat

# Or manually:
# Terminal 1 - OCR
python -m uvicorn pipeline.ocr_service:app --port 8000

# Terminal 2 - RAG
python -m uvicorn pipeline.rag_service:app --port 8001

# Terminal 3 - UI
streamlit run pipeline/app_microservices.py
```

---

## 📖 Full Documentation

See **`TECHNICAL_DOCUMENTATION.md`** for complete details on:
- Detailed model specifications
- Algorithm explanations
- Architecture patterns
- Design decisions
- Scalability options
- Security considerations

