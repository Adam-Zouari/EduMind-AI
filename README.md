# MLOps Project

This repository contains multiple components for the MLOps project.

## 📁 Project Structure

```
Project/
├── pipeline/              # 🚀 UNIFIED OCR-RAG PIPELINE (RECOMMENDED)
│   ├── orchestrator.py    # Orchestrates OCR + RAG
│   ├── app.py             # Streamlit web interface
│   ├── requirements.txt   # All dependencies
│   ├── run_app.bat        # Windows launcher
│   └── README.md          # Pipeline documentation
│
├── OCR/                   # OCR Extraction System
│   ├── core/              # Core pipeline and extractors
│   ├── extractors/        # Format-specific extractors (PDF, DOCX, etc.)
│   ├── processors/        # Text cleaning and processing
│   ├── utils/             # Utilities and logging
│   ├── examples/          # Example files
│   └── requirements.txt   # OCR dependencies
│
├── RAG/                   # RAG System (Standalone)
│   ├── src/               # Source code (embedder, vector store, etc.)
│   ├── config/            # Configuration files
│   ├── data/              # Data storage (raw, processed, vectordb)
│   ├── app.py             # Streamlit web interface
│   ├── requirements.txt   # RAG dependencies
│   ├── README.md          # Full RAG documentation
│   ├── run_app.bat        # Windows launcher
│   └── run_app.sh         # Linux/Mac launcher
│
├── venv/                  # Python virtual environment (shared)
└── tests/                 # Test files

```

## 🚀 Quick Start

### 🎯 Unified Pipeline (Recommended)

The **pipeline** folder combines OCR extraction and RAG into a single, easy-to-use interface.

**Run the complete pipeline:**
```bash
# Windows
cd pipeline
run_app.bat

# Or directly
streamlit run pipeline/app.py
```

**Features:**
- 📤 Upload any file format (PDF, DOCX, Images, Audio, Video, HTML)
- 🔍 Automatic text extraction with OCR
- 💬 Ask questions and get AI-powered answers
- 📊 Track processed files and chat history

### Individual Components

#### OCR System (`/OCR`)
Extract text from various file formats:
```python
from OCR.core.pipeline import DataIngestionPipeline

pipeline = DataIngestionPipeline()
result = pipeline.process_file("document.pdf")
print(result.text)
```

#### RAG System (`/RAG`)
Standalone RAG for pre-extracted text:
```bash
cd RAG
run_app.bat  # Windows
```

## 📚 Components

### 1. Unified Pipeline (`/pipeline`) ⭐ RECOMMENDED
- **Purpose:** Complete OCR-to-RAG solution with web interface
- **Tech Stack:** Python, Streamlit, OCR Pipeline, ChromaDB, Ollama (Qwen)
- **Features:**
  - Multi-format file upload (PDF, DOCX, Images, Audio, Video, HTML)
  - Automatic text extraction and cleaning
  - Semantic search and AI-powered Q&A
  - Batch processing
  - Chat history and source attribution

**Key Features:**
- ✅ Upload and process any file format
- ✅ Automatic OCR extraction
- ✅ Intelligent text chunking and embedding
- ✅ Vector database storage
- ✅ Natural language queries
- ✅ AI-generated answers with sources

### 2. OCR System (`/OCR`)
- **Purpose:** Extract text from various file formats
- **Tech Stack:** PyMuPDF, python-docx, Tesseract, Whisper, BeautifulSoup
- **Supported Formats:**
  - PDF (PyMuPDF)
  - DOCX (python-docx)
  - Images (Tesseract OCR)
  - Audio (Whisper)
  - Video (Whisper + FFmpeg)
  - HTML (BeautifulSoup)

### 3. RAG System (`/RAG`)
- **Purpose:** Standalone RAG for pre-extracted text
- **Tech Stack:** Python, ChromaDB, Sentence Transformers, Ollama (Qwen)
- **Features:**
  - Load OCR JSON files
  - Semantic search
  - AI-powered Q&A

## 🛠️ Setup

### Prerequisites
- Python 3.10+
- Ollama (for RAG system)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Project
```

2. **Create virtual environment** (if not exists)
```bash
python -m venv venv
```

3. **Activate virtual environment**
```bash
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. **Install dependencies**
```bash
# For unified pipeline (recommended)
cd pipeline
pip install -r requirements.txt

# Or for individual components
cd OCR
pip install -r requirements.txt

cd ../RAG
pip install -r requirements.txt
```

5. **Install Ollama (for AI features)**
- Download from: https://ollama.ai/download
- Start Ollama: `ollama serve`
- Pull model: `ollama pull qwen3:1.7b`

6. **Run the pipeline**
```bash
# Unified pipeline
streamlit run pipeline/app.py

# Or use the launcher
cd pipeline
run_app.bat  # Windows
```

## 📖 Documentation

Each component has its own detailed README:
- **Unified Pipeline:** `pipeline/README.md` ⭐
- **OCR System:** `OCR/README.md` (coming soon)
- **RAG System:** `RAG/README.md`

## 🎯 Use Cases

### Use the Unified Pipeline when:
- You want to upload files and ask questions immediately
- You need to process multiple file formats
- You want a complete end-to-end solution

### Use OCR System standalone when:
- You only need text extraction
- You want to integrate OCR into your own pipeline
- You need programmatic access to extraction

### Use RAG System standalone when:
- You already have extracted text
- You want to work with JSON files from OCR
- You need a lightweight Q&A system

## 🤝 Contributing

Please read the individual component documentation before contributing.

## 📄 License

[Add your license here]

