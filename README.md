# MLOps Project

This repository contains multiple components for the MLOps project.

## 📁 Project Structure

```
Project/
├── RAG/                    # OCR-to-RAG Pipeline System
│   ├── src/               # Source code (embedder, vector store, etc.)
│   ├── config/            # Configuration files
│   ├── data/              # Data storage (raw, processed, vectordb)
│   ├── app.py             # Streamlit web interface
│   ├── requirements.txt   # Python dependencies
│   ├── README.md          # Full RAG documentation
│   ├── run_app.bat        # Windows launcher
│   └── run_app.sh         # Linux/Mac launcher
│
├── venv/                  # Python virtual environment (shared)
└── tests/                 # Test files

```

## 🚀 Quick Start

### RAG Pipeline System

The RAG (Retrieval-Augmented Generation) system converts OCR-extracted text into a searchable knowledge base with AI-powered answers.

**Navigate to RAG folder:**
```bash
cd RAG
```

**Run the Streamlit interface:**
```bash
# Windows
run_app.bat

# Linux/Mac
chmod +x run_app.sh
./run_app.sh
```

**Or see the full documentation:**
```bash
cd RAG
cat README.md
```

## 📚 Components

### 1. RAG Pipeline (`/RAG`)
- **Purpose:** OCR text to searchable knowledge base with AI answers
- **Tech Stack:** Python, ChromaDB, Sentence Transformers, Ollama (Qwen)
- **Features:**
  - Streamlit web interface
  - Semantic search using embeddings
  - AI-powered Q&A with source attribution
  - Persistent vector database

**Key Features:**
- ✅ Load OCR JSON files
- ✅ Chunk and embed text
- ✅ Store in vector database
- ✅ Query with natural language
- ✅ Generate AI answers with Qwen

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

4. **Install dependencies for RAG**
```bash
cd RAG
pip install -r requirements.txt
```

5. **Install Ollama (for RAG)**
- Download from: https://ollama.ai/download
- Start Ollama: `ollama serve`
- Pull model: `ollama pull qwen3:1.7b`

## 📖 Documentation

Each component has its own detailed README:
- **RAG System:** `RAG/README.md`

## 🤝 Contributing

Please read the individual component documentation before contributing.

## 📄 License

[Add your license here]

