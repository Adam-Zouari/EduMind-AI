# 🏗️ Technical Architecture & Technologies

## 📋 Table of Contents
- [System Overview](#system-overview)
- [Machine Learning Models](#machine-learning-models)
- [Databases & Storage](#databases--storage)
- [Algorithms & Techniques](#algorithms--techniques)
- [Technology Stack](#technology-stack)
- [Architecture Patterns](#architecture-patterns)

---

## 🎯 System Overview

This project implements an **end-to-end OCR-to-RAG pipeline** that combines:
1. **Multi-format document extraction** (OCR System)
2. **Semantic search and retrieval** (RAG System)
3. **AI-powered question answering** (LLM Integration)

**Architecture Pattern**: Microservices-based with HTTP API communication

```
┌─────────────────┐      ┌─────────────────┐        ┌─────────────────┐
│  OCR Service    │─────▶│  RAG Service    │─────▶ │  Streamlit UI   │
│  (Port 8000)    │      │  (Port 8001)    │        │  (Port 8501)    │
└─────────────────┘      └─────────────────┘        └─────────────────┘
        │                        │                         │
        ▼                        ▼                         ▼
   Text Extraction        Vector Storage            User Interface
```

---

## 🤖 Machine Learning Models

### 1. **Large Language Model (LLM)**

**Model**: Qwen 3 (1.7B parameters)
- **Type**: Small Language Model (SLM) - Optimized for efficiency
- **Provider**: Ollama (Local inference)
- **Purpose**: Generate natural language answers from retrieved context
- **Configuration**:
  - Temperature: 0.7 (balanced creativity/accuracy)
  - Max tokens: 2048
  - Context window: Supports long-form answers
- **API**: Ollama REST API (http://localhost:11434)
- **Alternative Models Supported**:
  - `gemma3:1b` (Google's Gemma)
  - `llama3.2:1b` (Meta's Llama)
  - `deepseek-r1:1.5b` (DeepSeek)

**Why Qwen 3?**
- Excellent performance for its size
- Fast inference on CPU
- Strong multilingual support
- Good instruction following

---

### 2. **Embedding Model**

**Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Type**: Sentence Transformer (BERT-based)
- **Architecture**: 6-layer MiniLM
- **Embedding Dimension**: 384
- **Purpose**: Convert text to dense vector representations
- **Performance**:
  - Speed: ~14,000 sentences/sec (CPU)
  - Quality: 68.06% on STS benchmark
- **Use Cases**:
  - Document embedding for storage
  - Query embedding for retrieval
  - Semantic similarity computation

**Technical Details**:
- Framework: Sentence-Transformers (HuggingFace)
- Pooling: Mean pooling
- Normalization: L2 normalized vectors
- Device: CPU (configurable to CUDA)

---

### 3. **OCR Models**

#### **Tesseract OCR** (Primary)
- **Type**: Traditional OCR engine
- **Version**: 5.x
- **Engine**: LSTM-based neural network
- **Languages**: Multi-language support (eng, fra, spa, deu, etc.)
- **Purpose**: Extract text from images
- **Configuration**:
  - Confidence threshold: 60%
  - PSM (Page Segmentation Mode): Auto-detect
  - OEM (OCR Engine Mode): LSTM neural network

#### **PaddleOCR** (Optional)
- **Type**: Deep learning OCR
- **Architecture**: CNN + RNN + Attention
- **Components**:
  - Text detection: DB (Differentiable Binarization)
  - Text recognition: CRNN
  - Angle classification: ResNet
- **Status**: Optional (not installed by default)

---

### 4. **Speech Recognition Model**

**Model**: OpenAI Whisper
- **Type**: Transformer-based speech recognition
- **Default Size**: Base model (~74M parameters)
- **Architecture**: Encoder-decoder transformer
- **Purpose**: Transcribe audio/video to text
- **Features**:
  - Multilingual support (99 languages)
  - Automatic language detection
  - Timestamp generation
  - Robust to accents and noise
- **Loading**: Lazy-loaded (only when processing audio/video)

**Available Sizes**:
- `tiny` (39M) - Fastest, lower accuracy
- `base` (74M) - **Default** - Good balance
- `small` (244M) - Better accuracy
- `medium` (769M) - High accuracy
- `large` (1550M) - Best accuracy

---

## 💾 Databases & Storage

### 1. **Vector Database: ChromaDB**

**Version**: 0.4.22
- **Type**: Embedding database
- **Storage**: Persistent local storage
- **Purpose**: Store and retrieve document embeddings

**Technical Specifications**:
- **Distance Metric**: Cosine similarity (default)
  - Alternatives: L2 (Euclidean), IP (Inner Product)
- **Index Type**: HNSW (Hierarchical Navigable Small World)
- **Persistence**: SQLite + file-based storage
- **Collection**: `ocr_documents`
- **Storage Path**: `./data/vectordb`

**Features**:
- Metadata filtering
- Batch operations
- Incremental updates
- Automatic persistence

**Query Performance**:
- Top-K retrieval: O(log N) with HNSW
- Supports filtering by metadata
- Default K: 5 documents

---

### 2. **File Storage**

**Temporary Storage**:
- Uploaded files: System temp directory
- Cleanup: Automatic after processing

**Persistent Storage**:
- Vector database: `./data/vectordb/`
- Logs: Application logs (in-memory/console)

---

## 🧮 Algorithms & Techniques

### 1. **Text Chunking Algorithm**

**Algorithm**: Recursive Character Text Splitter (LangChain)

**Strategy**: Hierarchical splitting with overlap
```
Text → Split by "\n\n" → Split by "\n" → Split by " " → Character-level
```

**Parameters**:
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters (20%)
- **Separators**: `["\n\n", "\n", " ", ""]`

**Why Overlap?**
- Preserves context across chunk boundaries
- Improves retrieval quality
- Prevents information loss at boundaries

**Chunking Process**:
1. Try splitting by paragraph (`\n\n`)
2. If chunks too large, split by line (`\n`)
3. If still too large, split by space (` `)
4. Last resort: character-level splitting

---

### 2. **Retrieval Algorithm**

**Method**: Dense Vector Retrieval (Semantic Search)

**Process**:
1. **Query Encoding**: Convert query to 384-dim vector
2. **Similarity Search**: Cosine similarity in vector space
3. **Top-K Selection**: Retrieve K most similar chunks
4. **Score Filtering**: Filter by similarity threshold

**Similarity Metric**: Cosine Similarity
```
similarity = (A · B) / (||A|| × ||B||)
```

**Parameters**:
- **Top-K**: 5 documents
- **Score Threshold**: 0.5 (50% similarity minimum)

**Advantages**:
- Semantic understanding (not just keyword matching)
- Handles synonyms and paraphrasing
- Language-agnostic (within model's training)

---

### 3. **RAG (Retrieval-Augmented Generation)**

**Algorithm**: Context-Enhanced Generation

**Pipeline**:
```
Query → Embed → Retrieve → Rank → Context → LLM → Answer
```

**Steps**:
1. **Retrieval**: Get top-K relevant chunks
2. **Context Assembly**: Combine chunks into context
3. **Prompt Engineering**: Create structured prompt
4. **Generation**: LLM generates answer
5. **Source Attribution**: Track source documents

**Prompt Template**:
```
System: You are a helpful assistant...

Context:
[Retrieved chunks with sources]

Question: [User query]

Answer:
```

---

### 4. **OCR Processing Algorithms**

#### **PDF Extraction**
- **Primary**: PyMuPDF (fast text extraction)
- **Fallback**: pdfplumber (better for tables)
- **Strategy**: Try PyMuPDF first, use pdfplumber if text < 100 chars

#### **Image Preprocessing** (Tesseract)
1. Grayscale conversion
2. Noise reduction (optional)
3. Contrast enhancement (optional)
4. Binarization
5. OCR with confidence scoring

#### **Audio/Video Processing**
1. **Audio Extraction** (FFmpeg for video)
2. **Speech Recognition** (Whisper)
3. **Timestamp Alignment**
4. **Segment Extraction**

---

### 5. **Text Cleaning & Processing**

**Techniques**:
- **Whitespace Normalization**: Remove extra spaces/newlines
- **Unicode Normalization**: Handle special characters
- **LaTeX Preservation**: Keep mathematical notation
- **Metadata Extraction**: Extract page numbers, sources

---

## 🛠️ Technology Stack

### **Backend Frameworks**
| Technology | Version | Purpose |
|------------|---------|---------|
| **FastAPI** | Latest | REST API services (OCR, RAG) |
| **Uvicorn** | Latest | ASGI server for FastAPI |
| **Streamlit** | Latest | Web UI framework |

### **Machine Learning Libraries**
| Library | Version | Purpose |
|---------|---------|---------|
| **sentence-transformers** | Latest | Embedding generation |
| **openai-whisper** | 20250625 | Speech-to-text |
| **torch** | 2.9.1 | Deep learning backend |
| **tiktoken** | Latest | Tokenization for LLM |

### **OCR & Document Processing**
| Library | Version | Purpose |
|---------|---------|---------|
| **pytesseract** | Latest | Tesseract OCR wrapper |
| **PyMuPDF (fitz)** | Latest | PDF text extraction |
| **pdfplumber** | Latest | PDF table extraction |
| **python-docx** | Latest | DOCX processing |
| **opencv-python** | Latest | Image preprocessing |
| **Pillow** | Latest | Image handling |

### **Web Scraping**
| Library | Version | Purpose |
|---------|---------|---------|
| **trafilatura** | 2.0.0 | Web content extraction |
| **newspaper3k** | Latest | Article extraction |
| **beautifulsoup4** | Latest | HTML parsing |
| **lxml** | Latest | XML/HTML processing |

### **Vector Database**
| Technology | Version | Purpose |
|------------|---------|---------|
| **ChromaDB** | 0.4.22 | Vector storage & retrieval |
| **SQLite** | Built-in | ChromaDB persistence |

### **LLM Integration**
| Technology | Version | Purpose |
|------------|---------|---------|
| **Ollama** | Latest | Local LLM inference |
| **Qwen 3** | 1.7B | Answer generation |

### **Text Processing**
| Library | Version | Purpose |
|---------|---------|---------|
| **LangChain** | 0.1.0 | Text splitting utilities |
| **NLTK** | Latest | NLP utilities |
| **regex** | Latest | Pattern matching |
| **ftfy** | Latest | Text fixing |
| **unidecode** | Latest | Unicode normalization |

### **Utilities**
| Library | Version | Purpose |
|---------|---------|---------|
| **loguru** | Latest | Logging |
| **pydantic** | Latest | Data validation |
| **PyYAML** | Latest | Configuration |
| **requests** | Latest | HTTP client |
| **tqdm** | Latest | Progress bars |

---

## 🏛️ Architecture Patterns

### 1. **Microservices Architecture**

**Why Microservices?**
- **Dependency Isolation**: OCR and RAG have conflicting dependencies
- **Independent Scaling**: Scale services independently
- **Technology Flexibility**: Different Python environments
- **Fault Isolation**: One service failure doesn't crash entire system

**Services**:
```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (venv_main)             │
│                    Port: 8501                           │
└────────────────┬────────────────────────────────────────┘
                 │ HTTP REST API
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐  ┌──────────────┐
│ OCR Service  │  │ RAG Service  │
│ (venv_ocr)   │  │ (venv_rag)   │
│ Port: 8000   │  │ Port: 8001   │
└──────────────┘  └──────────────┘
```

**Communication**: RESTful HTTP APIs
- **OCR Service**: `/extract`, `/health`
- **RAG Service**: `/ingest`, `/query`, `/health`

---

### 2. **Lazy Loading Pattern**

**Purpose**: Optimize startup time and memory usage

**Implementation**:
- **Whisper Model**: Loaded only when first audio/video file is processed
- **Heavy Dependencies**: Imported only when needed

**Benefits**:
- Faster service startup
- Lower memory footprint
- Better resource utilization

**Example**:
```python
# Not loaded at startup
self.extractors["audio"] = None

# Loaded on first use
if format_type == "audio":
    from extractors.audio_extractor import AudioExtractor
    self.extractors["audio"] = AudioExtractor()
```

---

### 3. **Pipeline Pattern**

**OCR Pipeline**:
```
File → Format Detection → Extractor Selection → Extraction →
Text Cleaning → Math Extraction → Result
```

**RAG Pipeline**:
```
Text → Chunking → Embedding → Vector Storage →
Query → Retrieval → LLM Generation → Answer
```

---

### 4. **Strategy Pattern**

**Format-Specific Extractors**:
- Each file format has dedicated extractor
- Common interface (`BaseExtractor`)
- Runtime selection based on file type

**Extractors**:
- `PDFExtractor`: PyMuPDF + pdfplumber
- `DOCXExtractor`: python-docx
- `OCRExtractor`: Tesseract/PaddleOCR
- `AudioExtractor`: Whisper
- `VideoExtractor`: FFmpeg + Whisper
- `WebExtractor`: Trafilatura + Newspaper

---

# 🏗️ Technical Architecture - Part 2

## 5. **Repository Pattern**

**Vector Store Abstraction**:
- Encapsulates ChromaDB operations
- Provides clean API for storage/retrieval
- Handles persistence and configuration

**Benefits**:
- Easy to swap vector databases
- Testable without actual database
- Centralized database logic

---

## 📊 Data Flow

### **Document Ingestion Flow**

```
┌──────────────┐
│ Upload File  │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ OCR Service      │
│ - Detect format  │
│ - Extract text   │
│ - Clean text     │
└──────┬───────────┘
       │ JSON
       ▼
┌──────────────────┐
│ RAG Service      │
│ - Chunk text     │
│ - Generate embed │
│ - Store vectors  │
└──────────────────┘
```

### **Query Flow**

```
┌──────────────┐
│ User Query   │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ RAG Service      │
│ - Embed query    │
│ - Search vectors │
│ - Retrieve docs  │
└──────┬───────────┘
       │ Context
       ▼
┌──────────────────┐
│ LLM (Ollama)     │
│ - Generate ans   │
│ - Add sources    │
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│ Display UI   │
└──────────────┘
```

---

## 🔧 Configuration Management

**Configuration File**: `RAG/config/config.yaml`

**Configurable Parameters**:
- Embedding model and dimensions
- Chunk size and overlap
- Vector database settings
- LLM model and parameters
- Retrieval settings (top-K, threshold)
- Logging levels

**Environment Variables**:
- `TESSERACT_CMD`: Tesseract executable path
- `WHISPER_MODEL`: Whisper model size
- `OLLAMA_BASE_URL`: Ollama API endpoint

---

## 🎯 Key Design Decisions

### **1. Why ChromaDB?**
- ✅ Easy to use and deploy
- ✅ No separate server required
- ✅ Persistent local storage
- ✅ Good performance for small-medium datasets
- ✅ Built-in metadata filtering

### **2. Why Sentence Transformers?**
- ✅ State-of-the-art embeddings
- ✅ Fast inference on CPU
- ✅ Pre-trained models available
- ✅ Easy to use API
- ✅ Good multilingual support

### **3. Why Ollama + Qwen?**
- ✅ Runs locally (privacy)
- ✅ No API costs
- ✅ Fast inference
- ✅ Good quality for size
- ✅ Easy model switching

### **4. Why Microservices?**
- ✅ Dependency isolation
- ✅ Independent deployment
- ✅ Better fault tolerance
- ✅ Easier testing
- ✅ Scalability

### **5. Why Tesseract over PaddleOCR?**
- ✅ Easier installation
- ✅ No GPU required
- ✅ Mature and stable
- ✅ Good accuracy for most cases
- ✅ Wide language support

---

## 📈 Performance Characteristics

### **Embedding Generation**
- Speed: ~14,000 sentences/sec (CPU)
- Batch size: 32
- Memory: ~500MB (model loaded)

### **Vector Search**
- Query time: <100ms for 10K documents
- Scales: O(log N) with HNSW index

### **LLM Generation**
- Speed: ~20-50 tokens/sec (CPU)
- Latency: 2-5 seconds for typical answer
- Memory: ~2GB (Qwen 1.7B loaded)

### **OCR Processing**
- PDF: ~1-2 pages/sec
- Images: ~2-5 sec/image (Tesseract)
- Audio: ~0.1x realtime (Whisper base)

---

## 🔒 Security Considerations

- ✅ Local processing (no data sent to cloud)
- ✅ No API keys required
- ✅ Temporary file cleanup
- ✅ Input validation on uploads
- ⚠️ No authentication (local use only)
- ⚠️ No encryption at rest

---

## 🚀 Scalability

**Current Limitations**:
- Single-machine deployment
- CPU-only inference
- Local file storage

**Scaling Options**:
- Add GPU support for faster inference
- Deploy services on separate machines
- Use cloud vector database (Pinecone, Weaviate)
- Add load balancing
- Implement caching layer

---

## 📚 References

- **Sentence Transformers**: https://www.sbert.net/
- **ChromaDB**: https://www.trychroma.com/
- **Ollama**: https://ollama.ai/
- **Whisper**: https://github.com/openai/whisper
- **LangChain**: https://python.langchain.com/
- **Tesseract**: https://github.com/tesseract-ocr/tesseract
- **Qwen**: https://github.com/QwenLM/Qwen

