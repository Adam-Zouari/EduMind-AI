# OCR-to-RAG Pipeline with Qwen

A complete pipeline for converting OCR-extracted text into a searchable knowledge base using Retrieval-Augmented Generation (RAG) with Ollama's Qwen model and a Streamlit web interface.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start - Streamlit Interface](#quick-start---streamlit-interface)
3. [Quick Start - Python API](#quick-start---python-api)
4. [OCR JSON Format](#ocr-json-format)
5. [Pipeline Architecture](#pipeline-architecture)
6. [Step-by-Step Explanation](#step-by-step-explanation)
7. [Configuration](#configuration)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This pipeline transforms OCR-extracted text into a queryable knowledge base with AI-powered answers:

```
OCR JSON → Load → Chunk → Embed → Vector Store → Query → LLM (Qwen) → Answer
```

**Key Features:**
- ✅ **Streamlit Web Interface** - Easy-to-use UI for adding context and asking questions
- ✅ **AI-Powered Answers** - Uses Ollama's Qwen model for intelligent responses
- ✅ **Simple JSON Input** - Direct integration with OCR systems
- ✅ **Semantic Search** - Find relevant context using embeddings
- ✅ **Source Attribution** - Track where answers come from
- ✅ **Metadata Preservation** - Keep source, page, and custom metadata
- ✅ **Persistent Storage** - ChromaDB vector database

**Technology Stack:**
- **Web Interface:** Streamlit
- **LLM:** Ollama (Qwen 3:1.7b)
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB:** ChromaDB with persistent storage
- **Text Processing:** LangChain RecursiveCharacterTextSplitter
- **Language:** Python 3.10+

---

## 🚀 Quick Start - Streamlit Interface

### **1. Prerequisites**

**Install Ollama:**
- Download from: https://ollama.ai/download
- Start Ollama: `ollama serve`
- Pull Qwen model: `ollama pull qwen3:1.7b`

**Activate Virtual Environment:**
```bash
# Windows
./venv/Scripts/activate

# Linux/Mac
source venv/bin/activate
```

### **2. Run the Interface**

**Windows:**
```bash
run_app.bat
```

**Linux/Mac:**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### **3. Use the Interface**

**Step 1: Initialize Pipeline**
- Click **"🚀 Initialize Pipeline"** in the sidebar
- Wait for models to load

**Step 2: Add Context**
- Go to **"📝 Add Context"** tab
- Choose input method:
  - **Text Input:** Paste text directly
  - **JSON Input:** Enter JSON format
  - **Upload File:** Upload JSON file
- Click "Add to Knowledge Base"

**Step 3: Ask Questions**
- Go to **"💬 Ask Questions"** tab
- Enter your question
- Click **"🔍 Ask"**
- View AI-generated answer with sources

**Step 4: View History**
- Go to **"📚 Chat History"** tab
- See all previous Q&A

---

## 🐍 Quick Start - Python API

### **1. Installation**

```bash
# Activate virtual environment
./venv/Scripts/activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### **2. Basic Usage**

```python
from src.rag_pipeline import RAGPipeline

# Initialize pipeline with LLM
pipeline = RAGPipeline(use_llm=True)

# Add context from JSON
total_chunks = pipeline.ingest_from_json('ocr_output.json')
print(f"Ingested {total_chunks} chunks")

# Ask a question and get AI-generated answer
result = pipeline.generate_answer(
    query="What is this document about?",
    top_k=3
)

print(f"Answer: {result['answer']}")
print(f"\nSources:")
for source in result['sources']:
    print(f"  - {source['source']} (Page {source['page']}) - {source['similarity']}")
```

### **3. Without LLM (Just Retrieval)**

```python
from src.rag_pipeline import RAGPipeline

# Initialize without LLM
pipeline = RAGPipeline(use_llm=False)

# Ingest data
pipeline.ingest_from_json('ocr_output.json')

# Query for relevant documents
results = pipeline.query("What is machine learning?", top_k=5)

# Print results
for i, result in enumerate(results):
    print(f"\n{i+1}. {result['document'][:100]}...")
    print(f"   Source: {result['metadata']['source']}")
    print(f"   Similarity: {1 - result['distance']:.2%}")
```

---

## 📝 OCR JSON Format

### **Required Structure**

Your OCR system should output JSON in this format:

```json
[
  {
    "text": "Extracted text content...",
    "source": "filename.pdf",
    "page": 1
  }
]
```

### **Field Specifications**

| Field | Required? | Type | Description | Example |
|-------|-----------|------|-------------|---------|
| `text` | ✅ **YES** | string | The extracted text | `"This is the document text..."` |
| `source` | ⭐ Recommended | string | Source filename | `"invoice_001.pdf"` |
| `page` | ⭐ Recommended | int/string | Page number | `1` or `"1"` |
| `date` | 📝 Optional | string | Document date | `"2024-01-15"` |
| `document_type` | 📝 Optional | string | Type of document | `"invoice"`, `"contract"` |
| *custom fields* | 📝 Optional | any | Any metadata you need | `"author": "John Doe"` |

### **Multi-Page Documents**

**Option 1: One object per page** (Recommended)
```json
[
  {"text": "Page 1 content...", "source": "doc.pdf", "page": 1},
  {"text": "Page 2 content...", "source": "doc.pdf", "page": 2},
  {"text": "Page 3 content...", "source": "doc.pdf", "page": 3}
]
```

**Option 2: Combined text**
```json
[
  {
    "text": "Page 1...\n\n[PAGE BREAK]\n\nPage 2...",
    "source": "doc.pdf",
    "total_pages": 2
  }
]
```

### **Validation**

```python
import json

# Validate your JSON
with open('ocr_output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

assert isinstance(data, list), "Must be an array"
for i, doc in enumerate(data):
    assert 'text' in doc, f"Document {i} missing 'text'"
    assert isinstance(doc['text'], str), f"'text' must be string"

print(f"✅ Valid! {len(data)} documents")
```

---

## 🏗️ Pipeline Architecture

### **Components**

```
┌─────────────────┐
│  OCR Processor  │  Loads and validates JSON
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Chunker   │  Splits text into ~1000 char chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Embedder     │  Converts text to 384-dim vectors
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │  Stores in ChromaDB
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Query Engine   │  Similarity search & retrieval
└─────────────────┘
```

### **Data Flow**

```
Input:  {"text": "ML is...", "source": "doc.pdf", "page": 1}
   ↓
Step 1: Load & validate
   ↓
Step 2: Chunk into pieces (with overlap)
   ↓
Step 3: Generate embeddings [0.23, -0.45, 0.12, ..., 0.67]
   ↓
Step 4: Store in vector DB
   ↓
Query:  "What is machine learning?"
   ↓
Output: Top K most similar chunks with metadata
```

---

## 📖 Step-by-Step Explanation

### **STEP 1: OCR Processor - Load JSON** 📥

**File:** `src/ocr_processor.py`

**What it does:**
- Reads JSON file from OCR system
- Validates each document has a `text` field
- Returns list of document dictionaries

**Example:**
```python
processor = OCRProcessor()
documents = processor.load_from_json('ocr_output.json')
# Returns: [{'text': '...', 'source': '...', 'page': 1}, ...]
```


### **STEP 2: Text Chunker - Split into Pieces** ✂️

**File:** `src/text_chunker.py`

**Why chunking?**
- OCR documents can be very long (10,000+ words)
- Embedding models work best with smaller chunks (512-1024 tokens)
- Enables more precise retrieval (find exact relevant sections)

**How it works:**
1. Splits text into ~1000 character chunks
2. Overlaps chunks by 200 characters (preserves context)
3. Preserves all metadata from original document
4. Adds chunk-specific metadata (chunk_index, total_chunks)

**Configuration:**
```yaml
chunking:
  chunk_size: 1000        # Max characters per chunk
  chunk_overlap: 200      # Characters to overlap
  separators: ["\n\n", "\n", " ", ""]  # Split on paragraphs first
```

**Example:**
```python
# Input: 3000 character document
chunker = TextChunker()
chunks = chunker.chunk_text(text, metadata={'source': 'doc.pdf', 'page': 1})

# Output: 3 chunks with overlap
# Chunk 0: chars 0-1000
# Chunk 1: chars 800-1800 (200 char overlap)
# Chunk 2: chars 1600-2600 (200 char overlap)
```

**Why overlap?**
- Prevents losing context at chunk boundaries
- If important info is split, overlap captures it

---

### **STEP 3: Embedder - Convert Text to Vectors** 🔢

**File:** `src/embedder.py`

**What are embeddings?**
- Vector representations of text (lists of numbers)
- Similar meanings → Similar vectors
- Enables mathematical comparison of text

**The Model:**
- **Name:** `sentence-transformers/all-MiniLM-L6-v2`
- **Type:** Pre-trained neural network
- **Output:** 384-dimensional vector per text
- **Size:** ~90MB (downloaded once, cached locally)

**How it works:**
```
Text: "Machine learning is a subset of AI..."
  ↓ (Neural Network)
Vector: [0.234, -0.456, 0.123, 0.789, ..., 0.345]  ← 384 numbers
```

Each dimension captures a different aspect of meaning:
- Dimension 1: Topic (AI/ML)
- Dimension 2: Sentiment
- Dimension 3: Formality
- ... (384 total aspects)

**Example:**
```python
embedder = Embedder()
chunks_with_embeddings = embedder.embed_chunks(chunks)

# Each chunk now has:
# {
#   'text': '...',
#   'source': 'doc.pdf',
#   'embedding': [0.234, -0.456, ..., 0.345]  # 384 numbers
# }
```

**Performance:**
- CPU: ~100-500 chunks/second
- GPU: ~1000-5000 chunks/second

---

### **STEP 4: Vector Store - Save to Database** 💾

**File:** `src/vector_store.py`

**What is ChromaDB?**
- Specialized database for storing and searching vectors
- Optimized for similarity search
- Persistent storage (survives restarts)

**How it works:**
1. Generates unique UUID for each chunk
2. Stores embedding vector (384 dimensions)
3. Stores original text
4. Stores all metadata
5. Builds HNSW index for fast search

**Storage Location:**
```
./data/vectordb/
├── chroma.sqlite3          # Database file
└── [index files]           # HNSW index for fast search
```

**What gets stored:**
```
ID: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
Embedding: [0.234, -0.456, 0.123, ..., 0.345]
Document: "Machine learning is a subset of AI..."
Metadata: {
  'source': 'ml_textbook.pdf',
  'page': '1',
  'chunk_index': '0',
  'total_chunks': '3'
}
```

**Indexing:**
- Uses HNSW (Hierarchical Navigable Small World) algorithm
- Enables fast similarity search (milliseconds)
- Cosine distance metric for similarity

**Example:**
```python
vector_store = VectorStore()
vector_store.add_documents(chunks_with_embeddings)
# Stored in ./data/vectordb/
```

---

### **STEP 5: Query - Search for Similar Content** 🔍

**How querying works:**

1. **User asks a question:**
   ```python
   query = "What is machine learning?"
   ```

2. **Convert query to embedding:**
   ```python
   query_embedding = [0.189, -0.412, 0.765, ..., 0.234]  # 384 dims
   ```

3. **Compare to all stored embeddings:**
   ```
   Query:      [0.189, -0.412, 0.765, ...]
   Chunk 1:    [0.234, -0.456, 0.123, ...]  → Similarity: 85%
   Chunk 2:    [0.015, -0.032, 0.891, ...]  → Similarity: 42%
   Chunk 3:    [-0.567, 0.234, -0.123, ...] → Similarity: 12%
   ```

4. **Return top K most similar:**
   ```python
   results = [
     {
       'document': 'Machine learning is a subset of AI...',
       'metadata': {'source': 'ml_textbook.pdf', 'page': '1'},
       'distance': 0.15  # Lower = more similar
     },
     ...
   ]
   ```

**Cosine Similarity:**
- Measures angle between vectors
- 1.0 = Identical
- 0.0 = Unrelated
- -1.0 = Opposite

**Example:**
```python
results = pipeline.query("What is machine learning?", top_k=5)

for result in results:
    print(f"Text: {result['document'][:100]}...")
    print(f"Source: {result['metadata']['source']}")
    print(f"Similarity: {1 - result['distance']:.2%}")
```

---

### **STEP 6: RAG Context Generation** 📝

**What is RAG Context?**
- Combines retrieved chunks into formatted text
- Ready to feed to an LLM (GPT-4, Claude, etc.)
- LLM uses context to answer questions accurately

**Example:**
```python
context = pipeline.generate_context("What is machine learning?", top_k=3)

# Output:
# [Document 1]
# Machine learning is a subset of AI that enables computers to learn...
# Source: ml_textbook.pdf, Page: 1
#
# [Document 2]
# Supervised learning is a type of machine learning where...
# Source: ml_textbook.pdf, Page: 3
#
# [Document 3]
# Deep learning uses neural networks with multiple layers...
# Source: dl_guide.pdf, Page: 1
```

**Using with LLM:**
```python
import openai

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "system",
            "content": f"Answer using this context:\n\n{context}"
        },
        {
            "role": "user",
            "content": "What is machine learning?"
        }
    ]
)

print(response.choices[0].message.content)
# Output: "Machine learning is a subset of AI that enables computers
#          to learn from data without being explicitly programmed..."
```

---

## ⚙️ Configuration

**File:** `config/config.yaml`

```yaml
# Embedding Configuration
embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  embedding_dim: 384
  device: "cpu"  # or "cuda" for GPU

# Text Chunking Configuration
chunking:
  chunk_size: 1000
  chunk_overlap: 200
  separators: ["\n\n", "\n", " ", ""]

# Vector Database Configuration
vectordb:
  collection_name: "ocr_documents"
  persist_directory: "./data/vectordb"
  distance_metric: "cosine"

# RAG Configuration
rag:
  top_k: 5
  score_threshold: 0.5

# LLM Configuration (Ollama)
llm:
  model_name: "qwen3:1.7b"  # Available: qwen3:1.7b, gemma3:1b, llama3.2:1b, deepseek-r1:1.5b
  base_url: "http://localhost:11434"
  temperature: 0.7  # 0.0 = deterministic, 1.0 = creative
  max_tokens: 2048
```

**Customization:**

- **chunk_size:** Increase for longer chunks (better context), decrease for more precise retrieval
- **chunk_overlap:** Increase to preserve more context at boundaries
- **top_k:** Number of results to return per query
- **device:** Use "cuda" if you have a GPU for faster embeddings
- **model_name (LLM):** Choose from your installed Ollama models
- **temperature:** Control randomness in LLM responses (0.0-1.0)
- **max_tokens:** Maximum length of LLM responses

---

## 💡 Usage Examples

### **Example 1: AI-Powered Q&A (Recommended)**

```python
from src.rag_pipeline import RAGPipeline

# Initialize with LLM
pipeline = RAGPipeline(use_llm=True)

# Ingest from JSON
total_chunks = pipeline.ingest_from_json('ocr_output.json')
print(f"Ingested {total_chunks} chunks")

# Ask a question and get AI-generated answer
result = pipeline.generate_answer(
    query="What is machine learning?",
    top_k=3
)

print(f"Answer: {result['answer']}")
print(f"\nSources:")
for source in result['sources']:
    print(f"  - {source['source']} (Page {source['page']}) - {source['similarity']}")
```

### **Example 2: Basic Retrieval (Without LLM)**

```python
from src.rag_pipeline import RAGPipeline

# Initialize without LLM
pipeline = RAGPipeline(use_llm=False)

# Ingest from JSON
total_chunks = pipeline.ingest_from_json('ocr_output.json')
print(f"Ingested {total_chunks} chunks")

# Query for relevant documents
results = pipeline.query("What is this about?", top_k=5)

# Display results
for i, result in enumerate(results):
    print(f"\n{i+1}. {result['document'][:150]}...")
    print(f"   Source: {result['metadata']['source']}")
    print(f"   Page: {result['metadata'].get('page', 'N/A')}")
    print(f"   Similarity: {1 - result['distance']:.2%}")
```

### **Example 3: Multiple Documents with Metadata Filtering**

```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

# Ingest multiple documents
documents = [
    {
        'text': 'Machine learning enables computers to learn from data...',
        'source': 'ml_textbook.pdf',
        'page': 1,
        'category': 'AI/ML'
    },
    {
        'text': 'Natural language processing focuses on human language...',
        'source': 'nlp_guide.pdf',
        'page': 1,
        'category': 'AI/NLP'
    }
]

total_chunks = pipeline.ingest_documents(documents)
print(f"Ingested {total_chunks} chunks")

# Query with metadata filter
results = pipeline.query(
    "What is machine learning?",
    top_k=5,
    filter_metadata={'category': 'AI/ML'}
)
```

### **Example 4: Add Documents Programmatically**

```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(use_llm=True)

# Add a single document
document = {
    'text': 'Machine learning is a subset of AI...',
    'source': 'ml_guide.pdf',
    'page': 1,
    'category': 'AI/ML'
}

chunks = pipeline.ingest_document(document)
print(f"Created {chunks} chunks")

# Ask question
result = pipeline.generate_answer("What is machine learning?")
print(result['answer'])
```

### **Example 5: Pipeline Statistics**

```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.ingest_from_json('ocr_output.json')

# Get statistics
stats = pipeline.get_stats()
print(f"Total documents: {stats['total_documents']}")
print(f"Collection: {stats['collection_name']}")
print(f"Storage: {stats['persist_directory']}")
```

---

## 📚 API Reference

### **RAGPipeline**

Main class for the OCR-to-RAG pipeline.

#### **Initialization:**

**`__init__(config_path: str = "config/config.yaml", use_llm: bool = False)`**
- Initialize the pipeline with configuration
- **Parameters:**
  - `config_path`: Path to YAML configuration file
  - `use_llm`: Whether to initialize LLM generator (requires Ollama)

#### **Ingestion Methods:**

**`ingest_from_json(json_path: str) -> int`**
- Load and ingest documents from OCR JSON file
- **Returns:** Number of chunks created

**`ingest_documents(documents: List[Dict]) -> int`**
- Ingest multiple documents
- **Returns:** Number of chunks created

**`ingest_document(document: Dict) -> int`**
- Ingest a single document
- **Returns:** Number of chunks created

#### **Query Methods:**

**`query(query_text: str, top_k: int = 5, filter_metadata: Dict = None) -> List[Dict]`**
- Query the knowledge base for relevant documents
- **Returns:** List of results with document, metadata, and distance

**`generate_answer(query: str, top_k: int = None, filter_metadata: Dict = None, stream: bool = False, system_prompt: str = None) -> Dict`**
- Generate AI-powered answer using RAG + LLM (requires `use_llm=True`)
- **Returns:** Dictionary with 'answer', 'sources', and 'context'

**`generate_context(query_text: str, top_k: int = 5) -> str`**
- Generate formatted context for RAG
- **Returns:** Formatted string ready for LLM

#### **Utility Methods:**

**`get_stats() -> Dict`**
- Get pipeline statistics
- **Returns:** Dictionary with stats (total_documents, collection_name, etc.)

**`reset() -> None`**
- Reset the vector store (delete all data)

---

### **OCRProcessor**

Loads OCR JSON output.

**`load_from_json(json_path: str) -> List[Dict]`**
- Load documents from JSON file
- Returns: List of document dictionaries

---

### **TextChunker**

Splits text into chunks.

**`chunk_text(text: str, metadata: Dict = None) -> List[Dict]`**
- Chunk a single text
- Returns: List of chunk dictionaries

**`chunk_documents(documents: List[Dict]) -> List[Dict]`**
- Chunk multiple documents
- Returns: List of all chunks

---

### **Embedder**

Generates embeddings from text.

**`embed_text(text: str) -> np.ndarray`**
- Generate embedding for single text
- Returns: 384-dimensional numpy array

**`embed_texts(texts: List[str]) -> np.ndarray`**
- Generate embeddings for multiple texts
- Returns: (N, 384) numpy array

**`embed_chunks(chunks: List[Dict]) -> List[Dict]`**
- Add embeddings to chunk dictionaries
- Returns: Chunks with 'embedding' field added

---

### **VectorStore**

Manages ChromaDB vector storage.

**`add_documents(chunks: List[Dict]) -> None`**
- Add chunks to vector store

**`query(query_embedding: List[float], top_k: int = 5, filter_metadata: Dict = None) -> List[Dict]`**
- Query by embedding vector
- Returns: List of results

**`query_by_text(query_text: str, embedder: Embedder, top_k: int = 5) -> List[Dict]`**
- Query by text (generates embedding first)
- Returns: List of results

**`get_collection_count() -> int`**
- Get number of documents in collection

**`reset_collection() -> None`**
- Delete and recreate collection

---

## 🎯 Key Concepts

### **What are Embeddings?**

Embeddings are numerical representations of text that capture semantic meaning:

```
Text: "Machine learning is powerful"
  ↓
Embedding: [0.23, -0.45, 0.12, ..., 0.67]  (384 numbers)

Text: "ML is very useful"
  ↓
Embedding: [0.25, -0.43, 0.14, ..., 0.65]  (similar numbers!)
```

Similar meanings produce similar embeddings, enabling semantic search.

### **Why Chunking?**

Large documents are split into smaller chunks for better retrieval:

```
Long Document (10,000 words)
  ↓ Split into chunks
Chunk 1: Introduction (1000 chars)
Chunk 2: Methods (1000 chars)
Chunk 3: Results (1000 chars)
...
```

**Benefits:**
- More precise retrieval (find exact relevant section)
- Better embedding quality (focused content)
- Faster search (smaller vectors to compare)

### **How Similarity Search Works**

Vector similarity search finds semantically similar content:

```
Query: "What is ML?"
Query Embedding: [0.18, -0.41, 0.76, ...]

Compare to all stored chunks:
Chunk 1: [0.23, -0.45, 0.12, ...]  → 85% similar ✓
Chunk 2: [0.01, -0.03, 0.89, ...]  → 42% similar
Chunk 3: [-0.56, 0.23, -0.12, ...] → 12% similar

Return top K most similar chunks
```

Uses cosine similarity to measure vector closeness.

---

## 🔧 Troubleshooting

### **Issue: "Pipeline initialization failed" in Streamlit**

**Solution:** Make sure Ollama is running:
```bash
# Start Ollama
ollama serve

# Verify it's running
ollama list
```

### **Issue: "Connection refused" when using LLM**

**Solution:**
1. Check if Ollama is running: `ollama serve`
2. Verify the model is installed: `ollama list`
3. Pull the model if needed: `ollama pull qwen3:1.7b`

### **Issue: "Model not found" error**

**Solution:** Install the Qwen model:
```bash
ollama pull qwen3:1.7b
```

Or use a different model you have installed:
```yaml
# config/config.yaml
llm:
  model_name: "gemma3:1b"  # or llama3.2:1b, deepseek-r1:1.5b
```

### **Issue: ChromaDB telemetry errors**

```
ERROR:chromadb.telemetry.product.posthog:Failed to send telemetry event...
```

**Solution:** These are harmless warnings. ChromaDB tries to send usage statistics but fails. Doesn't affect functionality.

### **Issue: Import errors**

```
ModuleNotFoundError: No module named 'sentence_transformers'
```

**Solution:** Install dependencies:
```bash
./venv/Scripts/activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### **Issue: Slow embedding generation**

**Solution:** Use GPU acceleration:
```yaml
# config/config.yaml
embedding:
  device: "cuda"  # Change from "cpu" to "cuda"
```

### **Issue: Out of memory**

**Solution:** Reduce chunk size or batch size:
```yaml
chunking:
  chunk_size: 500  # Reduce from 1000
```

### **Issue: Streamlit won't start**

**Solution:** Install Streamlit:
```bash
pip install streamlit==1.28.0
```

---

## 📊 Performance

### **Benchmarks (CPU - Intel i7)**

| Operation | Speed | Notes |
|-----------|-------|-------|
| Text Chunking | ~10,000 chunks/sec | Very fast |
| Embedding Generation | ~100-500 chunks/sec | CPU bottleneck |
| Vector Storage | ~1,000 chunks/sec | Fast |
| Query | <100ms | Sub-second |

### **Benchmarks (GPU - NVIDIA RTX 3080)**

| Operation | Speed | Notes |
|-----------|-------|-------|
| Embedding Generation | ~1,000-5,000 chunks/sec | 10x faster |
| Query | <50ms | 2x faster |

### **Storage**

- **Per chunk:** ~1KB (text + embedding + metadata)
- **1,000 chunks:** ~1MB
- **100,000 chunks:** ~100MB

---

## 🚀 Next Steps

### **1. Connect to LLM**

Integrate with OpenAI, Anthropic, or local LLMs:

```python
import openai

context = pipeline.generate_context(query, top_k=3)
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": f"Use this context: {context}"},
        {"role": "user", "content": query}
    ]
)
```

### **2. Build API**

Create a REST API with FastAPI:

```python
from fastapi import FastAPI
from src.rag_pipeline import RAGPipeline

app = FastAPI()
pipeline = RAGPipeline()

@app.post("/ingest")
def ingest(json_path: str):
    return {"chunks": pipeline.ingest_from_json(json_path)}

@app.get("/query")
def query(q: str, top_k: int = 5):
    return pipeline.query(q, top_k)
```

### **3. Add Reranking**

Improve results with cross-encoder reranking:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = reranker.predict([(query, r['document']) for r in results])
```

### **4. Scale Up**

- Use GPU for faster embeddings
- Implement batch processing for large datasets
- Add caching for frequent queries
- Deploy with Docker

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📞 Support

For issues or questions:
- Check the troubleshooting section
- Review the examples in `example_usage.py`
- Test with `sample_ocr_output.json`

---

**Built with ❤️ for OCR-to-RAG applications**
