# OCR to RAG Integration Guide

## 🎯 Overview

The OCR system now outputs JSON in a **RAG-compatible format** with flattened metadata structure.

---

## ✅ New JSON Format

### **Single Document**
```json
{
  "text": "Extracted content...",
  "source": "document.pdf",
  "format_type": "pdf",
  "num_pages": 5,
  "title": "Sample Document",
  "author": "John Doe",
  "confidence": 95.5,
  "success": true
}
```

### **Multiple Documents (RAG Array Format)**
```json
[
  {
    "text": "Content from page 1...",
    "source": "document.pdf",
    "format_type": "pdf",
    "num_pages": 5,
    "success": true
  },
  {
    "text": "Content from another doc...",
    "source": "report.docx",
    "format_type": "docx",
    "num_paragraphs": 20,
    "success": true
  }
]
```

---

## 🔑 Key Features

### **1. Flattened Metadata**
All metadata fields are at the **top level** (not nested):
- ✅ `num_pages` instead of `metadata.num_pages`
- ✅ `confidence` instead of `metadata.confidence`
- ✅ `format_type` instead of `metadata.format_type`

### **2. RAG-Standard Fields**
- ✅ `text` - Required by RAG
- ✅ `source` - Recommended by RAG (filename only)
- ✅ All other fields become queryable metadata

### **3. Format-Specific Metadata**

**PDF Documents:**
```json
{
  "text": "...",
  "source": "doc.pdf",
  "format_type": "pdf",
  "num_pages": 10,
  "title": "Document Title",
  "author": "Author Name",
  "extractor": "pymupdf"
}
```

**DOCX Documents:**
```json
{
  "text": "...",
  "source": "doc.docx",
  "format_type": "docx",
  "num_paragraphs": 50,
  "num_tables": 3,
  "title": "Document Title"
}
```

**Images (OCR):**
```json
{
  "text": "...",
  "source": "image.png",
  "format_type": "image",
  "ocr_engine": "tesseract",
  "confidence": 92.5,
  "languages": ["eng"]
}
```

**Audio Files:**
```json
{
  "text": "...",
  "source": "audio.mp3",
  "format_type": "audio",
  "language": "en",
  "duration": 120.5,
  "num_segments": 15,
  "model": "large-v3"
}
```

---

## 💻 Usage Examples

### **Example 1: Process Single File**

```python
from core.pipeline import DataIngestionPipeline
import json

# Initialize OCR pipeline
pipeline = DataIngestionPipeline()

# Process file
result = pipeline.process_file("document.pdf")

# Get RAG-compatible JSON
json_output = result.to_dict()

# Save as JSON array (RAG expects array)
with open("ocr_output.json", "w", encoding="utf-8") as f:
    json.dump([json_output], f, indent=2, ensure_ascii=False)

print(f"✅ Processed: {json_output['source']}")
print(f"✅ Text length: {len(json_output['text'])} characters")
print(f"✅ Format: {json_output['format_type']}")
```

### **Example 2: Batch Processing**

```python
from core.pipeline import DataIngestionPipeline
import json

pipeline = DataIngestionPipeline()

# List of files to process
files = [
    "documents/report.pdf",
    "documents/notes.docx",
    "images/scan.png",
    "audio/lecture.mp3"
]

# Process all files
results = []
for file_path in files:
    result = pipeline.process_file(file_path)
    if result.success:
        results.append(result.to_dict())
        print(f"✅ {result.file_path}")
    else:
        print(f"❌ {result.file_path}: {result.error}")

# Save RAG-compatible JSON array
with open("batch_output.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Processed {len(results)} documents")
```

### **Example 3: Direct Integration with RAG**

```python
from core.pipeline import DataIngestionPipeline
from RAG.src.rag_pipeline import RAGPipeline
import json

# Step 1: Extract with OCR
ocr_pipeline = DataIngestionPipeline()
result = ocr_pipeline.process_file("document.pdf")

# Step 2: Save to JSON
with open("temp_ocr.json", "w") as f:
    json.dump([result.to_dict()], f)

# Step 3: Ingest into RAG
rag_pipeline = RAGPipeline()
chunks = rag_pipeline.ingest_from_json("temp_ocr.json")

print(f"✅ Ingested {chunks} chunks into RAG")

# Step 4: Query
results = rag_pipeline.query("What is this document about?", top_k=3)
for i, result in enumerate(results):
    print(f"\n{i+1}. {result['document'][:100]}...")
    print(f"   Source: {result['metadata']['source']}")
```

---

## 🔄 Complete Workflow

```
┌─────────────────┐
│  Input File     │
│  (PDF/DOCX/etc) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  OCR Pipeline           │
│  pipeline.process_file()│
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  ExtractionResult       │
│  .to_dict()             │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  RAG-Compatible JSON    │
│  {text, source, ...}    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Save as JSON Array     │
│  [{...}, {...}]         │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  RAG Pipeline           │
│  .ingest_from_json()    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Vector Database        │
│  (Ready for Queries)    │
└─────────────────────────┘
```

---

## 📝 Testing

Run the test script to verify the format:

```bash
cd OCR
python test_json_format.py
```

This will:
- Process sample files
- Display RAG-compatible JSON
- Validate format
- Save to `rag_compatible_output.json`

---

## 📚 Documentation

- **`JSON_FORMAT_CHANGE.md`** - Detailed change documentation
- **`RAG_FORMAT_ANSWER.md`** - RAG format specification
- **`RAG_FORMAT_COMPARISON.md`** - Format comparison
- **`QUICK_REFERENCE.md`** - Quick reference guide

