# Quick Reference: New JSON Format

## ✅ What Changed

The `ExtractionResult.to_dict()` method now returns a simplified JSON format optimized for the RAG pipeline.

## 📋 New JSON Structure

```json
{
  "text": "Extracted content...",
  "metadata": {
    "num_pages": 5,
    "confidence": 92.5,
    "format_type": "pdf"
  },
  "file_path": "document.pdf",
  "success": true
}
```

## 🔑 Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Extracted text content |
| `metadata` | object | Document metadata including `format_type` |
| `file_path` | string | Path to the source file |
| `success` | boolean | Whether extraction succeeded |

## 📊 Metadata Contents

The `metadata` object contains:
- **format_type**: File format (pdf, docx, image, audio, video, web)
- **Format-specific fields**:
  - **PDF**: `num_pages`, `title`, `author`, `subject`, `creator`, `producer`
  - **DOCX**: `num_paragraphs`, `num_tables`, `title`, `author`, `keywords`
  - **Image**: `confidence`, `ocr_engine`, `languages`
  - **Audio/Video**: `language`, `duration`, `num_segments`, `model`
  - **Web**: `title`, `author`, `description`, `url`

## 💻 Code Examples

### Basic Usage
```python
from core.pipeline import DataIngestionPipeline
import json

pipeline = DataIngestionPipeline()
result = pipeline.process_file("document.pdf")

# Get JSON format
json_data = result.to_dict()

# Save to file
with open("output.json", "w") as f:
    json.dump(json_data, f, indent=2)
```

### Accessing Data
```python
# Access text
text = json_data["text"]

# Access metadata
format_type = json_data["metadata"]["format_type"]
num_pages = json_data["metadata"].get("num_pages")

# Check success
if json_data["success"]:
    print(f"Successfully extracted {len(text)} characters")
```

### Batch Processing
```python
pipeline = DataIngestionPipeline()
files = ["doc1.pdf", "doc2.docx", "image.png"]

results = []
for file in files:
    result = pipeline.process_file(file)
    results.append(result.to_dict())

# Save all results
with open("batch_output.json", "w") as f:
    json.dump(results, f, indent=2)
```

## 🔄 Integration with RAG

The new format is designed to work seamlessly with the RAG pipeline:

```python
# OCR side
from core.pipeline import DataIngestionPipeline

pipeline = DataIngestionPipeline()
result = pipeline.process_file("document.pdf")

# Save for RAG
with open("ocr_output.json", "w") as f:
    json.dump([result.to_dict()], f)

# RAG side
from RAG.src.rag_pipeline import RAGPipeline

rag = RAGPipeline()
chunks = rag.ingest_from_json("ocr_output.json")
print(f"Ingested {chunks} chunks")
```

## ⚠️ Important Notes

1. **format_type location**: Now inside `metadata`, not top-level
2. **Removed fields**: `extraction_time`, `error`, `timestamp` not in JSON
3. **Still accessible**: These fields exist on the `ExtractionResult` object
4. **Metadata varies**: Different file types have different metadata fields

## 🧪 Testing

Test the new format:
```bash
cd OCR
python test_json_format.py
```

## 📖 Full Documentation

See `JSON_FORMAT_CHANGE.md` for complete details and migration guide.

