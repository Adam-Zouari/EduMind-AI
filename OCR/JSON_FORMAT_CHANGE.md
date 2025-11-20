# JSON Format Change Documentation

## Summary

Updated `ExtractionResult.to_dict()` method in `base_extractor.py` to output JSON in the **RAG-compatible format** with flattened metadata structure.

## Changes Made

### File Modified
- `OCR/core/base_extractor.py`

### Old Format (Nested Metadata)
```json
{
  "text": "Extracted content...",
  "metadata": {
    "num_pages": 5,
    "confidence": 92.5
  },
  "format_type": "pdf",
  "file_path": "document.pdf",
  "extraction_time": 1.23,
  "success": true,
  "error": null,
  "timestamp": "2025-11-06T21:33:32.201996"
}
```

### New Format (RAG-Compatible, Flattened)
```json
{
  "text": "Extracted content...",
  "source": "document.pdf",
  "format_type": "pdf",
  "num_pages": 5,
  "confidence": 92.5,
  "success": true
}
```

## Key Differences

### Removed Fields
- ❌ `extraction_time` - No longer in output
- ❌ `error` - No longer in output
- ❌ `timestamp` - No longer in output
- ❌ `file_path` - Replaced with `source`

### Changed Fields
- 🔄 `file_path` → `source` - Now uses just the filename (RAG convention)
- 🔄 `metadata` - **FLATTENED** to top level instead of nested object

### New Structure
- ✅ **Flat metadata**: All metadata fields at top level
- ✅ **`source` field**: Maps from `file_path` (RAG convention)
- ✅ **`format_type`**: At top level (not nested)
- ✅ **All extractor metadata**: Flattened (num_pages, confidence, etc.)

## Benefits

1. **RAG Compatible**: Matches exact format expected by RAG pipeline
2. **Flat Structure**: No nested metadata object - easier to query and filter
3. **Standard Fields**: Uses `source` instead of `file_path` (RAG convention)
4. **Cleaner Output**: Only essential fields, all at top level
5. **Better Filtering**: Metadata at top level enables easier filtering in RAG queries

## Implementation Details

The `to_dict()` method now:
1. Starts with the `text` field (required by RAG)
2. **Flattens all metadata** to the top level using `update()`
3. Adds `format_type` if not already present
4. Maps `file_path` to `source` (extracts just the filename)
5. Adds `success` status

```python
def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary in the format required for RAG pipeline"""
    # Start with text field (required by RAG)
    result = {"text": self.text}

    # Flatten all metadata fields to top level
    result.update(self.metadata)

    # Add format_type if not already in metadata
    if "format_type" not in result:
        result["format_type"] = self.format_type

    # Map file_path to 'source' (RAG convention)
    # Extract just the filename from the path
    if self.file_path:
        result["source"] = Path(self.file_path).name

    # Add success status
    result["success"] = self.success

    return result
```

## Testing

Run the test script to verify the new RAG-compatible format:

```bash
cd OCR
python test_json_format.py
```

This will:
- Process sample files (PDF, DOCX, images)
- Display the JSON output in RAG-compatible format
- Validate the format against RAG requirements
- Save results to `rag_compatible_output.json` as a JSON array

## Backward Compatibility

⚠️ **Breaking Change**: Code that relies on the following will need to be updated:

### Removed from JSON Output:
- `extraction_time` - No longer in JSON
- `error` - No longer in JSON
- `timestamp` - No longer in JSON

### Changed in JSON Output:
- `file_path` → `source` (now just filename, not full path)
- `metadata` object → Flattened to top level
- Access `num_pages` directly instead of `metadata.num_pages`

### Still Available:
These fields are still available in the `ExtractionResult` object itself, just not in the JSON output.

## Migration Guide

### Accessing Old Fields

If you need the old fields, access them directly from the `ExtractionResult` object:

```python
result = pipeline.process_file("document.pdf")

# Access fields directly (still available on object)
print(result.extraction_time)  # Still available
print(result.error)            # Still available
print(result.timestamp)        # Still available
print(result.format_type)      # Still available
print(result.file_path)        # Still available (full path)

# New JSON format (RAG-compatible)
json_output = result.to_dict()
print(json_output["format_type"])  # Now at top level (flattened)
print(json_output["source"])       # Filename only
print(json_output["num_pages"])    # Metadata flattened to top level
```

### Updating Code

**Old way (nested metadata):**
```python
json_output = result.to_dict()
num_pages = json_output["metadata"]["num_pages"]
format_type = json_output["format_type"]
file_path = json_output["file_path"]
```

**New way (flattened):**
```python
json_output = result.to_dict()
num_pages = json_output["num_pages"]      # Direct access
format_type = json_output["format_type"]  # Direct access
source = json_output["source"]            # Filename only
```

## Example Usage

### Single Document

```python
from core.pipeline import DataIngestionPipeline
import json

pipeline = DataIngestionPipeline()
result = pipeline.process_file("examples/document.pdf")

# Export to JSON (RAG-compatible)
with open("output.json", "w", encoding="utf-8") as f:
    json.dump([result.to_dict()], f, indent=2, ensure_ascii=False)
```

Output:
```json
[
  {
    "text": "Document content here...",
    "source": "document.pdf",
    "format_type": "pdf",
    "num_pages": 10,
    "title": "Sample Document",
    "author": "John Doe",
    "extractor": "pymupdf",
    "success": true
  }
]
```

### Multiple Documents (Batch)

```python
from core.pipeline import DataIngestionPipeline
import json

pipeline = DataIngestionPipeline()
files = ["doc1.pdf", "doc2.docx", "image.png"]

# Process all files
results = []
for file in files:
    result = pipeline.process_file(file)
    results.append(result.to_dict())

# Save as RAG-compatible JSON array
with open("rag_output.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Now use with RAG
from RAG.src.rag_pipeline import RAGPipeline
rag = RAGPipeline()
chunks = rag.ingest_from_json("rag_output.json")
print(f"Ingested {chunks} chunks into RAG")
```

