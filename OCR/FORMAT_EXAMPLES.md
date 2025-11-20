# JSON Format Examples by File Type

This document shows the exact JSON output format for each supported file type.

---

## 📄 PDF Document

**Input:** `document.pdf`

**Output:**
```json
{
  "text": "This is the extracted text from the PDF document...",
  "source": "document.pdf",
  "format_type": "pdf",
  "num_pages": 10,
  "title": "Annual Report 2024",
  "author": "John Doe",
  "subject": "Financial Report",
  "creator": "Microsoft Word",
  "producer": "Adobe PDF Library",
  "creation_date": "D:20240115120000",
  "extractor": "pymupdf",
  "success": true
}
```

---

## 📝 DOCX Document

**Input:** `report.docx`

**Output:**
```json
{
  "text": "This is the extracted text from the Word document...",
  "source": "report.docx",
  "format_type": "docx",
  "title": "Project Report",
  "author": "Jane Smith",
  "subject": "Q4 Analysis",
  "keywords": "report, analysis, Q4",
  "created": "2024-01-15 10:30:00",
  "modified": "2024-01-20 14:45:00",
  "num_paragraphs": 45,
  "num_tables": 3,
  "extractor": "python-docx",
  "success": true
}
```

---

## 🖼️ Image (OCR)

**Input:** `scanned_document.png`

**Output:**
```json
{
  "text": "This is the text extracted from the image using OCR...",
  "source": "scanned_document.png",
  "format_type": "image",
  "ocr_engine": "tesseract",
  "confidence": 92.5,
  "languages": ["eng", "fra", "spa", "deu"],
  "extractor": "ocr",
  "success": true
}
```

**With PaddleOCR:**
```json
{
  "text": "Text extracted using PaddleOCR...",
  "source": "image.jpg",
  "format_type": "image",
  "ocr_engine": "paddleocr",
  "confidence": 95.8,
  "languages": ["eng", "fra", "spa", "deu"],
  "extractor": "ocr",
  "success": true
}
```

---

## 🎵 Audio File

**Input:** `lecture.mp3`

**Output:**
```json
{
  "text": "This is the transcribed text from the audio file...",
  "source": "lecture.mp3",
  "format_type": "audio",
  "language": "en",
  "duration": 1825.5,
  "num_segments": 142,
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "Welcome to today's lecture on machine learning."
    },
    {
      "start": 5.2,
      "end": 12.8,
      "text": "We'll be discussing neural networks and deep learning."
    }
  ],
  "model": "large-v3",
  "extractor": "whisper",
  "success": true
}
```

---

## 🎬 Video File

**Input:** `presentation.mp4`

**Output:**
```json
{
  "text": "This is the transcribed audio from the video...",
  "source": "presentation.mp4",
  "format_type": "video",
  "language": "en",
  "duration": 3600.0,
  "num_segments": 280,
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "Hello everyone, welcome to this presentation."
    }
  ],
  "video_info": {
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "codec": "h264"
  },
  "model": "large-v3",
  "extractor": "video",
  "success": true
}
```

---

## 🌐 Web Page / HTML

**Input:** `article.html`

**Output:**
```json
{
  "text": "This is the extracted text from the web page...",
  "source": "article.html",
  "format_type": "web",
  "title": "Introduction to Machine Learning",
  "author": "Dr. Sarah Johnson",
  "description": "A comprehensive guide to machine learning basics",
  "keywords": "machine learning, AI, tutorial",
  "url": "https://example.com/ml-intro",
  "extractor": "beautifulsoup",
  "success": true
}
```

---

## ❌ Failed Extraction

**Input:** `corrupted_file.pdf`

**Output:**
```json
{
  "text": "",
  "source": "corrupted_file.pdf",
  "success": false
}
```

Note: When `success` is `false`, the `text` field will be empty. The error details are available on the `ExtractionResult` object but not in the JSON output.

---

## 📦 Batch Output (Multiple Files)

**Input:** Multiple files processed in batch

**Output:** JSON Array
```json
[
  {
    "text": "Content from first document...",
    "source": "doc1.pdf",
    "format_type": "pdf",
    "num_pages": 5,
    "success": true
  },
  {
    "text": "Content from second document...",
    "source": "doc2.docx",
    "format_type": "docx",
    "num_paragraphs": 20,
    "success": true
  },
  {
    "text": "OCR text from image...",
    "source": "scan.png",
    "format_type": "image",
    "confidence": 88.5,
    "success": true
  }
]
```

---

## 🔍 Field Reference

### Common Fields (All Formats)
| Field | Type | Always Present? | Description |
|-------|------|-----------------|-------------|
| `text` | string | ✅ Yes | Extracted text content |
| `source` | string | ✅ Yes | Source filename |
| `format_type` | string | ✅ Yes | File format (pdf, docx, image, etc.) |
| `success` | boolean | ✅ Yes | Whether extraction succeeded |

### Format-Specific Fields

**PDF:**
- `num_pages`, `title`, `author`, `subject`, `creator`, `producer`, `creation_date`

**DOCX:**
- `num_paragraphs`, `num_tables`, `title`, `author`, `keywords`, `created`, `modified`

**Image:**
- `ocr_engine`, `confidence`, `languages`

**Audio/Video:**
- `language`, `duration`, `num_segments`, `segments`, `model`
- Video only: `video_info` (width, height, fps, codec)

**Web:**
- `title`, `author`, `description`, `keywords`, `url`

---

## 💡 Tips

1. **Always wrap in array for RAG**: Even single documents should be `[{...}]`
2. **Source is filename only**: Full path is stripped to just the filename
3. **Metadata is flattened**: All fields are at top level, no nesting
4. **Success flag**: Check this before processing text
5. **Optional fields**: Not all fields are present for all file types

