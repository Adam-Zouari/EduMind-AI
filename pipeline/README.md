# OCR-RAG Pipeline

Complete end-to-end pipeline that combines OCR extraction with RAG (Retrieval-Augmented Generation) for intelligent document processing and question answering.

## Features

- 🔍 **Multi-format OCR**: Extract text from PDF, DOCX, Images, Audio, Video, and HTML
- 🧠 **Semantic Search**: Find relevant information using embeddings
- 💬 **AI-Powered Q&A**: Get answers using Qwen LLM
- 📊 **Batch Processing**: Process multiple files at once
- 🎯 **Source Attribution**: See where answers come from
- 📚 **Chat History**: Track questions and answers

## Architecture

```
pipeline/
├── orchestrator.py    # Integrates OCR and RAG pipelines
├── app.py            # Streamlit web interface
├── requirements.txt  # Dependencies
└── README.md         # This file
```

The orchestrator coordinates:
1. **OCR Pipeline** (`OCR/core/pipeline.py`) - Extracts content from files
2. **RAG Pipeline** (`RAG/src/rag_pipeline.py`) - Stores and queries documents

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Ollama and pull the model:
```bash
# Install Ollama from https://ollama.ai
ollama pull qwen3:1.7b
```

3. Start Ollama server:
```bash
ollama serve
```

## Usage

### Web Interface (Recommended)

Run the Streamlit app:
```bash
streamlit run pipeline/app.py
```

Then:
1. Click "Initialize Pipeline" in the sidebar
2. Upload files in the "Upload & Process" tab
3. Ask questions in the "Ask Questions" tab

### Python API

```python
from pipeline.orchestrator import OCRRAGOrchestrator

# Initialize
orchestrator = OCRRAGOrchestrator(use_llm=True)

# Process a file
result = orchestrator.process_file(
    file_path="document.pdf",
    ingest_to_rag=True
)

print(f"Extracted {len(result['text'])} characters")
print(f"Created {result['rag_chunks']} chunks")

# Ask questions
answer = orchestrator.query(
    query_text="What is this document about?",
    top_k=5,
    generate_answer=True
)

print(f"Answer: {answer['answer']}")
print(f"Sources: {answer['sources']}")
```

## Supported File Formats

| Format | Extensions | Description |
|--------|-----------|-------------|
| PDF | .pdf | Portable Document Format |
| Word | .docx | Microsoft Word documents |
| Images | .png, .jpg, .jpeg | OCR from images |
| Web | .html | HTML web pages |
| Audio | .mp3, .wav | Speech-to-text transcription |
| Video | .mp4, .avi | Extract audio and transcribe |

## Configuration

Edit `RAG/config/config.yaml` to customize:
- Embedding model
- Chunk size and overlap
- LLM model and parameters
- Vector database settings

## Requirements

- Python 3.8+
- Ollama running locally
- Tesseract OCR (for image processing)
- FFmpeg (for video processing)

## Troubleshooting

**Pipeline initialization fails:**
- Make sure Ollama is running: `ollama serve`
- Check that the model is available: `ollama list`

**OCR extraction fails:**
- For images: Install Tesseract OCR
- For videos: Install FFmpeg

**Out of memory:**
- Reduce chunk size in config.yaml
- Use a smaller LLM model
- Process files one at a time

## License

MIT

