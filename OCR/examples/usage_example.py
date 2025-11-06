"""
Example usage of the Data Ingestion Pipeline
"""
from pathlib import Path
from core.pipeline import DataIngestionPipeline
import json

def main():
    # Initialize pipeline
    pipeline = DataIngestionPipeline()
    
    # Example 1: Process a PDF file
    print("=" * 80)
    print("Example 1: Extract from PDF")
    print("=" * 80)
    
    pdf_result = pipeline.process_file("examples/Rebhi_Mohamed_Amine_Grades.pdf")
    
    if pdf_result.success:
        print(f"✓ Extracted {len(pdf_result.text)} characters")
        print(f"✓ Pages: {pdf_result.metadata.get('num_pages', 'N/A')}")
        print(f"✓ Extraction time: {pdf_result.extraction_time:.2f}s")
        print(f"\nFirst 500 characters:\n{pdf_result.text[:500]}...")
    else:
        print(f"✗ Extraction failed: {pdf_result.error}")
    
    # Example 2: Process a DOCX file
    print("\n" + "=" * 80)
    print("Example 2: Extract from DOCX")
    print("=" * 80)
    
    docx_result = pipeline.process_file("examples/architecture.docx")
    
    if docx_result.success:
        print(f"✓ Extracted {len(docx_result.text)} characters")
        print(f"✓ Paragraphs: {docx_result.metadata.get('num_paragraphs', 'N/A')}")
        print(f"✓ Tables: {docx_result.metadata.get('num_tables', 'N/A')}")
    else:
        print(f"✗ Extraction failed: {docx_result.error}")
    
    # Example 3: Process an image with OCR
    print("\n" + "=" * 80)
    print("Example 3: Extract from Image (OCR)")
    print("=" * 80)
    
    image_result = pipeline.process_file("examples/image.png")
    
    if image_result.success:
        print(f"✓ Extracted {len(image_result.text)} characters")
        print(f"✓ OCR Confidence: {image_result.metadata.get('confidence', 'N/A'):.2f}%")
        print(f"✓ OCR Engine: {image_result.metadata.get('ocr_engine', 'N/A')}")
    else:
        print(f"✗ Extraction failed: {image_result.error}")
    
    # Example 4: Process a web page
    print("\n" + "=" * 80)
    print("Example 4: Extract from Web/HTML")
    print("=" * 80)
    
    web_result = pipeline.process_file("examples/Machine learning - Wikipedia.html")
    
    if web_result.success:
        print(f"✓ Extracted {len(web_result.text)} characters")
        print(f"✓ Title: {web_result.metadata.get('title', 'N/A')}")
        print(f"✓ Author: {web_result.metadata.get('author', 'N/A')}")
    else:
        print(f"✗ Extraction failed: {web_result.error}")
    
    # Example 5: Process audio file
    print("\n" + "=" * 80)
    print("Example 5: Extract from Audio (Whisper)")
    print("=" * 80)
    
    audio_result = pipeline.process_file("examples/Jon_Worthy_-_Things_We_Cant_Ignore.mp3")
    
    if audio_result.success:
        print(f"✓ Transcribed {len(audio_result.text)} characters")
        print(f"✓ Language: {audio_result.metadata.get('language', 'N/A')}")
        print(f"✓ Duration: {audio_result.metadata.get('duration', 'N/A'):.2f}s")
        print(f"✓ Segments: {audio_result.metadata.get('num_segments', 'N/A')}")
    else:
        print(f"✗ Extraction failed: {audio_result.error}")
    
    # Example 6: Process video file
    print("\n" + "=" * 80)
    print("Example 6: Extract from Video")
    print("=" * 80)
    
    video_result = pipeline.process_file("lecture.mp4")
    
    if video_result.success:
        print(f"✓ Transcribed {len(video_result.text)} characters")
        print(f"✓ Language: {video_result.metadata.get('language', 'N/A')}")
        print(f"✓ Duration: {video_result.metadata.get('duration', 'N/A'):.2f}s")
        video_info = video_result.metadata.get('video_info', {})
        print(f"✓ Resolution: {video_info.get('width', 'N/A')}x{video_info.get('height', 'N/A')}")
    else:
        print(f"✗ Extraction failed: {video_result.error}")
    
    # Example 7: Batch processing
    print("\n" + "=" * 80)
    print("Example 7: Batch Processing")
    print("=" * 80)
    
    files = ["examples/Rebhi_Mohamed_Amine_Grades.pdf", "examples/architecture.docx", "examples/Machine learning - Wikipedia.html"]
    results = pipeline.process_batch(files)
    
    print(f"✓ Processed {len(results)} files")
    for i, result in enumerate(results):
        status = "✓" if result.success else "✗"
        print(f"{status} File {i+1}: {result.file_path} - {len(result.text)} chars")
    
    # Example 8: Export result to JSON
    print("\n" + "=" * 80)
    print("Example 8: Export to JSON")
    print("=" * 80)
    
    with open("extraction_result.json", "w", encoding="utf-8") as f:
        json.dump(pdf_result.to_dict(), f, indent=2, ensure_ascii=False)
    
    print("✓ Result exported to extraction_result.json")

if __name__ == "__main__":
    main()