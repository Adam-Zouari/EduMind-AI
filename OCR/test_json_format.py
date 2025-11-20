"""
Test script to demonstrate the new RAG-compatible JSON format output
"""
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.pipeline import DataIngestionPipeline

def test_json_format():
    """Test the new RAG-compatible JSON format for different file types"""

    pipeline = DataIngestionPipeline()

    # Test files
    test_files = [
        "examples/Rebhi_Mohamed_Amine_Grades.pdf",
        "examples/architecture.docx",
        "examples/image.png"
    ]

    print("=" * 80)
    print("Testing RAG-Compatible JSON Format Output")
    print("=" * 80)
    print("\nRAG expects a flat structure with 'text' and 'source' fields")
    print("All metadata should be at the top level (not nested)")
    print("=" * 80)

    all_results = []

    for file_path in test_files:
        if not Path(file_path).exists():
            print(f"\n⚠️  Skipping {file_path} (file not found)")
            continue

        print(f"\n📄 Processing: {file_path}")
        print("-" * 80)

        # Process file
        result = pipeline.process_file(file_path)

        # Convert to JSON format (RAG-compatible)
        json_output = result.to_dict()

        # Add to results array
        all_results.append(json_output)

        # Pretty print JSON
        print(json.dumps(json_output, indent=2, ensure_ascii=False))

        # Verify RAG format
        print("\n✅ RAG Format Validation:")

        # Check required field
        if "text" in json_output:
            print(f"  ✓ 'text' field present (REQUIRED by RAG)")
        else:
            print(f"  ✗ 'text' field MISSING (REQUIRED by RAG)")

        # Check recommended fields
        if "source" in json_output:
            print(f"  ✓ 'source' field present: {json_output['source']} (RECOMMENDED by RAG)")
        else:
            print(f"  ⚠️  'source' field missing (RECOMMENDED by RAG)")

        # Check format_type at top level
        if "format_type" in json_output:
            print(f"  ✓ 'format_type' at top level: {json_output['format_type']}")

        # Check metadata is flattened (not nested)
        if "metadata" in json_output and isinstance(json_output["metadata"], dict):
            print(f"  ⚠️  WARNING: 'metadata' is still nested (should be flattened)")
        else:
            print(f"  ✓ Metadata is flattened (no nested 'metadata' object)")

        # Show some metadata fields
        metadata_fields = [k for k in json_output.keys() if k not in ["text", "source", "success"]]
        if metadata_fields:
            print(f"  ✓ Metadata fields at top level: {', '.join(metadata_fields[:5])}")

        print()

    # Save as RAG-compatible JSON array
    if all_results:
        output_file = "rag_compatible_output.json"
        print("\n" + "=" * 80)
        print(f"💾 Saving RAG-compatible JSON array to: {output_file}")
        print("=" * 80)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Saved {len(all_results)} documents in RAG-compatible format")
        print(f"\nYou can now use this file with RAG:")
        print(f"  from RAG.src.rag_pipeline import RAGPipeline")
        print(f"  pipeline = RAGPipeline()")
        print(f"  chunks = pipeline.ingest_from_json('{output_file}')")

if __name__ == "__main__":
    test_json_format()

