import json
from uuid import uuid4
from src.config import PROCESSED_DIR

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 120

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if len(chunk) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk)

        if end >= text_length:
            break

        start += chunk_size - overlap

    return chunks

def build_chunks():
    input_file = PROCESSED_DIR / "sap_cleaned_documents.json"
    output_file = PROCESSED_DIR / "sap_chunks.json"

    with open(input_file, "r", encoding="utf-8") as f:
        docs = json.load(f)

    chunk_records = []

    for doc in docs:
        doc_chunks = chunk_text(doc["content"])

        for idx, chunk in enumerate(doc_chunks, start=1):
            chunk_records.append({
                "chunk_id": str(uuid4()),
                "chunk_index": idx,
                "doc_id": doc["doc_id"],
                "company": doc.get("company", "SAP"),
                "title": doc.get("title", ""),
                "source": doc.get("source", ""),
                "source_type": doc.get("source_type", "unknown"),
                "url": doc.get("url", ""),
                "published": doc.get("published"),
                "author": doc.get("author"),
                "category": doc.get("category", "general"),
                "subcategory": doc.get("subcategory"),
                "competitor": doc.get("competitor"),
                "sentiment": doc.get("sentiment"),
                "language": doc.get("language", "en"),
                "tags": doc.get("tags", []),
                "collected_at": doc.get("collected_at"),
                "content_hash": doc.get("content_hash"),
                "content": chunk
            })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunk_records, f, ensure_ascii=False, indent=2)

    print(f"Input docs: {len(docs)}")
    print(f"Total chunks created: {len(chunk_records)}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    build_chunks()