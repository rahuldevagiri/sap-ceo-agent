import json
import re
from pathlib import Path
from src.config import RAW_DIR, PROCESSED_DIR
from src.collector.utils import make_hash

MIN_CONTENT_LENGTH = 120

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\u00a0", " ", text)
    return text.strip()

def normalize_tags(tags):
    if not tags:
        return []
    cleaned = []
    for tag in tags:
        if not tag:
            continue
        tag = str(tag).strip().lower().replace(" ", "_")
        if tag and tag not in cleaned:
            cleaned.append(tag)
    return cleaned

def clean_documents():
    input_file = RAW_DIR / "sap_raw_documents.json"
    output_file = PROCESSED_DIR / "sap_cleaned_documents.json"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        docs = json.load(f)

    cleaned_docs = []
    seen_hashes = set()

    for doc in docs:
        title = normalize_text(doc.get("title", ""))
        content = normalize_text(doc.get("content", ""))

        if len(content) < MIN_CONTENT_LENGTH:
            continue

        content_hash = doc.get("content_hash") or make_hash(f"{title} {content[:1000]}")
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        cleaned_doc = {
            "doc_id": doc.get("doc_id"),
            "company": doc.get("company", "SAP"),
            "title": title,
            "source": normalize_text(doc.get("source", "")),
            "source_type": normalize_text(doc.get("source_type", "unknown")),
            "url": normalize_text(doc.get("url", "")),
            "published": doc.get("published"),
            "author": normalize_text(doc.get("author", "")) or None,
            "content": content,
            "category": normalize_text(doc.get("category", "general")),
            "subcategory": normalize_text(doc.get("subcategory", "")) or None,
            "competitor": normalize_text(doc.get("competitor", "")) or None,
            "sentiment": normalize_text(doc.get("sentiment", "")) or None,
            "language": normalize_text(doc.get("language", "en")) or "en",
            "tags": normalize_tags(doc.get("tags", [])),
            "collected_at": doc.get("collected_at"),
            "content_hash": content_hash
        }

        cleaned_docs.append(cleaned_doc)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_docs, f, ensure_ascii=False, indent=2)

    print(f"Input docs: {len(docs)}")
    print(f"Cleaned docs: {len(cleaned_docs)}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    clean_documents()