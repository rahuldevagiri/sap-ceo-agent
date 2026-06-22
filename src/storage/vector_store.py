import os

# Use the local HuggingFace cache for the embedding model (avoids a network
# call to huggingface.co on startup, which hangs/fails when offline).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json  # noqa: E402
import chromadb  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from src.config import PROCESSED_DIR, EMBEDDINGS_DIR, COLLECTION_NAME, EMBEDDING_MODEL  # noqa: E402

def safe_str(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)

def build_vector_store():
    input_file = PROCESSED_DIR / "sap_chunks.json"

    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(EMBEDDINGS_DIR))

    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(COLLECTION_NAME)

    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL)

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["content"])

        metadata = {
            "doc_id": safe_str(chunk.get("doc_id")),
            "chunk_index": safe_str(chunk.get("chunk_index")),
            "company": safe_str(chunk.get("company")),
            "title": safe_str(chunk.get("title")),
            "source": safe_str(chunk.get("source")),
            "source_type": safe_str(chunk.get("source_type")),
            "url": safe_str(chunk.get("url")),
            "published": safe_str(chunk.get("published")),
            "author": safe_str(chunk.get("author")),
            "category": safe_str(chunk.get("category")),
            "subcategory": safe_str(chunk.get("subcategory")),
            "competitor": safe_str(chunk.get("competitor")),
            "sentiment": safe_str(chunk.get("sentiment")),
            "language": safe_str(chunk.get("language")),
            "tags": safe_str(chunk.get("tags")),
            "collected_at": safe_str(chunk.get("collected_at")),
            "content_hash": safe_str(chunk.get("content_hash"))
        }
        metadatas.append(metadata)

    print("Loading embedding model...")
    chunk_embeddings = model.encode(documents, show_progress_bar=True)

    for idx, emb in enumerate(chunk_embeddings):
        embeddings.append(emb.tolist())

    batch_size = 100

    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end]
        )

    print(f"Indexed chunks: {len(ids)}")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Vector DB path: {EMBEDDINGS_DIR}")

if __name__ == "__main__":
    build_vector_store()