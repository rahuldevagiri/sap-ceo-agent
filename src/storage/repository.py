import os
import logging

# Disable ChromaDB's anonymous telemetry before importing it. With some
# chromadb/posthog version combinations the telemetry client raises
# "capture() takes 1 positional argument but 3 were given" and spams the
# console; the setting + logger silence both the calls and the noise.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# Load the embedding model from the local HuggingFace cache only. Without this,
# sentence-transformers makes a network HEAD request to huggingface.co on every
# startup; if the machine is offline that retries for ~25s and then fails. The
# model is already cached locally, so force offline mode (overridable via env if
# a fresh download is ever needed).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from src.config import COLLECTION_NAME, EMBEDDING_MODEL  # noqa: E402


PERSIST_DIR = os.path.join("data", "embeddings")


class Repository:
    def __init__(self):
        os.makedirs(PERSIST_DIR, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

        # Pin the embedding model to CPU. On a small GPU (e.g. 4 GB GTX 1650)
        # loading it on CUDA steals VRAM + creates a CUDA context that the LLM
        # (Ollama) needs, causing host-buffer allocation failures. MiniLM is
        # tiny, so CPU embedding of a few queries is fast enough.
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    def add_documents(self, ids, embeddings, metadatas, documents):
        if not ids:
            return

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def search(self, query, top_k=5, where=None):
        query_embedding = self.embedder.encode(query).tolist()

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k
        }

        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        formatted_results = []
        for i in range(len(ids)):
            metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            formatted_results.append({
                "doc_id": ids[i],
                "content": documents[i] if i < len(documents) else "",
                "score": distances[i] if i < len(distances) else None,
                "title": metadata.get("title", "No title"),
                "source": metadata.get("source", ""),
                "source_type": metadata.get("source_type", "unknown"),
                "url": metadata.get("url", ""),
                "published": metadata.get("published", ""),
                "category": metadata.get("category", "general"),
                "competitor": metadata.get("competitor", ""),
            })

        return formatted_results

    def count(self):
        return self.collection.count()

    def corpus_stats(self):
        """Corpus-wide totals across everything indexed (not just a query).

        Counts distinct source documents (many chunks share one doc_id) and
        the distribution of chunks by source type and source.
        """
        from collections import Counter

        total_chunks = self.collection.count()
        if total_chunks == 0:
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "source_type_counts": {},
                "source_counts": {},
            }

        data = self.collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", []) or []

        doc_ids = {m.get("doc_id") for m in metadatas if m.get("doc_id")}
        source_type_counts = Counter(m.get("source_type", "unknown") for m in metadatas)
        source_counts = Counter(m.get("source", "unknown") for m in metadatas)

        published_values = [m.get("published", "") for m in metadatas if m.get("published")]
        collected_values = [m.get("collected_at", "") for m in metadatas if m.get("collected_at")]
        last_updated = max(collected_values) if collected_values else (
            max(published_values) if published_values else ""
        )

        return {
            "total_chunks": total_chunks,
            "total_documents": len(doc_ids),
            "distinct_sources": len(source_counts),
            "last_updated": last_updated,
            "source_type_counts": dict(source_type_counts),
            "source_counts": dict(source_counts),
        }

    def all_documents(self):
        """Return one representative record per indexed document (deduped by
        doc_id), used for corpus-wide analysis such as sentiment over the whole
        collection rather than just the retrieved subset."""
        if self.collection.count() == 0:
            return []

        data = self.collection.get(include=["metadatas", "documents"])
        metadatas = data.get("metadatas", []) or []
        documents = data.get("documents", []) or []

        by_doc = {}
        for i, meta in enumerate(metadatas):
            doc_id = meta.get("doc_id") or f"_{i}"
            content = documents[i] if i < len(documents) else ""
            # Prefer the first chunk (chunk_index "1") as the representative text.
            existing = by_doc.get(doc_id)
            if existing is None or str(meta.get("chunk_index")) == "1":
                by_doc[doc_id] = {
                    "doc_id": doc_id,
                    "title": meta.get("title", "No title"),
                    "content": content,
                    "source": meta.get("source", ""),
                    "source_type": meta.get("source_type", "unknown"),
                    "url": meta.get("url", ""),
                    "published": meta.get("published", ""),
                    "competitor": meta.get("competitor", ""),
                }
        return list(by_doc.values())

    def peek(self, limit=5):
        return self.collection.peek(limit=limit)

    def reset_collection(self):
        self.client.delete_collection(name=COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )