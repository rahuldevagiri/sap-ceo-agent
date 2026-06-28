import json
from pathlib import Path
from src.config import RAW_DIR, MIN_TARGET_DOCUMENTS
from src.collector.company_collector import collect_company_sources
from src.collector.news_collector import collect_news_sources
from src.collector.market_collector import collect_market_sources
from src.collector.utils import make_hash

def deduplicate_documents(documents):
    seen = set()
    unique_docs = []

    for doc in documents:
        text_for_hash = f"{doc.title} {doc.content[:1000]}"
        content_hash = make_hash(text_for_hash)

        if content_hash in seen:
            continue

        seen.add(content_hash)
        doc.content_hash = content_hash
        unique_docs.append(doc)

    return unique_docs

def save_documents(documents, filename="sap_raw_documents.json"):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([doc.model_dump() for doc in documents], f, ensure_ascii=False, indent=2)

    return output_path

def run_collection():
    company_docs = collect_company_sources()
    news_docs = collect_news_sources(max_results_per_query=15)
    market_docs = collect_market_sources(max_results_per_query=12)

    all_docs = company_docs + news_docs + market_docs
    unique_docs = deduplicate_documents(all_docs)

    output_path = save_documents(unique_docs)

    print(f"Company docs: {len(company_docs)}")
    print(f"News docs: {len(news_docs)}")
    print(f"Market docs: {len(market_docs)}")
    print(f"Unique docs saved: {len(unique_docs)}")
    print(f"Saved to: {output_path}")

    if len(unique_docs) < MIN_TARGET_DOCUMENTS:
        print( 
            f"WARNING: collected {len(unique_docs)} unique docs, "
            f"below target of {MIN_TARGET_DOCUMENTS}. "
            f"DuckDuckGo may be rate-limiting — re-run to top up."
        )

    return unique_docs

if __name__ == "__main__":
    run_collection()