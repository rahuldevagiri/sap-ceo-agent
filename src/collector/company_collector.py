import feedparser
from uuid import uuid4
from src.schemas import DocumentRecord
from src.collector.utils import fetch_url_text, now_utc_iso

SAP_COMPANY_RSS = [
    "https://news.sap.com/feed/"
]

def collect_company_sources():
    documents = []

    for rss_url in SAP_COMPANY_RSS:
        feed = feedparser.parse(rss_url)

        for entry in feed.entries:
            url = entry.get("link", "")
            content = fetch_url_text(url)
            if not content:
                content = entry.get("summary", "")

            if len(content.strip()) < 200:
                continue

            doc = DocumentRecord(
                doc_id=str(uuid4()),
                title=entry.get("title", "No title"),
                source="SAP Newsroom",
                source_type="company",
                url=url,
                published=entry.get("published", None),
                author=entry.get("author", None),
                content=content,
                category="company_news",
                tags=["sap", "official", "company"],
                collected_at=now_utc_iso()
            )
            documents.append(doc)

    return documents