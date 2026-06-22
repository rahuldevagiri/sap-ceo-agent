import feedparser
import requests
from bs4 import BeautifulSoup
from uuid import uuid4
from src.schemas import DocumentRecord
from src.config import SAP_SOURCES

def collect_rss_documents():
    documents = []
    for rss_url in SAP_SOURCES["company_rss"]:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries:
            content = ""
            try:
                page = requests.get(entry.link, timeout=20)
                soup = BeautifulSoup(page.text, "lxml")
                paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
                content = " ".join(paragraphs)
            except Exception:
                content = entry.get("summary", "")

            doc = DocumentRecord(
                doc_id=str(uuid4()),
                title=entry.get("title", "No title"),
                source="SAP RSS",
                url=entry.get("link", ""),
                published=entry.get("published", None),
                content=content,
                category="company_news",
                tags=["sap", "official"]
            )
            documents.append(doc)
    return documents