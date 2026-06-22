from ddgs import DDGS
from uuid import uuid4
from src.schemas import DocumentRecord
from src.collector.utils import fetch_url_text, now_utc_iso, ddgs_search

NEWS_QUERIES = [
    "SAP earnings",
    "SAP AI strategy",
    "SAP cloud business",
    "SAP enterprise software competition",
    "SAP partnership technology news",
    "SAP S/4HANA migration",
    "SAP RISE cloud transformation",
    "SAP Business AI Joule",
    "SAP quarterly results revenue",
    "SAP acquisition deal",
    "SAP data analytics platform",
    "SAP customer enterprise deployment",
    "SAP stock outlook analyst",
    "SAP product launch announcement",
]

def collect_news_sources(max_results_per_query: int = 15):
    documents = []
    ddgs = DDGS()

    for query in NEWS_QUERIES:
        results = ddgs_search(ddgs.news, query, max_results_per_query)

        for item in results:
            url = item.get("url", "")
            title = item.get("title", "No title")
            source = item.get("source", "News")
            published = item.get("date", None)

            content = fetch_url_text(url)
            if len(content.strip()) < 200:
                body = item.get("body", "")
                content = body

            if len(content.strip()) < 80:
                continue

            doc = DocumentRecord(
                doc_id=str(uuid4()),
                title=title,
                source=source,
                source_type="news",
                url=url,
                published=published,
                content=content,
                category="industry_news",
                tags=["sap", "news", "industry"],
                collected_at=now_utc_iso()
            )
            documents.append(doc)

    return documents