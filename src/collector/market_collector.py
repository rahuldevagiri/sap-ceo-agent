from ddgs import DDGS
from uuid import uuid4
from src.schemas import DocumentRecord
from src.collector.utils import fetch_url_text, now_utc_iso, ddgs_search

MARKET_QUERIES = [
    "Oracle product announcement enterprise software",
    "Salesforce AI enterprise announcement",
    "Microsoft cloud enterprise software announcement",
    "Workday enterprise AI announcement",
    "ServiceNow enterprise platform update",
    "Oracle cloud ERP news",
    "Salesforce Agentforce AI",
    "Microsoft Dynamics 365 enterprise",
    "Workday financial HR cloud news",
    "ServiceNow AI automation announcement",
    "enterprise software market trends 2025",
    "ERP cloud competition analysis",
]

COMPETITOR_MAP = {
    "oracle": "Oracle",
    "salesforce": "Salesforce",
    "microsoft": "Microsoft",
    "workday": "Workday",
    "servicenow": "ServiceNow"
}

def detect_competitor(text: str):
    lower = text.lower()
    for key, value in COMPETITOR_MAP.items():
        if key in lower:
            return value
    return None

def collect_market_sources(max_results_per_query: int = 12):
    documents = []
    ddgs = DDGS()

    for query in MARKET_QUERIES:
        results = ddgs_search(ddgs.text, query, max_results_per_query)

        for item in results:
            url = item.get("href", "")
            title = item.get("title", "No title")
            source = item.get("hostname", "Market Source")
            body = item.get("body", "")

            content = fetch_url_text(url)
            if len(content.strip()) < 150:
                content = body

            if len(content.strip()) < 80:
                continue

            combined_text = f"{title} {content}"
            competitor = detect_competitor(combined_text)

            doc = DocumentRecord(
                doc_id=str(uuid4()),
                title=title,
                source=source,
                source_type="market",
                url=url,
                content=content,
                category="competitor_activity",
                competitor=competitor,
                tags=["market", "competitor", "enterprise_software"],
                collected_at=now_utc_iso()
            )
            documents.append(doc)

    return documents