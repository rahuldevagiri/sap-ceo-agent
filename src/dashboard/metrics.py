from collections import Counter


def build_dashboard_metrics(retrieved: dict, analysis: dict):
    all_items = []
    for _, items in retrieved.items():
        all_items.extend(items)

    total_docs = len({item.get("doc_id", "") for item in all_items if item.get("doc_id")})
    total_sources = len({item.get("source", "") for item in all_items if item.get("source")})

    published_values = [item.get("published", "") for item in all_items if item.get("published")]
    last_update = max(published_values) if published_values else "N/A"

    source_type_counts = Counter(item.get("source_type", "unknown") for item in all_items)
    source_counts = Counter(item.get("source", "unknown") for item in all_items)

    metrics = {
        "total_docs": total_docs,
        "total_sources": total_sources,
        "last_update": last_update,
        "opportunity_count": len(analysis.get("opportunities", [])),
        "risk_count": len(analysis.get("risks", [])),
        "competitor_count": len(analysis.get("competitor_signals", [])),
        "trend_count": len(analysis.get("trends", [])),
        "source_type_counts": dict(source_type_counts),
        "source_counts": dict(source_counts),
    }

    return metrics