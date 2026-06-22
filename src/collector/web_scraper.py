# src/collector/web_scraper.py
import requests
from bs4 import BeautifulSoup
import json
import os
import hashlib
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SCRAPING_TARGETS, DATA_RAW_DIR

# Mimic a real browser so websites don't block us
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def generate_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]

def scrape_sap_newsroom(url: str, source_config: dict) -> list:
    """
    Scrape SAP's newsroom page for press releases.
    Extracts article titles, links, and summaries.
    """
    print(f"  Scraping: {source_config['description']}...")
    articles = []

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()  # Raise error if request failed

        soup = BeautifulSoup(response.text, "html.parser")

        # SAP newsroom uses article cards — find all of them
        # We look for common article container patterns
        article_elements = (
            soup.find_all("article") or
            soup.find_all("div", class_=lambda x: x and "news" in x.lower()) or
            soup.find_all("div", class_=lambda x: x and "card" in x.lower())
        )

        print(f"    Found {len(article_elements)} article elements")

        for elem in article_elements[:25]:
            # Extract title
            title_tag = (
                elem.find("h1") or elem.find("h2") or
                elem.find("h3") or elem.find("h4")
            )
            title = title_tag.get_text(strip=True) if title_tag else ""

            # Extract link
            link_tag = elem.find("a", href=True)
            link = link_tag["href"] if link_tag else ""
            if link and not link.startswith("http"):
                link = "https://news.sap.com" + link

            # Extract summary/description
            para = elem.find("p")
            summary = para.get_text(strip=True) if para else ""

            if not title or len(title) < 10:
                continue

            content = f"{title}. {summary}".strip()

            articles.append({
                "id": generate_id(link or title),
                "title": title,
                "content": content,
                "url": link,
                "published_date": datetime.now().isoformat(),
                "source_name": "sap_investor_relations",
                "source_category": source_config["category"],
                "source_description": source_config["description"],
                "collected_at": datetime.now().isoformat(),
                "company_relevance": "SAP"
            })

        print(f"    ✓ Scraped {len(articles)} items")
        return articles

    except Exception as e:
        print(f"    ✗ Scraping error: {e}")
        return []

def scrape_all_targets() -> dict:
    """Run all web scrapers."""
    print("\n" + "="*50)
    print("WEB SCRAPING STARTED")
    print("="*50)

    all_articles = []
    results = {}

    for source_name, config in SCRAPING_TARGETS.items():
        articles = scrape_sap_newsroom(config["url"], config)
        all_articles.extend(articles)
        results[source_name] = {"count": len(articles)}

    # Save
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    filepath = os.path.join(DATA_RAW_DIR, "web_scraped.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2, ensure_ascii=False)

    print(f"\nSCRAPING COMPLETE: {len(all_articles)} items saved")
    return results

if __name__ == "__main__":
    scrape_all_targets()