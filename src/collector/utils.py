import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
import hashlib
import re
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
}

def fetch_url_text(url: str, timeout: int = 20) -> str:
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(paragraphs)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return ""

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def make_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def ddgs_search(search_fn, query, max_results, retries: int = 3, backoff: float = 2.0):
    """Call a DDGS search method with retry/backoff.

    DuckDuckGo search frequently rate-limits or returns empty results, which
    is the main reason collection volume is unreliable. Retrying with a short
    backoff makes the per-query yield far more consistent.
    """
    for attempt in range(retries):
        try:
            results = list(search_fn(query, max_results=max_results))
            if results:
                return results
        except Exception:
            pass
        time.sleep(backoff * (attempt + 1))
    return []