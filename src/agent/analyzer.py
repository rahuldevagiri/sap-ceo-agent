import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List


class IntelligenceAnalyzer:
    def __init__(self):
        self.opportunity_keywords = [
            "growth", "ai", "cloud", "partnership", "acquire", "acquisition",
            "expansion", "innovation", "opportunity", "revenue", "momentum",
            "platform", "automation", "data", "productivity", "backlog"
        ]

        self.risk_keywords = [
            "risk", "threat", "regulation", "decline", "negative", "slowdown",
            "supply", "uncertainty", "pressure", "lawsuit", "security",
            "breach", "competition", "debt", "volatility", "geopolitical",
            "downside", "exposure"
        ]

        self.trend_keywords = [
            "trend", "ai", "governance", "cloud", "automation", "data",
            "enterprise", "platform", "migration", "transformation",
            "digital", "erp", "analytics", "sovereign", "assistant"
        ]

        self.positive_keywords = [
            "growth", "beat", "increase", "momentum", "strong", "surge",
            "innovation", "opportunity", "expand", "accelerate", "gain"
        ]

        self.negative_keywords = [
            "risk", "uncertainty", "pressure", "decline", "threat",
            "slowdown", "volatility", "exposure", "breach", "downside"
        ]

        # Ordered risk taxonomy for the dashboard Risk Monitor "category" field.
        self.risk_category_keywords = {
            "Competitive": [
                "competition", "competitor", "rival", "market share",
                "oracle", "salesforce", "microsoft", "workday", "servicenow",
            ],
            "Regulatory": [
                "regulation", "regulatory", "lawsuit", "antitrust", "compliance",
                "gdpr", "investigation", "fine", "probe", "ruling",
            ],
            "Security": [
                "breach", "security", "cyber", "hack", "vulnerability", "outage", "ransomware",
            ],
            "Supply chain": [
                "supply", "logistics", "shortage", "disruption", "vendor", "procurement",
            ],
            "Financial": [
                "debt", "decline", "miss", "cost", "margin", "volatility",
                "downgrade", "loss", "guidance", "slowdown",
            ],
            "Reputation / Sentiment": [
                "negative", "backlash", "criticism", "reputation", "controversy", "complaint",
            ],
        }

    def classify_risk_category(self, text: str) -> str:
        text = text.lower()
        best_category = None
        best_score = 0
        for category, keywords in self.risk_category_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > best_score:
                best_score = score
                best_category = category
        return best_category or "General"

    def parse_published(self, published: str):
        if not published:
            return None
        try:
            dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except Exception:
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(published)
            except Exception:
                return None
        if dt is not None and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def build_sentiment_trend(self, analyzed_items: List[Dict], window_days: int = 1095) -> List[Dict]:
        """Group sentiment by calendar month using each document's publish date.

        Scraped pages sometimes carry junk publish dates (e.g. 2000, 2004, or
        future dates). We keep only dates within `window_days` of now (default
        ~3 years) and not in the future, so the trend reflects real recent data.
        """
        now = datetime.now(timezone.utc)
        buckets = defaultdict(lambda: {"Positive": 0, "Negative": 0, "Neutral": 0})
        for item in analyzed_items:
            dt = self.parse_published(item.get("published", ""))
            if dt is None:
                continue
            age_days = (now - dt).days
            if age_days < 0 or age_days > window_days:
                continue
            period = dt.strftime("%Y-%m")
            sentiment = item.get("sentiment", "Neutral")
            buckets[period][sentiment] = buckets[period].get(sentiment, 0) + 1

        rows = []
        for period in sorted(buckets):
            counts = buckets[period]
            rows.append({
                "period": period,
                "Positive": counts["Positive"],
                "Negative": counts["Negative"],
                "Neutral": counts["Neutral"],
                "net": counts["Positive"] - counts["Negative"],
            })
        return rows

    def normalize_title(self, title: str) -> str:
        return " ".join(title.lower().strip().split())

    # Sentences containing these markers are page boilerplate, not content.
    BOILERPLATE_MARKERS = (
        "media contact", "news center", "press room", "press@", "@sap.com",
        "forward-looking", "this document contains", "for more information",
        "all rights reserved", "subscribe", "cookie", "follow us", "via linkedin",
        "via bluesky", "get sap news", "read more", "click here", "sign up",
        "subject to risks and uncertainties", "these statements are based",
        "expectations, forecasts", "buy, sell or hold", "simply wall st",
        "company report for", "is not investment advice", "terms of use",
        "no longer supported", "upgrade to microsoft edge", "partner newsletter",
        "search search", "enable javascript", "skip to", "main navigation",
        "privacy policy", "menu menu", "requires authorization", "access to this page",
        "signing in", "changing directories",
    )

    # Common English stopwords used to tell prose from navigation word-lists.
    _STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "in",
        "for", "on", "with", "that", "this", "as", "by", "at", "its", "it",
        "will", "has", "have", "from", "their", "which", "but", "they", "we",
    }

    def _is_boilerplate_sentence(self, sentence: str) -> bool:
        low = sentence.lower()
        if any(marker in low for marker in self.BOILERPLATE_MARKERS):
            return True
        # Navigation menus / category lists are long runs of Capitalized words
        # with almost no stopwords and little punctuation.
        words = sentence.split()
        if len(words) >= 12 and sentence.count(",") <= 1:
            stop_ratio = sum(1 for w in words if w.lower().strip(".,") in self._STOPWORDS) / len(words)
            if stop_ratio < 0.12:
                return True
        return False

    def build_clean_summary(self, content: str, max_len: int = 420) -> str:
        """Produce a readable, on-topic excerpt from a retrieved chunk.

        Chunks are split at fixed character offsets, so they often begin in the
        middle of a sentence and may contain page boilerplate (contact details,
        legal disclaimers, social links). We drop the broken leading fragment and
        boilerplate sentences, then keep the first substantive sentences.
        """
        text = re.sub(r"\s+", " ", content or "").strip()
        if not text:
            return ""

        # If it begins mid-sentence, drop the broken leading fragment.
        if not text[0].isupper():
            match = re.search(r"[.!?]\s+(?=[A-Z0-9])", text)
            if match:
                text = text[match.end():].strip()
            else:
                parts = text.split(" ", 1)
                text = parts[1].strip() if len(parts) > 1 else text

        # Split into sentences and drop boilerplate / navigation ones.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept = []
        for sentence in sentences:
            if self._is_boilerplate_sentence(sentence):
                continue
            kept.append(sentence.strip())
            if sum(len(s) for s in kept) >= max_len:
                break

        # If nothing substantive remains, show no summary rather than junk.
        if not kept:
            return ""
        summary = " ".join(kept).strip()

        if len(summary) > max_len:
            window = summary[:max_len]
            cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
            if cut >= int(max_len * 0.4):
                summary = window[:cut + 1].strip()
            else:
                summary = window.rstrip(" ,;:") + "…"

        return summary.strip()

    def deduplicate_evidence(self, evidence_list: List[Dict]) -> List[Dict]:
        seen = set()
        unique_items = []

        for item in evidence_list:
            title_key = self.normalize_title(item.get("title", ""))
            url_key = item.get("url", "").strip().lower()
            dedup_key = title_key or url_key

            if dedup_key and dedup_key not in seen:
                seen.add(dedup_key)
                unique_items.append(item)

        return unique_items

    def score_text(self, text: str, keywords: List[str]) -> int:
        text = text.lower()
        return sum(1 for kw in keywords if kw in text)

    def get_sentiment_label(self, text: str) -> str:
        positive = self.score_text(text, self.positive_keywords)
        negative = self.score_text(text, self.negative_keywords)

        if positive > negative:
            return "Positive"
        if negative > positive:
            return "Negative"
        return "Neutral"

    def get_freshness_score(self, published: str) -> float:
        if not published:
            return 0.3

        try:
            published_clean = published.replace("Z", "+00:00")
            dt = datetime.fromisoformat(published_clean)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            age_days = max((now - dt).days, 0)

            if age_days <= 30:
                return 1.0
            if age_days <= 90:
                return 0.8
            if age_days <= 180:
                return 0.6
            if age_days <= 365:
                return 0.4
            return 0.2
        except Exception:
            return 0.3

    def build_evidence_summary(self, item: Dict) -> Dict:
        title = item.get("title", "No title")
        content = item.get("content", "")
        combined = f"{title} {content}".lower()

        opportunity_score = self.score_text(combined, self.opportunity_keywords)
        risk_score = self.score_text(combined, self.risk_keywords)
        trend_score = self.score_text(combined, self.trend_keywords)

        source_type = item.get("source_type", "unknown")
        competitor = item.get("competitor", "")
        category = item.get("category", "general")
        published = item.get("published", "")

        freshness_score = self.get_freshness_score(published)
        sentiment = self.get_sentiment_label(combined)

        confidence = min(
            0.45
            + 0.05 * max(opportunity_score, risk_score, trend_score)
            + (0.10 * freshness_score)
            + (0.05 if source_type == "company" else 0.03 if source_type == "news" else 0.02),
            0.95
        )

        impact_level = "Low"
        if max(opportunity_score, trend_score) >= 4:
            impact_level = "High"
        elif max(opportunity_score, trend_score) >= 2:
            impact_level = "Medium"

        severity_level = "Low"
        if risk_score >= 4:
            severity_level = "High"
        elif risk_score >= 2:
            severity_level = "Medium"

        return {
            "title": title,
            "summary": self.build_clean_summary(content),
            "source": item.get("source", ""),
            "source_type": source_type,
            "url": item.get("url", ""),
            "published": published,
            "category": category,
            "competitor": competitor,
            "sentiment": sentiment,
            "freshness_score": round(freshness_score, 2),
            "opportunity_score": opportunity_score,
            "risk_score": risk_score,
            "trend_score": trend_score,
            "impact_level": impact_level,
            "severity_level": severity_level,
            "risk_category": self.classify_risk_category(combined),
            "confidence_score": round(confidence, 2)
        }

    def analyze_section(self, evidence_list: List[Dict], section_name: str) -> List[Dict]:
        unique_items = self.deduplicate_evidence(evidence_list)
        analyzed = [self.build_evidence_summary(item) for item in unique_items]

        if section_name == "opportunities":
            analyzed = sorted(
                analyzed,
                key=lambda x: (x["opportunity_score"], x["freshness_score"], x["confidence_score"]),
                reverse=True
            )
        elif section_name == "risks":
            analyzed = sorted(
                analyzed,
                key=lambda x: (x["risk_score"], x["freshness_score"], x["confidence_score"]),
                reverse=True
            )
        elif section_name == "trends":
            analyzed = sorted(
                analyzed,
                key=lambda x: (x["trend_score"], x["freshness_score"], x["confidence_score"]),
                reverse=True
            )
        elif section_name == "competitors":
            analyzed = sorted(
                analyzed,
                key=lambda x: (
                    1 if x["competitor"] and x["competitor"] != "Unknown" else 0,
                    x["freshness_score"],
                    x["confidence_score"]
                ),
                reverse=True
            )

        return analyzed[:5]

    COMPETITOR_NAMES = ["Oracle", "Salesforce", "Microsoft", "Workday", "ServiceNow"]

    def detect_primary_competitor(self, item: Dict):
        """Determine which competitor a document is really about.

        The stored `competitor` field was set at collection time by first-match,
        which over-attributes everything to whichever name comes first. Here we
        prefer the competitor named in the title, then fall back to the most
        frequently mentioned competitor in the body.
        """
        title = (item.get("title", "") or "").lower()
        for name in self.COMPETITOR_NAMES:
            if name.lower() in title:
                return name

        content = (item.get("content", "") or "").lower()
        best_name = None
        best_count = 0
        for name in self.COMPETITOR_NAMES:
            count = content.count(name.lower())
            if count > best_count:
                best_count = count
                best_name = name
        return best_name

    GENERIC_TITLE_MARKERS = ("latest news", "news and insights", "earnings preview")

    def competitor_in_title(self, item: Dict):
        """Return the competitor ONLY if it is named in the title.

        A document is real competitor intelligence when the competitor is its
        subject (named in the headline). Documents that merely mention a rival
        in passing are usually about the home company and produce noise.
        """
        title = (item.get("title", "") or "").lower()
        if any(marker in title for marker in self.GENERIC_TITLE_MARKERS):
            return None
        for name in self.COMPETITOR_NAMES:
            if name.lower() in title:
                return name
        return None

    def build_competitor_signals(self, evidence_list: List[Dict], corpus_docs: List[Dict] = None) -> List[Dict]:
        # Scan the whole corpus (if available) for documents whose HEADLINE is
        # about a competitor, so signals are real competitor news, not articles
        # that merely mention a rival in passing.
        source_items = corpus_docs if corpus_docs else evidence_list

        grouped = defaultdict(list)
        for item in source_items:
            competitor = self.competitor_in_title(item)
            if not competitor:
                continue
            grouped[competitor].append(item)

        min_dt = datetime.min.replace(tzinfo=timezone.utc)
        signals = []
        for competitor, items in grouped.items():
            unique_items = self.deduplicate_evidence(items)
            unique_items.sort(
                key=lambda x: self.parse_published(x.get("published", "")) or min_dt,
                reverse=True,
            )
            headlines = [
                {
                    "title": it.get("title", "No title"),
                    "source": it.get("source", ""),
                    "url": it.get("url", ""),
                    "published": it.get("published", ""),
                    "summary": self.build_clean_summary(it.get("content", ""), max_len=220),
                }
                for it in unique_items[:4]
            ]
            latest_published = unique_items[0].get("published", "") if unique_items else ""
            signals.append({
                "competitor": competitor,
                "count": len(unique_items),
                "latest_published": latest_published,
                "headlines": headlines,
                "titles": [h["title"] for h in headlines],  # backward-compat
            })

        signals = sorted(signals, key=lambda x: x["count"], reverse=True)
        return signals[:6]

    def build_sentiment_summary(self, analyzed_items: List[Dict]) -> Dict:
        counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for item in analyzed_items:
            sentiment = item.get("sentiment", "Neutral")
            counts[sentiment] = counts.get(sentiment, 0) + 1

        dominant = max(counts, key=counts.get) if counts else "Neutral"

        return {
            "counts": counts,
            "dominant_sentiment": dominant
        }

    def build_sentiment_by_source_type(self, analyzed_items: List[Dict]) -> Dict:
        result = defaultdict(lambda: {"Positive": 0, "Negative": 0, "Neutral": 0})
        for item in analyzed_items:
            source_type = item.get("source_type", "unknown")
            sentiment = item.get("sentiment", "Neutral")
            result[source_type][sentiment] = result[source_type].get(sentiment, 0) + 1
        return {k: dict(v) for k, v in result.items()}

    def analyze(self, retrieved: Dict[str, List[Dict]], corpus_docs: List[Dict] = None) -> Dict:
        opportunities = self.analyze_section(retrieved.get("opportunities", []), "opportunities")
        risks = self.analyze_section(retrieved.get("risks", []), "risks")
        competitors = self.analyze_section(retrieved.get("competitors", []), "competitors")
        trends = self.analyze_section(retrieved.get("trends", []), "trends")
        competitor_signals = self.build_competitor_signals(
            retrieved.get("competitors", []), corpus_docs=corpus_docs
        )

        # Sentiment is computed over the ENTIRE collected corpus when available
        # (every indexed document), so the sentiment charts reflect all sources
        # we gathered, not just the documents retrieved for this run. Falls back
        # to the retrieved set if the corpus wasn't passed.
        all_retrieved = []
        for items in retrieved.values():
            all_retrieved.extend(items)
        unique_retrieved = self.deduplicate_evidence(all_retrieved)

        sentiment_source = corpus_docs if corpus_docs else unique_retrieved
        sentiment_source = self.deduplicate_evidence(sentiment_source)
        analyzed_sentiment = [self.build_evidence_summary(item) for item in sentiment_source]

        sentiment_summary = self.build_sentiment_summary(analyzed_sentiment)
        sentiment_by_source_type = self.build_sentiment_by_source_type(analyzed_sentiment)
        sentiment_trend = self.build_sentiment_trend(analyzed_sentiment)

        return {
            "opportunities": opportunities,
            "risks": risks,
            "competitors": competitors,
            "trends": trends,
            "competitor_signals": competitor_signals,
            "sentiment_summary": sentiment_summary,
            "sentiment_by_source_type": sentiment_by_source_type,
            "sentiment_trend": sentiment_trend,
            "documents_analyzed": len(unique_retrieved),
            "sentiment_documents": len(sentiment_source),
        }