import json
from src.agent.retriever import RAGRetriever
from src.agent.analyzer import IntelligenceAnalyzer
from src.agent.llm_adapter import OllamaLLM


class AICEOStrategist:
    def __init__(self, llm_enabled=True, retriever=None, analyzer=None, llm=None):
        # Heavy components are lazy: created only when actually needed. This lets
        # other modules (e.g. the agent's Recommender) reuse the pure
        # prompt/fallback builders without loading the embedding model.
        self._retriever = retriever
        self._analyzer = analyzer
        self.llm_enabled = llm_enabled
        self.llm = llm or OllamaLLM()

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = RAGRetriever()
        return self._retriever

    @property
    def analyzer(self):
        if self._analyzer is None:
            self._analyzer = IntelligenceAnalyzer()
        return self._analyzer

    def build_prompt(self, company: str, analysis: dict) -> str:
        def slim(items, keys):
            slimmed = []
            for it in items:
                slimmed.append({k: it.get(k) for k in keys if it.get(k) is not None})
            return slimmed

        # Keep the prompt small so the model has plenty of context budget left
        # to write the JSON response (full evidence summaries are not needed for
        # the model to reason about headlines).
        compact_analysis = {
            "opportunities": slim(analysis.get("opportunities", []), ["title", "impact_level", "source"]),
            "risks": slim(analysis.get("risks", []), ["title", "severity_level", "risk_category", "source"]),
            "competitor_signals": slim(
                analysis.get("competitor_signals", []), ["competitor", "count", "titles"]
            ),
            "trends": slim(analysis.get("trends", []), ["title", "source"]),
        }

        return f"""
You are the AI CEO Strategic Intelligence Agent for {company}.

Analyze the structured intelligence below and return ONLY valid JSON.

Required JSON structure:
{{
  "what_happened": "string",
  "why_it_matters": "string",
  "opportunities": ["string", "string"],
  "risks": ["string", "string"],
  "competitor_activity": ["string", "string"],
  "trends_to_monitor": ["string", "string"],
  "strategic_recommendations": [
    {{
      "title": "string",
      "priority": "High or Medium or Low",
      "expected_impact": "string",
      "risk_level": "High or Medium or Low",
      "confidence_score": 0.0,
      "supporting_evidence": ["string", "string"]
    }}
  ],
  "ceo_briefing": "string"
}}

Use only this structured intelligence:
{json.dumps(compact_analysis, ensure_ascii=False, indent=2)}
"""

    def build_fallback_recommendations(self, company: str, analysis: dict):
        recommendations = []

        if analysis.get("opportunities"):
            top_opportunity = analysis["opportunities"][0]
            recommendations.append({
                "title": f"Accelerate investment around {top_opportunity['title'][:80]}",
                "priority": "High" if top_opportunity["impact_level"] == "High" else "Medium",
                "expected_impact": "Revenue growth, stronger AI/cloud positioning, and market differentiation",
                "risk_level": "Medium",
                "confidence_score": top_opportunity["confidence_score"],
                "supporting_evidence": [top_opportunity["title"]],
            })

        if analysis.get("risks"):
            top_risk = analysis["risks"][0]
            recommendations.append({
                "title": f"Mitigate risk related to {top_risk['title'][:80]}",
                "priority": "High" if top_risk["severity_level"] == "High" else "Medium",
                "expected_impact": "Reduced operational, strategic, or reputational downside",
                "risk_level": top_risk["severity_level"],
                "confidence_score": top_risk["confidence_score"],
                "supporting_evidence": [top_risk["title"]],
            })

        if analysis.get("competitor_signals"):
            top_competitor = analysis["competitor_signals"][0]
            competitor_name = top_competitor.get("competitor", "a competitor")
            recommendations.append({
                "title": f"Increase executive monitoring of {competitor_name}'s market activity",
                "priority": "Medium",
                "expected_impact": "Faster response to competitive moves and better strategic positioning",
                "risk_level": "Medium",
                "confidence_score": round(min(0.6 + 0.05 * top_competitor.get("count", 0), 0.9), 2),
                "supporting_evidence": top_competitor.get("titles", []),
            })

        return recommendations[:3]

    def build_fallback_response(self, company: str, analysis: dict):
        return {
            "what_happened": (
                f"Recent intelligence suggests that {company} is operating in a fast-moving environment shaped by "
                f"AI adoption, cloud transformation, competitive activity, and evolving enterprise technology trends."
            ),
            "why_it_matters": (
                f"These developments may influence {company}'s growth trajectory, product strategy, market positioning, "
                f"and executive decision priorities."
            ),
            "opportunities": [x["title"] for x in analysis.get("opportunities", [])[:5]],
            "risks": [x["title"] for x in analysis.get("risks", [])[:5]],
            "competitor_activity": [x["title"] for x in analysis.get("competitors", [])[:5]],
            "trends_to_monitor": [x["title"] for x in analysis.get("trends", [])[:5]],
            "strategic_recommendations": self.build_fallback_recommendations(company, analysis),
            "ceo_briefing": (
                f"For {company}, the immediate priority should be to capture high-value AI and cloud opportunities, "
                f"while monitoring competitor activity and reducing exposure to strategic and operational risks."
            )
        }

    def answer(self, company: str = "SAP"):
        retrieved = self.retriever.retrieve_multi_view(company=company)
        corpus_docs = self.retriever.repo.all_documents()
        analysis = self.analyzer.analyze(retrieved, corpus_docs=corpus_docs)
        fallback_response = self.build_fallback_response(company, analysis)
        corpus = self.retriever.repo.corpus_stats()

        result = {
            "mode": "fallback",
            "retrieved": retrieved,
            "analysis": analysis,
            "corpus": corpus,
            "response": fallback_response,
            "llm_error": None,
            "llm_raw_output": None
        }

        if not self.llm_enabled:
            return result

        prompt = self.build_prompt(company, analysis)
        llm_result = self.llm.generate(prompt)

        result["llm_raw_output"] = llm_result.get("raw_output")

        if llm_result.get("ok") and isinstance(llm_result.get("data"), dict):
            # Overlay the LLM's fields onto the always-complete fallback so that
            # any field the (small) model leaves empty keeps a sensible value.
            merged = dict(fallback_response)
            for key, value in llm_result["data"].items():
                if value not in (None, "", [], {}):
                    merged[key] = value
            result["mode"] = "llm"
            result["response"] = merged
            return result

        result["llm_error"] = llm_result.get("error")
        return result