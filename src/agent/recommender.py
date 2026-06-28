"""Recommender - turn the agent's prioritised decisions into recommendations.

This is the "Recommend" phase. By now the agent has retrieved evidence, analyzed
it, and (in the Decide phase) ranked the strategically critical signals. The
Recommender converts those ranked decisions into an executive briefing with
evidence-backed, prioritized recommendations.

Design:
  * It is DECISION-AWARE: the prompt is seeded with the Decider's ranked
    decisions, so the LLM's recommendations follow what the agent decided was
    important - not just the raw analysis. This is what links autonomous
    decision-making to the final advice.
  * It reuses the project's proven reliability pattern: build a deterministic
    fallback briefing from the analysis, then OVERLAY the LLM's non-empty fields
    on top. So even if the small local model returns sparse output, every field
    is complete and evidence-grounded.
  * Each recommendation carries title, priority, expected impact, risk level,
    confidence, and supporting evidence (the assignment's required shape).
"""

import json

from src.agent.llm_adapter import OllamaLLM
from src.agent.strategist import AICEOStrategist


class Recommender:
    def __init__(self, llm: OllamaLLM = None):
        self.llm = llm or OllamaLLM()
        # Lazy strategist (no embedder load) - reused for its prompt + fallback builders.
        self.strategist = AICEOStrategist(llm=self.llm)

    def build_prompt(self, company: str, analysis: dict, decisions: list) -> str:
        """Reuse the strategist's briefing-shaped prompt, seeded with the
        agent's prioritised decisions so recommendations address them."""
        base = self.strategist.build_prompt(company, analysis)
        compact_decisions = [
            {
                "type": d.get("type"),
                "title": d.get("title"),
                "level": d.get("level"),
                "priority_score": d.get("priority_score"),
                "rationale": d.get("rationale"),
            }
            for d in (decisions or [])
        ]
        return (
            base
            + "\n\nThe agent has already PRIORITISED these decisions (highest priority first). "
            + "Base your strategic_recommendations primarily on the top items, and cite the "
            + "relevant titles as supporting_evidence:\n"
            + json.dumps(compact_decisions, ensure_ascii=False, indent=2)
        )

    def recommend(self, state) -> dict:
        company = state.company
        analysis = state.analysis or {}
        decisions = state.decisions or []

        # 1) Deterministic, always-complete fallback briefing.
        fallback = self.strategist.build_fallback_response(company, analysis)

        # 2) Ask the LLM for a decision-grounded briefing.
        prompt = self.build_prompt(company, analysis, decisions)
        llm_result = self.llm.generate(prompt)

        # 3) Overlay LLM output onto the fallback (LLM wins where it produced content).
        if llm_result.get("ok") and isinstance(llm_result.get("data"), dict):
            response = dict(fallback)
            for key, value in llm_result["data"].items():
                if value not in (None, "", [], {}):
                    response[key] = value
            source = "llm"
        else:
            response = fallback
            source = "fallback"
            if llm_result.get("error"):
                state.notes.append(f"recommender LLM error: {llm_result['error']}")

        state.response = response
        state.recommendations = response.get("strategic_recommendations", [])
        state.notes.append(f"recommendation source: {source}")
        state.log(
            "Recommend", source,
            "Draft evidence-backed recommendations from prioritised decisions",
            f"{len(state.recommendations)} recommendations ({source})",
        )
        return response
