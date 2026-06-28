"""Planner - the agent decides WHAT to investigate before it acts.

"Planning before execution" is the first explicit agent capability. Given a
goal (e.g. "advise the CEO of <company>"), the Planner produces an ordered list
of investigation steps - each step names an intelligence *view*, a concrete
*search query*, the *tool* to use, and a short *rationale*.

Design choices:
  * The plan is produced by the LLM (it reasons about the goal), but a
    deterministic fallback plan guarantees the agent can always proceed even if
    the small local model returns nothing usable - graceful degradation.
  * The plan ALWAYS covers the four core views (opportunities, risks,
    competitors, trends) so the downstream analyzer has the buckets it needs.
    The LLM may tailor the queries; the structure is guaranteed.
  * The Planner is a pure function goal -> plan. The orchestrator (later) decides
    what to do with the plan and records it in the agent trace.
"""

from src.agent.llm_adapter import OllamaLLM

CORE_VIEWS = ["opportunities", "risks", "competitors", "trends"]


class Planner:
    def __init__(self, llm: OllamaLLM = None, registry=None):
        self.llm = llm or OllamaLLM()
        self.registry = registry  # optional ToolRegistry, used to list tools in the prompt

    # ------------------------------------------------------------------ #
    # Deterministic fallback plan (the four standard intelligence views)
    # ------------------------------------------------------------------ #
    def fallback_plan(self, company: str):
        queries = {
            "opportunities": f"{company} business opportunities growth AI cloud partnerships product expansion",
            "risks": f"{company} risks threats regulation competition negative sentiment operational issues",
            "competitors": f"{company} competitors Oracle Salesforce Microsoft Workday ServiceNow market moves",
            "trends": f"{company} enterprise software trends AI cloud ERP automation data platforms",
        }
        objectives = {
            "opportunities": "Identify growth opportunities (AI, cloud, partnerships, new markets)",
            "risks": "Surface risks and threats (competition, regulation, sentiment, operations)",
            "competitors": "Track competitor activity and market moves",
            "trends": "Detect technology and industry trends to monitor",
        }
        steps = []
        for i, view in enumerate(CORE_VIEWS, start=1):
            steps.append({
                "id": i,
                "view": view,
                "objective": objectives[view],
                "tool": "search_evidence",
                "query": queries[view],
                "rationale": "Standard intelligence view required for a CEO briefing.",
            })
        return steps

    # ------------------------------------------------------------------ #
    # LLM prompt
    # ------------------------------------------------------------------ #
    def build_prompt(self, goal: str, company: str) -> str:
        tool_lines = ""
        if self.registry is not None:
            tool_lines = "\n".join(
                f"- {t['name']}: {t['description']}" for t in self.registry.describe()
            )

        return f"""
You are the planning module of a strategic intelligence AGENT for {company}.

GOAL: {goal}

Before gathering any data, produce an investigation PLAN. The plan must cover
these four intelligence views: {", ".join(CORE_VIEWS)}.

For each view, write ONE focused semantic-search query (keywords a retriever
will embed to find relevant documents about {company}) and a one-line rationale.

Available tools the agent can use:
{tool_lines or "- search_evidence: semantic search over the corpus"}

Return ONLY valid JSON in exactly this shape:
{{
  "steps": [
    {{
      "id": 1,
      "view": "opportunities",
      "objective": "string",
      "tool": "search_evidence",
      "query": "string",
      "rationale": "string"
    }}
  ]
}}
"""

    # ------------------------------------------------------------------ #
    # Validate / repair an LLM plan so the executor can always run it
    # ------------------------------------------------------------------ #
    def _normalize_steps(self, raw_steps, company: str):
        fallback = {s["view"]: s for s in self.fallback_plan(company)}
        by_view = {}

        if isinstance(raw_steps, list):
            for step in raw_steps:
                if not isinstance(step, dict):
                    continue
                view = str(step.get("view", "")).strip().lower()
                if view not in CORE_VIEWS:
                    continue
                query = str(step.get("query", "")).strip()
                if not query:
                    query = fallback[view]["query"]
                by_view[view] = {
                    "id": len(by_view) + 1,
                    "view": view,
                    "objective": str(step.get("objective") or fallback[view]["objective"]),
                    "tool": step.get("tool") or "search_evidence",
                    "query": query,
                    "rationale": str(step.get("rationale") or fallback[view]["rationale"]),
                }

        # Guarantee all four core views exist (fill any missing from fallback).
        steps = []
        for i, view in enumerate(CORE_VIEWS, start=1):
            step = by_view.get(view, dict(fallback[view]))
            step["id"] = i
            steps.append(step)
        return steps

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def plan(self, goal: str, company: str = "SAP") -> dict:
        """Return {'steps': [...], 'source': 'llm'|'fallback', 'error': ...}."""
        prompt = self.build_prompt(goal, company)
        result = self.llm.run_json(
            prompt,
            system_prompt="You are a strategic planning module. Return only one valid JSON object describing the plan.",
        )

        if result.get("ok") and isinstance(result.get("data"), dict):
            steps = self._normalize_steps(result["data"].get("steps"), company)
            return {"steps": steps, "source": "llm", "error": None,
                    "raw_output": result.get("raw_output")}

        # LLM unavailable/failed -> deterministic plan
        return {"steps": self.fallback_plan(company), "source": "fallback",
                "error": result.get("error"), "raw_output": result.get("raw_output")}
