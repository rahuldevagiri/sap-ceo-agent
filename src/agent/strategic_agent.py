"""StrategicAgent - the orchestrator that runs the full agent loop.

This ties the seven phases together into the workflow the assignment asks for:

    Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend -> Validate

and adds the behaviour that makes it an *agent* rather than a pipeline: an
**autonomous re-plan branch**. After deciding, the agent inspects its own
evidence-sufficiency judgement; if the evidence is too thin, it autonomously
broadens its retrieval and re-analyzes before committing to recommendations -
bounded by `max_replans` so it always terminates.

Everything is recorded in the AgentState trace, so the whole reasoning process
is observable in the dashboard.

The components are created once (the tool registry owns the embedding model and
Chroma client) and a single LLM instance is shared by the Planner and
Recommender.
"""

from src.agent.agent_state import AgentState
from src.agent.tools import build_default_registry
from src.agent.planner import Planner
from src.agent.executor import Executor
from src.agent.decider import Decider
from src.agent.recommender import Recommender
from src.agent.validator import Validator
from src.agent.llm_adapter import OllamaLLM
from src.config import RETRIEVAL_TOP_K


class StrategicAgent:
    def __init__(self, llm: OllamaLLM = None, max_replans: int = 1):
        self.llm = llm or OllamaLLM()
        self.registry = build_default_registry()           # loads embedder + Chroma once
        self.planner = Planner(llm=self.llm, registry=self.registry)
        self.executor = Executor(self.registry)
        self.decider = Decider()
        self.recommender = Recommender(llm=self.llm)
        self.validator = Validator()
        self.max_replans = max_replans

    def run(self, goal: str = None, company: str = "SAP") -> AgentState:
        goal = goal or f"Act as the CEO advisor for {company}: decide what to do next, and why."
        state = AgentState(goal=goal, company=company)
        state.log("Goal", "input", goal, f"company={company}")

        # --- Plan -------------------------------------------------------
        plan_out = self.planner.plan(goal, company)
        state.plan = plan_out["steps"]
        state.log("Plan", plan_out["source"],
                  f"Planned {len(state.plan)} investigation steps",
                  " | ".join(s["view"] for s in state.plan))

        # --- Retrieve + Analyze + Decide (with autonomous re-plan) ------
        top_k = RETRIEVAL_TOP_K
        self.executor.execute(state, top_k=top_k)
        self.decider.decide(state)

        replans = 0
        while state.decision_summary.get("needs_more_evidence") and replans < self.max_replans:
            replans += 1
            top_k = min(top_k * 2, 120)
            state.log("Re-plan", "reason",
                      f"Evidence insufficient: {state.decision_summary.get('reason')}",
                      f"autonomously broadening retrieval to top_k={top_k}")
            self.executor.execute(state, top_k=top_k)
            self.decider.decide(state)

        if replans == 0:
            state.log("Decide", "reason", "Evidence judged sufficient; no re-planning needed",
                      state.decision_summary.get("reason", ""))

        # --- Recommend --------------------------------------------------
        self.recommender.recommend(state)

        # --- Validate (before presenting) ------------------------------
        self.validator.validate(state)

        # --- Finalize ---------------------------------------------------
        source = "fallback"
        for note in state.notes:
            if note.startswith("recommendation source:"):
                source = note.split(":", 1)[1].strip()
        state.mode = "agent" if source == "llm" else "agent-fallback"
        state.log("Done", "finalize",
                  f"Briefing ready (recommendation source: {source})",
                  f"{len(state.recommendations)} validated recommendations · "
                  f"validation={'OK' if state.validation.get('validated') else 'partial'}")
        return state
