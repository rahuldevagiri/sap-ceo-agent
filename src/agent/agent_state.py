"""Agent working memory and reasoning trace.

This is the "scratchpad" the StrategicAgent carries through the
Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend -> Validate loop.

Two ideas live here:

1. AgentState  - the shared state every phase reads from and writes to. It is
   what turns a one-shot pipeline into a multi-step agent: each phase builds on
   the accumulated state instead of starting from scratch.

2. TraceEntry  - one record of "the agent did X". Logging every plan step, tool
   call, decision, and validation makes the agent's behaviour *observable* (we
   render the trace in the dashboard so an examiner can see it reasoning).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class TraceEntry:
    """A single step in the agent's reasoning trace."""

    phase: str            # Plan | Retrieve | Analyze | Decide | Recommend | Validate
    action: str           # tool name, "llm", or "reason"
    detail: str           # human-readable description of what was attempted
    output_summary: str = ""   # short summary of the result/observation
    ok: bool = True       # whether the step succeeded
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "action": self.action,
            "detail": self.detail,
            "output_summary": self.output_summary,
            "ok": self.ok,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentState:
    """Everything the agent knows during a single run."""

    goal: str = ""
    company: str = "SAP"

    plan: List[Dict[str, Any]] = field(default_factory=list)        # ordered planned steps
    evidence: Dict[str, List[Dict]] = field(default_factory=dict)   # {view/query: [evidence items]}
    corpus_docs: List[Dict] = field(default_factory=list)           # whole-corpus docs (for sentiment)
    corpus: Dict[str, Any] = field(default_factory=dict)            # corpus-wide coverage stats
    analysis: Dict[str, Any] = field(default_factory=dict)          # analyzer output
    decisions: List[Dict[str, Any]] = field(default_factory=list)   # prioritized decisions
    decision_summary: Dict[str, Any] = field(default_factory=dict)  # sufficiency gate result
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)        # validation report
    response: Dict[str, Any] = field(default_factory=dict)          # final briefing for the UI

    mode: str = "agent"               # "agent" | "fallback"
    notes: List[str] = field(default_factory=list)
    trace: List[TraceEntry] = field(default_factory=list)

    def log(self, phase: str, action: str, detail: str,
            output_summary: str = "", ok: bool = True) -> "AgentState":
        """Append a trace entry and return self (chainable)."""
        self.trace.append(TraceEntry(phase, action, detail, output_summary, ok))
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for the dashboard / session_state."""
        return {
            "goal": self.goal,
            "company": self.company,
            "plan": self.plan,
            "evidence": self.evidence,
            "corpus": self.corpus,
            "analysis": self.analysis,
            "decisions": self.decisions,
            "decision_summary": self.decision_summary,
            "recommendations": self.recommendations,
            "validation": self.validation,
            "response": self.response,
            "mode": self.mode,
            "notes": self.notes,
            "trace": [t.to_dict() for t in self.trace],
        }
