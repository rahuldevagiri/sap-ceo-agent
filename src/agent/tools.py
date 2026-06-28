"""Tool registry for the StrategicAgent.

An agent needs "hands": capabilities beyond the LLM that it can invoke to act on
the world and gather facts. Here we expose the project's existing deterministic
components as named, described, callable **tools**:

    search_evidence    - semantic vector search over the corpus (RAG)
    multi_view_search  - the 4 intelligence-view queries at once
    analyze_evidence   - score evidence into opportunities/risks/trends/sentiment
    corpus_stats       - corpus-wide coverage statistics
    all_documents      - one record per indexed document (for corpus sentiment)
    assess_freshness   - recency score for a publish date
    detect_competitor  - which competitor a document is about (light NER)

Why a registry (instead of calling the methods directly)?
  * Each tool has a NAME + DESCRIPTION, so the LLM planner can be told which
    tools exist and choose among them (tool selection / function calling).
  * Calls go through one place, so every tool use can be recorded in the agent
    trace, making "tool usage beyond the LLM" explicit and auditable.
  * It decouples the agent's reasoning from the concrete implementations.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from src.agent.retriever import RAGRetriever
from src.agent.analyzer import IntelligenceAnalyzer
from src.config import RETRIEVAL_TOP_K


@dataclass
class Tool:
    """A named capability the agent can invoke."""

    name: str
    description: str
    func: Callable
    args_hint: str = ""   # human-readable hint of expected arguments

    def run(self, **kwargs) -> Any:
        return self.func(**kwargs)


class ToolRegistry:
    """Holds the available tools and runs them by name."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def describe(self) -> List[Dict[str, str]]:
        """Tool catalogue for the planner prompt / dashboard."""
        return [
            {"name": t.name, "description": t.description, "args": t.args_hint}
            for t in self._tools.values()
        ]

    def run(self, name: str, **kwargs) -> Any:
        return self.get(name).run(**kwargs)


def build_default_registry(retriever: RAGRetriever = None,
                           analyzer: IntelligenceAnalyzer = None) -> ToolRegistry:
    """Construct the registry wired to the real project components.

    A single RAGRetriever (which owns the embedding model + Chroma client) and a
    single IntelligenceAnalyzer are shared across the tools so heavy resources
    load only once.
    """
    retriever = retriever or RAGRetriever()
    analyzer = analyzer or IntelligenceAnalyzer()
    repo = retriever.repo

    registry = ToolRegistry()

    registry.register(Tool(
        name="search_evidence",
        description="Semantic vector search over the indexed corpus; returns the most relevant document chunks for a query.",
        args_hint="query: str, top_k: int = RETRIEVAL_TOP_K",
        func=lambda query, top_k=RETRIEVAL_TOP_K: retriever.retrieve(query=query, top_k=top_k),
    ))

    registry.register(Tool(
        name="multi_view_search",
        description="Run the four standard intelligence queries (opportunities, risks, competitors, trends) for a company and return a dict of evidence lists.",
        args_hint="company: str = 'SAP', top_k: int = RETRIEVAL_TOP_K",
        func=lambda company="SAP", top_k=RETRIEVAL_TOP_K: retriever.retrieve_multi_view(company=company, top_k=top_k),
    ))

    registry.register(Tool(
        name="analyze_evidence",
        description="Score retrieved evidence into opportunities, risks, trends, competitor signals, and sentiment.",
        args_hint="retrieved: dict, corpus_docs: list = None",
        func=lambda retrieved, corpus_docs=None: analyzer.analyze(retrieved, corpus_docs=corpus_docs),
    ))

    registry.register(Tool(
        name="corpus_stats",
        description="Corpus-wide coverage statistics: document count, distinct sources, source-type mix, last updated.",
        args_hint="(no args)",
        func=lambda: repo.corpus_stats(),
    ))

    registry.register(Tool(
        name="all_documents",
        description="Return one representative record per indexed document, used for corpus-wide sentiment.",
        args_hint="(no args)",
        func=lambda: repo.all_documents(),
    ))

    registry.register(Tool(
        name="assess_freshness",
        description="Given a publish-date string, return a 0-1 recency score (newer is higher).",
        args_hint="published: str",
        func=lambda published: analyzer.get_freshness_score(published),
    ))

    registry.register(Tool(
        name="detect_competitor",
        description="Detect which competitor (Oracle/Salesforce/Microsoft/Workday/ServiceNow) a document is about, from its title/content.",
        args_hint="item: dict (with 'title' and 'content')",
        func=lambda item: analyzer.detect_primary_competitor(item),
    ))

    return registry
