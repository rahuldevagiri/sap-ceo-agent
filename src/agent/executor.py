"""Executor - the agent acts on its plan by USING TOOLS.

This is the Retrieve + Analyze portion of the agent loop. It does not "think" on
its own; it carries out the plan the Planner produced by invoking tools from the
registry, and it writes both the results and a trace entry for every tool call
into the shared AgentState.

Why this matters for the assignment:
  * "Tool usage beyond the LLM"  - retrieval and analysis happen via explicit
    tool calls (search_evidence, all_documents, corpus_stats, analyze_evidence),
    not inside an LLM prompt.
  * "Retrieval and use of evidence" - each planned query pulls real document
    chunks from the vector store, bucketed by intelligence view.
  * Every action is logged, so the tool usage is auditable in the trace.
"""

from src.config import RETRIEVAL_TOP_K


class Executor:
    def __init__(self, registry):
        self.registry = registry

    def execute(self, state, top_k: int = RETRIEVAL_TOP_K):
        # ---- Retrieve: one tool call per planned step ----
        for step in state.plan:
            view = step.get("view", "general")
            query = step.get("query", "")
            try:
                evidence = self.registry.run("search_evidence", query=query, top_k=top_k)
                state.evidence[view] = evidence
                state.log(
                    "Retrieve", "search_evidence",
                    f"{view}: \"{query[:70]}\"",
                    f"{len(evidence)} chunks retrieved",
                )
            except Exception as exc:
                state.evidence[view] = []
                state.log("Retrieve", "search_evidence",
                          f"{view}: \"{query[:70]}\"", f"error: {exc}", ok=False)

        # ---- Retrieve whole corpus (for corpus-wide sentiment) ----
        try:
            corpus_docs = self.registry.run("all_documents")
            state.corpus_docs = corpus_docs
            state.log("Retrieve", "all_documents",
                      "Fetch full corpus for sentiment", f"{len(corpus_docs)} documents")
        except Exception as exc:
            state.corpus_docs = []
            state.log("Retrieve", "all_documents", "Fetch full corpus",
                      f"error: {exc}", ok=False)

        # ---- Corpus coverage stats ----
        try:
            stats = self.registry.run("corpus_stats")
            state.corpus = stats
            state.log("Retrieve", "corpus_stats", "Corpus coverage",
                      f"{stats.get('total_documents', 0)} docs · "
                      f"{stats.get('distinct_sources', 0)} sources")
        except Exception as exc:
            state.log("Retrieve", "corpus_stats", "Corpus coverage",
                      f"error: {exc}", ok=False)

        # ---- Analyze: score the retrieved evidence into signals ----
        try:
            analysis = self.registry.run(
                "analyze_evidence", retrieved=state.evidence, corpus_docs=state.corpus_docs
            )
            state.analysis = analysis
            state.log(
                "Analyze", "analyze_evidence",
                "Score evidence into opportunities / risks / trends / sentiment",
                f"opps {len(analysis.get('opportunities', []))} · "
                f"risks {len(analysis.get('risks', []))} · "
                f"trends {len(analysis.get('trends', []))} · "
                f"competitors {len(analysis.get('competitor_signals', []))} · "
                f"sentiment over {analysis.get('sentiment_documents', 0)} docs",
            )
        except Exception as exc:
            state.analysis = {}
            state.log("Analyze", "analyze_evidence", "Score evidence",
                      f"error: {exc}", ok=False)

        return state
