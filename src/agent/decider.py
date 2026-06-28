"""Decider - the agent autonomously decides what matters most.

This is the "Decide" phase. After evidence has been retrieved and analyzed, the
agent must make a judgement: of all the opportunities and risks surfaced, which
ones are strategically critical and worth acting on?

Two autonomous behaviours live here:

1. Prioritisation - each signal is scored by a transparent composite of
   level (impact/severity) x confidence x freshness, then ranked. The agent
   picks the top items rather than dumping everything. This is a decision, not a
   summary.

2. Sufficiency gating - the agent assesses whether the evidence is strong
   enough to make confident recommendations. If signals are too few, confidence
   is low, or a whole view returned nothing, it flags `needs_more_evidence` so
   the orchestrator can loop back and retrieve more. Deciding *whether to act or
   gather more data* is the essence of autonomy.

The scoring is deterministic on purpose: it is fast, reproducible, and fully
explainable in a viva ("this risk ranked #1 because it is High severity, 0.93
confidence, and recent"). The LLM later turns these decisions into prose.
"""


class Decider:
    LEVEL_WEIGHT = {"High": 3, "Medium": 2, "Low": 1}

    # Sufficiency thresholds (tunable).
    MIN_TOTAL_SIGNALS = 4
    MIN_AVG_CONFIDENCE = 0.55

    def __init__(self, top_n: int = 6):
        self.top_n = top_n

    def _score(self, item: dict, level_key: str):
        level = item.get(level_key, "Medium")
        weight = self.LEVEL_WEIGHT.get(level, 2)
        confidence = float(item.get("confidence_score", 0.5) or 0.5)
        freshness = float(item.get("freshness_score", 0.3) or 0.3)
        # Multiplicative so an item must be strong on all three to rank top.
        score = round(weight * confidence * (0.6 + 0.4 * freshness), 3)
        return score, level, confidence, freshness

    def _decision(self, item: dict, dtype: str, level_key: str) -> dict:
        score, level, confidence, freshness = self._score(item, level_key)
        recency = "recent" if freshness >= 0.8 else ("dated" if freshness < 0.4 else "moderately recent")
        dimension = "impact" if dtype == "opportunity" else "severity"
        rationale = f"{level} {dimension} {dtype}; confidence {confidence}; {recency} evidence."
        return {
            "type": dtype,
            "title": item.get("title", "Untitled"),
            "level": level,
            "confidence": confidence,
            "freshness": freshness,
            "priority_score": score,
            "source": item.get("source", ""),
            "summary": item.get("summary", ""),
            "rationale": rationale,
        }

    def decide(self, state) -> dict:
        analysis = state.analysis or {}

        decisions = []
        for item in analysis.get("opportunities", []):
            decisions.append(self._decision(item, "opportunity", "impact_level"))
        for item in analysis.get("risks", []):
            decisions.append(self._decision(item, "risk", "severity_level"))

        # Rank by composite priority, highest first.
        decisions.sort(key=lambda d: d["priority_score"], reverse=True)
        selected = decisions[: self.top_n]

        # ---- Autonomous sufficiency gate ----
        total_signals = len(analysis.get("opportunities", [])) + len(analysis.get("risks", []))
        avg_confidence = (
            round(sum(d["confidence"] for d in selected) / len(selected), 3) if selected else 0.0
        )
        empty_views = [view for view, ev in (state.evidence or {}).items() if not ev]

        reasons = []
        if total_signals < self.MIN_TOTAL_SIGNALS:
            reasons.append(f"only {total_signals} opportunity/risk signals")
        if avg_confidence < self.MIN_AVG_CONFIDENCE:
            reasons.append(f"low average confidence ({avg_confidence})")
        if empty_views:
            reasons.append(f"no evidence for: {', '.join(empty_views)}")
        needs_more = bool(reasons)

        sufficiency = {
            "needs_more_evidence": needs_more,
            "reason": "; ".join(reasons) if reasons else "evidence sufficient to recommend",
            "total_signals": total_signals,
            "avg_confidence": avg_confidence,
            "considered": len(decisions),
            "selected": len(selected),
        }

        state.decisions = selected
        state.decision_summary = sufficiency
        state.log(
            "Decide", "prioritize",
            f"Ranked {len(decisions)} signals; selected top {len(selected)}",
            f"avg confidence {avg_confidence} · "
            f"{'NEEDS MORE EVIDENCE' if needs_more else 'evidence sufficient'}",
        )

        return {"decisions": selected, "sufficiency": sufficiency}
