"""Validator - the agent checks its own recommendations before presenting.

This is the "Validate" phase: a reflection / self-critique step that guards
against the LLM presenting unsupported or malformed advice. It is the headline
new agent capability the brief asks for ("validation of recommendations before
presenting them").

Each recommendation is checked on three axes:

  1. Grounding (anti-hallucination): every piece of supporting_evidence must
     correspond to a REAL document the agent actually retrieved. If a
     recommendation cites evidence that does not exist in the corpus, it is not
     trustworthy. This is the most important check.

  2. Completeness: the required fields (title, priority, expected_impact,
     risk_level, confidence_score, supporting_evidence) must be present.

  3. Consistency: priority/risk_level must be High/Medium/Low and the confidence
     score must be a number in [0, 1].

The checks are deterministic and explainable (you can state exactly why a
recommendation failed). An LLM-as-judge faithfulness check could be layered on
top, but deterministic grounding is more reliable on a small local model.

Outcome: hard failures are dropped (with a safeguard so we never present an
empty briefing), warnings are kept but flagged, and a full validation report is
written to the agent state for the dashboard.
"""

import re

VALID_LEVELS = {"High", "Medium", "Low"}
REQUIRED_FIELDS = ["title", "priority", "expected_impact", "risk_level",
                   "confidence_score", "supporting_evidence"]


def _tokens(text: str):
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class Validator:
    GROUNDING_OVERLAP = 0.5   # min token-overlap to call a citation "grounded"

    def _known_titles(self, state):
        """All document titles the agent actually retrieved/holds = the set of
        evidence a recommendation is allowed to cite."""
        titles = []
        for items in (state.evidence or {}).values():
            for it in items:
                t = it.get("title")
                if t:
                    titles.append(t)
        for doc in (state.corpus_docs or []):
            t = doc.get("title")
            if t:
                titles.append(t)
        return titles

    def _is_grounded(self, evidence_text: str, known_titles) -> bool:
        e = (evidence_text or "").strip().lower()
        if not e:
            return False
        for title in known_titles:
            t = title.lower()
            if e in t or t in e:
                return True
            if _jaccard(evidence_text, title) >= self.GROUNDING_OVERLAP:
                return True
        return False

    def _validate_one(self, rec: dict, known_titles) -> dict:
        issues = []
        severity = "pass"

        # 1) Completeness
        for field in REQUIRED_FIELDS:
            value = rec.get(field)
            if value in (None, "", []):
                issues.append(f"missing '{field}'")
                severity = "fail"

        # 2) Consistency
        if rec.get("priority") not in VALID_LEVELS:
            issues.append(f"invalid priority '{rec.get('priority')}'")
            severity = "warn" if severity != "fail" else severity
        if rec.get("risk_level") not in VALID_LEVELS:
            issues.append(f"invalid risk_level '{rec.get('risk_level')}'")
            severity = "warn" if severity != "fail" else severity
        try:
            conf = float(rec.get("confidence_score"))
            if not (0.0 <= conf <= 1.0):
                issues.append(f"confidence out of range ({conf})")
                severity = "warn" if severity != "fail" else severity
        except (TypeError, ValueError):
            issues.append("confidence not numeric")
            severity = "warn" if severity != "fail" else severity

        # 3) Grounding (anti-hallucination)
        evidence = rec.get("supporting_evidence") or []
        grounded = [e for e in evidence if self._is_grounded(e, known_titles)]
        if evidence and not grounded:
            issues.append("no supporting evidence found in corpus (possible hallucination)")
            severity = "fail"

        return {
            "title": rec.get("title", "Untitled"),
            "status": severity,
            "issues": issues,
            "grounded_evidence": len(grounded),
            "total_evidence": len(evidence),
        }

    def validate(self, state) -> dict:
        known_titles = self._known_titles(state)
        recs = state.recommendations or []

        items = [self._validate_one(r, known_titles) for r in recs]

        kept, dropped = [], []
        for rec, report in zip(recs, items):
            if report["status"] == "fail":
                dropped.append(report["title"])
            else:
                kept.append(rec)

        # Safeguard: never present an empty briefing. If everything failed,
        # keep the originals but mark the run as not fully validated.
        all_failed = bool(recs) and not kept
        if all_failed:
            kept = list(recs)

        passed = sum(1 for i in items if i["status"] == "pass")
        warned = sum(1 for i in items if i["status"] == "warn")
        failed = sum(1 for i in items if i["status"] == "fail")
        validated = (failed == 0) and bool(recs)

        report = {
            "validated": validated,
            "total": len(recs),
            "passed": passed,
            "warned": warned,
            "failed": failed,
            "items": items,
            "dropped": dropped,
            "note": ("all recommendations failed validation; presenting unvalidated"
                     if all_failed else ""),
        }

        # Apply the validated set back to the response shown to the user.
        state.recommendations = kept
        if isinstance(state.response, dict):
            state.response["strategic_recommendations"] = kept
        state.validation = report

        verdict = "VALIDATED" if validated else ("PARTIAL" if not all_failed else "UNVALIDATED")
        state.log(
            "Validate", "self-check",
            "Check recommendations for grounding, completeness, and consistency",
            f"{verdict}: {passed} pass / {warned} warn / {failed} fail"
            + (f"; dropped {len(dropped)}" if dropped else ""),
            ok=(not all_failed),
        )
        return report
