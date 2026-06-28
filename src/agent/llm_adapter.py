import json
import re

from src.config import LLM_MODEL, LLM_TEMPERATURE, LLM_NUM_CTX, LLM_KEEP_ALIVE

try:
    from ollama import chat
    OLLAMA_AVAILABLE = True
except ImportError:
    chat = None
    OLLAMA_AVAILABLE = False


def _dict_to_text(d: dict) -> str:
    """Render a dict the model wrongly returned (instead of a string) as a
    readable one-line summary. Handles the competitor-signal shape it tends to
    echo, then falls back to a title-like field, then to key:value pairs."""
    if "competitor" in d:
        text = str(d.get("competitor", "")).strip()
        if d.get("count") is not None:
            text += f" ({d.get('count')} signals)"
        titles = d.get("titles") or []
        if titles:
            text += ": " + "; ".join(str(t) for t in titles[:2])
        return text
    for key in ("title", "name", "text", "summary", "activity", "description"):
        if d.get(key):
            return str(d[key])
    return "; ".join(f"{k}: {v}" for k, v in d.items())


def _coerce_text_list(value) -> list:
    """Guarantee a list of readable strings (the model sometimes returns dicts
    or nested objects where plain strings are expected)."""
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    out = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            out.append(_dict_to_text(item))
        else:
            out.append(str(item))
    return out


class OllamaLLM:
    def __init__(self, model=None, temperature=None, num_ctx=None):
        self.model = model or LLM_MODEL
        self.temperature = LLM_TEMPERATURE if temperature is None else temperature
        self.num_ctx = num_ctx or LLM_NUM_CTX

    def _extract_json_block(self, text: str):
        if not text:
            return None

        text = text.strip()

        if text.startswith("{") and text.endswith("}"):
            return text

        fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced_match:
            return fenced_match.group(1)

        fenced_generic = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced_generic:
            return fenced_generic.group(1)

        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return text[first_brace:last_brace + 1]

        return None

    def _normalize_response_shape(self, data: dict):
        if not isinstance(data, dict):
            return None

        normalized = {
            "what_happened": data.get("what_happened", ""),
            "why_it_matters": data.get("why_it_matters", ""),
            "opportunities": data.get("opportunities", []),
            "risks": data.get("risks", []),
            "competitor_activity": data.get("competitor_activity", []),
            "trends_to_monitor": data.get("trends_to_monitor", data.get("trends", [])),
            "strategic_recommendations": data.get("strategic_recommendations", []),
            "ceo_briefing": data.get("ceo_briefing", "")
        }

        # Guarantee these are lists of readable strings (the model sometimes
        # echoes structured dicts here, e.g. competitor-signal objects).
        normalized["opportunities"] = _coerce_text_list(normalized["opportunities"])
        normalized["risks"] = _coerce_text_list(normalized["risks"])
        normalized["competitor_activity"] = _coerce_text_list(normalized["competitor_activity"])
        normalized["trends_to_monitor"] = _coerce_text_list(normalized["trends_to_monitor"])

        if not isinstance(normalized["strategic_recommendations"], list):
            normalized["strategic_recommendations"] = []

        if not normalized["ceo_briefing"]:
            normalized["ceo_briefing"] = (
                "SAP should focus on cloud and AI growth opportunities, while monitoring "
                "competitive pressure and execution risks."
            )

        if not normalized["strategic_recommendations"]:
            normalized["strategic_recommendations"] = [
                {
                    "title": "Accelerate high-value AI and cloud initiatives",
                    "priority": "High",
                    "expected_impact": "Growth acceleration and stronger product differentiation",
                    "risk_level": "Medium",
                    "confidence_score": 0.75,
                    "supporting_evidence": normalized["opportunities"][:2]
                },
                {
                    "title": "Strengthen monitoring of competitive and market risks",
                    "priority": "Medium",
                    "expected_impact": "Improved strategic response and lower downside exposure",
                    "risk_level": "Medium",
                    "confidence_score": 0.7,
                    "supporting_evidence": normalized["risks"][:2] + normalized["competitor_activity"][:1]
                }
            ]

        required_keys = [
            "what_happened",
            "why_it_matters",
            "opportunities",
            "risks",
            "competitor_activity",
            "trends_to_monitor",
            "strategic_recommendations",
            "ceo_briefing",
        ]

        if not all(key in normalized for key in required_keys):
            return None

        return normalized

    def run_json(self, prompt: str, system_prompt: str = None) -> dict:
        """Generic 'ask the model for one JSON object' call.

        Unlike generate() (which normalizes into the CEO-briefing shape), this
        returns whatever JSON the model produced, so the agent's Planner,
        Decider, and Validator can each request their own structure. Uses
        constrained JSON decoding (format='json') for reliability.
        """
        if not OLLAMA_AVAILABLE:
            return {"ok": False, "error": "Ollama Python package is not installed.",
                    "raw_output": None, "data": None}

        system_prompt = system_prompt or (
            "You are a precise reasoning assistant. Return only one valid JSON object. "
            "Do not add markdown, comments, or text before or after the JSON."
        )
        try:
            response = chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": self.temperature, "num_ctx": self.num_ctx},
                format="json",
                keep_alive=LLM_KEEP_ALIVE,
            )
            content = (response["message"]["content"] or "").strip()
            json_text = self._extract_json_block(content)
            if not json_text:
                return {"ok": False, "error": "No JSON object found in model output.",
                        "raw_output": content, "data": None}
            try:
                parsed = json.loads(json_text)
            except Exception as exc:
                return {"ok": False, "error": f"JSON parsing failed: {repr(exc)}",
                        "raw_output": content, "data": None}
            return {"ok": True, "data": parsed, "raw_output": content, "error": None}
        except Exception as exc:
            return {"ok": False, "error": f"Ollama chat failed: {repr(exc)}",
                    "raw_output": None, "data": None}

    def generate(self, prompt: str) -> dict:
        if not OLLAMA_AVAILABLE:
            return {
                "ok": False,
                "error": "Ollama Python package is not installed.",
                "raw_output": None
            }

        try:
            response = chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an AI CEO Strategic Intelligence Agent. "
                            "Return only one valid JSON object. "
                            "Do not add markdown or explanation before or after the JSON. "
                            "Use only the provided evidence."
                        )
                    },
                    {
                        "role": "user",
                       "content": prompt
                    }
                ],
                options={"temperature": self.temperature, "num_ctx": self.num_ctx},
                format="json",
                keep_alive=LLM_KEEP_ALIVE,
            )

            content = response["message"]["content"].strip()
            json_text = self._extract_json_block(content)

            if not json_text:
                return {
                    "ok": False,
                    "error": "No JSON object found in model output.",
                    "raw_output": content
                }

            try:
                parsed = json.loads(json_text)
            except Exception as e:
                return {
                    "ok": False,
                    "error": f"JSON parsing failed: {repr(e)}",
                    "raw_output": content
                }

            normalized = self._normalize_response_shape(parsed)

            if not normalized:
                return {
                    "ok": False,
                    "error": "Parsed JSON could not be normalized into expected response structure.",
                    "raw_output": content
                }

            return {
                "ok": True,
                "data": normalized,
                "raw_output": content
            }

        except Exception as e:
            return {
                "ok": False,
                "error": f"Ollama chat failed: {repr(e)}",
                "raw_output": None
            }