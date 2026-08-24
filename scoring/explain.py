"""Turn a risk score + its contributing features into a reviewer-facing
explanation using Gemini 2.5 Flash.

This module NEVER influences risk_score or decision — those are already
final by the time explain_decision() is called. If GEMINI_API_KEY is
missing or the API call fails, we fall back to a template so the review
queue never breaks on a missing key or a rate limit.
"""
import os

from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        from google import genai
        _client = genai.Client(api_key=api_key)
    return _client


def _template_explanation(score_result: dict) -> str:
    decision = score_result["decision"]
    risk_score = score_result["risk_score"]
    features = score_result.get("top_features", [])

    if not features:
        return f"Risk score {risk_score:.2f} -> {decision.replace('_', ' ')}."

    reasons = "; ".join(f"{f['feature']} ({f['direction']}, value={f['value']:.2f})" for f in features)
    return f"Risk score {risk_score:.2f} -> {decision.replace('_', ' ')}. Top signals: {reasons}."


def explain_decision(score_result: dict) -> str:
    client = _get_client()
    if client is None:
        return _template_explanation(score_result)

    features = score_result.get("top_features", [])
    feature_lines = "\n".join(
        f"- {f['feature']}: value={f['value']:.2f}, {f['direction']}" for f in features
    ) or "- no per-feature breakdown available for this model"

    prompt = f"""You are writing a short note for a human fraud-review agent at an e-commerce merchant.
A return has been scored by a fraud model — DO NOT change or second-guess the score or decision, only explain it.

Risk score: {score_result['risk_score']:.2f} (0=low risk, 1=high risk)
Routing decision: {score_result['decision']}
Top contributing signals:
{feature_lines}

Write 2-3 plain-English sentences telling the reviewer why this return was routed this way, referencing only the signals above. Do not invent facts not listed. No preamble, just the explanation."""

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text = (response.text or "").strip()
        return text if text else _template_explanation(score_result)
    except Exception:
        return _template_explanation(score_result)
