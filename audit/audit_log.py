"""Append-only audit trail for every routing decision, auto or manual."""
import datetime
import json
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "decisions.jsonl"


def log_decision(
    return_id: str,
    risk_score: float,
    decision: str,
    decision_source: str,
    model_name: str,
    explanation: str = "",
    reviewer: str = "",
):
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "return_id": return_id,
        "risk_score": risk_score,
        "decision": decision,
        "decision_source": decision_source,  # "auto" or "manual"
        "model_name": model_name,
        "explanation": explanation,
        "reviewer": reviewer,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def read_log():
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
