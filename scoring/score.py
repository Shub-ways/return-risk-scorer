"""Pure scoring function: a return record -> risk score + routing decision.

No LLM involved anywhere in this module — the money decision comes only
from the trained model and the fixed thresholds saved at train time.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model.features import build_feature_matrix

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "model" / "artifacts"

_model = None
_feature_columns = None
_threshold_config = None


def _load_artifacts():
    global _model, _feature_columns, _threshold_config
    if _model is None:
        _model = joblib.load(ARTIFACTS_DIR / "model.pkl")
        _feature_columns = json.load(open(ARTIFACTS_DIR / "feature_columns.json"))
        _threshold_config = json.load(open(ARTIFACTS_DIR / "threshold_config.json"))
    return _model, _feature_columns, _threshold_config


def _top_contributing_features(model, x_row: pd.Series, feature_columns: list[str], top_n: int = 3):
    """Exact per-feature contribution for a linear model: coef * scaled value.

    Only defined for the logistic_regression pipeline (scaler + linear
    classifier) — this is why train.py prefers it over gradient boosting
    whenever the performance gap is small: a tree ensemble has no equally
    cheap, exact per-record attribution.
    """
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    x_scaled = scaler.transform(x_row.to_frame().T)[0]
    contributions = clf.coef_[0] * x_scaled
    order = np.argsort(-np.abs(contributions))[:top_n]
    return [
        {
            "feature": feature_columns[i],
            "value": float(x_row.iloc[i]),
            "contribution": float(contributions[i]),
            "direction": "raises risk" if contributions[i] > 0 else "lowers risk",
        }
        for i in order
    ]


def score_record(record: dict) -> dict:
    """record: raw return fields (see model/features.py RAW_FIELDS).

    Returns {risk_score, decision, top_features}.
    """
    model, feature_columns, cfg = _load_artifacts()

    df = pd.DataFrame([record])
    X = build_feature_matrix(df, feature_columns)
    risk_score = float(model.predict_proba(X)[:, 1][0])

    t1, t2 = cfg["flag_threshold_t1"], cfg["deny_threshold_t2"]
    if risk_score >= t2:
        decision = "auto_deny"
    elif risk_score >= t1:
        decision = "hold_for_review"
    else:
        decision = "auto_approve"

    top_features = []
    if cfg["model_name"] == "logistic_regression":
        top_features = _top_contributing_features(model, X.iloc[0], feature_columns)

    return {
        "risk_score": round(risk_score, 4),
        "decision": decision,
        "top_features": top_features,
        "model_name": cfg["model_name"],
    }
