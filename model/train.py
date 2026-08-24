"""Train the return-risk classifier and tune routing thresholds.

Split strategy (fixed seed, reused by eval/evaluate.py so the test set stays
untouched throughout): 60% train / 20% val (model selection + threshold
tuning) / 20% held-out test (touched only by eval/evaluate.py).
"""
import json

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model.features import build_feature_matrix

SEED = 42
LABEL_COL = "label_is_fraudulent_return"

TARGET_RECALL = 0.85       # T1: catch at least this much fraud outside auto-approve
TARGET_PRECISION = 0.85    # T2: only auto-deny when this confident


def split_data(df: pd.DataFrame, seed: int = SEED):
    train_val, test = train_test_split(df, test_size=0.20, stratify=df[LABEL_COL], random_state=seed)
    train, val = train_test_split(train_val, test_size=0.25, stratify=train_val[LABEL_COL], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def pick_flag_threshold(y_true, y_score, target_recall: float) -> float:
    """Lowest score threshold whose recall is >= target_recall (T1)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns thresholds of len(n-1); pair with recall[:-1]
    candidates = [(t, r) for t, r in zip(thresholds, recall[:-1]) if r >= target_recall]
    if not candidates:
        return float(thresholds.min()) if len(thresholds) else 0.5
    # Highest threshold that still meets the recall target -> fewest false positives.
    return float(max(candidates, key=lambda tr: tr[0])[0])


def pick_deny_threshold(y_true, y_score, target_precision: float) -> float:
    """Lowest score threshold whose precision is >= target_precision (T2)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    candidates = [(t, p) for t, p in zip(thresholds, precision[:-1]) if p >= target_precision]
    if not candidates:
        return float(thresholds.max()) if len(thresholds) else 0.9
    return float(min(candidates, key=lambda tp: tp[0])[0])


def main():
    df = pd.read_csv("data/returns_dataset.csv")
    train, val, test = split_data(df)

    X_train = build_feature_matrix(train)
    feature_columns = list(X_train.columns)
    X_val = build_feature_matrix(val, feature_columns)
    y_train, y_val = train[LABEL_COL], val[LABEL_COL]

    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    ])
    logreg.fit(X_train, y_train)
    logreg_score = average_precision_score(y_val, logreg.predict_proba(X_val)[:, 1])

    gbc = GradientBoostingClassifier(random_state=SEED)
    gbc.fit(X_train, y_train)
    gbc_score = average_precision_score(y_val, gbc.predict_proba(X_val)[:, 1])

    print(f"Logistic Regression val PR-AUC: {logreg_score:.4f}")
    print(f"Gradient Boosting val PR-AUC:  {gbc_score:.4f}")

    # Default to the transparent model unless boosting clears a real margin —
    # the money decision should stay explainable end-to-end, and logistic
    # regression's per-feature coefficients are what scoring/explain.py uses
    # to build the reviewer-facing "why flagged" explanation.
    MODEL_SELECTION_MARGIN = 0.15
    if gbc_score - logreg_score > MODEL_SELECTION_MARGIN:
        model, model_name, val_score = gbc, "gradient_boosting", gbc.predict_proba(X_val)[:, 1]
    else:
        model, model_name, val_score = logreg, "logistic_regression", logreg.predict_proba(X_val)[:, 1]
    print(f"Selected model: {model_name}")

    t1 = pick_flag_threshold(y_val, val_score, TARGET_RECALL)
    t2 = pick_deny_threshold(y_val, val_score, TARGET_PRECISION)
    t2 = max(t2, t1)  # deny threshold can never be below the flag threshold
    print(f"Flag threshold T1 (auto_approve below this): {t1:.4f}")
    print(f"Deny threshold T2 (auto_deny at/above this):  {t2:.4f}")

    joblib.dump(model, "model/artifacts/model.pkl")
    with open("model/artifacts/feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)
    with open("model/artifacts/threshold_config.json", "w") as f:
        json.dump({
            "model_name": model_name,
            "flag_threshold_t1": t1,
            "deny_threshold_t2": t2,
            "target_recall": TARGET_RECALL,
            "target_precision": TARGET_PRECISION,
        }, f, indent=2)

    print("Saved model.pkl, feature_columns.json, threshold_config.json to model/artifacts/")


if __name__ == "__main__":
    main()
