"""Held-out evaluation: precision/recall + a cost-weighted confusion matrix.

Uses the exact same seeded split as model/train.py, so this only ever
touches the 20% test slice that training and threshold tuning never saw.
Writes eval/report.md and eval/pr_curve.png.
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score

from model.features import build_feature_matrix
from model.train import split_data

LABEL_COL = "label_is_fraudulent_return"
REVIEW_FRICTION_COST = 150  # INR: support/goodwill cost of holding or denying a legit return


def evaluate():
    df = pd.read_csv("data/returns_dataset.csv")
    _, _, test = split_data(df)

    feature_columns = json.load(open("model/artifacts/feature_columns.json"))
    cfg = json.load(open("model/artifacts/threshold_config.json"))
    model = joblib.load("model/artifacts/model.pkl")

    X_test = build_feature_matrix(test, feature_columns)
    y_test = test[LABEL_COL].values
    score = model.predict_proba(X_test)[:, 1]

    t1, t2 = cfg["flag_threshold_t1"], cfg["deny_threshold_t2"]
    flagged = score >= t1
    denied = score >= t2

    ap = average_precision_score(y_test, score)
    precision = precision_score(y_test, flagged)
    recall = recall_score(y_test, flagged)
    f1 = f1_score(y_test, flagged)

    tp = int(((flagged) & (y_test == 1)).sum())
    fp = int(((flagged) & (y_test == 0)).sum())
    fn = int(((~flagged) & (y_test == 1)).sum())
    tn = int(((~flagged) & (y_test == 0)).sum())

    fn_order_values = test.loc[(~flagged) & (y_test == 1), "order_value"]
    model_cost = fn_order_values.sum() + fp * REVIEW_FRICTION_COST

    approve_everything_cost = test.loc[y_test == 1, "order_value"].sum()
    flag_everything_cost = int((y_test == 0).sum()) * REVIEW_FRICTION_COST

    deny_precision = float(y_test[denied].mean()) if denied.sum() else None

    prec_curve, rec_curve, _ = precision_recall_curve(y_test, score)
    plt.figure(figsize=(5, 4))
    plt.plot(rec_curve, prec_curve, label=f"AP={ap:.3f}")
    plt.axhline(y_test.mean(), color="gray", linestyle="--", label=f"base rate={y_test.mean():.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Return-Risk Scorer — held-out PR curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("eval/pr_curve.png", dpi=150)
    plt.close()

    report = f"""# Held-out evaluation report

Model: **{cfg['model_name']}** | Test set size: {len(test)} ({int(y_test.sum())} fraudulent, {int((y_test == 0).sum())} legit)
Thresholds: flag (T1) = {t1:.4f} | auto-deny (T2) = {t2:.4f}

## Flag decision (score >= T1) vs. auto-approve

| | Predicted fraud (flagged) | Predicted legit (auto-approved) |
|---|---|---|
| **Actual fraud** | TP = {tp} | FN = {fn} |
| **Actual legit** | FP = {fp} | TN = {tn} |

- Precision: {precision:.3f}
- Recall: {recall:.3f}
- F1: {f1:.3f}
- PR-AUC (average precision): {ap:.3f}
- Auto-deny band (score >= T2) precision: {f"{deny_precision:.3f}" if deny_precision is not None else "n/a (no test records in this band)"} ({int(denied.sum())} records)

## Cost-weighted comparison (INR)

Cost model: a false negative (fraud auto-approved) costs the full order value lost; a false positive (legit return flagged/held) costs a flat Rs {REVIEW_FRICTION_COST} in review/support friction.

| Strategy | Cost |
|---|---|
| Approve everything (no model) | Rs {approve_everything_cost:,.0f} |
| Flag everything (review every return) | Rs {flag_everything_cost:,.0f} |
| **This model** | **Rs {model_cost:,.0f}** |

Fraud loss still slipping through (false negatives, {fn} records): Rs {fn_order_values.sum():,.0f}
Legit customers held unnecessarily (false positives): {fp} records (Rs {fp * REVIEW_FRICTION_COST:,.0f} friction cost)

## Honest caveats

- Only {int(denied.sum())} of {tp} correctly-flagged fraud cases meet the auto-deny bar ({cfg['target_precision']:.0%} precision); the rest are routed to human review, not auto-denied, because the model isn't confident enough on its own.
- Trained on synthetic data with deliberately overlapping (not perfectly separable) legit/fraud feature distributions to approximate real-world signal quality; absolute numbers will shift on real data.
"""
    with open("eval/report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    return dict(precision=precision, recall=recall, f1=f1, ap=ap, model_cost=model_cost,
                approve_everything_cost=approve_everything_cost, flag_everything_cost=flag_everything_cost)


if __name__ == "__main__":
    evaluate()
