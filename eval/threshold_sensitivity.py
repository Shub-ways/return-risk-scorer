"""Sweep the flag threshold T1 across the score range and report how
precision, recall, and expected cost move — so the chosen threshold in
threshold_config.json is a documented trade-off, not an arbitrary pick.

Runs on the same held-out test split as evaluate.py. Writes
eval/threshold_sensitivity.csv and eval/threshold_sensitivity.png.
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.features import build_feature_matrix
from model.train import split_data

LABEL_COL = "label_is_fraudulent_return"
REVIEW_FRICTION_COST = 150


def sweep():
    df = pd.read_csv("data/returns_dataset.csv")
    _, _, test = split_data(df)

    feature_columns = json.load(open("model/artifacts/feature_columns.json"))
    cfg = json.load(open("model/artifacts/threshold_config.json"))
    model = joblib.load("model/artifacts/model.pkl")

    X_test = build_feature_matrix(test, feature_columns)
    y_test = test[LABEL_COL].values
    order_values = test["order_value"].values
    score = model.predict_proba(X_test)[:, 1]

    # Include the exact deployed thresholds in the grid so the reported
    # "chosen T1" row matches evaluate.py's numbers exactly, not a nearest
    # neighbor on a fixed grid.
    thresholds = np.unique(np.concatenate([
        np.linspace(0.0, 1.0, 101), [cfg["flag_threshold_t1"], cfg["deny_threshold_t2"]],
    ]))
    rows = []
    for t in thresholds:
        flagged = score >= t
        tp = int((flagged & (y_test == 1)).sum())
        fp = int((flagged & (y_test == 0)).sum())
        fn_mask = (~flagged) & (y_test == 1)
        fn = int(fn_mask.sum())
        tn = int((~flagged & (y_test == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        cost = order_values[fn_mask].sum() + fp * REVIEW_FRICTION_COST

        rows.append(dict(threshold=t, precision=precision, recall=recall,
                          flagged_count=int(flagged.sum()), tp=tp, fp=fp, fn=fn, tn=tn, cost=cost))

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv("eval/threshold_sensitivity.csv", index=False)

    current_t1 = cfg["flag_threshold_t1"]
    best_row = sweep_df.loc[sweep_df["cost"].idxmin()]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
    ax1.plot(sweep_df["threshold"], sweep_df["precision"], label="precision")
    ax1.plot(sweep_df["threshold"], sweep_df["recall"], label="recall")
    ax1.axvline(current_t1, color="red", linestyle="--", label=f"chosen T1={current_t1:.2f}")
    ax1.set_ylabel("precision / recall")
    ax1.legend()
    ax1.set_title("Flag-threshold sensitivity (held-out test set)")

    ax2.plot(sweep_df["threshold"], sweep_df["cost"], color="black", label="expected cost (Rs)")
    ax2.axvline(current_t1, color="red", linestyle="--")
    ax2.axvline(best_row["threshold"], color="green", linestyle=":", label=f"cost-minimizing T1={best_row['threshold']:.2f}")
    ax2.set_xlabel("flag threshold")
    ax2.set_ylabel("expected cost (Rs)")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("eval/threshold_sensitivity.png", dpi=150)
    plt.close()

    current_row = sweep_df.loc[sweep_df["threshold"] == current_t1].iloc[0]

    summary = f"""
## Threshold sensitivity (held-out test set)

Full sweep: [eval/threshold_sensitivity.csv](threshold_sensitivity.csv) | chart: [eval/threshold_sensitivity.png](threshold_sensitivity.png)

- Chosen T1 = {current_t1:.3f} (picked on validation to hit >=85% recall): precision {current_row['precision']:.3f}, recall {current_row['recall']:.3f}, cost Rs {current_row['cost']:,.0f}
- Cost-minimizing T1 = {best_row['threshold']:.3f} on this test set: precision {best_row['precision']:.3f}, recall {best_row['recall']:.3f}, cost Rs {best_row['cost']:,.0f}

We deliberately chose the recall-first threshold over the cost-minimizing one: minimizing cost alone would let more fraud through in exchange for fewer reviewer hours, which trades a hard, recoverable cost (staff time) for a soft, less recoverable one (fraud loss + repeat-offender risk). This gap is small on the current dataset ({current_row['cost'] - best_row['cost']:,.0f} Rs) but is exactly the kind of trade-off worth stating explicitly rather than silently picking whichever threshold makes the top-line number look best.
"""
    with open("eval/report.md", "a", encoding="utf-8") as f:
        f.write(summary)

    print(summary)
    return sweep_df


if __name__ == "__main__":
    sweep()
