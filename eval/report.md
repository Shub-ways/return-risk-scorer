# Held-out evaluation report

Model: **logistic_regression** | Test set size: 800 (80 fraudulent, 720 legit)
Thresholds: flag (T1) = 0.4466 | auto-deny (T2) = 0.9999

## Flag decision (score >= T1) vs. auto-approve

| | Predicted fraud (flagged) | Predicted legit (auto-approved) |
|---|---|---|
| **Actual fraud** | TP = 69 | FN = 11 |
| **Actual legit** | FP = 128 | TN = 592 |

- Precision: 0.350
- Recall: 0.863
- F1: 0.498
- PR-AUC (average precision): 0.522
- Auto-deny band (score >= T2) precision: n/a (no test records in this band) (0 records)

## Cost-weighted comparison (INR)

Cost model: a false negative (fraud auto-approved) costs the full order value lost; a false positive (legit return flagged/held) costs a flat Rs 150 in review/support friction.

| Strategy | Cost |
|---|---|
| Approve everything (no model) | Rs 99,680 |
| Flag everything (review every return) | Rs 108,000 |
| **This model** | **Rs 28,718** |

Fraud loss still slipping through (false negatives, 11 records): Rs 9,518
Legit customers held unnecessarily (false positives): 128 records (Rs 19,200 friction cost)

## Honest caveats

- Only 0 of 69 correctly-flagged fraud cases meet the auto-deny bar (85% precision); the rest are routed to human review, not auto-denied, because the model isn't confident enough on its own.
- Trained on synthetic data with deliberately overlapping (not perfectly separable) legit/fraud feature distributions to approximate real-world signal quality; absolute numbers will shift on real data.

## Threshold sensitivity (held-out test set)

Full sweep: [eval/threshold_sensitivity.csv](threshold_sensitivity.csv) | chart: [eval/threshold_sensitivity.png](threshold_sensitivity.png)

- Chosen T1 = 0.447 (picked on validation to hit >=85% recall): precision 0.350, recall 0.863, cost Rs 28,718
- Cost-minimizing T1 = 0.540 on this test set: precision 0.439, recall 0.812, cost Rs 27,596

We deliberately chose the recall-first threshold over the cost-minimizing one: minimizing cost alone would let more fraud through in exchange for fewer reviewer hours, which trades a hard, recoverable cost (staff time) for a soft, less recoverable one (fraud loss + repeat-offender risk). This gap is small on the current dataset (1,122 Rs) but is exactly the kind of trade-off worth stating explicitly rather than silently picking whichever threshold makes the top-line number look best.
