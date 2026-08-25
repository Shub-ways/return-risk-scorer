# Architecture — Return-Risk Scorer

## Problem

Return fraud/abuse (wardrobing, serial returners, refund-to-different-account, device-reuse rings) quietly eats merchant margin. A human reviewing every return doesn't scale; auto-approving everything bleeds money. This system scores every return and routes only the ambiguous middle to a human.

## Flow

```
Return request
      |
      v
[ features.py ]  raw fields -> numeric feature vector
      |
      v
[ Logistic Regression + fixed thresholds ]   <-- the ONLY thing that decides
      |
      +--> risk_score, decision (auto_approve / hold_for_review / auto_deny)
      |
      v
[ Gemini 2.5 Flash ]  score + top-3 feature contributions -> plain-English
                       explanation for the reviewer (falls back to a
                       template if no API key / call fails)
      |
      v
[ audit_log.py ]  every decision (auto or manual) appended, immutable
      |
      v
[ Streamlit ]  Review Queue | Batch Metrics | Audit Trail
```

## Why the LLM never touches the decision

A merchant's money decision has to be auditable and reproducible. A classical model with fixed, cost-tuned thresholds gives:
- **Determinism** — same input, same decision, every time.
- **Exact per-record attribution** — logistic regression coefficients times the (scaled) feature value are an exact, cheap explanation, unlike a black-box model that would need SHAP or similar to explain individual predictions.
- **No hallucination risk in the money path** — the LLM's only failure mode (a bad explanation) is cosmetic, never a wrong auto-approve/deny.

This is the "AI judgment: right tool in right place, and where you chose not to use one" call for this build: statistics for the decision, LLM for the narrative.

## Model selection

`model/train.py` trains both a logistic regression pipeline and a gradient boosting classifier, and prefers logistic regression unless gradient boosting's held-out PR-AUC clears it by a large margin (0.15) — because gradient boosting's performance edge isn't worth losing exact per-record explainability for a money decision. On the current synthetic dataset, gradient boosting scores higher (PR-AUC ~0.54 vs ~0.41) but not by enough to give up interpretability.

## Thresholds and the cost model

Two thresholds, tuned on a validation split the test set never touches:
- **T1 (flag)**: lowest score that still catches ≥85% of validation fraud. Below T1 → auto-approve.
- **T2 (auto-deny)**: lowest score with ≥85% validation precision. At/above T2 → auto-deny; between T1 and T2 → hold for human review.

On this dataset, **no score band ever reaches 85% precision** — roughly 4% of legitimate customers are, by construction, feature-identical to fraud (shared devices, quick returns for genuine reasons), which caps precision even at the top of the score distribution. So T2 collapses to an unreachable threshold and the system never auto-denies; it only ever flags for human review. We consider this a feature: the asymmetric cost of wrongly denying a real customer means the system should never pull that trigger on ambiguous signal alone. See the "Honest caveats" section of [eval/report.md](../eval/report.md).

## Honest metrics

`eval/evaluate.py` runs once, on the untouched 20% test split, and reports precision/recall/F1/PR-AUC plus a **cost-weighted confusion matrix**: false negatives cost the full order value lost to fraud, false positives cost a flat review/friction fee. This is compared against two baselines ("approve everything," "flag everything") so the reported win is a real cost reduction, not a cherry-picked accuracy number.

## What broke

1. **The first dataset was perfectly separable** — PR-AUC 1.0 — because the injected fraud patterns had no overlap with legit behavior. Fixed by deliberately blending a fraction of "quiet fraud" (fraud with legit-looking features) and "loud legit" (legit customers who trip fraud-shaped features) into the generator, which is also what produced the T2-never-reachable finding above.
2. **The threshold-sensitivity report and the main eval report disagreed on the model's own cost number** (Rs 28,718 vs Rs 28,568) even though both read the same model and the same test set. Cause: `threshold_sensitivity.py` swept a fixed grid of 101 evenly-spaced thresholds and picked the nearest one to the deployed T1 for its headline row — close, but not the exact value `evaluate.py` used. Fixed by inserting the exact deployed T1/T2 into the sweep grid instead of approximating. Lesson: two reports quoting "the same" number from two code paths need to be checked against each other, not just individually sanity-checked.
