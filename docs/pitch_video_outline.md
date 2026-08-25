# Pitch video outline (5 minutes)

Numbers below are pulled from the current [eval/report.md](../eval/report.md) — re-check them if you retrain before recording, since the dataset/thresholds may have moved.

## 0:00–0:30 — Hook + problem

"Return fraud — wardrobing, serial returners, refund-to-different-account, device-reuse rings — quietly eats merchant margin. You can't review every return by hand, and you can't approve everything either. I built a system that scores every return and only bothers a human with the ambiguous ones."

State the track and the one-line pitch: **Return-Risk Scorer — routes every return to auto-approve / hold-for-review / auto-deny, with a plain-English reason and a full audit trail.**

## 0:30–1:00 — Architecture in one breath

Show `docs/architecture.md`'s flow diagram or just say it on camera:

"The risk score and routing decision come entirely from a transparent logistic regression model with fixed, cost-tuned thresholds. The LLM — Gemini 2.5 Flash — does exactly one thing: it turns the score and the top contributing features into a plain-English note for the reviewer. It never sees the decision before it's made, and it can never change it. That's a deliberate choice: money decisions need to be deterministic and auditable, not something a language model can quietly steer."

## 1:00–3:00 — Live demo (Streamlit)

1. **Batch Metrics tab first** — show the held-out numbers before the live demo, so the audience trusts what they're about to watch isn't cherry-picked:
   - Precision 0.350 / Recall 0.863 / PR-AUC 0.522 on an 800-record held-out test set the model never trained or tuned thresholds on.
   - The cost table: approve-everything costs Rs 99,680, review-everything costs Rs 108,000, this model costs Rs 28,718 — roughly a 70% reduction versus both naive baselines.
   - Point at the "Honest caveats" section live: 128 legit customers get held unnecessarily, 11 fraud cases still slip through. Say the false-positive number out loud — don't hide it.
   - Threshold sensitivity chart: show that the chosen threshold isn't the cost-minimizing one, and explain why (recall-first, because fraud loss is less recoverable than reviewer time).

2. **Review Queue tab** — load a batch, show the auto-approve/hold/auto-deny counts. Explain live why auto-deny is 0 for this dataset (referenced caption in the UI) — this is a finding, not a bug.
   - Open a couple of flagged cards, read the Gemini-generated explanation out loud, click Approve or Deny.
   - Turn on "Show ground truth" and reveal whether the call was right — this is the single most convincing 20 seconds of the video.

3. **Audit Trail tab** — scroll it, point out every decision (auto and manual) is logged with a timestamp and an explanation, immutable, filterable by source.

## 3:00–4:00 — What broke (this is the answer they read first on the form)

Say this out loud, not just as a footnote:

"My first synthetic dataset was perfectly separable — the model hit 100% PR-AUC on held-out data. That's a red flag, not a win: real fraud signal is noisy. I went back and deliberately blended in 'quiet fraud' — fraud with legit-looking features — and 'loud legit' — legitimate customers who happen to trip fraud-shaped signals — until the numbers looked like a real, imperfect detection problem. That's also what produced the finding that the model can never clear the auto-deny confidence bar on this data, which became a real design decision instead of an accident."

(Optional second beat if time allows: the numeric mismatch between the two eval reports caught by cross-checking two code paths against each other.)

## 4:00–4:45 — Why this AI judgment call matters

"The easy version of this project uses an LLM to score fraud directly. I didn't do that, on purpose. A classifier gives exact, cheap, reproducible per-record explanations — the coefficient times the feature value, no hallucination possible. The LLM's only failure mode here is a slightly worse sentence, never a wrong money decision. That's the 'right tool in the right place' bar for this track."

## 4:45–5:00 — Close

"Repo's public, README has the quickstart, eval/report.md has the numbers un-cherry-picked. Thanks."

## Recording checklist

- [ ] Re-run `python -m eval.evaluate && python -m eval.threshold_sensitivity` right before recording so on-screen numbers match what you say
- [ ] Load a fresh batch in the Review Queue with `Load new batch` before recording (don't reuse stale session state)
- [ ] Turn `show_ground_truth` on only for the reveal beat, not the whole demo — it should look like a blind review first
- [ ] Keep the audit trail non-empty going into the recording (approve/deny at least 2-3 records first)
- [ ] Unlisted YouTube link is fine per the application form — just make sure the repo URL and video are both in the form
