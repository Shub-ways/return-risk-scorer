# Return-Risk Scorer

**Track:** AI Risk Manager — Razorpay AI Buildathon

Scores incoming e-commerce return requests for fraud/abuse risk (wardrobing, serial returners, refund-to-different-account, device-reuse rings) and routes each one to **auto-approve**, **hold for human review**, or **auto-deny** — with a plain-English explanation for the reviewer and a full audit trail.

## Design principle

The risk score and routing decision come **entirely** from a transparent logistic-regression classifier and fixed, cost-tuned thresholds. The LLM (Gemini 2.5 Flash) is used for exactly one thing — turning the score and its top contributing features into a reviewer-facing explanation. It never sees or influences the actual decision. See [docs/architecture.md](docs/architecture.md).

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate        # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

python data/generate_synthetic_data.py   # generate the labeled dataset
python -m model.train                    # train + select model, tune thresholds
python -m eval.evaluate                  # held-out precision/recall + cost report
python -m eval.threshold_sensitivity     # threshold trade-off sweep (appends to eval/report.md)
pytest tests/                            # decision-boundary unit tests

streamlit run app.py                     # review queue / metrics / audit trail UI
```

Optional: copy `.env.example` to `.env` and set `GEMINI_API_KEY` to get live LLM explanations. Without a key, the app falls back to a template-based explanation — it never breaks.

## Repo layout

```text
data/       synthetic labeled returns dataset generator
model/      feature engineering, training, model comparison, threshold tuning
scoring/    pure scoring function + LLM explanation layer
eval/       held-out precision/recall + cost-weighted confusion matrix
audit/      append-only decision log
app.py      Streamlit dashboard
tests/      unit tests
docs/       architecture writeup for the submission
```

## Results (held-out test set)

See [eval/report.md](eval/report.md) for the current numbers (regenerate with `python -m eval.evaluate`). Headline: the model cuts expected fraud+friction cost by roughly 70% versus both "approve everything" and "flag everything" baselines, at ~86% recall and ~35% precision on the flag decision — with honest reporting of the false positives that recall costs.
