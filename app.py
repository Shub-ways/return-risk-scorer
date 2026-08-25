"""Return-Risk Scorer — Streamlit demo.

Review Queue: score a fresh batch of returns, auto-route obvious cases,
put ambiguous ones in front of a human reviewer with an LLM-written
explanation. Batch Metrics: held-out precision/recall + cost-weighted
confusion matrix. Audit Trail: every decision ever logged, auto or manual.
"""
import pandas as pd
import streamlit as st

from audit.audit_log import log_decision, read_log
from eval.evaluate import evaluate
from model.features import RAW_FIELDS
from scoring.explain import explain_decision
from scoring.score import score_record

st.set_page_config(page_title="Return-Risk Scorer", layout="wide")
st.title("Return-Risk Scorer")
st.caption("AI Risk Manager — Razorpay AI Buildathon | risk score & routing come from a transparent classifier; the LLM only explains the decision, it never makes it.")

with st.sidebar:
    st.header("Session")
    reviewer_name = st.text_input("Reviewer name", value="demo_reviewer")
    show_ground_truth = st.checkbox(
        "Show ground truth (synthetic data)", value=False,
        help="This demo runs on labeled synthetic data, so we know the real answer. "
             "Off by default to simulate a real blind review; turn on to check the system's calls.",
    )

if "batch" not in st.session_state:
    st.session_state.batch = None
if "results" not in st.session_state:
    st.session_state.results = {}
if "explanations" not in st.session_state:
    st.session_state.explanations = {}
if "auto_logged_batch_id" not in st.session_state:
    st.session_state.auto_logged_batch_id = None

tab_queue, tab_metrics, tab_audit = st.tabs(["Review Queue", "Batch Metrics", "Audit Trail"])

with tab_queue:
    col1, col2 = st.columns([1, 3])
    with col1:
        n = st.number_input("Batch size", min_value=10, max_value=200, value=40, step=10)
        if st.button("Load new batch", type="primary"):
            df = pd.read_csv("data/returns_dataset.csv")
            batch = df.sample(n=n).reset_index(drop=True)
            batch_id = str(batch["return_id"].tolist())
            st.session_state.batch = batch
            st.session_state.results = {}
            for _, row in batch.iterrows():
                record = {k: row[k] for k in RAW_FIELDS}
                st.session_state.results[row["return_id"]] = score_record(record)
            st.session_state.auto_logged_batch_id = None

    batch = st.session_state.batch
    if batch is None:
        st.info("Load a batch to simulate an incoming stream of return requests.")
    else:
        results = st.session_state.results
        decisions = pd.Series({rid: r["decision"] for rid, r in results.items()})
        counts = decisions.value_counts()

        with col2:
            m1, m2, m3 = st.columns(3)
            m1.metric("Auto-approved", int(counts.get("auto_approve", 0)))
            m2.metric("Held for review", int(counts.get("hold_for_review", 0)))
            m3.metric("Auto-denied", int(counts.get("auto_deny", 0)))
            if counts.get("auto_deny", 0) == 0:
                st.caption(
                    "Auto-deny is empty by design, not a bug: on this dataset no score band reaches the "
                    "precision bar required to auto-deny (see 'Honest caveats' in Batch Metrics), so every "
                    "flagged return goes to a human instead of being denied automatically."
                )

        batch_id = str(batch["return_id"].tolist())
        if st.session_state.auto_logged_batch_id != batch_id:
            for _, row in batch.iterrows():
                rid = row["return_id"]
                r = results[rid]
                if r["decision"] in ("auto_approve", "auto_deny"):
                    log_decision(rid, r["risk_score"], r["decision"], "auto", r["model_name"])
            st.session_state.auto_logged_batch_id = batch_id

        st.subheader("Pending human review")
        already_decided = {e["return_id"] for e in read_log() if e["decision_source"] == "manual"}
        pending = [rid for rid, r in results.items() if r["decision"] == "hold_for_review" and rid not in already_decided]

        if not pending:
            st.success("Nothing pending — every flagged return in this batch has been reviewed.")

        for rid in pending:
            r = results[rid]
            row = batch[batch["return_id"] == rid].iloc[0]
            if rid not in st.session_state.explanations:
                st.session_state.explanations[rid] = explain_decision(r)
            explanation = st.session_state.explanations[rid]

            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                with c1:
                    label = f"**{rid}** — risk score `{r['risk_score']:.2f}` — {row['item_category']}, order value {row['order_value']:.0f}, reason: {row['return_reason_code']}"
                    if show_ground_truth:
                        truth = "FRAUD" if row["label_is_fraudulent_return"] == 1 else "legit"
                        label += f" — *ground truth: {truth}*"
                    st.markdown(label)
                    st.write(explanation)
                with c2:
                    if st.button("Approve", key=f"approve_{rid}"):
                        log_decision(rid, r["risk_score"], "approved", "manual", r["model_name"], explanation, reviewer=reviewer_name)
                        st.rerun()
                    if st.button("Deny", key=f"deny_{rid}"):
                        log_decision(rid, r["risk_score"], "denied", "manual", r["model_name"], explanation, reviewer=reviewer_name)
                        st.rerun()

        reviewed_this_batch = [
            e for e in read_log()
            if e["decision_source"] == "manual" and e["return_id"] in set(batch["return_id"])
        ]
        if reviewed_this_batch:
            st.subheader("Reviewed this batch")
            rows = []
            for e in reviewed_this_batch:
                row = batch[batch["return_id"] == e["return_id"]].iloc[0]
                entry = {"return_id": e["return_id"], "reviewer_decision": e["decision"], "reviewer": e["reviewer"]}
                if show_ground_truth:
                    truth = "fraud" if row["label_is_fraudulent_return"] == 1 else "legit"
                    entry["ground_truth"] = truth
                    entry["correct"] = (e["decision"] == "denied") == (truth == "fraud")
                rows.append(entry)
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tab_metrics:
    st.subheader("Held-out evaluation")
    st.caption("Runs on the 20% test split the model never saw during training or threshold tuning.")
    if st.button("Run evaluation"):
        with st.spinner("Scoring held-out test set..."):
            metrics = evaluate()
        st.session_state.last_metrics = metrics

    try:
        with open("eval/report.md", encoding="utf-8") as f:
            report = f.read()
        st.markdown(report)
        st.image("eval/pr_curve.png")
    except FileNotFoundError:
        st.info("Run the evaluation at least once (via `python -m eval.evaluate` or the button above) to see the report.")

with tab_audit:
    st.subheader("Audit trail")
    log = read_log()
    if not log:
        st.info("No decisions logged yet — load a batch in the Review Queue tab.")
    else:
        log_df = pd.DataFrame(log)
        source_filter = st.multiselect("Decision source", options=sorted(log_df["decision_source"].unique()), default=list(log_df["decision_source"].unique()))
        filtered = log_df[log_df["decision_source"].isin(source_filter)]
        st.dataframe(filtered.sort_values("timestamp", ascending=False), use_container_width=True)
