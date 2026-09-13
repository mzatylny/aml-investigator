"""Local Streamlit analyst workbench. Start with: streamlit run app.py"""

import pandas as pd
import plotly.express as px
import streamlit as st
from aml.__main__ import read_transactions
from aml.data import generate_transactions
from aml.features import FEATURES
from aml.model import alert_mask, evaluate, run_experiment, score_transactions
from aml.plots import GOLD, TEAL, gnn_network_figure, network_figure, pr_figure, style

st.set_page_config(page_title="AML Investigator", page_icon="◎", layout="wide")
st.markdown("""<style>
  .block-container {padding-top:2rem; max-width:1500px}
  [data-testid="stMetric"] {background:#152333; border:1px solid #283b4e; border-radius:12px; padding:18px}
  [data-testid="stMetricValue"] {color:#4de0c1}
  h1 {letter-spacing:-1.5px}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Generating synthetic transactions and evaluating the model…")
def load_demo(seed, include_gnn):
    return run_experiment(generate_transactions(seed=seed), seed=seed, with_gnn=include_gnn)


with st.sidebar:
    st.markdown("### ◎ AML Investigator")
    st.caption("TRANSACTION INTELLIGENCE")
    st.divider()
    seed = st.number_input("Dataset seed", min_value=0, max_value=999999, value=42, step=1)
    include_gnn = st.checkbox("Train GraphSAGE", value=True, key="include_gnn")
    st.caption("Changing the seed generates a different synthetic dataset and retrains the model.")
    st.divider()
    st.markdown("**Demo environment**")
    st.caption("500 accounts · 90 days · EUR\n\nNo bank connections or external APIs.")
    st.caption("Portfolio research prototype. Scores prioritise review; they are not proof of wrongdoing.")

experiment = load_demo(int(seed), include_gnn)
tx = experiment.transactions
test = tx.query("split == 'test'").copy()
st.caption("MONITOR / INVESTIGATE / UNDERSTAND")
st.title("Follow the money.")
st.markdown("Explore transaction risk, inspect account connections, and measure the cost of false alarms.")
st.caption("SYNTHETIC DATA · Evaluation on the last 20% of calendar days · Random forest / rules / temporal GraphSAGE")
detectors = ["Random forest", "GraphSAGE"] if experiment.gnn else ["Random forest"]
detector = st.radio("Review model", detectors, index=len(detectors)-1, horizontal=True, key="review_model")
score_column = "gnn_score" if detector == "GraphSAGE" else "risk_score"
selected_threshold = experiment.gnn.threshold if detector == "GraphSAGE" else experiment.threshold

overview, investigation, graph_tab, validation, imported = st.tabs(["Overview", "Investigation", "Graph neural network", "Model & evaluation", "Import CSV"])

with overview:
    threshold = st.slider("Review threshold", min_value=0.0, max_value=1.01,
                          value=1.01 if selected_threshold > 1 else float(selected_threshold),
                          step=0.01, key=f"threshold_{detector}_{seed}")
    st.caption(f"{detector} validation-selected threshold: {selected_threshold:.4f}. Moving this control explores test outcomes; it does not change the saved evaluation.")
    st.caption("Set the threshold to 1.01 to disable all alerts, including scores equal to 1.")
    review = alert_mask(test[score_column], threshold)
    metrics = evaluate(test.is_laundering, test[score_column], threshold)
    cols = st.columns(4)
    for column, label, value in zip(cols, ["Test transactions", "Review queue", "Precision", "Recall"],
                                    [f"{len(test):,}", f"{metrics['alerts']:,}", f"{metrics['precision']:.1%}", f"{metrics['recall']:.1%}"]):
        column.metric(label, value)
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Daily review volume")
        daily = test.assign(day=test.timestamp.dt.date, review=review).groupby("day").agg(
            Transactions=("transaction_id", "size"), Alerts=("review", "sum")).reset_index()
        st.plotly_chart(style(px.bar(daily, x="day", y="Alerts", color_discrete_sequence=[TEAL])), width="stretch")
    with right:
        st.subheader("Risk score distribution")
        st.plotly_chart(style(px.histogram(test, x=score_column, nbins=35, color_discrete_sequence=[TEAL])), width="stretch")
    st.subheader("Prioritised review queue")
    queue = test[review].sort_values(score_column, ascending=False)
    columns = ["transaction_id", "timestamp", "sender", "receiver", "amount", score_column, "evidence"]
    st.dataframe(queue[columns], hide_index=True, width="stretch")
    st.download_button("Download review queue", queue[columns].to_csv(index=False), "aml_review_queue.csv", "text/csv")

with investigation:
    st.subheader("Inspect a transaction")
    ordered = test.sort_values(score_column, ascending=False)
    selected_id = st.selectbox("Transaction", ordered.transaction_id.tolist())
    selected = test.loc[test.transaction_id.eq(selected_id)].iloc[0]
    left, right = st.columns([2, 1])
    with left:
        graph, context = network_figure(tx, selected_id)
        st.plotly_chart(graph, width="stretch")
        st.caption("Gold: selected transfer and its accounts. Arrows show direction. Context includes up to 60 adjacent transfers from the preceding 24 hours; no future transfers are shown.")
    with right:
        st.metric(f"{detector} risk score", f"{selected[score_column]:.3f}")
        st.metric("Transfer amount", f"€{selected.amount:,.2f}")
        st.markdown("**Observed rule evidence**")
        st.write(selected.evidence)
        st.caption("These rules describe the transaction context. They are separate from the model and do not explain its score causally.")
        if st.checkbox("Show synthetic ground truth", key="show_truth"):
            st.write({"injected_label": int(selected.is_laundering), "pattern": selected.pattern})
    st.dataframe(context[["timestamp", "sender", "receiver", "amount", "transaction_id"]], hide_index=True, width="stretch")

with graph_tab:
    st.subheader("Learn from connected transactions")
    if experiment.gnn is None:
        st.info("Enable Train GraphSAGE in the sidebar to train and inspect the neural network.")
    else:
        st.write("Each node is a transaction. A directed edge brings information from an earlier transaction sharing a sender or receiver account. Two GraphSAGE layers aggregate this history before assigning a score.")
        st.caption("This differs from the Investigation graph, whose nodes are accounts and whose arrows are money transfers.")
        network, neighbourhood = gnn_network_figure(tx, experiment.graph_edges, selected_id)
        st.plotly_chart(network, width="stretch")
        st.caption("Uses the transaction selected in Investigation. Gold outline: selected transaction. Purple: higher GNN score. Arrows show past-to-later information flow, not movement of funds. Up to 45 nodes from the two-hop receptive field are shown. This view is context, not a learned attribution or evidence of a criminal network.")
        left, right = st.columns(2)
        left.metric("Graph nodes", f"{experiment.report['gnn']['graph_nodes']:,}")
        right.metric("Temporal edges", f"{experiment.report['gnn']['graph_edges']:,}")
        history_table = pd.DataFrame(experiment.report["gnn"]["training_history"])
        st.plotly_chart(style(px.line(history_table, x="epoch", y="validation_average_precision",
                                     color_discrete_sequence=["#b6a0ff"])), width="stretch")
        st.caption(f"Checkpoint chosen at epoch {experiment.report['gnn']['best_epoch']} using validation average precision. Test labels do not select weights or thresholds.")
        st.dataframe(neighbourhood[["transaction_id", "timestamp", "sender", "receiver", "amount", "gnn_score"]], hide_index=True, width="stretch")

with validation:
    st.subheader("Evaluate beyond accuracy")
    st.write("Models train on the first 60% of days. The next 20% select the GNN checkpoint and set each detector's threshold for a 5% alert budget. The final 20% are held out for evaluation.")
    table = pd.DataFrame(experiment.report["test"]).T
    st.dataframe(table[["average_precision", "precision", "recall", "alert_rate", "precision_at_budget", "false_positives_per_1000"]].round(3), width="stretch")
    st.caption("Average precision summarises the precision–recall curve. Ties can make the rule baseline use less than its validation budget. Precision at budget ranks exactly the top 5% (ties follow chronological order).")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(pr_figure(experiment), width="stretch")
    with right:
        importance = pd.DataFrame({"Feature": FEATURES, "Importance": experiment.model.feature_importances_}).sort_values("Importance")
        st.plotly_chart(style(px.bar(importance, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[GOLD])), width="stretch")
        st.caption("Random forest global impurity importance can favour continuous features. It does not explain individual scores or the GNN.")
    st.markdown("**What the model can and cannot show**")
    st.write("The generator includes fan-in, fan-out, chains and cycles, alongside legitimate lookalikes. Labels and scenario IDs never enter the model. Features are computed from strictly earlier timestamps; simultaneous transfers do not see each other.")
    st.write("Synthetic labels are complete by construction. Real AML labels are incomplete and delayed, and transaction patterns differ. This experiment has no external-bank validation, calibrated probability estimates, or production monitoring.")
    st.json(experiment.report["recall_by_pattern"])
    if experiment.gnn:
        st.markdown("**GraphSAGE recall by injected pattern**")
        st.json(experiment.report["gnn_recall_by_pattern"])

with imported:
    st.subheader("Score a transaction file")
    st.write("Upload a CSV with transaction_id, timestamp, sender, receiver, amount and currency. EUR only. The demo-trained model scores your file locally; its results may not transfer to other data.")
    template = tx[["transaction_id", "timestamp", "sender", "receiver", "amount", "currency"]].tail(100)
    st.download_button("Download example CSV", template.to_csv(index=False), "example_transactions.csv", "text/csv")
    file = st.file_uploader("Transactions to score", type=["csv"])
    past_file = st.file_uploader("Earlier history (optional)", type=["csv"])
    st.caption("Without history, early transactions have a cold start. History must end before the first new transaction. GNN replay may need up to 72 hours of raw history (two 24h hops plus node features). Up to 50,000 rows in total.")
    if st.button("Score uploaded transactions", disabled=file is None):
        try:
            current = read_transactions(file)
            history = read_transactions(past_file) if past_file else None
            if len(current) + (len(history) if history is not None else 0) > 50000:
                raise ValueError("Limit the combined input to 50,000 rows.")
            with st.spinner("Computing transaction history and risk scores…"):
                scored = score_transactions(experiment.model, current, experiment.threshold, history)
                if experiment.gnn:
                    from aml.gnn import score_graphsage
                    gnn_scores = score_graphsage(experiment.gnn, current, history)
                    scored = scored.merge(gnn_scores[["transaction_id", "gnn_score", "gnn_alert"]], on="transaction_id", validate="one_to_one")
            st.session_state["uploaded_scores"] = scored
        except (ValueError, OSError, pd.errors.ParserError) as error:
            st.session_state.pop("uploaded_scores", None)
            st.error(str(error))
    if "uploaded_scores" in st.session_state:
        scored = st.session_state["uploaded_scores"]
        st.caption("Results from the last successful scoring run.")
        st.dataframe(scored, hide_index=True, width="stretch")
        st.download_button("Download scored CSV", scored.to_csv(index=False), "scored_transactions.csv", "text/csv")
