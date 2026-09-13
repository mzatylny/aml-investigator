"""Charts shared by the local dashboard and the exported report."""

from html import escape
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .model import precision_recall_data

TEAL, GOLD = "#4de0c1", "#ffca70"


def style(figure, height=400):
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", color="#dce5f0"), height=height,
        margin=dict(l=15, r=15, t=35, b=25), legend=dict(orientation="h", y=1.12),
    )
    return figure


def pr_figure(experiment):
    fig = px.line(precision_recall_data(experiment), x="recall", y="precision", color="model",
                  color_discrete_sequence=[TEAL, GOLD, "#b6a0ff"], labels={"recall": "Recall", "precision": "Precision"})
    baseline = experiment.report["test"]["random_forest"]["prevalence"]
    fig.add_hline(y=baseline, line_dash="dot", annotation_text="Positive prevalence")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    return style(fig)


def network_figure(transactions, transaction_id, max_edges=60):
    selected = transactions.loc[transactions.transaction_id.eq(transaction_id)].iloc[0]
    seeds = {selected.sender, selected.receiver}
    past = transactions[
        ((transactions.timestamp < selected.timestamp) | transactions.transaction_id.eq(transaction_id))
        & (transactions.timestamp >= selected.timestamp - pd.Timedelta(hours=24))
    ]
    context = past[past.sender.isin(seeds) | past.receiver.isin(seeds)].tail(max_edges)
    graph = nx.DiGraph()
    for row in context.itertuples(index=False):
        graph.add_edge(row.sender, row.receiver)
    positions = nx.spring_layout(graph, seed=42, k=1.4)
    fig = go.Figure()
    for row in context.itertuples(index=False):
        x0, y0 = positions[row.sender]
        x1, y1 = positions[row.receiver]
        active = row.transaction_id == transaction_id
        color = GOLD if active else "#52728f"
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines", showlegend=False,
            line=dict(color=color, width=3.5 if active else 1.2),
            text=[f"{escape(row.sender)} → {escape(row.receiver)}<br>EUR {row.amount:,.2f}"] * 2,
            hoverinfo="text",
        ))
        fig.add_annotation(x=x1 * .8 + x0 * .2, y=y1 * .8 + y0 * .2,
                           ax=x1 * .68 + x0 * .32, ay=y1 * .68 + y0 * .32,
                           xref="x", yref="y", axref="x", ayref="y", text="",
                           showarrow=True, arrowhead=2, arrowsize=1.1, arrowcolor=color)
    nodes = list(graph.nodes)
    fig.add_trace(go.Scatter(
        x=[positions[n][0] for n in nodes], y=[positions[n][1] for n in nodes],
        mode="markers+text", text=[escape(n) for n in nodes], textposition="top center",
        marker=dict(size=[21 if n in seeds else 12 for n in nodes],
                    color=[GOLD if n in seeds else TEAL for n in nodes], line=dict(width=2, color="#122334")),
        hovertext=[f"Account {escape(n)}" for n in nodes], hoverinfo="text", showlegend=False,
    ))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, 510), context


def gnn_network_figure(transactions, edges, transaction_id, max_nodes=45):
    """Show a capped two-hop ancestor subgraph used for GraphSAGE message passing."""
    root = int(transactions.index[transactions.transaction_id.eq(transaction_id)][0])
    nodes, frontier = {root}, {root}
    for _ in range(2):
        parents = {int(a) for a, b in edges.T if int(b) in frontier} - nodes
        frontier = set(sorted(parents, reverse=True)[:max_nodes - len(nodes)])
        nodes.update(frontier)
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(nodes))
    graph.add_edges_from((int(a), int(b)) for a, b in edges.T if int(a) in nodes and int(b) in nodes)
    positions = nx.spring_layout(graph, seed=42, k=1.2)
    figure = go.Figure()
    for source, target in graph.edges:
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        figure.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                    line=dict(color="#657392", width=1.1), hoverinfo="skip", showlegend=False))
        figure.add_annotation(x=.8*x1+.2*x0, y=.8*y1+.2*y0, ax=.65*x1+.35*x0, ay=.65*y1+.35*y0,
                               xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True,
                               arrowhead=2, arrowcolor="#8b9bb6", arrowsize=1)
    ordered = sorted(nodes)
    rows = transactions.loc[ordered]
    figure.add_trace(go.Scatter(
        x=[positions[i][0] for i in ordered], y=[positions[i][1] for i in ordered],
        mode="markers+text", text=[escape(row.transaction_id) if i == root else "" for i, row in rows.iterrows()],
        textposition="top center", showlegend=False,
        marker=dict(size=[23 if i == root else 13 for i in ordered],
                    color=rows.gnn_score, colorscale=[[0, "#315879"], [1, "#b6a0ff"]], cmin=0, cmax=1,
                    colorbar=dict(title="GNN score"), line=dict(color=GOLD, width=[3 if i == root else 0 for i in ordered])),
        hovertext=[f"{escape(row.transaction_id)}<br>{escape(row.sender)} → {escape(row.receiver)}"
                   f"<br>EUR {row.amount:,.2f}<br>{row.timestamp}<br>GNN {row.gnn_score:.3f}" for row in rows.itertuples()],
        hoverinfo="text",
    ))
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return style(figure, 500), rows
