import numpy as np
import pandas as pd
import pytest
import torch
from aml.data import generate_transactions
from aml.features import build_features
from aml.gnn import MeanSAGE, feature_values, load_graphsage, save_graphsage, score_graphsage, temporal_edges
from aml.model import run_experiment
from aml.plots import gnn_network_figure


def tiny_transactions():
    return pd.DataFrame({
        "transaction_id": ["T0", "T1", "T2", "T3", "T4"],
        "timestamp": ["2025-01-01T10:00Z", "2025-01-01T10:00Z", "2025-01-01T10:10Z", "2025-01-01T10:20Z", "2025-01-02T10:21Z"],
        "sender": ["A", "B", "B", "C", "C"], "receiver": ["B", "A", "C", "D", "D"],
        "amount": [100, 100, 95, 90, 90], "currency": ["EUR"] * 5,
    })


@pytest.fixture(scope="module")
def experiment():
    data = generate_transactions(seed=9, days=12, accounts=50, daily_transactions=30)
    return run_experiment(data, seed=9, with_gnn=True, gnn_epochs=4)


def test_edges_are_strictly_causal_and_expire():
    tx = tiny_transactions()
    edges = temporal_edges(tx)
    assert edges.shape[0] == 2
    times = pd.to_datetime(tx.timestamp, utc=True)
    assert all(times[a] < times[b] for a, b in edges.T)
    assert not (edges[1] < 2).any()
    assert not (edges[1] == 4).any()
    assert {(0, 2), (1, 2), (2, 3)}.issubset(set(map(tuple, edges.T)))


def test_edges_are_unique_capped_and_prefix_invariant():
    tx = generate_transactions(seed=2, days=10, accounts=25, daily_transactions=25)
    edges = temporal_edges(tx, max_neighbors=3)
    _, counts = np.unique(edges[1], return_counts=True)
    assert max(counts) <= 3
    assert len(set(map(tuple, edges.T))) == edges.shape[1]
    expected = temporal_edges(tx.iloc[:200], max_neighbors=3)
    np.testing.assert_array_equal(expected, edges[:, edges[1] < 200])


def test_two_layers_actually_propagate_information_across_two_hops():
    layer = MeanSAGE(1, 1)
    with torch.no_grad():
        layer.combine.weight.copy_(torch.tensor([[0., 1.]]))
        layer.combine.bias.zero_()
    nodes = torch.tensor([[2.], [0.], [0.], [0.]])
    edges = torch.tensor([[0, 1], [1, 2]])
    first = layer(nodes, edges)
    second = layer(first, edges)
    assert first[1].item() == 2 and first[2].item() == 0
    assert second[2].item() == 2 and second[3].item() == 0
    empty = layer(nodes, torch.empty((2, 0), dtype=torch.long))
    assert empty.eq(0).all()


def test_training_normalization_uses_only_training_rows(experiment):
    train = experiment.transactions.split.eq("train")
    expected = feature_values(experiment.features.loc[train]).mean(axis=0)
    np.testing.assert_allclose(experiment.gnn.mean, expected)
    assert experiment.report["gnn"]["best_epoch"] <= 4
    validation = experiment.transactions.query("split == 'validation'")
    assert validation.gnn_alert.mean() <= .05


def test_gnn_predictions_do_not_change_when_future_is_appended(experiment):
    prefix = experiment.transactions.iloc[:200]
    _, features = build_features(prefix)
    scores = experiment.gnn.predict(features, temporal_edges(prefix))
    np.testing.assert_allclose(scores, experiment.transactions.gnn_score.iloc[:200], atol=1e-6)


def test_graph_connections_affect_the_learned_scores(experiment):
    without_messages = experiment.gnn.predict(experiment.features, np.empty((2, 0), dtype=np.int64))
    assert np.max(np.abs(without_messages - experiment.transactions.gnn_score.to_numpy())) > 1e-4


def test_checkpoint_and_history_inference_match_replay(experiment, tmp_path):
    checkpoint = tmp_path / "graphsage.pt"
    save_graphsage(experiment.gnn, checkpoint)
    restored = load_graphsage(checkpoint)
    tx = experiment.transactions
    current = tx.iloc[200:].drop(columns=["is_laundering", "pattern", "scenario_id"])
    scored = score_graphsage(restored, current, tx.iloc[:200])
    np.testing.assert_allclose(scored.gnn_score, tx.gnn_score.iloc[200:], atol=1e-6)
    assert restored.threshold == experiment.gnn.threshold
    assert scored.transaction_id.tolist() == current.transaction_id.tolist()
    with pytest.raises(ValueError, match="strictly before"):
        score_graphsage(restored, current, tx)


def test_test_labels_do_not_select_weights_or_threshold(experiment):
    frame = experiment.transactions.copy()
    test = frame.split.eq("test")
    frame.loc[test, "is_laundering"] = 1 - frame.loc[test, "is_laundering"]
    rerun = run_experiment(frame, seed=9, with_gnn=True, gnn_epochs=4)
    assert rerun.gnn.threshold == experiment.gnn.threshold
    for key, value in experiment.gnn.model.state_dict().items():
        torch.testing.assert_close(value, rerun.gnn.model.state_dict()[key], rtol=0, atol=0)


def test_gnn_neighbourhood_contains_target_and_only_earlier_nodes(experiment):
    tx = experiment.transactions
    selected = tx.iloc[400]
    figure, context = gnn_network_figure(tx, experiment.graph_edges, selected.transaction_id)
    assert selected.transaction_id in set(context.transaction_id)
    assert len(context) <= 45
    assert context.timestamp.max() <= selected.timestamp
    assert figure.data
