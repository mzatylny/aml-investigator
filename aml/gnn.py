"""Two-layer temporal GraphSAGE, implemented with ordinary PyTorch tensors.

Nodes are transactions; directed messages flow only from past transactions sharing
an account. This is a temporal transaction graph, not the account-transfer graph.
"""

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score
from .data import validate_transactions
from .features import FEATURES, build_features
from .model import alert_mask

GRAPH_SCHEMA = 1


def temporal_edges(transactions, max_neighbors=8, window_hours=24):
    """Return [source, target] indices in validated chronological transaction order."""
    if max_neighbors < 1 or window_hours <= 0:
        raise ValueError("Graph neighbour count and history window must be positive.")
    tx = validate_transactions(transactions)
    # asi8 follows the array's resolution; Timedelta.value is always nanoseconds.
    times = tx.timestamp.dt.as_unit("ns").array.asi8
    window = pd.Timedelta(hours=window_hours).value
    histories = defaultdict(lambda: deque(maxlen=max_neighbors))
    edges = []
    start = 0
    while start < len(tx):
        end = start + 1
        while end < len(tx) and times[end] == times[start]:
            end += 1
        for i in range(start, end):
            candidates = set()
            for account in (tx.sender.iat[i], tx.receiver.iat[i]):
                history = histories[account]
                while history and times[history[0]] < times[i] - window:
                    history.popleft()
                candidates.update(history)
            for previous in sorted(candidates)[-max_neighbors:]:
                edges.append((previous, i))
        # Updating only after the batch excludes simultaneous transfers.
        for i in range(start, end):
            histories[tx.sender.iat[i]].append(i)
            histories[tx.receiver.iat[i]].append(i)
        start = end
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2).T.copy()


def feature_values(features):
    values = features[FEATURES].to_numpy(dtype=np.float32, copy=True)
    # Reduce count/ratio/lag tails before fitting train-only mean and scale.
    values[:, 3:11] = np.log1p(values[:, 3:11])
    return values


class MeanSAGE(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.combine = nn.Linear(inputs * 2, outputs)

    def forward(self, nodes, edges):
        neighbours = torch.zeros_like(nodes)
        degree = nodes.new_zeros((len(nodes), 1))
        if edges.shape[1]:
            source, target = edges
            neighbours.index_add_(0, target, nodes[source])
            degree.index_add_(0, target, nodes.new_ones((len(source), 1)))
        average = neighbours / degree.clamp(min=1)
        return torch.relu(self.combine(torch.cat([nodes, average], dim=1)))


class TemporalGraphSAGE(nn.Module):
    def __init__(self, inputs=len(FEATURES), hidden=32):
        super().__init__()
        self.layer1 = MeanSAGE(inputs, hidden)
        self.layer2 = MeanSAGE(hidden, hidden)
        self.classifier = nn.Sequential(nn.Linear(hidden + inputs, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, nodes, edges):
        first = self.layer1(nodes, edges)
        second = self.layer2(first, edges)
        return self.classifier(torch.cat([nodes, second], dim=1)).squeeze(1)


@dataclass
class GraphSAGEBundle:
    model: TemporalGraphSAGE
    mean: np.ndarray
    scale: np.ndarray
    threshold: float = 0.5
    max_neighbors: int = 8
    window_hours: int = 24
    hidden: int = 32
    run_id: str | None = None

    def predict(self, features, edges):
        values = np.clip((feature_values(features) - self.mean) / self.scale, -5, 5)
        self.model.eval()
        with torch.inference_mode():
            return torch.sigmoid(self.model(torch.from_numpy(values), torch.as_tensor(edges))).numpy()


def train_graphsage(tx, features, seed=42, epochs=40, patience=8):
    if epochs < 1 or patience < 1:
        raise ValueError("GNN epochs and patience must be positive.")
    splits = tx.split.to_numpy()
    train_n = int(np.sum(splits == "train"))
    validation_end = train_n + int(np.sum(splits == "validation"))
    if not (np.all(splits[:train_n] == "train") and np.all(splits[train_n:validation_end] == "validation")
            and np.all(splits[validation_end:] == "test")):
        raise ValueError("GraphSAGE requires contiguous chronological train/validation/test splits.")
    values = feature_values(features)
    mean, scale = values[:train_n].mean(axis=0), values[:train_n].std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    nodes = torch.from_numpy(np.clip((values - mean) / scale, -5, 5))
    edges = temporal_edges(tx)
    train_edges = torch.from_numpy(edges[:, edges[1] < train_n])
    validation_edges = torch.from_numpy(edges[:, edges[1] < validation_end])
    y_train = torch.tensor(tx.is_laundering.iloc[:train_n].to_numpy(), dtype=torch.float32)
    y_validation = tx.is_laundering.iloc[train_n:validation_end].to_numpy()
    if y_train.unique().numel() != 2 or len(np.unique(y_validation)) != 2:
        raise ValueError("GNN training and validation require both classes.")
    records, best_epoch, best_ap, stalled = [], 0, -1.0, 0
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = TemporalGraphSAGE()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss(pos_weight=(len(y_train) - y_train.sum()) / y_train.sum())
            best_state = deepcopy(model.state_dict())
            for epoch in range(1, epochs + 1):
                model.train()
                optimizer.zero_grad()
                # The optimisation graph physically excludes validation and test nodes.
                logits = model(nodes[:train_n], train_edges)
                loss = criterion(logits, y_train)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                model.eval()
                with torch.inference_mode():
                    scores = torch.sigmoid(model(nodes[:validation_end], validation_edges))[train_n:].numpy()
                ap = float(average_precision_score(y_validation, scores))
                records.append({"epoch": epoch, "train_loss": float(loss.detach()), "validation_average_precision": ap})
                if ap > best_ap + 1e-5:
                    best_ap, best_epoch, stalled = ap, epoch, 0
                    best_state = deepcopy(model.state_dict())
                else:
                    stalled += 1
                if stalled >= patience:
                    break
            model.load_state_dict(best_state)
            bundle = GraphSAGEBundle(model, mean, scale)
            predictions = bundle.predict(features, edges)
    finally:
        torch.set_num_threads(old_threads)
    metadata = {
        "architecture": "Two-layer temporal GraphSAGE (mean aggregation)",
        "node_type": "transaction", "edge_type": "past shared-account transaction -> later transaction",
        "hidden_dimensions": 32, "layers": 2, "max_neighbors": 8, "window_hours_per_edge": 24,
        "max_epochs": epochs, "epochs_run": len(records), "best_epoch": best_epoch,
        "best_validation_average_precision": best_ap, "patience": patience,
        "optimizer": "AdamW", "learning_rate": 0.003, "weight_decay": 0.0001,
        "device": "cpu", "torch_version": str(torch.__version__),
        "graph_nodes": len(tx), "graph_edges": int(edges.shape[1]), "training_history": records,
        "normalization": "log1p counts/ratios/lag; mean/std fitted only on training; clip [-5,5]",
    }
    return bundle, predictions, edges, metadata


def save_graphsage(bundle, path):
    torch.save({
        "schema": GRAPH_SCHEMA, "features": FEATURES, "state_dict": bundle.model.state_dict(),
        "mean": torch.from_numpy(bundle.mean), "scale": torch.from_numpy(bundle.scale),
        "threshold": bundle.threshold, "max_neighbors": bundle.max_neighbors,
        "window_hours": bundle.window_hours, "hidden": bundle.hidden,
        "torch_version": str(torch.__version__),
        "run_id": bundle.run_id,
    }, path)


def load_graphsage(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema") != GRAPH_SCHEMA or state.get("features") != FEATURES:
        raise ValueError("Unsupported GNN checkpoint schema; retrain with this version.")
    model = TemporalGraphSAGE(hidden=state["hidden"])
    model.load_state_dict(state["state_dict"])
    return GraphSAGEBundle(model, state["mean"].numpy(), state["scale"].numpy(),
                           state["threshold"], state["max_neighbors"], state["window_hours"], state["hidden"],
                           state.get("run_id"))


def score_graphsage(bundle, transactions, history=None):
    current = validate_transactions(transactions)
    ids = set(current.transaction_id)
    if history is not None:
        past = validate_transactions(history)
        if past.timestamp.max() >= current.timestamp.min():
            raise ValueError("History must end strictly before the first transaction to score.")
        if ids.intersection(past.transaction_id):
            raise ValueError("History and new transactions must have distinct IDs.")
        current = pd.concat([past, current], ignore_index=True)
    current = current.drop(columns=["is_laundering", "pattern", "scenario_id"], errors="ignore")
    tx, features = build_features(current)
    edges = temporal_edges(tx, bundle.max_neighbors, bundle.window_hours)
    tx["gnn_score"] = bundle.predict(features, edges)
    tx["gnn_alert"] = alert_mask(tx.gnn_score, bundle.threshold)
    return tx[tx.transaction_id.isin(ids)].reset_index(drop=True)
