# Temporal GraphSAGE

## Graph definition

Every transaction is a node with 13 causal numeric features. A directed edge `j -> i` means that transaction `j` happened strictly before transaction `i`, is at most 24 hours old, and shares at least one sender/receiver account with it. At most the eight most recent qualifying neighbours send messages to a node. Neighbours are deduplicated; ties among earlier events use chronological transaction-ID order. Same-timestamp transactions never connect to each other.

This is a transaction-neighbourhood graph. Its edges represent information flow, not the direction of money. The account graph in the Investigation tab still shows sender-to-receiver money transfers. Sharing an account is not sufficient evidence that transactions belong to a laundering operation.

Version 0.2.1 converts timestamps to nanoseconds before applying the window. Equivalent timestamps supplied at second, millisecond, microsecond or nanosecond resolution now produce the same edges, including the exact 24-hour boundary.

For the default seed-42 experiment the graph has 26,327 transaction nodes and 73,433 temporal edges.

## Architecture

For node `i` in each layer:

```text
neighbours_i = mean(h_j for j in earlier_neighbours(i))
h_i_next = ReLU(W · concat(h_i, neighbours_i) + b)
```

An empty neighbourhood contributes a zero vector. Two independently learned layers use 32 hidden dimensions. The classifier concatenates the second-layer representation with the original normalised node features, applies a 32-unit ReLU layer, and outputs a single logit. A sigmoid produces the uncalibrated review score. Aggregation is implemented using PyTorch `index_add_`; no precomputed neighbour averages replace the learned message-passing layers.

The mean aggregator is based on [GraphSAGE](https://arxiv.org/abs/1706.02216). This implementation uses deterministic recent-neighbour truncation, temporal directed edges, a skip connection to input features, and supervised transaction labels. It is not a reproduction of the paper's full training/benchmark setup.

## Temporal safeguards

- Graph edges always point from earlier to later timestamps; no undirected or reverse-edge expansion is performed.
- Features only use transaction history available before each node's timestamp.
- Train/validation/test use the existing whole-day 60/20/20 split.
- Training optimisation physically uses only the training graph and training labels.
- Counts, ratios and lag features receive a log transform. Mean and scale use training rows only; transformed values are clipped to [-5, 5].
- The validation prefix is used to choose a checkpoint and review threshold. Test nodes and labels are absent from checkpoint selection.
- Earlier validation and test transactions may be unlabelled neighbours of later transactions during final scoring. This reflects sequential availability, not target-label propagation.

Tests change future test labels and confirm that learned weights and the threshold stay identical. A separate test appends future nodes and confirms earlier scores remain unchanged. A controlled two-edge chain verifies actual two-hop message propagation, and removing graph edges changes the trained network's scores.

## Training

CPU; seed 42; up to 40 epochs; AdamW with learning rate 0.003 and weight decay 0.0001; positive-class-weighted binary cross entropy; gradient norm clipped at 5. Validation average precision selects the best state, with patience 8. The default run selected epoch 40. No test-set tuning was performed.

```bash
python -m aml demo --gnn --gnn-epochs 40 --seed 42 --output artifacts
```

The forest, rules and GNN each get a validation threshold constrained by the same 5% alert budget. Ranking precision at the top 5% additionally provides a fixed-capacity comparison. All three are measured on the same 5,242 test transactions, including 378 injected positive transfers.

## Results and interpretation

| Detector | Average precision | Threshold precision | Threshold recall | Alert rate | Precision at top 5% |
|---|---:|---:|---:|---:|---:|
| Random forest | 0.610 | 72.4% | 54.2% | 5.4% | 73.3% |
| GraphSAGE | 0.457 | 53.8% | 35.4% | 4.8% | 54.6% |
| Rules | 0.254 | 48.6% | 4.8% | 0.7% | 45.4% |

The GNN exceeds the simple rule baseline in this run but underperforms the random forest. A possible explanation is that the engineered features already expose useful signals, while shared-account neighbourhoods mix relevant and irrelevant activity. This is a hypothesis, not a measured causal explanation. A proper architecture ablation and independent benchmark would be needed to establish why.

The node colours and displayed graph show context and model scores. They are not attention weights, learned feature attributions, a predicted criminal organisation, or a guarantee that a transaction is illicit.

## Saving and scoring

Training writes `graphsage.pt` with model weights, normalisation tensors, feature names, graph settings and threshold. Loading uses `torch.load(..., weights_only=True, map_location="cpu")` and checks the schema. The archive ships source and example results, not pretrained binary checkpoints.

Checkpoints exported by 0.2.1 also include the experiment's `run_id`, shared with the forest and metrics report. Combined CLI scoring requires matching, non-empty IDs. Retrain legacy checkpoint pairs together to obtain these identifiers. Baseline-only exports remove previous generated GNN outputs from the selected directory, so a stale network cannot be silently reused alongside a new forest.

```bash
python -m aml score --input new_transactions.csv --history earlier_transactions.csv --model artifacts/model.joblib --gnn-model artifacts/graphsage.pt --output scores.csv
```

The output includes forest scores/alerts and `gnn_score`/`gnn_alert`. Full earlier history reproduces the replay. The GNN's effective raw-data context can span 72 hours: two 24-hour graph hops plus 24-hour engineered features at the outer neighbours. Shorter history can cause cold-start differences. The two-hop graph itself spans at most 48 hours; a window per edge is not the whole receptive field.

## Scope and extensions

This module predicts transaction labels on the existing synthetic AML generator. It does not add real credit-card records, merchant/device relationships or card-fraud labels. Applying it to card fraud would require a dataset with stable relationship identifiers, a suitable graph schema and fresh evaluation; anonymised feature-only card datasets cannot directly provide those links.

Potential next experiments: distinguish money-flow and shared-account edge types, include time/amount information on edges, train a feature-only MLP under the same budget, compare against a relational GNN, evaluate unseen accounts, and test on an external AML benchmark. Select these using training/validation data and reserve a new test period for final measurement.
