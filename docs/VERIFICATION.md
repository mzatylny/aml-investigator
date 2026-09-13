# Verification — version 0.2.1

Validated on Python 3.11.14 with the versions pinned in requirements.txt, including PyTorch 2.11.0 on CPU.

- 44 automated tests passed, including GraphSAGE propagation, strict temporal edges, simultaneous timestamps, neighbourhood caps, prefix-invariant predictions, training-only normalisation, independence from test labels and checkpoint round trips.
- Review regressions verify tied-score budgets for float32/float64, consistency between saved flags and reports, exact graph-window boundaries across four timestamp resolutions, cleanup of stale GNN outputs, and rejection of mismatched checkpoint pairs.
- Dashboard tests exercised threshold changes, switching between GraphSAGE and Random Forest, disabling GraphSAGE, and showing synthetic labels.
- A saturated-score dashboard test verifies that the no-alert policy stays disabled until the user lowers the threshold.
- The complete seed-42, 90-day experiment trained all three detectors and exported their comparison, GNN training history, checkpoints and offline network chart.
- CLI scoring of 100 new transactions with earlier history reproduced both forest and GraphSAGE full-replay scores (absolute tolerance 1e-6). GNN alert flags matched exactly.
- The graph chart was constructed and exported successfully. Browser rendering was not visually inspected; the earlier environment restricted local-server port binding.

Re-run tests with `python -m pytest -q` after installing requirements-dev.txt.
Reproduce the full experiment with `python -m aml demo --gnn --seed 42`.
The archive contains code and example results. It excludes environments, caches,
complete generated experiment directories and binary model checkpoints. Models are
trained on the first dashboard load or via the command-line experiment.
