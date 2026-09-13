# Synthetic data and model card

## Origin

`aml/data.py` is an original simulator included in this project. No personal records, real account numbers, banking datasets or IBM source code are embedded. It generates EUR-denominated transfers starting on 1 January 2025 UTC. Account and transaction identifiers are fictitious.

The default configuration has 500 accounts, 90 days and a Poisson mean of 250 background transfers per day. Each day also contains four labelled motifs and four legitimate lookalikes. The positive rate is a consequence of that construction; it is not intended to match real AML prevalence.

## Patterns

| Pattern | Construction |
|---|---|
| Fan-in | Five accounts send to one account, followed by an onward transfer |
| Fan-out | One account receives funds, then sends to five accounts |
| Cycle | Four transfers form a directed loop |
| Chain | Five successive transfers pass funds through six accounts |

Legitimate lookalikes use the same motif structures, account pool, base amount distribution, currency and start-time range. Their transfer spacing has an overlapping but wider range. Background activity includes repeated counterparties and variable amounts. This creates false positives and ambiguity, but does not establish realism. Labels are assigned to all transfers in injected positive motifs, including early transfers that may be impossible to distinguish from normal activity at that time.

The simulator does not enforce account balances or conservation of funds, except approximate amounts within individual motifs. It does not model foreign exchange, KYC information, institutional account roles, investigation outcomes, seasonality, real reporting obligations or adaptive behaviour. It is intended for testing an ML workflow.

## Feature availability

The model sees 13 numeric features. Rolling aggregates and cycle closure use only timestamps strictly before the scored transfer; the active window includes exactly 24 hours of history. A cycle feature searches for an existing path from receiver to sender of at most three edges. It therefore detects closure of directed cycles of length two to four. It does not enumerate all laundering networks.

Ratios are capped at 100. An account without outgoing history gets an amount-to-mean ratio of 1; missing incoming history gets a lag of 1,440 minutes. These explicit defaults simplify the prototype but can bias early scores. History persists across evaluation split boundaries because earlier unlabelled activity would be available operationally.

Labels, scenario names and identifiers cannot affect feature construction. Simultaneous transfers see the same pre-batch history. The dashboard graph shows a capped local neighbourhood, not a full connected component or an exhaustive investigation.

## Evaluation scope

Split by whole calendar days: first 60% train, next 20% validation, final 20% test. The generator keeps each motif within one day, so synthetic scenarios do not cross the boundaries. Account identities can recur across splits but are excluded as features.

The random forest uses 160 trees, maximum depth 10, minimum leaf size 12 and balanced subsample weights. No hyperparameter search is performed. A validation-only 5% alert budget sets each detector's threshold. Threshold ties are kept intact. Coarse rule scores may leave unused capacity, so the report also includes ranking precision at a fixed budget. Test labels are only used for measurement.

The risk score is an uncalibrated classifier output. Average precision, threshold precision/recall, false positives per 1,000 transactions, alert rate and per-pattern recall quantify this synthetic experiment only. No independent AML dataset, real deployment, fairness evaluation, adversarial robustness validation or regulatory certification has been performed.

## Graph neural network extension

Version 0.2 adds temporal GraphSAGE without changing the generator or labels. Transactions become nodes connected to up to eight earlier shared-account transactions in a 24-hour window per edge. Two learned mean-aggregation layers use train-only feature normalisation. Validation selects the checkpoint and threshold; test labels do not affect training. See [GNN.md](GNN.md) for architecture, history requirements, exact results and scope.

The GNN does not transform this synthetic transfer benchmark into a credit-card-fraud benchmark. Relationship identifiers and representative labels would be required for that adaptation. It also does not automatically discover or label entire criminal networks; it scores individual transactions with relational context.

## Reproducibility

Seed 42 is the default. Dependencies are pinned in `requirements.txt`; Python 3.11 is the tested runtime. Reproduce all three detectors with `python -m aml demo --gnn --seed 42`. Omit `--gnn` for the original forest/rule experiment. Compare the split boundaries and row counts in `metrics.json` as well as metrics. Minor floating-point differences may occur on different systems.
