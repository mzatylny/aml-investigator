"""Train on the past, select an alert budget on validation, evaluate once on the future."""

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from .data import validate_transactions
from .features import FEATURES, build_features, evidence, rule_scores


@dataclass
class Experiment:
    transactions: pd.DataFrame
    features: pd.DataFrame
    model: RandomForestClassifier
    report: dict
    threshold: float
    gnn: object = None
    graph_edges: object = None


def chronological_split(tx):
    days = tx.timestamp.dt.floor("D")
    dates = sorted(days.unique())
    if len(dates) < 10:
        raise ValueError("Training requires at least 10 distinct calendar days.")
    train_end, validation_end = dates[int(len(dates) * 0.6)], dates[int(len(dates) * 0.8)]
    return pd.Series(np.where(days < train_end, "train", np.where(days < validation_end, "validation", "test")))


def threshold_for_budget(scores, budget=0.05):
    """Most inclusive threshold with <= budget alerts on validation, including ties."""
    scores = np.asarray(scores, dtype=float)
    if not 0 < budget <= 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Budget must be in (0, 1] and scores finite and non-empty.")
    permitted = math.floor(len(scores) * budget)
    values, counts = np.unique(scores, return_counts=True)
    totals = np.cumsum(counts[::-1])
    allowed = values[::-1][totals <= permitted]
    return float(allowed[-1]) if len(allowed) else float(np.nextafter(scores.max(), np.inf))


def evaluate(y, scores, threshold, budget=0.05):
    y, scores = np.asarray(y, dtype=int), np.asarray(scores, dtype=float)
    predicted = scores >= threshold
    tp, fp = int(np.sum(predicted & (y == 1))), int(np.sum(predicted & (y == 0)))
    fn, tn = int(np.sum(~predicted & (y == 1))), int(np.sum(~predicted & (y == 0)))
    k = max(1, math.floor(len(y) * budget))
    top = np.argsort(-scores, kind="stable")[:k]
    return {
        "transactions": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, scores)) if y.sum() else None,
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "alert_rate": float(predicted.mean()), "alerts": int(predicted.sum()),
        "precision_at_budget": float(y[top].mean()), "budget_k": k,
        "false_positives_per_1000": fp / len(y) * 1000,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "threshold": threshold,
    }


def run_experiment(transactions, seed=42, budget=0.05, with_gnn=False, gnn_epochs=40):
    validated = validate_transactions(transactions, require_labels=True)
    tx, features = build_features(validated)
    tx["split"] = chronological_split(tx)
    train, validation, test = [tx.split.eq(name) for name in ("train", "validation", "test")]
    for name, mask in (("train", train), ("validation", validation), ("test", test)):
        if tx.loc[mask, "is_laundering"].nunique() != 2:
            raise ValueError(f"The {name} split must contain both classes.")
    model = RandomForestClassifier(
        n_estimators=160, max_depth=10, min_samples_leaf=12,
        class_weight="balanced_subsample", random_state=seed, n_jobs=1,
    )
    model.fit(features.loc[train, FEATURES], tx.loc[train, "is_laundering"])
    tx["risk_score"] = model.predict_proba(features[FEATURES])[:, 1]
    tx["rule_score"] = rule_scores(features)
    threshold = threshold_for_budget(tx.loc[validation, "risk_score"], budget)
    rule_threshold = threshold_for_budget(tx.loc[validation, "rule_score"], budget)
    tx["alert"] = tx.risk_score.ge(threshold)
    tx["evidence"] = evidence(features)
    report = {
        "seed": seed, "data": "synthetic", "validation_alert_budget": budget,
        "features": FEATURES, "model_parameters": model.get_params(),
        "splits": {
            name: {"rows": int(mask.sum()), "start": tx.loc[mask, "timestamp"].min().isoformat(),
                   "end": tx.loc[mask, "timestamp"].max().isoformat()}
            for name, mask in (("train", train), ("validation", validation), ("test", test))
        },
        "test": {
            "random_forest": evaluate(tx.loc[test, "is_laundering"], tx.loc[test, "risk_score"], threshold, budget),
            "rules": evaluate(tx.loc[test, "is_laundering"], tx.loc[test, "rule_score"], rule_threshold, budget),
        },
    }
    if "pattern" in tx:
        positives = tx[test & tx.is_laundering.eq(1)]
        report["recall_by_pattern"] = {
            name: {"transactions": len(group), "recall": float(group.alert.mean())}
            for name, group in positives.groupby("pattern")
        }
    experiment = Experiment(tx, features, model, report, threshold)
    if with_gnn:
        from .gnn import train_graphsage
        bundle, scores, edges, metadata = train_graphsage(tx, features, seed=seed, epochs=gnn_epochs)
        bundle.threshold = threshold_for_budget(scores[validation], budget)
        tx["gnn_score"] = scores
        tx["gnn_alert"] = tx.gnn_score.ge(bundle.threshold)
        report["gnn"] = metadata
        report["test"]["graphsage"] = evaluate(tx.loc[test, "is_laundering"], scores[test], bundle.threshold, budget)
        if "pattern" in tx:
            report["gnn_recall_by_pattern"] = {
                name: {"transactions": len(group), "recall": float(group.gnn_alert.mean())}
                for name, group in tx[test & tx.is_laundering.eq(1)].groupby("pattern")
            }
        experiment.gnn, experiment.graph_edges = bundle, edges
    return experiment


def score_transactions(model, transactions, threshold, history=None):
    current = validate_transactions(transactions)
    ids = set(current.transaction_id)
    if history is not None:
        past = validate_transactions(history)
        if past.timestamp.max() >= current.timestamp.min():
            raise ValueError("History must end strictly before the first transaction to score.")
        if ids.intersection(past.transaction_id):
            raise ValueError("History and new transactions must have distinct IDs.")
        current = pd.concat([past, current], ignore_index=True)
    # Labels are never necessary for inference, including mixed labelled history.
    current = current.drop(columns=["is_laundering", "pattern", "scenario_id"], errors="ignore")
    tx, features = build_features(current)
    tx["risk_score"] = model.predict_proba(features[FEATURES])[:, 1]
    tx["alert"] = tx.risk_score.ge(threshold)
    tx["rule_score"] = rule_scores(features)
    tx["evidence"] = evidence(features)
    return tx[tx.transaction_id.isin(ids)].reset_index(drop=True)


def precision_recall_data(experiment):
    test = experiment.transactions.query("split == 'test'")
    rows = []
    detectors = [("Random forest", "risk_score"), ("Rules", "rule_score")]
    if "gnn_score" in test:
        detectors.append(("GraphSAGE", "gnn_score"))
    for label, column in detectors:
        precision, recall, _ = precision_recall_curve(test.is_laundering, test[column])
        rows.extend({"model": label, "precision": float(p), "recall": float(r)} for p, r in zip(precision, recall))
    return pd.DataFrame(rows)
