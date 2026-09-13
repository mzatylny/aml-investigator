import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from aml.data import generate_transactions, validate_transactions
from aml.features import FEATURES, build_features
from aml.model import alert_mask, chronological_split, evaluate, run_experiment, score_transactions, threshold_for_budget
from aml.plots import network_figure


def transfers(edges):
    return pd.DataFrame([
        {"transaction_id": f"T{i}", "timestamp": time, "sender": source,
         "receiver": destination, "amount": amount, "currency": "EUR"}
        for i, (time, source, destination, amount) in enumerate(edges)
    ])


@pytest.fixture(scope="module")
def data():
    return generate_transactions(seed=7, days=15, accounts=60, daily_transactions=40)


@pytest.fixture(scope="module")
def experiment(data):
    return run_experiment(data, seed=7)


def test_generator_is_reproducible_and_has_benign_lookalikes(data):
    assert_frame_equal(data, generate_transactions(seed=7, days=15, accounts=60, daily_transactions=40))
    assert len(data.pattern.unique()) == 9
    assert data.transaction_id.is_unique
    assert not data.sender.eq(data.receiver).any()
    assert data.timestamp.is_monotonic_increasing


def test_features_cannot_see_future_or_labels(data):
    prefix = data.iloc[:120].copy()
    _, expected = build_features(prefix)
    changed = data.copy()
    changed.loc[120:, "amount"] *= 10000
    changed["is_laundering"] = 1 - changed.is_laundering
    changed["scenario_id"] = "ignore-this"
    changed["pattern"] = "also-ignored"
    _, actual = build_features(changed)
    assert_frame_equal(expected, actual.iloc[:120].reset_index(drop=True))
    assert set(FEATURES).isdisjoint({"is_laundering", "pattern", "scenario_id", "sender", "receiver", "transaction_id"})


def test_simultaneous_events_cannot_see_each_other():
    frame = transfers([
        ("2025-01-01T12:00:00Z", "A", "B", 100),
        ("2025-01-01T12:00:00Z", "B", "A", 100),
    ])
    _, features = build_features(frame)
    assert features.sender_in_count_24h.eq(0).all()
    assert features.closes_cycle_24h.eq(0).all()
    _, shuffled = build_features(frame.iloc[::-1])
    assert_frame_equal(features, shuffled)


def test_cycle_and_window_expiry_use_only_available_edges():
    frame = transfers([
        ("2025-01-01T12:00:00Z", "A", "B", 100),
        ("2025-01-01T12:10:00Z", "B", "C", 95),
        ("2025-01-01T12:20:00Z", "C", "A", 90),
        ("2025-01-02T12:21:00Z", "C", "A", 90),
    ])
    _, features = build_features(frame)
    assert features.closes_cycle_24h.tolist() == [0, 0, 1, 0]
    assert features.minutes_since_inflow.iloc[2] == 10
    assert features.sender_in_count_24h.iloc[3] == 0


@pytest.mark.parametrize("column,value", [("amount", -1), ("amount", np.inf), ("timestamp", "invalid"),
                                         ("sender", ""), ("currency", "USD"), ("is_laundering", 2)])
def test_invalid_inputs_rejected(data, column, value):
    frame = data.head(3).copy()
    if column == "timestamp":
        frame[column] = frame[column].astype(str)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_transactions(frame)


def test_duplicate_ids_and_empty_input_rejected(data):
    with pytest.raises(ValueError, match="unique"):
        validate_transactions(pd.concat([data.head(1)] * 2))
    with pytest.raises(ValueError, match="empty"):
        validate_transactions(data.head(0))


def test_threshold_respects_budget_with_ties():
    scores = np.array([0.1] * 90 + [0.8] * 6 + [0.9] * 4)
    threshold = threshold_for_budget(scores, .05)
    assert threshold == .9
    assert (scores >= threshold).sum() == 4
    assert threshold_for_budget(np.ones(20), .05) > 1


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("value", [.7, 1.0])
def test_tied_scores_cannot_exceed_budget_in_any_comparison(dtype, value):
    scores = np.full(20, value, dtype=dtype)
    threshold = threshold_for_budget(scores, .05)
    assert not (scores >= threshold).any()
    assert not pd.Series(scores).ge(threshold).any()
    assert not alert_mask(scores, threshold).any()
    assert evaluate([1] + [0] * 19, scores, threshold)["alerts"] == 0


def test_float32_budget_smaller_than_one_review_slot():
    scores = np.array([.6, .7], dtype=np.float32)
    threshold = threshold_for_budget(scores, .05)
    assert not (scores >= threshold).any()


def test_existing_float64_threshold_is_applied_consistently_to_float32_scores():
    scores = np.full(20, .7, dtype=np.float32)
    old_threshold = float(np.nextafter(float(scores.max()), np.inf))
    assert not alert_mask(scores, old_threshold).any()
    assert evaluate([1] + [0] * 19, scores, old_threshold)["alerts"] == 0


def test_split_is_chronological_and_scenarios_do_not_cross(data):
    tx = validate_transactions(data)
    split = chronological_split(tx)
    groups = [tx[split.eq(name)] for name in ("train", "validation", "test")]
    assert groups[0].timestamp.max() < groups[1].timestamp.min()
    assert groups[1].timestamp.max() < groups[2].timestamp.min()
    scenarios = [set(group.scenario_id) - {""} for group in groups]
    assert not scenarios[0] & scenarios[1] and not scenarios[1] & scenarios[2]


def test_evaluation_counts_known_errors():
    result = evaluate([1, 1, 0, 0], [.9, .2, .8, .1], .5)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)
    assert result["precision"] == result["recall"] == .5


def test_inference_with_history_matches_full_replay(experiment):
    tx = experiment.transactions
    raw = tx.drop(columns=["split", "risk_score", "rule_score", "alert", "evidence"])
    current = raw.iloc[500:].drop(columns=["is_laundering", "pattern", "scenario_id"])
    scored = score_transactions(experiment.model, current, experiment.threshold, raw.iloc[:500])
    np.testing.assert_allclose(scored.risk_score, tx.iloc[500:].risk_score)
    assert len(scored) == len(current)
    assert "is_laundering" not in scored
    with pytest.raises(ValueError, match="strictly before"):
        score_transactions(experiment.model, current, experiment.threshold, raw)


def test_model_and_rules_are_evaluated_on_same_holdout(experiment):
    report = experiment.report["test"]
    assert report["random_forest"]["transactions"] == report["rules"]["transactions"]
    validation = experiment.transactions.query("split == 'validation'")
    assert validation.alert.mean() <= .05
    assert np.isfinite(experiment.features.to_numpy()).all()


def test_network_excludes_future_transfers(experiment):
    tx = experiment.transactions
    selected = tx.iloc[800]
    figure, context = network_figure(tx, selected.transaction_id)
    assert context.timestamp.max() <= selected.timestamp
    assert selected.transaction_id in set(context.transaction_id)
    assert figure.data
